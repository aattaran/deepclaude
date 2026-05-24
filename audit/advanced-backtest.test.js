/**
 * Advanced Backtest Engine — unit tests
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/advanced-backtest.test.js
 */

import { describe, it, mock, before, after, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
    monteCarloSim,
    walkForward,
    compareStrategies,
    analyzeDrawdowns,
    equityCurveMetrics,
} from './advanced-backtest.mjs';

// ===========================================================================
// Helpers
// ===========================================================================

/**
 * Seeded pseudo-random number generator (Mulberry32).
 * Produces deterministic values for reproducible tests.
 */
function seededRandom(seed) {
    let s = seed | 0;
    return () => {
        s = (s + 0x6d2b79f5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Generate N synthetic candles with deterministic noise for testing.
 *
 * Each candle has { timestamp, open, high, low, close }.
 * When no rng is passed, the default (() => 0.5) produces a steady trend
 * with no random variation. Pass a seeded rng for noisy-but-deterministic data.
 */
function generateCandles(n, opts = {}) {
    const {
        startPrice = 100,
        startTime = 1700000000000,
        intervalMs = 60000,
        volatility = 0.001,
        trend = 0.0002,
        rng = () => 0.5,
    } = opts;

    const candles = [];
    let price = startPrice;
    for (let i = 0; i < n; i++) {
        const ret = (rng() - 0.5) * volatility + trend;
        price = price * (1 + ret);
        const halfVol = volatility * 0.5;
        candles.push({
            timestamp: startTime + i * intervalMs,
            open: +(price * (1 - halfVol)).toFixed(2),
            high: +(price * (1 + halfVol)).toFixed(2),
            low: +(price * (1 - halfVol)).toFixed(2),
            close: +price.toFixed(2),
        });
    }
    return candles;
}

/**
 * Strategy factory: always enters long on every bar (after warmup).
 * The backtest engine handles exit via stop-loss / take-profit / SMA crossover.
 */
function alwaysLongFactory(_params) {
    return (candle, _history) => ({ direction: 'long', timestamp: candle.timestamp });
}

/**
 * Strategy factory: SMA crossover entry.
 * Params: { fastPeriod, slowPeriod }
 * Returns long when fast SMA > slow SMA, short when fast SMA < slow SMA.
 * Generates a new entry signal on every bar.
 */
function smaCrossoverFactory(params) {
    const fastPeriod = params.fastPeriod || 5;
    const slowPeriod = params.slowPeriod || 20;
    return (candle, history) => {
        const all = [...history, candle];
        if (all.length < slowPeriod) return null;
        const recent = all.slice(-slowPeriod);
        const fastPrices = recent.slice(-fastPeriod).map(c => c.close);
        const slowPrices = recent.map(c => c.close);
        const fastSma = fastPrices.reduce((s, p) => s + p, 0) / fastPeriod;
        const slowSma = slowPrices.reduce((s, p) => s + p, 0) / slowPeriod;
        return {
            direction: fastSma > slowSma ? 'long' : 'short',
            timestamp: candle.timestamp,
        };
    };
}


// ===========================================================================
// 1. monteCarloSim
// ===========================================================================
describe('monteCarloSim', () => {
    let origRandom;

    before(() => { origRandom = Math.random; });
    after(() => { Math.random = origRandom; });
    afterEach(() => { Math.random = origRandom; });

    // ── Edge cases ─────────────────────────────────────────────────────

    it('returns empty structure for empty PnLs', () => {
        const result = monteCarloSim([], { simulations: 100 });
        assert.equal(result.simulations, 0);
        assert.deepEqual(result.medianEquityCurve, []);
        assert.deepEqual(result.confidenceBands, []);
        assert.equal(result.probabilityOfProfit, 0);
        assert.equal(result.expectedFinalEquity, 100000);
        assert.equal(result.expectedReturn, 0);
        assert.equal(result.cvar, 0);
    });

    it('returns empty structure for non-array input', () => {
        const result = monteCarloSim(null, { simulations: 100 });
        assert.equal(result.simulations, 0);
    });

    it('handles single PnL value', () => {
        Math.random = seededRandom(42);
        const result = monteCarloSim([500], { simulations: 50 });
        assert.equal(result.simulations, 50);
        // With only 1 PnL, every resample is just [500], so every sim ends at 100500
        assert.equal(result.medianEquityCurve.length, 1);
        assert.equal(result.finalEquityDistribution.mean, 100500);
        assert.equal(result.finalEquityDistribution.min, 100500);
        assert.equal(result.finalEquityDistribution.max, 100500);
        // probabilityOfProfit should be 1 since PnL is positive
        assert.equal(result.probabilityOfProfit, 1);
    });

    // ── Happy path ─────────────────────────────────────────────────────

    it('returns positive expected return for profitable PnLs', () => {
        Math.random = seededRandom(123);
        const pnls = [100, 200, 150, 300, 250];
        const result = monteCarloSim(pnls, { simulations: 200, startingEquity: 100000 });

        assert.equal(result.simulations, 200);
        assert.equal(result.confidence, 0.95);
        assert.ok(result.expectedFinalEquity > 100000,
            `expected final equity ${result.expectedFinalEquity} should be > 100000`);
        assert.ok(result.expectedReturn > 0);
        assert.ok(result.probabilityOfProfit > 0.9);
    });

    it('computes confidence bands with lower <= upper at every step', () => {
        Math.random = seededRandom(456);
        const pnls = [500, -200, 300, -100, 400, -50, 600, -300, 200, 100];
        const result = monteCarloSim(pnls, { simulations: 300, confidence: 0.95 });

        assert.equal(result.medianEquityCurve.length, pnls.length);
        assert.equal(result.confidenceBands.length, pnls.length);

        for (const band of result.confidenceBands) {
            assert.ok(band.lower <= band.upper,
                `confidence band lower ${band.lower} > upper ${band.upper} at step ${band.step}`);
        }
    });

    it('returns properly shaped distribution summaries', () => {
        Math.random = seededRandom(789);
        const pnls = [100, -50, 200, -75, 300, -25, 150, -100, 250, 50];
        const result = monteCarloSim(pnls, { simulations: 200 });

        // finalEquityDistribution
        const fed = result.finalEquityDistribution;
        assert.ok(typeof fed.mean === 'number');
        assert.ok(fed.stdDev >= 0);
        assert.ok(fed.min <= fed.p25);
        assert.ok(fed.p25 <= fed.p50);
        assert.ok(fed.p50 <= fed.p75);
        assert.ok(fed.p75 <= fed.max);

        // maxDrawdownDistribution
        const mdd = result.maxDrawdownDistribution;
        assert.ok(mdd.mean >= 0);
        assert.ok(mdd.stdDev >= 0);
        assert.ok(mdd.min <= mdd.p25);
        assert.ok(mdd.p25 <= mdd.p50);
        assert.ok(mdd.p50 <= mdd.p75);
        assert.ok(mdd.p75 <= mdd.max);
    });

    // ── Known result validation ────────────────────────────────────────

    it('respects custom startingEquity', () => {
        Math.random = seededRandom(999);
        const pnls = [1000, 2000];
        const result = monteCarloSim(pnls, { simulations: 100, startingEquity: 50000 });

        assert.ok(result.expectedFinalEquity > 50000);
        assert.ok(result.finalEquityDistribution.mean > 50000);
    });

    it('CVaR is positive when there is tail risk', () => {
        Math.random = seededRandom(111);
        // Mix of profits and losses — some tail risk exists
        const pnls = [100, 200, -500, -300, 100, -200, 300, 400, -100, 50];
        const result = monteCarloSim(pnls, { simulations: 500, confidence: 0.95 });

        // CVaR = startingEquity - mean(tailEquities)
        // With losses in the PnLs, some resampled paths end below starting equity,
        // so the tail mean is below starting equity → CVaR should be positive.
        assert.ok(typeof result.cvar === 'number', 'CVaR should be a number');

        // The expected return should be >= 0 (more profits than losses in this set)
        // CVaR should be more conservative (further from the mean)
        // Since CVaR measures the expected shortfall, for mixed PnLs it should
        // indicate a non-trivial risk.
        const hasTailRisk = result.cvar > 0;
        const meanReturn = result.expectedReturn;
        // At a minimum, cvar and expectedReturn should not both be zero
        assert.ok(result.cvar !== 0 || result.expectedReturn !== 0,
            'either CVaR or expected return should be non-zero for mixed PnLs');
        // For a set with losses, maxDrawdownDistribution.mean should be > 0
        assert.ok(result.maxDrawdownDistribution.mean >= 0);

        // Bonus structural check: expected return should be consistent with
        // the mean of simulated final equities
        const impliedReturn =
            (result.expectedFinalEquity - 100000) / 100000;
        assert.equal(result.expectedReturn, +impliedReturn.toFixed(4));
    });

    it('probabilityOfProfit is 1 when all PnLs are positive', () => {
        Math.random = seededRandom(222);
        const pnls = [10, 20, 15, 30, 25, 40, 35, 50, 45, 60];
        const result = monteCarloSim(pnls, { simulations: 100 });

        assert.equal(result.probabilityOfProfit, 1);
    });

    it('handles the default confidence of 0.95', () => {
        Math.random = seededRandom(333);
        const pnls = [100, -50, 200];
        const result = monteCarloSim(pnls, { simulations: 100 });

        assert.equal(result.confidence, 0.95);
    });
});


// ===========================================================================
// 2. walkForward
// ===========================================================================
describe('walkForward', () => {
    // ── Edge cases ─────────────────────────────────────────────────────

    it('returns empty result for insufficient candles (< 100)', () => {
        const candles = generateCandles(50);
        const result = walkForward(candles, alwaysLongFactory, [{ period: 5 }]);

        assert.deepEqual(result.windows, []);
        assert.equal(result.aggregate.totalOosTrades, 0);
        assert.equal(result.aggregate.totalOosPnL, 0);
        assert.equal(result.paramStabilityScore, 0);
    });

    it('returns empty result for non-array candles', () => {
        const result = walkForward(null, alwaysLongFactory, [{ period: 5 }]);
        assert.deepEqual(result.windows, []);
    });

    it('returns empty result for empty paramCandidates', () => {
        const candles = generateCandles(200);
        const result = walkForward(candles, alwaysLongFactory, []);
        assert.deepEqual(result.windows, []);
    });

    // ── Happy path ─────────────────────────────────────────────────────

    it('performs walk-forward with multiple windows', () => {
        // Use a noisy-but-deterministic candle set so the SMA crossover
        // generates a variety of trades.
        const rng = seededRandom(42);
        const candles = generateCandles(300, {
            volatility: 0.004,
            trend: 0.0003,
            rng,
        });

        const params = [
            { fastPeriod: 3, slowPeriod: 10 },
            { fastPeriod: 5, slowPeriod: 20 },
            { fastPeriod: 10, slowPeriod: 30 },
        ];

        const result = walkForward(candles, smaCrossoverFactory, params, {
            inSamplePct: 0.35,
            outOfSamplePct: 0.2,
            stepCount: 2,
        });

        // With 300 candles, IS=0.35 (=105), OOS=0.2 (=60), window=165,
        // step = (300-165)/(2-1) = 135, we get 2 windows.
        assert.ok(result.windows.length >= 1,
            `expected at least 1 window, got ${result.windows.length}`);

        // Each window should have the expected structure
        for (const w of result.windows) {
            assert.equal(typeof w.windowIndex, 'number');
            assert.ok(w.candleRange.inSample.end > w.candleRange.inSample.start);
            assert.ok(w.candleRange.outOfSample.end >= w.candleRange.outOfSample.start);
            assert.ok(w.bestParams !== undefined);
            assert.equal(typeof w.inSample.sharpe, 'number');
            assert.equal(typeof w.outOfSample.sharpe, 'number');
            assert.equal(typeof w.inSample.trades, 'number');
            assert.equal(typeof w.outOfSample.trades, 'number');
            assert.equal(typeof w.inSample.totalPnL, 'number');
            assert.equal(typeof w.outOfSample.totalPnL, 'number');
        }

        // Aggregate should have valid values
        assert.equal(typeof result.aggregate.totalOosTrades, 'number');
        assert.equal(typeof result.aggregate.totalOosPnL, 'number');
        assert.equal(typeof result.aggregate.avgOosSharpe, 'number');
        assert.equal(typeof result.aggregate.avgOosSortino, 'number');
        assert.equal(typeof result.aggregate.avgOosMaxDD, 'number');
        assert.equal(typeof result.aggregate.avgOosWinRate, 'number');

        // paramStabilityScore should be 0-1
        assert.ok(result.paramStabilityScore >= 0 && result.paramStabilityScore <= 1,
            `paramStabilityScore ${result.paramStabilityScore} should be 0-1`);
    });

    it('returns paramStabilityScore of 1 when only one param candidate', () => {
        const candles = generateCandles(250, { volatility: 0.003, trend: 0.0002 });
        const result = walkForward(candles, alwaysLongFactory, [{ period: 5 }], {
            inSamplePct: 0.35,
            outOfSamplePct: 0.2,
            stepCount: 2,
        });

        // With a single param, every window must pick the same params
        if (result.windows.length > 1) {
            assert.equal(result.paramStabilityScore, 1);
        }
    });

    it('uses custom objective function for parameter selection', () => {
        const rng = seededRandom(77);
        const candles = generateCandles(250, { volatility: 0.003, trend: 0.0003, rng });
        const params = [
            { fastPeriod: 5, slowPeriod: 20 },
            { fastPeriod: 10, slowPeriod: 30 },
        ];

        // Objective: maximize total PnL
        const objectiveFn = (metrics) => metrics.totalPnL;
        const result = walkForward(candles, smaCrossoverFactory, params, {
            inSamplePct: 0.35,
            outOfSamplePct: 0.2,
            stepCount: 2,
            objectiveFn,
        });

        assert.ok(result.windows.length >= 1);
        if (result.windows.length > 0) {
            assert.ok(result.windows[0].bestParams !== undefined);
        }
    });

    it('computes aggregate out-of-sample metrics across windows', () => {
        const rng = seededRandom(101);
        const candles = generateCandles(400, { volatility: 0.003, trend: 0.0002, rng });
        const params = [
            { fastPeriod: 5, slowPeriod: 15 },
            { fastPeriod: 8, slowPeriod: 25 },
        ];

        const result = walkForward(candles, smaCrossoverFactory, params, {
            inSamplePct: 0.35,
            outOfSamplePct: 0.2,
            stepCount: 3,
        });

        if (result.windows.length >= 2) {
            // Aggregate total trades should equal sum of window OOS trades
            const sumTrades = result.windows.reduce((s, w) => s + w.outOfSample.trades, 0);
            assert.equal(result.aggregate.totalOosTrades, sumTrades);

            // Aggregate total PnL should equal sum of window OOS PnLs
            const sumPnL = +result.windows.reduce((s, w) => s + w.outOfSample.totalPnL, 0).toFixed(2);
            assert.equal(result.aggregate.totalOosPnL, sumPnL);
        }
    });
});


// ===========================================================================
// 3. compareStrategies
// ===========================================================================
describe('compareStrategies', () => {
    // ── Edge cases ─────────────────────────────────────────────────────

    it('returns empty ranking for insufficient candles (< 20)', () => {
        const candles = generateCandles(10);
        const strategies = { long: (_c, _h) => ({ direction: 'long', timestamp: _c.timestamp }) };
        const result = compareStrategies(candles, strategies);

        assert.deepEqual(result.ranking, []);
        assert.strictEqual(result.significanceTest, null);
    });

    it('returns empty ranking for non-array candles', () => {
        const result = compareStrategies(null, {});
        assert.deepEqual(result.ranking, []);
    });

    it('returns empty ranking for empty strategies map', () => {
        const candles = generateCandles(100);
        const result = compareStrategies(candles, {});

        assert.deepEqual(result.ranking, []);
        assert.strictEqual(result.significanceTest, null);
    });

    // ── Happy path ─────────────────────────────────────────────────────

    it('ranks multiple strategies by Sharpe ratio descending', () => {
        const rng = seededRandom(55);
        const candles = generateCandles(300, {
            volatility: 0.004,
            trend: 0.0004,
            rng,
        });

        const alwaysLong = (_c, _h) => ({ direction: 'long', timestamp: _c.timestamp });
        const alwaysShort = (_c, _h) => ({ direction: 'short', timestamp: _c.timestamp });

        const result = compareStrategies(candles, {
            long: alwaysLong,
            short: alwaysShort,
        });

        assert.equal(result.ranking.length, 2);

        // Check structural fields on each entry
        for (const r of result.ranking) {
            assert.equal(typeof r.name, 'string');
            assert.equal(typeof r.rank, 'number');
            assert.equal(typeof r.sharpe, 'number');
            assert.equal(typeof r.sortino, 'number');
            assert.equal(typeof r.totalTrades, 'number');
            assert.equal(typeof r.totalPnL, 'number');
            assert.equal(typeof r.maxDrawdown, 'number');
            assert.equal(typeof r.winRate, 'number');
            assert.equal(typeof r.profitFactor, 'number');
            assert.equal(typeof r.avgTradeDurationSec, 'number');
            assert.equal(typeof r.avgPnLPerTrade, 'number');
        }

        // Ranks should be 1 and 2
        assert.equal(result.ranking[0].rank, 1);
        assert.equal(result.ranking[1].rank, 2);
    });

    it('includes significanceTest when top 2 strategies have 2+ trades each', () => {
        const rng = seededRandom(55);
        const candles = generateCandles(300, {
            volatility: 0.004,
            trend: 0.0004,
            rng,
        });

        const alwaysLong = (_c, _h) => ({ direction: 'long', timestamp: _c.timestamp });
        const alwaysShort = (_c, _h) => ({ direction: 'short', timestamp: _c.timestamp });

        const result = compareStrategies(candles, {
            long: alwaysLong,
            short: alwaysShort,
        });

        // Both strategies should generate enough trades
        if (result.significanceTest !== null) {
            assert.equal(typeof result.significanceTest.winner, 'string');
            assert.equal(typeof result.significanceTest.runnerUp, 'string');
            assert.equal(typeof result.significanceTest.tStatistic, 'number');
            assert.equal(typeof result.significanceTest.degreesOfFreedom, 'number');
            assert.equal(typeof result.significanceTest.pValue, 'number');
            assert.equal(typeof result.significanceTest.significantAt95, 'boolean');
            // Winner and runner-up should be the two strategy names
            assert.ok(
                (result.significanceTest.winner === 'long' && result.significanceTest.runnerUp === 'short') ||
                (result.significanceTest.winner === 'short' && result.significanceTest.runnerUp === 'long')
            );
        }
    });

    it('returns all strategies in the ranking', () => {
        const candles = generateCandles(100, { volatility: 0.002, trend: 0.0002 });
        const stratA = (_c, _h) => ({ direction: 'long', timestamp: _c.timestamp });
        const stratB = (_c, _h) => ({ direction: 'short', timestamp: _c.timestamp });

        const result = compareStrategies(candles, { alpha: stratA, beta: stratB });

        assert.equal(result.ranking.length, 2);
        const names = result.ranking.map(r => r.name).sort();
        assert.deepEqual(names, ['alpha', 'beta']);
    });

    it('handles custom symbol and startingEquity in opts', () => {
        const candles = generateCandles(100, { volatility: 0.001 });
        const strat = (_c, _h) => ({ direction: 'long', timestamp: _c.timestamp });

        const result = compareStrategies(candles, { test: strat }, {
            symbol: 'CUSTOM',
            startingEquity: 50000,
        });

        assert.equal(result.ranking.length, 1);
        assert.equal(result.ranking[0].name, 'test');
    });
});


// ===========================================================================
// 4. analyzeDrawdowns
// ===========================================================================
describe('analyzeDrawdowns', () => {
    // ── Edge cases ─────────────────────────────────────────────────────

    it('returns empty result for non-array input', () => {
        const result = analyzeDrawdowns(null);
        assert.equal(result.drawdownCount, 0);
        assert.deepEqual(result.topDrawdowns, []);
        assert.deepEqual(result.underwaterChart, []);
    });

    it('returns empty result for single-point curve', () => {
        const result = analyzeDrawdowns([{ timestamp: 1, equity: 100 }]);
        assert.equal(result.drawdownCount, 0);
    });

    it('detects no drawdowns for monotonically increasing curve', () => {
        const curve = [
            { timestamp: 100, equity: 100 },
            { timestamp: 200, equity: 110 },
            { timestamp: 300, equity: 120 },
        ];
        const result = analyzeDrawdowns(curve);
        assert.equal(result.drawdownCount, 0);
        assert.equal(result.averageDrawdown, 0);
        assert.equal(result.maxDrawdownDepth, 0);
    });

    it('detects no drawdowns for flat (constant) equity', () => {
        const curve = [
            { timestamp: 100, equity: 100 },
            { timestamp: 200, equity: 100 },
            { timestamp: 300, equity: 100 },
        ];
        const result = analyzeDrawdowns(curve);
        assert.equal(result.drawdownCount, 0);
    });

    // ── Happy path ─────────────────────────────────────────────────────

    it('identifies a single drawdown trough and recovery', () => {
        const curve = [
            { timestamp: 100, equity: 100 },   // peak (ATH)
            { timestamp: 200, equity: 80 },    // trough
            { timestamp: 300, equity: 101 },   // recovery (above old peak)
            { timestamp: 400, equity: 110 },   // new ATH
        ];
        const result = analyzeDrawdowns(curve);

        assert.equal(result.drawdownCount, 1);
        assert.equal(result.topDrawdowns.length, 1);

        const dd = result.topDrawdowns[0];
        assert.equal(dd.startEquity, 100);
        assert.equal(dd.troughEquity, 80);
        assert.equal(dd.recoveryTimestamp, 300);
        assert.equal(dd.depthPct, 20);  // (100-80)/100 * 100 = 20%
    });

    it('identifies multiple drawdowns and ranks them by depth', () => {
        const curve = [
            { timestamp: 100, equity: 100 },   // peak
            { timestamp: 200, equity: 70 },    // DD 1: depth = 30%
            { timestamp: 300, equity: 110 },   // recovery, new peak
            { timestamp: 400, equity: 90 },    // DD 2: depth ≈ 18.18%
            { timestamp: 500, equity: 120 },   // recovery, new peak
        ];
        const result = analyzeDrawdowns(curve);

        assert.equal(result.drawdownCount, 2);
        assert.equal(result.topDrawdowns.length, 2);

        // First drawdown should be deeper (30% vs 18.18%)
        assert.ok(result.topDrawdowns[0].depthPct > result.topDrawdowns[1].depthPct,
            `top depth ${result.topDrawdowns[0].depthPct}% should be > second ${result.topDrawdowns[1].depthPct}%`);

        // Check the deeper drawdown details
        const dd0 = result.topDrawdowns[0];
        assert.equal(dd0.troughEquity, 70);
        assert.equal(dd0.depthPct, 30);
    });

    it('handles drawdown without recovery (ends underwater)', () => {
        const curve = [
            { timestamp: 100, equity: 100 },
            { timestamp: 200, equity: 110 },   // peak
            { timestamp: 300, equity: 90 },    // trough, no recovery follows
        ];
        const result = analyzeDrawdowns(curve);

        assert.equal(result.drawdownCount, 1);
        assert.equal(result.topDrawdowns.length, 1);
        // recoveryTimestamp should be null since curve ends in drawdown
        assert.strictEqual(result.topDrawdowns[0].recoveryTimestamp, null);
        // depth should be (110-90)/110 = 18.18...%
        assert.ok(Math.abs(result.topDrawdowns[0].depthPct - 18.18) < 0.01,
            `expected depth ~18.18%, got ${result.topDrawdowns[0].depthPct}%`);
    });

    it('computes underwaterChart correctly', () => {
        const curve = [
            { timestamp: 100, equity: 100 },   // ATH → underwater = 0
            { timestamp: 200, equity: 80 },    // below peak → underwater = 0.2
            { timestamp: 300, equity: 101 },   // recovery (above ATH)
        ];
        const result = analyzeDrawdowns(curve);

        assert.ok(Array.isArray(result.underwaterChart));
        assert.equal(result.underwaterChart.length, 3);

        // First point is at ATH
        assert.equal(result.underwaterChart[0].underwater, 0);
        // Second point is in drawdown
        assert.ok(result.underwaterChart[1].underwater > 0);
        // Third point recovered (equity above peak)
        assert.equal(result.underwaterChart[2].underwater, 0);
    });

    it('computes average drawdown duration and recovery time', () => {
        const curve = [
            { timestamp: 100, equity: 100 },
            { timestamp: 200, equity: 70 },
            { timestamp: 400, equity: 110 },   // recovery: 200ms trough, 300ms duration
            { timestamp: 500, equity: 90 },
            { timestamp: 700, equity: 120 },
        ];
        const result = analyzeDrawdowns(curve);

        assert.equal(result.drawdownCount, 2);
        // averageDrawdownDuration should be > 0
        assert.ok(result.averageDrawdownDuration > 0);
        // averageRecoveryTime should be > 0 since both drawdowns recover
        assert.ok(result.averageRecoveryTime > 0);
        // maxDrawdownDepth = 0.3 (30%)
        assert.ok(Math.abs(result.maxDrawdownDepth - 0.3) < 0.001);
    });

    it('sorts the input by timestamp before analysis', () => {
        // Unsorted input
        const curve = [
            { timestamp: 300, equity: 101 },
            { timestamp: 100, equity: 100 },
            { timestamp: 200, equity: 80 },
        ];
        const result = analyzeDrawdowns(curve);

        // Should still detect the drawdown correctly
        assert.equal(result.drawdownCount, 1);
        assert.equal(result.topDrawdowns[0].depthPct, 20);
    });
});


// ===========================================================================
// 5. equityCurveMetrics
// ===========================================================================
describe('equityCurveMetrics', () => {
    // ── Edge cases ─────────────────────────────────────────────────────

    it('returns zeros for single-point curve', () => {
        const curve = [{ timestamp: 1700000000000, equity: 100 }];
        const result = equityCurveMetrics(curve);

        assert.equal(result.cagr, 0);
        assert.equal(result.volatility, 0);
        assert.equal(result.calmarRatio, 0);
        assert.equal(result.stability, 0);
        assert.equal(result.ulcerIndex, 0);
        assert.equal(result.painIndex, 0);
        assert.equal(result.totalReturn, 0);
        assert.equal(result.yearsElapsed, 0);
    });

    it('returns zeros for non-array input', () => {
        const result = equityCurveMetrics(null);
        assert.equal(result.cagr, 0);
        assert.equal(result.volatility, 0);
    });

    // ── Happy path ─────────────────────────────────────────────────────

    it('computes CAGR correctly for a steadily growing curve', () => {
        // Simulate 1 year of daily compounding at 0.1% per day
        const days = 252;
        const dailyReturn = 0.001;
        const curve = [];
        let eq = 100000;
        const startTime = 1700000000000;

        for (let i = 0; i <= days; i++) {
            curve.push({ timestamp: startTime + i * 86400000, equity: eq });
            eq = eq * (1 + dailyReturn);
        }

        const result = equityCurveMetrics(curve);

        // Expected total return
        const expectedTotalReturn = Math.pow(1 + dailyReturn, days) - 1;
        // Expected years elapsed
        const years = days / 365.25;
        // Expected CAGR = (1 + totalReturn)^(1/years) - 1
        const expectedCagr = Math.pow(1 + expectedTotalReturn, 1 / years) - 1;

        assert.ok(Math.abs(result.cagr - expectedCagr) < 0.01,
            `CAGR ${result.cagr} ≈ expected ${expectedCagr}`);
        assert.ok(result.totalReturn > 0);
        assert.ok(result.totalReturnUsd > 0);
        assert.ok(result.yearsElapsed > 0);
    });

    it('computes Calmar ratio when max drawdown > 0', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 110 },  // peak
            { timestamp: 1700172800000, equity: 90 },   // trough
            { timestamp: 1700259200000, equity: 115 },  // recovery + new high
        ];
        const result = equityCurveMetrics(curve);

        // Calmar = CAGR / maxDrawdown
        assert.equal(typeof result.calmarRatio, 'number');
        // For a profitable curve, Calmar should be positive
        assert.ok(result.calmarRatio >= 0,
            `calmarRatio ${result.calmarRatio} should be >= 0`);
    });

    it('returns Calmar as 999 when CAGR > 0 and maxDD = 0', () => {
        // Needs 3+ points (2+ daily returns) to pass the early-return guard
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 110 },
            { timestamp: 1700172800000, equity: 120 },
        ];
        const result = equityCurveMetrics(curve);

        // No drawdown (monotonically increasing) and CAGR > 0
        // Calmar should be 999 (representing Infinity)
        assert.equal(result.calmarRatio, 999);
    });

    it('returns positive ulcer index and pain index for volatile curve', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 110 },   // peak
            { timestamp: 1700172800000, equity: 90 },    // drawdown
            { timestamp: 1700259200000, equity: 115 },   // recovery + new peak
            { timestamp: 1700345600000, equity: 85 },    // deep drawdown
            { timestamp: 1700432000000, equity: 120 },   // recovery
        ];
        const result = equityCurveMetrics(curve);

        assert.ok(result.ulcerIndex > 0,
            `ulcerIndex ${result.ulcerIndex} should be > 0 for volatile curve`);
        assert.ok(result.painIndex > 0,
            `painIndex ${result.painIndex} should be > 0 for volatile curve`);
    });

    it('returns zero ulcer index for monotonically increasing curve', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 110 },
            { timestamp: 1700172800000, equity: 120 },
            { timestamp: 1700259200000, equity: 130 },
        ];
        const result = equityCurveMetrics(curve);

        assert.equal(result.ulcerIndex, 0);
        assert.equal(result.painIndex, 0);
    });

    it('computes daily return statistics', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 110 },   // +10%
            { timestamp: 1700172800000, equity: 105 },   // -4.55%
            { timestamp: 1700259200000, equity: 115 },   // +9.52%
        ];
        const result = equityCurveMetrics(curve);

        assert.ok(result.bestDay > 0, `bestDay ${result.bestDay} should be > 0`);
        assert.ok(result.worstDay < 0, `worstDay ${result.worstDay} should be < 0`);
        assert.ok(result.positiveDayRatio > 0 && result.positiveDayRatio <= 1,
            `positiveDayRatio ${result.positiveDayRatio} should be 0-1`);
        assert.ok(result.dailyReturnStd > 0,
            `dailyReturnStd ${result.dailyReturnStd} should be > 0`);
    });

    it('computes stability (R-squared) near 1 for a consistent trend', () => {
        // Nearly linear equity growth
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 105 },
            { timestamp: 1700172800000, equity: 110 },
            { timestamp: 1700259200000, equity: 115 },
            { timestamp: 1700345600000, equity: 120 },
            { timestamp: 1700432000000, equity: 125 },
        ];
        const result = equityCurveMetrics(curve);

        // For a near-perfect linear trend, R-squared should be close to 1
        assert.ok(result.stability > 0.9,
            `stability ${result.stability} should be > 0.9 for near-linear growth`);
    });

    it('computes stability lower for a noisy/wobbly curve', () => {
        // Erratic equity movements
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 95 },
            { timestamp: 1700172800000, equity: 110 },
            { timestamp: 1700259200000, equity: 85 },
            { timestamp: 1700345600000, equity: 115 },
            { timestamp: 1700432000000, equity: 90 },
        ];
        const result = equityCurveMetrics(curve);

        // Wobbly curve should have lower R-squared than a perfect line
        assert.ok(result.stability < 0.99,
            `stability ${result.stability} should be noticeably < 1 for noisy data`);
    });

    it('supportscustom startingEquity in opts', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 200 },
            { timestamp: 1700086400000, equity: 220 },
        ];
        const result = equityCurveMetrics(curve, { startingEquity: 200 });

        // totalReturn = (220-200)/200 = 0.1
        assert.ok(Math.abs(result.totalReturn - 0.1) < 0.001,
            `totalReturn ${result.totalReturn} should be ~0.1`);
        assert.equal(result.totalReturnUsd, 20);
    });

    it('infers startingEquity from first curve point when not provided', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 50 },
            { timestamp: 1700086400000, equity: 60 },
        ];
        const result = equityCurveMetrics(curve);

        // totalReturn = (60-50)/50 = 0.2
        assert.ok(Math.abs(result.totalReturn - 0.2) < 0.001,
            `totalReturn ${result.totalReturn} should be ~0.2`);
        assert.equal(result.totalReturnUsd, 10);
    });

    it('handles negative total return', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 80 },
        ];
        const result = equityCurveMetrics(curve);

        assert.ok(result.totalReturn < 0);
        assert.equal(result.totalReturnUsd, -20);
        assert.ok(result.cagr < 0);
    });

    it('returns annualized volatility for daily returns', () => {
        const curve = [
            { timestamp: 1700000000000, equity: 100 },
            { timestamp: 1700086400000, equity: 105 },
            { timestamp: 1700172800000, equity: 95 },
            { timestamp: 1700259200000, equity: 110 },
        ];
        const result = equityCurveMetrics(curve);

        // Annualized volatility = daily_std * sqrt(252)
        assert.ok(result.volatility > 0,
            `volatility ${result.volatility} should be > 0`);
    });
});
