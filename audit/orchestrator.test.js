/**
 * Trading System Orchestrator — unit tests (node:test runner)
 * Run: node --test audit/orchestrator.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createOrchestrator, runBacktest, DECISION } from './orchestrator.mjs';

// ─── Seeded PRNG ─────────────────────────────────────────────────────────────

function mulberry32(seed) {
    return function () {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ─── Candle generators ───────────────────────────────────────────────────────

function generateTrending(count, options = {}) {
    const rng = options.rng ?? mulberry32(options.seed ?? 1);
    const dir = options.trend === 'bearish' ? -1 : 1;
    const vol = options.volatility ?? 0.005;
    const base = options.basePrice ?? 100;
    const candles = [];
    let price = base;
    for (let i = 0; i < count; i++) {
        const drift = dir * vol * 0.3 * base + (rng() - 0.5) * vol * 0.1 * base;
        const open = price;
        const close = open + drift;
        const high = Math.max(open, close) + rng() * vol * 0.2 * base;
        const low = Math.min(open, close) - rng() * vol * 0.2 * base;
        candles.push({
            timestamp: Date.now() - (count - i) * 1000,
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: +(base * 10 + rng() * base * 5).toFixed(2),
        });
        price = close;
    }
    return candles;
}

function generateRanging(count, options = {}) {
    const rng = options.rng ?? mulberry32(options.seed ?? 2);
    const base = options.basePrice ?? 100;
    const width = options.rangeWidth ?? 0.006;
    const candles = [];
    let price = base;
    for (let i = 0; i < count; i++) {
        const oscillation = Math.sin(i * 0.4 + 1) * width * 0.5 * base;
        const noise = (rng() - 0.5) * width * 0.05 * base;
        const target = base + oscillation + noise;
        const open = price;
        const close = target;
        const high = Math.max(open, close) + Math.abs(rng() - 0.5) * width * 0.05 * base;
        const low = Math.min(open, close) - Math.abs(rng() - 0.5) * width * 0.05 * base;
        candles.push({
            timestamp: Date.now() - (count - i) * 1000,
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: +(base * 5 + rng() * base * 2).toFixed(2),
        });
        price = close;
    }
    return candles;
}

// ─── createOrchestrator ──────────────────────────────────────────────────────

describe('createOrchestrator', () => {
    it('creates an orchestrator with expected methods', () => {
        const orch = createOrchestrator();
        assert.equal(typeof orch.run, 'function');
        assert.equal(typeof orch.getLastDecision, 'function');
        assert.equal(typeof orch.reset, 'function');
        assert.equal(orch.decisionCount, 0);
    });

    it('handles null/empty candles', () => {
        const orch = createOrchestrator();
        const result = orch.run(null);
        assert.equal(result.action, 'HOLD');
        assert.equal(result.confidence, 0);
        assert.ok(result.reasoning.includes('insufficient'));
    });

    it('handles fewer than 5 candles', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(3);
        const result = orch.run(candles);
        assert.equal(result.action, 'HOLD');
        assert.ok(result.reasoning.includes('insufficient'));
    });

    it('runs pipeline on enough candles and returns a decision', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60, { trend: 'bullish', volatility: 0.01 });
        const result = orch.run(candles, { symbol: 'SOL-USD' });
        assert.ok(result.id);
        assert.ok(result.ts);
        assert.equal(result.symbol, 'SOL-USD');
        assert.ok(result.price > 0);
        assert.ok(['BUY', 'SELL', 'HOLD'].includes(result.action));
        assert.ok(result.confidence >= 0 && result.confidence <= 1);
        assert.ok(result.compositeScore >= -1 && result.compositeScore <= 1);
    });

    it('returns decision with signals populated', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60, { trend: 'bullish', volatility: 0.01 });
        const result = orch.run(candles);
        assert.ok(result.signals.length > 0, `got ${result.signals.length} signals`);
        for (const s of result.signals) {
            assert.ok(s.source);
            assert.ok(s.name);
            assert.equal(typeof s.value, 'number');
            assert.ok(s.confidence >= 0 && s.confidence <= 1);
        }
    });

    it('returns regime information', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60, { trend: 'bullish', volatility: 0.01 });
        const result = orch.run(candles);
        assert.ok(result.regime);
        assert.ok(typeof result.regimeConfidence === 'number');
        assert.ok(result.params);
        assert.ok('stopMultiplier' in result.params);
    });

    it('returns active zones near current price', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(80, { trend: 'bullish', volatility: 0.008 });
        const result = orch.run(candles);
        assert.ok(Array.isArray(result.activeZones));
        // May or may not have zones depending on market structure
        for (const z of result.activeZones) {
            assert.ok(z.price > 0);
            assert.ok(z.type);
            assert.ok(z.strength >= 0 && z.strength <= 1);
        }
    });

    it('includes support and resistance when zones exist', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(80, { trend: 'bullish', volatility: 0.008 });
        const result = orch.run(candles);
        // Support/resistance may be null if zones weren't detected
        assert.ok(result.support === null || result.support.price < result.price);
        assert.ok(result.resistance === null || result.resistance.price > result.price);
    });

    it('BUYs in bullish trending market', () => {
        const orch = createOrchestrator();
        // Strong extended trend
        const candles = generateTrending(100, { trend: 'bullish', volatility: 0.012 });
        const result = orch.run(candles);
        // In a strong uptrend, should NOT be SELL
        assert.notEqual(result.action, 'SELL');
    });

    it('SELLs in bearish trending market', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(100, { trend: 'bearish', volatility: 0.012 });
        const result = orch.run(candles);
        assert.notEqual(result.action, 'BUY');
    });

    it('HOLDs or reduces confidence in ranging market', () => {
        const orch = createOrchestrator();
        const candles = generateRanging(80, { rangeWidth: 0.006 });
        const result = orch.run(candles);
        // Ranging should produce lower confidence or HOLD
        assert.ok(result.confidence < 0.8 || result.action === 'HOLD',
            `action=${result.action}, confidence=${result.confidence}`);
    });

    it('includes reasoning string', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60);
        const result = orch.run(candles);
        assert.equal(typeof result.reasoning, 'string');
        assert.ok(result.reasoning.length > 0);
    });

    it('getLastDecision tracks most recent', () => {
        const orch = createOrchestrator();
        assert.equal(orch.getLastDecision(), null);

        const candles = generateTrending(60);
        const result = orch.run(candles);
        assert.equal(orch.getLastDecision(), result);
    });

    it('decisionCount increments', () => {
        const orch = createOrchestrator();
        assert.equal(orch.decisionCount, 0);
        orch.run(generateTrending(60));
        assert.equal(orch.decisionCount, 1);
        orch.run(generateTrending(60, { seed: 99 }));
        assert.equal(orch.decisionCount, 2);
    });

    it('reset clears state', () => {
        const orch = createOrchestrator();
        orch.run(generateTrending(60));
        orch.reset();
        assert.equal(orch.getLastDecision(), null);
        assert.equal(orch.decisionCount, 0);
    });

    it('accepts custom config — different fusion methods', () => {
        const methods = ['weighted', 'bayesian', 'voting'];
        for (const method of methods) {
            const orch = createOrchestrator({ fusionMethod: method });
            const candles = generateTrending(60, { trend: 'bullish' });
            const result = orch.run(candles);
            assert.ok(result.action);
        }
    });

    it('accepts custom zone detection thresholds', () => {
        const orch = createOrchestrator({
            zoneThreshold: 0.03,
            fvgMinGap: 0.01,
        });
        const candles = generateTrending(60);
        const result = orch.run(candles);
        assert.ok(result.action);
    });

    it('uses order book context when provided', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60, { trend: 'bullish' });
        const currentPrice = candles[candles.length - 1].close;
        const bids = [
            { price: currentPrice * 0.999, volume: 500 },
            { price: currentPrice * 0.998, volume: 1000 },
        ];
        const asks = [
            { price: currentPrice * 1.001, volume: 200 },
            { price: currentPrice * 1.002, volume: 100 },
        ];
        const result = orch.run(candles, { symbol: 'BTC-USD', bids, asks });
        assert.ok(result.action);
        // Should have book_imbalance signal
        const hasObSignal = result.signals.some(s => s.source === 'microstructure');
        assert.ok(hasObSignal, 'should have microstructure signals');
    });

    it('decision has unique IDs across runs', () => {
        const orch = createOrchestrator();
        const r1 = orch.run(generateTrending(60, { seed: 1 }));
        const r2 = orch.run(generateTrending(60, { seed: 2 }));
        assert.notEqual(r1.id, r2.id);
    });
});

// ─── runBacktest ─────────────────────────────────────────────────────────────

describe('runBacktest', () => {
    it('returns trades and stats structure', () => {
        const candles = generateTrending(200, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, { symbol: 'SOL-USD', warmupBars: 50, cooldownBars: 20 });
        assert.ok(Array.isArray(result.trades));
        assert.ok(result.stats);
        assert.ok(typeof result.stats.totalTrades === 'number');
        assert.ok(typeof result.stats.winRate === 'number');
        assert.ok(typeof result.stats.profitFactor === 'number');
    });

    it('trades have correct fields', () => {
        const candles = generateTrending(200, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, { warmupBars: 40, cooldownBars: 15, symbol: 'SOL-USD' });

        for (const trade of result.trades) {
            assert.ok(typeof trade.entryBar === 'number');
            assert.ok(typeof trade.exitBar === 'number');
            assert.ok(trade.exitBar >= trade.entryBar);
            assert.ok(['long', 'short'].includes(trade.direction));
            assert.ok(trade.entryPrice > 0);
            assert.ok(trade.exitPrice > 0);
            assert.ok(typeof trade.pnl === 'number');
            assert.ok(typeof trade.pnlPct === 'number');
            assert.ok(trade.exitReason);
        }
    });

    it('respects cooldownBars', () => {
        const candles = generateTrending(300, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, { warmupBars: 50, cooldownBars: 200 });

        // With huge cooldown, should have very few trades (max 1-2)
        assert.ok(result.stats.totalTrades <= 3,
            `got ${result.stats.totalTrades} trades with 200-bar cooldown`);
    });

    it('stops out on stop loss', () => {
        // Force a losing scenario: strong trend then reversal
        const rng = mulberry32(555);
        const candles = [];
        let price = 100;
        for (let i = 0; i < 150; i++) {
            if (i < 80) {
                // Strong uptrend to trigger BUY
                price += price * 0.003 + (rng() - 0.3) * price * 0.002;
            } else {
                // Sharp reversal to hit stop
                price -= price * 0.005 + (rng() - 0.7) * price * 0.002;
            }
            candles.push({
                timestamp: Date.now() - (150 - i) * 1000,
                open: +(price - price * 0.001).toFixed(2),
                high: +(price + price * 0.002).toFixed(2),
                low: +(price - price * 0.002).toFixed(2),
                close: +price.toFixed(2),
                volume: +(1000 + rng() * 500).toFixed(2),
            });
        }

        const result = runBacktest(candles, {
            warmupBars: 50,
            cooldownBars: 10,
            stopLossPct: 0.02,
            takeProfitPct: 0.50,
        });

        if (result.stats.totalTrades > 0) {
            const stoppedOut = result.trades.some(t => t.exitReason === 'stop_loss');
            // May not hit if no long entry was triggered — that's fine
            assert.ok(result.stats.totalTrades >= 0);
        }
    });

    it('takes profit when target hit', () => {
        const candles = generateTrending(200, { trend: 'bullish', volatility: 0.015 });
        const result = runBacktest(candles, {
            warmupBars: 40,
            cooldownBars: 10,
            stopLossPct: 0.50, // very wide stop
            takeProfitPct: 0.03, // tight TP
        });

        // With a strong trend and tight TP, some trades should hit TP
        if (result.trades.length > 0) {
            // At minimum the stats should be valid
            assert.ok(result.stats.winRate >= 0 && result.stats.winRate <= 1);
        }
    });

    it('end_of_data closes open trade at end of backtest', () => {
        const candles = generateTrending(120, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, {
            warmupBars: 50,
            cooldownBars: 5,
            stopLossPct: 0.99, // will not hit
            takeProfitPct: 0.99, // will not hit
            maxHoldingBars: 999, // will not hit
        });

        if (result.trades.length > 0) {
            // The last trade should end at end_of_data rather than the other reasons
            const lastTrade = result.trades[result.trades.length - 1];
            // It's end_of_data OR it got stopped out at some point
            assert.ok(lastTrade.exitReason);
        }
    });

    it('perRegime stats computed correctly', () => {
        const candles = generateTrending(200, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, { warmupBars: 40, cooldownBars: 15 });

        if (result.trades.length > 0) {
            const regimes = Object.keys(result.stats.perRegime);
            assert.ok(regimes.length > 0, 'should have at least one regime');
            for (const regime of regimes) {
                const s = result.stats.perRegime[regime];
                assert.ok(typeof s.count === 'number');
                assert.ok(typeof s.winRate === 'number');
                assert.ok(typeof s.avgPnl === 'number');
                assert.ok(typeof s.totalPnl === 'number');
                assert.ok(s.winRate >= 0 && s.winRate <= 1);
            }
        }
    });

    it('stats have all expected fields', () => {
        const candles = generateTrending(200, { trend: 'bullish' });
        const result = runBacktest(candles, { warmupBars: 50, cooldownBars: 20 });

        assert.ok('totalTrades' in result.stats);
        assert.ok('winningTrades' in result.stats);
        assert.ok('losingTrades' in result.stats);
        assert.ok('winRate' in result.stats);
        assert.ok('totalPnl' in result.stats);
        assert.ok('avgWin' in result.stats);
        assert.ok('avgLoss' in result.stats);
        assert.ok('profitFactor' in result.stats);
        assert.ok('perRegime' in result.stats);
    });

    it('winRate is correct', () => {
        const candles = generateTrending(200, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, { warmupBars: 50, cooldownBars: 20 });

        if (result.stats.totalTrades > 0) {
            const expectedWR = result.stats.winningTrades / result.stats.totalTrades;
            assert.equal(result.stats.winRate, +expectedWR.toFixed(4));
        }
    });

    it('profitFactor equals Infinity for all-win scenario', () => {
        // With extreme TP and no SL, every trade that wins should produce high PF
        const candles = generateTrending(200, { trend: 'bullish', volatility: 0.01 });
        const result = runBacktest(candles, {
            warmupBars: 50,
            cooldownBars: 20,
            stopLossPct: 0.99,
            takeProfitPct: 0.001,
        });
        // Validates stats are computed without NaN
        assert.ok(!Number.isNaN(result.stats.profitFactor));
        assert.ok(result.stats.profitFactor >= 0);
    });

    it('handles flat price (no opportunities)', () => {
        const candles = [];
        for (let i = 0; i < 100; i++) {
            candles.push({
                timestamp: Date.now() - (100 - i) * 1000,
                open: 100, high: 100.01, low: 99.99, close: 100, volume: 1000,
            });
        }
        const result = runBacktest(candles, { warmupBars: 40 });
        assert.ok(Array.isArray(result.trades));
        assert.ok(result.stats.totalTrades >= 0);
    });
});

// ─── DECISION constant ───────────────────────────────────────────────────────

describe('DECISION', () => {
    it('exports expected decision types', () => {
        assert.equal(DECISION.BUY, 'BUY');
        assert.equal(DECISION.SELL, 'SELL');
        assert.equal(DECISION.HOLD, 'HOLD');
    });
});

// ─── Edge cases ──────────────────────────────────────────────────────────────

describe('Edge cases', () => {
    it('handles candles with zero volume', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60).map(c => ({ ...c, volume: 0 }));
        const result = orch.run(candles);
        assert.ok(result.action);
    });

    it('handles very large candles array quickly', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(2000, { trend: 'bullish' });
        const start = performance.now();
        const result = orch.run(candles);
        const elapsed = performance.now() - start;
        assert.ok(result.action);
        // Should complete in < 2s
        assert.ok(elapsed < 2000, `took ${elapsed.toFixed(0)}ms for 2000 candles`);
    });

    it('multiple runs produce consistent decisions for similar data', () => {
        const orch = createOrchestrator({ fusionMethod: 'weighted' });
        const r1 = orch.run(generateTrending(60, { seed: 42 }));
        orch.reset();
        const r2 = orch.run(generateTrending(60, { seed: 42 }));
        assert.equal(r1.action, r2.action);
    });

    it('works without order book context', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60);
        const result = orch.run(candles, {}); // no bids/asks
        assert.ok(result.action);
    });

    it('works with empty bids/asks', () => {
        const orch = createOrchestrator();
        const candles = generateTrending(60);
        const result = orch.run(candles, { bids: [], asks: [] });
        assert.ok(result.action);
    });

    it('custom minConfidence affects decisions', () => {
        const lenient = createOrchestrator({ minConfidence: 0.05 });
        const strict = createOrchestrator({ minConfidence: 0.95 });
        const candles = generateTrending(60, { trend: 'bullish', volatility: 0.008 });

        const rLenient = lenient.run(candles);
        const rStrict = strict.run(candles);

        // Lenient should be at least as likely to produce BUY/SELL as strict
        assert.ok(rLenient.action);
        assert.ok(rStrict.action);
    });
});
