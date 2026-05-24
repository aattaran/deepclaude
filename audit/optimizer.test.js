/**
 * Strategy Parameter Optimizer — unit tests (node:test runner)
 * Run: node --test audit/optimizer.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { gridSearch, hillClimb, optimize, optimizePerRegime, sensitivity } from './optimizer.mjs';
import { runBacktest } from './orchestrator.mjs';

// ─── Seeded PRNG ────────────────────────────────────────────────────────────────

function mulberry32(seed) {
    return function () {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// ─── Candle generator (must match trade.mjs semantics) ──────────────────────────

function generateCandles(count, options = {}) {
    const rng = mulberry32(options.seed ?? 42);
    const basePrice = options.basePrice ?? 100;
    const volatility = options.volatility ?? 0.01;
    const startTs = options.startTs ?? Date.now() - count * 60000;
    const candles = [];
    let price = basePrice;
    const regimeLength = Math.max(20, Math.floor(count / 5));
    let regime = 'trending_bullish';
    let regimeBars = 0;
    let trendDir = 1;

    for (let i = 0; i < count; i++) {
        if (regimeBars >= regimeLength) {
            regimeBars = 0;
            const roll = rng();
            if (roll < 0.35) {
                trendDir = trendDir > 0 ? -1 : 1;
                regime = trendDir > 0 ? 'trending_bullish' : 'trending_bearish';
            } else if (roll < 0.6) {
                regime = 'ranging';
            } else if (roll < 0.8) {
                regime = trendDir > 0 ? 'breakout' : 'breakdown';
            } else {
                regime = 'volatile';
            }
        }
        regimeBars++;

        let drift;
        const volFactor = regime === 'volatile' ? 2.5 : regime === 'breakout' || regime === 'breakdown' ? 1.8 : 1.0;
        switch (regime) {
            case 'trending_bullish': drift = volatility * 0.15 + rng() * volatility * 0.2; break;
            case 'trending_bearish': drift = -volatility * 0.15 - rng() * volatility * 0.2; break;
            case 'ranging': drift = (rng() - 0.5) * volatility * 0.3; break;
            case 'breakout': drift = volatility * 0.4 + rng() * volatility * 0.3; break;
            case 'breakdown': drift = -volatility * 0.4 - rng() * volatility * 0.3; break;
            case 'volatile': drift = (rng() - 0.5) * volatility * 2; break;
            default: drift = (rng() - 0.5) * volatility * 0.3;
        }

        const wickFactor = regime === 'volatile' ? 0.6 : 0.3;
        const open = price;
        const close = open + drift * basePrice * volFactor;
        const high = Math.max(open, close) + rng() * volatility * wickFactor * basePrice * volFactor;
        const low = Math.min(open, close) - rng() * volatility * wickFactor * basePrice * volFactor;
        const volume = basePrice * 10 + rng() * basePrice * 5 * volFactor;

        candles.push({
            timestamp: startTs + i * 60000,
            open: +open.toFixed(4),
            high: +high.toFixed(4),
            low: +low.toFixed(4),
            close: +close.toFixed(4),
            volume: +volume.toFixed(2),
            _regime: regime,
        });
        price = close;
    }
    return candles;
}

// ─── 1. gridSearch ──────────────────────────────────────────────────────────────

describe('gridSearch', () => {
    const candles = generateCandles(300, { seed: 42, volatility: 0.012 });

    it('returns bestParams and bestScore', () => {
        const result = gridSearch(candles, {
            paramSpace: {
                stopLossPct: [0.02, 0.04],
                takeProfitPct: [0.04, 0.08],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxCombinations: 100,
        });

        assert.ok(result.bestParams, 'has bestParams');
        assert.ok(typeof result.bestScore === 'number', 'bestScore is number');
        assert.ok(result.bestStats, 'has bestStats');
        assert.ok(Array.isArray(result.allResults), 'allResults is array');
        assert.equal(result.allResults.length, 4, '2x2x1x1x1x1 = 4 combinations');
        assert.ok(Array.isArray(result.topN), 'topN is array');
        assert.ok(result.topN.length <= 10, 'topN capped at 10');
        assert.equal(result.objective, 'profitFactor');
        assert.equal(result.totalEvaluations, 4);
    });

    it('returns sorted results (best first)', () => {
        const result = gridSearch(candles, {
            paramSpace: {
                stopLossPct: [0.02, 0.04],
                takeProfitPct: [0.04, 0.08],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxCombinations: 100,
        });

        for (let i = 1; i < result.allResults.length; i++) {
            assert.ok(
                result.allResults[i - 1].score >= result.allResults[i].score,
                `result ${i - 1} score >= result ${i} score`,
            );
        }
    });

    it('bestParams is the first in allResults', () => {
        const result = gridSearch(candles, {
            paramSpace: {
                stopLossPct: [0.02],
                takeProfitPct: [0.04, 0.08],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxCombinations: 100,
        });

        assert.deepEqual(result.bestParams, result.allResults[0].params);
        assert.equal(result.bestScore, result.allResults[0].score);
    });

    it('supports different objective functions', () => {
        for (const obj of ['profitFactor', 'sharpe', 'totalPnl', 'winRate', 'expectancy', 'calmar']) {
            const result = gridSearch(candles, {
                objective: obj,
                paramSpace: {
                    stopLossPct: [0.02],
                    takeProfitPct: [0.04],
                    cooldownBars: [10],
                    warmupBars: [50],
                    minConfidence: [0.2],
                    fusionMethod: ['weighted'],
                },
                maxCombinations: 100,
            });
            assert.equal(result.objective, obj, `objective ${obj}`);
            assert.ok(Number.isFinite(result.bestScore), `${obj} score is finite`);
        }
    });

    it('throws when grid exceeds maxCombinations', () => {
        assert.throws(() => {
            gridSearch(candles, {
                paramSpace: {
                    stopLossPct: [0.01, 0.02, 0.03, 0.04, 0.05],
                    takeProfitPct: [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                    cooldownBars: [3, 5, 10, 15, 20, 30],
                    warmupBars: [30, 50, 70],
                    minConfidence: [0.1, 0.15, 0.2, 0.25, 0.3],
                    fusionMethod: ['weighted', 'bayesian', 'voting'],
                },
                maxCombinations: 50,
            });
        }, /exceeds maxCombinations/);
    });

    it('handles single param combination', () => {
        const result = gridSearch(candles, {
            paramSpace: {
                stopLossPct: [0.02],
                takeProfitPct: [0.04],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
        });

        assert.equal(result.allResults.length, 1);
        assert.equal(result.totalEvaluations, 1);
    });
});

// ─── 2. hillClimb ───────────────────────────────────────────────────────────────

describe('hillClimb', () => {
    const candles = generateCandles(300, { seed: 42, volatility: 0.012 });

    it('returns bestParams, bestScore, bestStats, and path', () => {
        const result = hillClimb(candles, {
            paramSpace: {
                stopLossPct: [0.01, 0.02, 0.03, 0.04, 0.05],
                takeProfitPct: [0.02, 0.04, 0.06, 0.08, 0.10],
                cooldownBars: [5, 10, 20],
                warmupBars: [50],
                minConfidence: [0.15, 0.25],
                fusionMethod: ['weighted'],
            },
            maxIterations: 10,
            neighborsPerIteration: 4,
        });

        assert.ok(result.bestParams, 'has bestParams');
        assert.ok(typeof result.bestScore === 'number', 'bestScore is number');
        assert.ok(result.bestStats, 'has bestStats');
        assert.ok(Array.isArray(result.path), 'path is array');
        assert.ok(result.path.length >= 1, 'path has at least initial state');
        assert.ok(result.iterations >= 0);
        assert.ok(result.iterations <= 10);
    });

    it('initialParams are used as starting point', () => {
        const initial = {
            stopLossPct: 0.05,
            takeProfitPct: 0.10,
            cooldownBars: 20,
            warmupBars: 50,
            minConfidence: 0.25,
            fusionMethod: 'weighted',
        };

        const result = hillClimb(candles, {
            initialParams: initial,
            paramSpace: {
                stopLossPct: [0.01, 0.02, 0.03, 0.04, 0.05],
                takeProfitPct: [0.02, 0.04, 0.06, 0.08, 0.10],
                cooldownBars: [5, 10, 20],
                warmupBars: [50],
                minConfidence: [0.15, 0.25],
                fusionMethod: ['weighted'],
            },
            maxIterations: 5,
            neighborsPerIteration: 3,
        });

        assert.equal(result.path[0].params.stopLossPct, 0.05);
        assert.equal(result.path[0].params.takeProfitPct, 0.10);
    });

    it('path score is non-decreasing', () => {
        const result = hillClimb(candles, {
            paramSpace: {
                stopLossPct: [0.01, 0.02, 0.03],
                takeProfitPct: [0.02, 0.04, 0.06],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxIterations: 8,
            neighborsPerIteration: 4,
        });

        for (let i = 1; i < result.path.length; i++) {
            assert.ok(
                result.path[i].score >= result.path[i - 1].score,
                `step ${i} score >= step ${i - 1} score (${result.path[i].score} >= ${result.path[i - 1].score})`,
            );
        }
    });

    it('supports different objectives', () => {
        const result = hillClimb(candles, {
            objective: 'sharpe',
            paramSpace: {
                stopLossPct: [0.02],
                takeProfitPct: [0.04],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxIterations: 3,
        });
        assert.equal(result.objective, 'sharpe');
    });
});

// ─── 3. optimize (grid + hill) ──────────────────────────────────────────────────

describe('optimize', () => {
    const candles = generateCandles(300, { seed: 42, volatility: 0.012 });

    it('returns bestParams, bestScore, gridResult, hillResult', () => {
        const result = optimize(candles, {
            coarseSpace: {
                stopLossPct: [0.02, 0.04],
                takeProfitPct: [0.04, 0.08],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxIterations: 5,
            neighborsPerIteration: 3,
        });

        assert.ok(result.bestParams, 'has bestParams');
        assert.ok(typeof result.bestScore === 'number', 'bestScore is number');
        assert.ok(result.gridResult, 'has gridResult');
        assert.ok(result.hillResult, 'has hillResult');
        assert.ok(result.hillResult.bestScore >= result.gridResult.bestScore, 'hill improved or matched grid');
    });
});

// ─── 4. sensitivity ─────────────────────────────────────────────────────────────

describe('sensitivity', () => {
    const candles = generateCandles(300, { seed: 42, volatility: 0.012 });

    it('returns results for each varied value', () => {
        const baseParams = {
            stopLossPct: 0.02,
            takeProfitPct: 0.04,
            cooldownBars: 10,
            warmupBars: 50,
            minConfidence: 0.2,
            fusionMethod: 'weighted',
        };

        const results = sensitivity(candles, baseParams, 'stopLossPct', [0.01, 0.02, 0.03, 0.04]);

        assert.equal(results.length, 4);
        for (const r of results) {
            assert.ok(typeof r.score === 'number');
            assert.ok(r.params.stopLossPct !== undefined);
            assert.equal(r.params.takeProfitPct, 0.04); // unchanged
        }
    });

    it('varying param takes on each test value', () => {
        const baseParams = {
            stopLossPct: 0.02,
            takeProfitPct: 0.04,
            cooldownBars: 10,
            warmupBars: 50,
            minConfidence: 0.2,
            fusionMethod: 'weighted',
        };

        const testValues = [0.01, 0.02, 0.03];
        const results = sensitivity(candles, baseParams, 'stopLossPct', testValues);

        for (let i = 0; i < testValues.length; i++) {
            assert.equal(results[i].params.stopLossPct, testValues[i]);
        }
    });
});

// ─── 5. optimizePerRegime ───────────────────────────────────────────────────────

describe('optimizePerRegime', () => {
    it('returns per-regime optimization results', () => {
        // Generate enough candles per regime
        const candles = generateCandles(600, { seed: 42, volatility: 0.012 });
        const regimes = new Set(candles.map(c => c._regime));

        const results = optimizePerRegime(candles, {
            coarseSpace: {
                stopLossPct: [0.02, 0.04],
                takeProfitPct: [0.04, 0.08],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxIterations: 3,
            neighborsPerIteration: 2,
        });

        assert.ok(Object.keys(results).length >= 1, 'at least one regime optimized');
        for (const [regime, r] of Object.entries(results)) {
            assert.ok(regimes.has(regime) || regime === 'unknown', `regime ${regime} exists`);
            assert.ok(r.bestParams, `${regime}: has bestParams`);
            assert.ok(typeof r.bestScore === 'number', `${regime}: score is number`);
        }
    });
});

// ─── 6. Edge Cases ──────────────────────────────────────────────────────────────

describe('edge cases', () => {
    it('gridSearch handles single-element param arrays', () => {
        const candles = generateCandles(200, { seed: 1 });
        const result = gridSearch(candles, {
            paramSpace: {
                stopLossPct: [0.02],
                takeProfitPct: [0.04],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
        });
        assert.equal(result.totalEvaluations, 1);
    });

    it('hillClimb converges within maxIterations', () => {
        const candles = generateCandles(200, { seed: 1 });
        const result = hillClimb(candles, {
            paramSpace: {
                stopLossPct: [0.02],
                takeProfitPct: [0.04],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
            maxIterations: 5,
        });
        assert.ok(result.iterations <= 5);
    });

    it('objective functions return 0 for zero trades', () => {
        // Flat candles produce 0 trades
        const flatCandles = [];
        let price = 100;
        for (let i = 0; i < 200; i++) {
            flatCandles.push({
                timestamp: Date.now() + i * 60000,
                open: price,
                high: price + 0.01,
                low: price - 0.01,
                close: price,
                volume: 1000,
                _regime: 'ranging',
            });
        }

        const result = gridSearch(flatCandles, {
            paramSpace: {
                stopLossPct: [0.02],
                takeProfitPct: [0.04],
                cooldownBars: [10],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
        });

        // With flat candles and 0 trades, all objectives should return 0
        assert.equal(result.bestScore, 0);
    });
});

// ─── 7. Performance ─────────────────────────────────────────────────────────────

describe('performance', () => {
    it('gridSearch with 8 combinations runs within 30s', () => {
        const candles = generateCandles(200, { seed: 42 });

        const start = Date.now();
        const result = gridSearch(candles, {
            paramSpace: {
                stopLossPct: [0.02, 0.04],
                takeProfitPct: [0.04, 0.08],
                cooldownBars: [10, 20],
                warmupBars: [50],
                minConfidence: [0.2],
                fusionMethod: ['weighted'],
            },
        });
        const elapsed = Date.now() - start;

        assert.equal(result.totalEvaluations, 8);
        assert.ok(elapsed < 30000, `8 combinations took ${elapsed}ms`);
    });
});
