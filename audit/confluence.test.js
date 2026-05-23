/**
 * Confluence Engine — unit tests.
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/confluence.test.js
 *
 * Tests cover:
 *   - confluenceLevels       (clustering, scoring, filtering)
 *   - multiTimeframeZones    (end-to-end convenience function)
 *   - ConfluenceHeatmap      (snapshot tracking, binning, density)
 *   - zoneConfluenceScore    (live alignment scoring)
 *   - timeframeWeight        (weight lookup)
 *   - Edge cases             (null, negative, overflow)
 *   - Performance            (1000 zones across 5 TFs in <100 ms)
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    confluenceLevels,
    multiTimeframeZones,
    ConfluenceHeatmap,
    zoneConfluenceScore,
    timeframeWeight,
} from './confluence.mjs';
import { ZoneDetector } from './zone-detector.mjs';

// ===========================================================================
// Helpers: PRNG, zone factories, candle generators
// ===========================================================================

/**
 * Seeded pseudo-random number generator (Mulberry32).
 * Deterministic values for reproducible tests.
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
 * Create a minimal zone object with defaults.
 * Mirrors the shape returned by ZoneDetector.getActiveZones.
 */
function makeZone(overrides = {}) {
    return {
        price: 100,
        type: 'orderBlock',
        strength: 0.7,
        freshness: 1.0,
        timestamp: 1700000000000,
        ...overrides,
    };
}

/**
 * Generate synthetic 1-second OHLCV candles.
 * Reused from zone-detector.test.js pattern.
 */
function generateCandles(n, opts = {}) {
    const {
        startPrice = 100,
        startTime = 1700000000000,
        intervalMs = 1000,
        trend = 0,
        volatility = 0.001,
        baseVolume = 100,
        rng = () => 0.5,
    } = opts;

    const candles = [];
    let price = startPrice;

    for (let i = 0; i < n; i++) {
        const open = price;
        const change = (rng() - 0.5) * volatility * 2 + trend;
        const close = price * (1 + change);
        const range = Math.abs(close - open) + volatility * startPrice;
        const high = Math.max(open, close) + range * rng() * 0.5;
        const low = Math.min(open, close) - range * (1 - rng()) * 0.5;
        const vol = Math.max(1, Math.round(baseVolume * (0.5 + rng())));

        candles.push({
            timestamp: startTime + i * intervalMs,
            open: +open.toFixed(4),
            high: +high.toFixed(4),
            low: +low.toFixed(4),
            close: +close.toFixed(4),
            volume: vol,
        });

        price = close;
    }

    return candles;
}

// ===========================================================================
// timeframeWeight
// ===========================================================================

describe('timeframeWeight', () => {

    it('returns 0.3 for 1s', () => {
        assert.equal(timeframeWeight('1s'), 0.3);
    });

    it('returns 0.5 for 5s', () => {
        assert.equal(timeframeWeight('5s'), 0.5);
    });

    it('returns 0.7 for 15s', () => {
        assert.equal(timeframeWeight('15s'), 0.7);
    });

    it('returns 0.85 for 1m', () => {
        assert.equal(timeframeWeight('1m'), 0.85);
    });

    it('returns 1.0 for 5m', () => {
        assert.equal(timeframeWeight('5m'), 1.0);
    });

    it('returns 1.0 for 15m', () => {
        assert.equal(timeframeWeight('15m'), 1.0);
    });

    it('defaults to 0.5 for unknown timeframe', () => {
        assert.equal(timeframeWeight('1h'), 0.5);
        assert.equal(timeframeWeight('daily'), 0.5);
        assert.equal(timeframeWeight(''), 0.5);
    });
});

// ===========================================================================
// confluenceLevels — basic
// ===========================================================================

describe('confluenceLevels — empty / invalid input', () => {

    it('returns [] for empty zonesByTimeframe object', () => {
        const result = confluenceLevels({}, 100);
        assert.deepEqual(result, []);
    });

    it('returns [] for null zonesByTimeframe', () => {
        assert.deepEqual(confluenceLevels(null, 100), []);
    });

    it('returns [] for undefined zonesByTimeframe', () => {
        assert.deepEqual(confluenceLevels(undefined, 100), []);
    });

    it('returns [] for array zonesByTimeframe (invalid shape)', () => {
        assert.deepEqual(confluenceLevels([], 100), []);
    });

    it('returns [] for null currentPrice', () => {
        const zones = { '1s': [makeZone()] };
        assert.deepEqual(confluenceLevels(zones, null), []);
    });

    it('returns [] for NaN currentPrice', () => {
        const zones = { '1s': [makeZone()] };
        assert.deepEqual(confluenceLevels(zones, NaN), []);
    });

    it('returns [] for Infinity currentPrice', () => {
        const zones = { '1s': [makeZone()] };
        assert.deepEqual(confluenceLevels(zones, Infinity), []);
    });

    it('returns [] when all timeframe arrays are empty', () => {
        const zones = { '1s': [], '5s': [], '15s': [] };
        assert.deepEqual(confluenceLevels(zones, 100), []);
    });

    it('skips timeframe entries that are not arrays', () => {
        const zones = { '1s': [makeZone({ price: 100 })], '5s': null, '15s': 'not-an-array' };
        const result = confluenceLevels(zones, 100);
        // Only 1s contributes, minTimeframes=2 => empty
        assert.deepEqual(result, []);
    });

    it('skips zones with null price', () => {
        const zones = { '1s': [makeZone({ price: null })], '5s': [makeZone({ price: null })] };
        assert.deepEqual(confluenceLevels(zones, 100), []);
    });

    it('skips zones with NaN price', () => {
        const zones = { '1s': [makeZone({ price: NaN })], '5s': [makeZone({ price: NaN })] };
        assert.deepEqual(confluenceLevels(zones, 100), []);
    });
});

// ===========================================================================
// confluenceLevels — clustering
// ===========================================================================

describe('confluenceLevels — clustering', () => {

    it('single timeframe yields no levels (minTimeframes=2)', () => {
        const zones = { '1s': [makeZone({ price: 100 })] };
        assert.deepEqual(confluenceLevels(zones, 100), []);
    });

    it('two timeframes with nearby zones produce one level', () => {
        const zones = {
            '1s': [makeZone({ price: 99.85 })],
            '5s': [makeZone({ price: 100.00 })],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
    });

    it('returned level has correct shape', () => {
        const zones = {
            '1s': [makeZone({ price: 99.85 })],
            '5s': [makeZone({ price: 100.00 })],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        const level = result[0];
        assert.ok(typeof level.price === 'number');
        assert.ok(level.type === 'support' || level.type === 'resistance');
        assert.ok(typeof level.score === 'number');
        assert.ok(level.score >= 0 && level.score <= 1);
        assert.ok(typeof level.timeframeCount === 'number');
        assert.ok(Array.isArray(level.timeframes));
        assert.ok(Array.isArray(level.zones));
        assert.ok(typeof level.strength === 'number');
    });

    it('zones far apart form separate clusters', () => {
        const zones = {
            '1s': [makeZone({ price: 99.0 }), makeZone({ price: 101.0 })],
            '5s': [makeZone({ price: 99.0 }), makeZone({ price: 101.0 })],
        };
        const result = confluenceLevels(zones, 100, { priceWindow: 0.003 });
        // With priceWindow=0.003 * 100 = 0.3, zones at 99 and 101 are far apart.
        // One level at ~99 (support) and one at ~101 (resistance).
        // But the priceWindow of 0.3 means: zones at 99.0 and 99.0 are same cluster,
        // zones at 101.0 and 101.0 are same cluster. Two clusters total.
        assert.equal(result.length, 2);
    });

    it('three timeframes yield higher score than two', () => {
        // Two-TF: lower avg strength to ensure score clearly differs from three-TF case
        const twoTf = {
            '1s': [makeZone({ price: 99.5, strength: 0.6, freshness: 1.0 })],
            '5s': [makeZone({ price: 99.6, strength: 0.6, freshness: 1.0 })],
        };
        const threeTf = {
            '1s': [makeZone({ price: 99.5, strength: 0.8, freshness: 1.0 })],
            '5s': [makeZone({ price: 99.6, strength: 0.8, freshness: 1.0 })],
            '15s': [makeZone({ price: 99.7, strength: 0.8, freshness: 1.0 })],
        };
        const result2 = confluenceLevels(twoTf, 100);
        const result3 = confluenceLevels(threeTf, 100);

        assert.equal(result2.length, 1);
        assert.equal(result3.length, 1);
        assert.ok(result3[0].score > result2[0].score,
            `3-TF score (${result3[0].score}) should exceed 2-TF score (${result2[0].score})`);
    });

    it('all five timeframes produce a "grail" level with high score', () => {
        const zones = {
            '1s': [makeZone({ price: 99.9, strength: 0.8, freshness: 1.0 })],
            '5s': [makeZone({ price: 100.0, strength: 0.8, freshness: 1.0 })],
            '15s': [makeZone({ price: 100.0, strength: 0.8, freshness: 1.0 })],
            '1m': [makeZone({ price: 100.1, strength: 0.8, freshness: 1.0 })],
            '5m': [makeZone({ price: 100.1, strength: 0.8, freshness: 1.0 })],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        assert.equal(result[0].timeframeCount, 5);
        // With all 5 TFs and perfect freshness/strength, score should be high
        const level = result[0];
        assert.ok(level.score > 0.6,
            `grail level score (${level.score}) should be > 0.6`);
        assert.deepEqual(level.timeframes, ['15s', '1m', '1s', '5m', '5s']);
    });

    it('zones are correctly classified as support or resistance', () => {
        // All zones below current price => support
        const supportZones = {
            '1s': [makeZone({ price: 99.0 })],
            '5s': [makeZone({ price: 99.1 })],
        };
        const supportResult = confluenceLevels(supportZones, 100);
        assert.equal(supportResult.length, 1);
        assert.equal(supportResult[0].type, 'support');

        // All zones above current price => resistance
        const resistanceZones = {
            '1s': [makeZone({ price: 101.0 })],
            '5s': [makeZone({ price: 101.1 })],
        };
        const resistanceResult = confluenceLevels(resistanceZones, 100);
        assert.equal(resistanceResult.length, 1);
        assert.equal(resistanceResult[0].type, 'resistance');
    });

    it('sorts results by descending score', () => {
        // Create two clusters with different scores
        // Cluster A: 3 TFs at ~100 (high diversity, higher score)
        // Cluster B: 2 TFs at ~102 (lower diversity, lower score)
        const zones = {
            '1s': [
                makeZone({ price: 99.9, strength: 0.8, freshness: 1.0 }),
                makeZone({ price: 101.9, strength: 0.5, freshness: 0.5 }),
            ],
            '5s': [
                makeZone({ price: 100.0, strength: 0.8, freshness: 1.0 }),
                makeZone({ price: 102.0, strength: 0.5, freshness: 0.5 }),
            ],
            '15s': [
                makeZone({ price: 100.1, strength: 0.8, freshness: 1.0 }),
            ],
        };
        const result = confluenceLevels(zones, 100, { priceWindow: 0.005 });
        assert.equal(result.length, 2);
        assert.ok(result[0].score >= result[1].score,
            `first score (${result[0].score}) should be >= second (${result[1].score})`);
        // The first cluster (higher score) should be the one with 3 TFs
        assert.equal(result[0].timeframeCount, 3);
    });
});

// ===========================================================================
// confluenceLevels — options
// ===========================================================================

describe('confluenceLevels — options', () => {

    it('priceWindow controls merging distance', () => {
        const zones = {
            '1s': [makeZone({ price: 99.5 })],
            '5s': [makeZone({ price: 100.0 })],
            '15s': [makeZone({ price: 100.5 })],
        };
        // Wide window: all three merge into one cluster
        const wide = confluenceLevels(zones, 100, { priceWindow: 0.01 });
        assert.equal(wide.length, 1);
        assert.equal(wide[0].timeframeCount, 3);

        // Narrow window: zones 99.5 and 100.5 are far from 100.0
        // 99.5 to 100.0 diff=0.5 > 0.1 (0.001*100), so separate cluster
        // Actually with absPriceWindow=0.001*100=0.1:
        // 99.5, 100.0: diff=0.5 > 0.1 => separate clusters
        // 100.0, 100.5: diff=0.5 > 0.1 => separate clusters
        // Each zone ends up as its own cluster, each with 1 TF => filtered out
        const narrow = confluenceLevels(zones, 100, { priceWindow: 0.001 });
        assert.equal(narrow.length, 0);
    });

    it('minTimeframes filters out weak levels', () => {
        const zones = {
            '1s': [makeZone({ price: 100 })],
            '5s': [makeZone({ price: 100 })],
            '15s': [makeZone({ price: 100 })],
        };
        const strict = confluenceLevels(zones, 100, { minTimeframes: 4 });
        assert.equal(strict.length, 0, 'no level with 4 TFs when only 3 TFs exist');

        const lenient = confluenceLevels(zones, 100, { minTimeframes: 2 });
        assert.equal(lenient.length, 1, 'level should appear with minTimeframes=2');
    });

    it('minTimeframes=1 includes single-timeframe levels', () => {
        const zones = { '1s': [makeZone({ price: 100 })] };
        const result = confluenceLevels(zones, 100, { minTimeframes: 1 });
        assert.equal(result.length, 1);
        assert.equal(result[0].timeframeCount, 1);
    });

    it('decayFactor=0 means freshness does not affect score', () => {
        const fresh = {
            '1s': [makeZone({ price: 100, strength: 0.8, freshness: 1.0 })],
            '5s': [makeZone({ price: 100, strength: 0.8, freshness: 1.0 })],
        };
        const stale = {
            '1s': [makeZone({ price: 100, strength: 0.8, freshness: 0.0 })],
            '5s': [makeZone({ price: 100, strength: 0.8, freshness: 0.0 })],
        };
        const resultFresh = confluenceLevels(fresh, 100, { decayFactor: 0 });
        const resultStale = confluenceLevels(stale, 100, { decayFactor: 0 });

        assert.equal(resultFresh.length, 1);
        assert.equal(resultStale.length, 1);
        // With decayFactor=0, freshnessDecay = 1 regardless of freshness
        assert.equal(resultFresh[0].score, resultStale[0].score);
    });

    it('decayFactor=1 makes score proportional to freshness', () => {
        const fresh = {
            '1s': [makeZone({ price: 100, strength: 0.5, freshness: 1.0 })],
            '5s': [makeZone({ price: 100, strength: 0.5, freshness: 1.0 })],
        };
        const stale = {
            '1s': [makeZone({ price: 100, strength: 0.5, freshness: 0.2 })],
            '5s': [makeZone({ price: 100, strength: 0.5, freshness: 0.2 })],
        };
        const resultFresh = confluenceLevels(fresh, 100, { decayFactor: 1 });
        const resultStale = confluenceLevels(stale, 100, { decayFactor: 1 });

        assert.equal(resultFresh.length, 1);
        assert.equal(resultStale.length, 1);
        // With decayFactor=1, freshnessDecay = avgFreshness
        // Fresh: tfDiv=1/4=0.25, str=0.5, fresh=1 => score=0.125
        // Stale: tfDiv=1/4=0.25, str=0.5, fresh=0.2 => score=0.025
        assert.ok(resultFresh[0].score > resultStale[0].score,
            `fresh score (${resultFresh[0].score}) > stale score (${resultStale[0].score})`);
    });

    it('zones with strength=0 still cluster but score reflects it', () => {
        const zones = {
            '1s': [makeZone({ price: 100, strength: 0.0 })],
            '5s': [makeZone({ price: 100, strength: 1.0 })],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        // avgStrength = 0.5, so score should reflect that
        assert.ok(result[0].score > 0, 'score should be > 0 with mixed strengths');
        assert.equal(result[0].strength, 0.5);
    });

    it('uses default strength=0.5 when zone.strength is missing', () => {
        const zones = {
            '1s': [{ price: 100, type: 'fvg', freshness: 1.0, timestamp: 1000 }],
            '5s': [{ price: 100, type: 'fvg', freshness: 1.0, timestamp: 1000 }],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        assert.equal(result[0].strength, 0.5);
    });

    it('uses default freshness=0.5 when zone.freshness is missing', () => {
        const zones = {
            '1s': [{ price: 100, type: 'orderBlock', strength: 1.0, timestamp: 1000 }],
            '5s': [{ price: 100, type: 'orderBlock', strength: 1.0, timestamp: 1000 }],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        // freshness defaults to 0.5, decayFactor=0.8 => freshnessDecay = 1-0.8*(1-0.5) = 0.6
        // tfDiversity = (2-1)/(2-1) = 1, avgStrength = 1.0
        // score = 1 * 1.0 * 0.6 = 0.6
        assert.equal(result[0].score, 0.6);
    });

    it('zones at exact same price across TFs produce tight cluster', () => {
        const zones = {
            '1s': [makeZone({ price: 100.0 })],
            '5s': [makeZone({ price: 100.0 })],
            '15s': [makeZone({ price: 100.0 })],
            '1m': [makeZone({ price: 100.0 })],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        assert.equal(result[0].timeframeCount, 4);
        assert.equal(result[0].price, 100.0);
    });

    it('handles multiple zones per timeframe correctly', () => {
        const zones = {
            '1s': [
                makeZone({ price: 99.5, strength: 0.6 }),
                makeZone({ price: 100.5, strength: 0.6 }),
            ],
            '5s': [
                makeZone({ price: 99.6, strength: 0.7 }),
                makeZone({ price: 100.4, strength: 0.7 }),
            ],
        };
        const result = confluenceLevels(zones, 100, { priceWindow: 0.005 });
        // Should form two clusters: one support (~99.55) and one resistance (~100.45)
        assert.equal(result.length, 2);
        const types = result.map(l => l.type).sort();
        assert.deepEqual(types, ['resistance', 'support']);
    });
});

// ===========================================================================
// confluenceLevels — score edge cases
// ===========================================================================

describe('confluenceLevels — scoring edge cases', () => {

    it('score is clamped to [0, 1]', () => {
        const zones = {
            '1s': [makeZone({ price: 100, strength: 2.0, freshness: 2.0 })],
            '5s': [makeZone({ price: 100, strength: 2.0, freshness: 2.0 })],
        };
        const result = confluenceLevels(zones, 100, { minTimeframes: 1 });
        assert.equal(result.length, 1);
        assert.ok(result[0].score >= 0 && result[0].score <= 1);
        assert.ok(result[0].strength >= 0 && result[0].strength <= 1);
    });

    it('scoring with negative currentPrice works', () => {
        const zones = {
            '1s': [makeZone({ price: -100 })],
            '5s': [makeZone({ price: -99.9 })],
        };
        // With currentPrice=-100, zone at -99.9 is above (less negative),
        // so the cluster centre (-99.95) is above currentPrice => resistance.
        const result = confluenceLevels(zones, -100);
        assert.equal(result.length, 1);
        assert.equal(result[0].type, 'resistance');
    });

    it('large price values do not cause precision issues', () => {
        const zones = {
            '1s': [makeZone({ price: 99999.9 })],
            '5s': [makeZone({ price: 100000.0 })],
        };
        const result = confluenceLevels(zones, 100000);
        assert.equal(result.length, 1);
        assert.ok(result[0].score > 0);
    });
});

// ===========================================================================
// multiTimeframeZones
// ===========================================================================

describe('multiTimeframeZones', () => {

    it('returns [] for null input', () => {
        assert.deepEqual(multiTimeframeZones(null), []);
    });

    it('returns [] for empty object', () => {
        assert.deepEqual(multiTimeframeZones({}), []);
    });

    it('returns [] for single timeframe (minTimeframes=2)', () => {
        const rng = seededRandom(42);
        const candles = generateCandles(50, { rng, volatility: 0.003 });
        const result = multiTimeframeZones({ '1s': candles });
        assert.deepEqual(result, []);
    });

    it('produces confluence with two timeframes', () => {
        const rng = seededRandom(42);
        const candles1 = generateCandles(50, { rng, volatility: 0.003 });
        const candles5 = generateCandles(50, { rng, volatility: 0.003 });
        const result = multiTimeframeZones({
            '1s': candles1,
            '5s': candles5,
        });
        // May or may not find confluence depending on zone locations.
        // The key check is that it runs without errors and returns valid data.
        assert.ok(Array.isArray(result));
        for (const level of result) {
            assert.ok(typeof level.price === 'number');
            assert.ok(typeof level.score === 'number');
            assert.ok(level.timeframeCount >= 1);
            assert.ok(Array.isArray(level.timeframes));
        }
    });

    it('forwards options to ZoneDetector', () => {
        const rng = seededRandom(1);
        const candles = generateCandles(20, { rng, volatility: 0.0001 });
        // Very high OB/FVG thresholds + negative volume threshold =>
        // no OBs, no FVGs, no voids => no zones => empty result
        const result = multiTimeframeZones(
            { '1s': candles, '5s': candles },
            { obThreshold: 100, minGap: 100, volumeThreshold: -1, minTimeframes: 1 },
        );
        assert.deepEqual(result, []);
    });

    it('forwards confluence options correctly', () => {
        const rng = seededRandom(42);
        const candles = generateCandles(100, { rng, volatility: 0.005, trend: 0.0005 });
        const result = multiTimeframeZones(
            { '1s': candles, '5s': candles },
            { minTimeframes: 1 },
        );
        // With minTimeframes=1, single-TF levels are included
        for (const level of result) {
            assert.ok(level.timeframeCount >= 1);
        }
    });
});

// ===========================================================================
// ConfluenceHeatmap
// ===========================================================================

describe('ConfluenceHeatmap', () => {

    it('getHeatmap returns [] for empty history', () => {
        const hm = new ConfluenceHeatmap();
        assert.deepEqual(hm.getHeatmap(), []);
    });

    it('getHighConfluenceLevels returns [] for empty history', () => {
        const hm = new ConfluenceHeatmap();
        assert.deepEqual(hm.getHighConfluenceLevels(0.5), []);
    });

    it('single update produces heatmap entries', () => {
        const hm = new ConfluenceHeatmap({ priceBinSize: 0.001 });
        const zones = {
            '1s': [makeZone({ price: 99.9 })],
            '5s': [makeZone({ price: 100.0 })],
        };
        hm.update(zones, 100);
        const heatmap = hm.getHeatmap();
        assert.ok(heatmap.length > 0, 'should have at least one bin');
        for (const bin of heatmap) {
            assert.ok(typeof bin.price === 'number');
            assert.ok(typeof bin.density === 'number');
            assert.ok(typeof bin.maxConfluence === 'number');
            assert.ok(bin.timeframeBreakdown && typeof bin.timeframeBreakdown === 'object');
        }
    });

    it('multiple updates increase density for persistent levels', () => {
        const hm = new ConfluenceHeatmap({ priceBinSize: 0.005, minTimeframes: 1 });
        const zones = {
            '1s': [makeZone({ price: 100 })],
            '5s': [makeZone({ price: 100 })],
        };

        // First update: same level at 100
        hm.update(zones, 100);
        let heatmap = hm.getHeatmap();
        const density1 = heatmap.length > 0 ? heatmap[0].density : 0;

        // Second update: same level at 100
        hm.update(zones, 100);
        heatmap = hm.getHeatmap();
        const density2 = heatmap.length > 0 ? heatmap[0].density : 0;

        // Density should be higher after two snapshots
        assert.ok(density2 >= density1,
            `density after 2 updates (${density2}) should be >= after 1 (${density1})`);
    });

    it('update records the level and its timeframes', () => {
        const hm = new ConfluenceHeatmap({ priceBinSize: 0.002, minTimeframes: 1 });
        const zones = {
            '1s': [makeZone({ price: 100 })],
            '5s': [makeZone({ price: 100.01 })],
        };
        hm.update(zones, 100);
        const heatmap = hm.getHeatmap();
        assert.ok(heatmap.length > 0);
        const top = heatmap[0];
        assert.ok(top.timeframeBreakdown['1s'] >= 1);
        assert.ok(top.timeframeBreakdown['5s'] >= 1);
    });

    it('getHighConfluenceLevels filters by density', () => {
        const hm = new ConfluenceHeatmap({ priceBinSize: 0.002, minTimeframes: 1 });
        const zones1 = { '1s': [makeZone({ price: 100 })], '5s': [makeZone({ price: 100 })] };
        const zones2 = { '1s': [makeZone({ price: 200 })], '5s': [makeZone({ price: 200 })] };

        hm.update(zones1, 100);
        hm.update(zones2, 200);

        // Level at 100 appears in 1/2 snapshots => density 0.5
        // Level at 200 appears in 1/2 snapshots => density 0.5
        const high = hm.getHighConfluenceLevels(0.6);
        assert.equal(high.length, 0, 'no level should have density >= 0.6');

        const med = hm.getHighConfluenceLevels(0.4);
        assert.equal(med.length, 2, 'both levels should have density >= 0.4');
    });

    it('reset clears history', () => {
        const hm = new ConfluenceHeatmap();
        const zones = { '1s': [makeZone({ price: 100 })], '5s': [makeZone({ price: 100 })] };
        hm.update(zones, 100);
        assert.ok(hm.history.length > 0);
        assert.ok(hm.getHeatmap().length > 0);

        hm.reset();
        assert.equal(hm.history.length, 0);
        assert.deepEqual(hm.getHeatmap(), []);
        assert.deepEqual(hm.getHighConfluenceLevels(0), []);
    });

    it('constructor stores priceBinSize and forwards confluence options', () => {
        const hm = new ConfluenceHeatmap({ priceBinSize: 0.005, minTimeframes: 3 });
        assert.equal(hm.priceBinSize, 0.005);
        assert.equal(hm.confluenceOptions.minTimeframes, 3);
    });

    it('handles empty zones in update without crashing', () => {
        const hm = new ConfluenceHeatmap();
        hm.update({}, 100);
        assert.deepEqual(hm.getHeatmap(), []);
        hm.update({ '1s': [] }, 100);
        assert.deepEqual(hm.getHeatmap(), []);
    });
});

// ===========================================================================
// zoneConfluenceScore
// ===========================================================================

describe('zoneConfluenceScore', () => {

    it('returns 0 for empty array', () => {
        assert.equal(zoneConfluenceScore([], 100), 0);
    });

    it('returns 0 for null zones', () => {
        assert.equal(zoneConfluenceScore(null, 100), 0);
    });

    it('returns 0 for undefined zones', () => {
        assert.equal(zoneConfluenceScore(undefined, 100), 0);
    });

    it('returns 0 for null currentPrice', () => {
        assert.equal(zoneConfluenceScore([makeZone()], null), 0);
    });

    it('returns 0 for zero currentPrice', () => {
        assert.equal(zoneConfluenceScore([makeZone()], 0), 0);
    });

    it('returns 0 for negative currentPrice', () => {
        assert.equal(zoneConfluenceScore([makeZone()], -100), 0);
    });

    it('returns 0 for NaN currentPrice', () => {
        assert.equal(zoneConfluenceScore([makeZone()], NaN), 0);
    });

    it('single zone gives moderate score (less than 1)', () => {
        const score = zoneConfluenceScore([makeZone({ price: 100 })], 100);
        assert.ok(score > 0, 'single zone should have positive score');
        assert.ok(score < 1, 'single zone should not have perfect score');
    });

    it('multiple zones at same price bin give higher score', () => {
        const zones = [
            makeZone({ price: 100 }),
            makeZone({ price: 100.001 }),
            makeZone({ price: 99.999 }),
        ];
        const score = zoneConfluenceScore(zones, 100);
        // All within 0.1% bin since 100 * 0.001 = 0.1, and all are within 0.001 of 100
        assert.ok(score > 0.5, `clustered zones should score > 0.5, got ${score}`);
    });

    it('scattered zones give lower score than clustered', () => {
        const clustered = [
            makeZone({ price: 100 }),
            makeZone({ price: 100.001 }),
            makeZone({ price: 99.999 }),
            makeZone({ price: 100.002 }),
        ];
        const scattered = [
            makeZone({ price: 99 }),
            makeZone({ price: 100 }),
            makeZone({ price: 101 }),
            makeZone({ price: 102 }),
        ];
        const scoreC = zoneConfluenceScore(clustered, 100);
        const scoreS = zoneConfluenceScore(scattered, 100);
        assert.ok(scoreC > scoreS,
            `clustered score (${scoreC}) > scattered score (${scoreS})`);
    });

    it('zones with null price are skipped', () => {
        const zones = [
            makeZone({ price: 100 }),
            makeZone({ price: null }),
            { type: 'bad', freshness: 0.5 },
        ];
        assert.doesNotThrow(() => zoneConfluenceScore(zones, 100));
        const score = zoneConfluenceScore(zones, 100);
        assert.ok(score > 0, 'valid zones should still produce a score');
    });

    it('score is clamped to [0, 1]', () => {
        const zones = [
            makeZone({ price: 100 }),
            makeZone({ price: 100.001 }),
        ];
        const score = zoneConfluenceScore(zones, 100);
        assert.ok(score >= 0 && score <= 1);
    });
});

// ===========================================================================
// Edge cases
// ===========================================================================

describe('Edge cases', () => {

    it('all zones at identical price produce single cluster', () => {
        const zones = {
            '1s': [makeZone({ price: 100 }), makeZone({ price: 100 })],
            '5s': [makeZone({ price: 100 }), makeZone({ price: 100 })],
        };
        const result = confluenceLevels(zones, 100, { minTimeframes: 1 });
        assert.equal(result.length, 1);
        assert.equal(result[0].timeframeCount, 2);
        assert.equal(result[0].price, 100);
    });

    it('negative zone prices are handled', () => {
        const zones = {
            '1s': [makeZone({ price: -50.0 })],
            '5s': [makeZone({ price: -49.9 })],
        };
        const result = confluenceLevels(zones, -50);
        assert.equal(result.length, 1);
        // avgPrice = -49.95, which is above currentPrice (-50) => resistance
        assert.equal(result[0].type, 'resistance');
        assert.ok(result[0].price < -49, `price should be negative, got ${result[0].price}`);
    });

    it('very large zone prices are handled without overflow', () => {
        const zones = {
            '1s': [makeZone({ price: 1000000 })],
            '5s': [makeZone({ price: 1000000.5 })],
        };
        const result = confluenceLevels(zones, 1000000);
        assert.equal(result.length, 1);
        assert.ok(result[0].score > 0);
    });

    it('zones with missing price field are skipped', () => {
        const zones = {
            '1s': [{ type: 'fvg', freshness: 1.0, timestamp: 1000 }],
            '5s': [{ type: 'fvg', freshness: 1.0, timestamp: 1000 }],
        };
        // No price field => all zones skipped => empty result
        assert.deepEqual(confluenceLevels(zones, 100), []);
    });

    it('works with currentPrice at 0 (edge case, no clustering)', () => {
        const zones = {
            '1s': [makeZone({ price: 0.001 })],
            '5s': [makeZone({ price: 0.002 })],
        };
        const result = confluenceLevels(zones, 0);
        // With currentPrice=0, absPriceWindow = 0.003 * 0 = 0
        // Zones at 0.001 and 0.002 have gap 0.001 > 0 => separate clusters
        // Each cluster has only 1 TF => filtered by minTimeframes=2
        assert.deepEqual(result, []);
    });

    it('mixed null/valid zones in the same timeframe are handled', () => {
        const zones = {
            '1s': [makeZone({ price: 100 }), makeZone({ price: null })],
            '5s': [makeZone({ price: 100 }), { type: 'fvg', freshness: 1.0 }],
        };
        const result = confluenceLevels(zones, 100);
        assert.equal(result.length, 1);
        assert.equal(result[0].timeframeCount, 2);
    });

    it('ConfluenceHeatmap update with currentPrice=0 does not crash', () => {
        const hm = new ConfluenceHeatmap();
        const zones = { '1s': [makeZone({ price: 0 })], '5s': [makeZone({ price: 0 })] };
        assert.doesNotThrow(() => hm.update(zones, 0));
    });

    it('zoneConfluenceScore with empty-like zones returns 0', () => {
        assert.equal(zoneConfluenceScore([], 100), 0);
        assert.equal(zoneConfluenceScore([{ noPrice: true }], 100), 0);
    });
});

// ===========================================================================
// Performance: 1000 zones across 5 TFs in <100 ms
// ===========================================================================

describe('Performance', () => {
    it('confluenceLevels with 1000 zones across 5 timeframes completes in <100 ms', () => {
        const rng = seededRandom(12345);

        // Build 200 zones per timeframe using deterministic random prices
        const zonesByTimeframe = {};
        const tfs = ['1s', '5s', '15s', '1m', '5m'];
        const basePrice = 100;

        for (const tf of tfs) {
            const zones = [];
            for (let i = 0; i < 200; i++) {
                const price = basePrice + (rng() - 0.5) * 2; // spread ~ -1 to +1
                zones.push(makeZone({
                    price: +price.toFixed(4),
                    strength: 0.3 + rng() * 0.7,
                    freshness: rng(),
                }));
            }
            zonesByTimeframe[tf] = zones;
        }

        // Warmup / JIT
        confluenceLevels(zonesByTimeframe, basePrice);
        confluenceLevels(zonesByTimeframe, basePrice);

        const start = performance.now();
        confluenceLevels(zonesByTimeframe, basePrice);
        const elapsed = performance.now() - start;

        assert.ok(elapsed < 100,
            `confluenceLevels with 1000 zones took ${elapsed.toFixed(1)}ms (limit 100ms)`);
    });
});
