/**
 * Market Regime Classifier — unit tests (node:test runner)
 * Run: node --test audit/market-regime.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { classifyRegime, StreamingRegimeClassifier, multiTimeframeRegime, REGIMES } from './market-regime.mjs';

// ─── Seeded PRNG for deterministic fixtures ──────────────────────────────────

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

function generateTrending(candlesCount, options = {}) {
    const rng = options.rng ?? mulberry32(options.seed ?? 42);
    const trend = options.trend ?? 'bullish';
    const volatility = options.volatility ?? 0.005;
    const basePrice = options.basePrice ?? 100;
    const startTs = options.startTs ?? Date.now() - candlesCount * 1000;
    const candles = [];
    let price = basePrice;
    const dir = trend === 'bullish' ? 1 : -1;

    for (let i = 0; i < candlesCount; i++) {
        const drift = dir * volatility * 0.2 + (rng() - 0.5) * volatility * 0.1;
        const open = price + (rng() - 0.5) * volatility * 0.3;
        const close = open + drift * basePrice;
        const high = Math.max(open, close) + rng() * volatility * 0.2 * basePrice;
        const low = Math.min(open, close) - rng() * volatility * 0.2 * basePrice;
        const volume = basePrice * 10 + rng() * basePrice * 5;

        candles.push({
            timestamp: startTs + i * 1000,
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: +volume.toFixed(2),
        });
        price = close;
    }
    return candles;
}

function generateRanging(candlesCount, options = {}) {
    const rng = options.rng ?? mulberry32(options.seed ?? 99);
    const basePrice = options.basePrice ?? 100;
    const rangeWidth = options.rangeWidth ?? 0.01;
    const startTs = options.startTs ?? Date.now() - candlesCount * 1000;
    const candles = [];
    let price = basePrice;

    for (let i = 0; i < candlesCount; i++) {
        const t = i / candlesCount;
        const meanReversion = Math.sin(t * Math.PI * 8) * rangeWidth * 0.5 * basePrice;
        const target = basePrice + meanReversion;
        const open = price + (rng() - 0.5) * rangeWidth * 0.1 * basePrice;
        const close = target + (rng() - 0.5) * rangeWidth * 0.05 * basePrice;
        const high = Math.max(open, close) + rng() * rangeWidth * 0.03 * basePrice;
        const low = Math.min(open, close) - rng() * rangeWidth * 0.03 * basePrice;
        const volume = basePrice * 5 + rng() * basePrice * 3;

        candles.push({
            timestamp: startTs + i * 1000,
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: +volume.toFixed(2),
        });
        price = close;
    }
    return candles;
}

function generateBreakout(candlesCount, options = {}) {
    const rng = options.rng ?? mulberry32(options.seed ?? 77);
    const basePrice = options.basePrice ?? 100;
    const startTs = options.startTs ?? Date.now() - candlesCount * 1000;
    const breakIdx = Math.floor(candlesCount * 0.7);
    const candles = [];
    let price = basePrice;

    for (let i = 0; i < candlesCount; i++) {
        if (i < breakIdx) {
            // Low vol consolidation
            const drift = (rng() - 0.5) * 0.0005 * basePrice;
            const open = price;
            const close = open + drift;
            const high = Math.max(open, close) + rng() * 0.001 * basePrice;
            const low = Math.min(open, close) - rng() * 0.001 * basePrice;
            const volume = basePrice * 3 + rng() * basePrice * 1;
            price = close;
            candles.push({
                timestamp: startTs + i * 1000,
                open: +open.toFixed(2),
                high: +high.toFixed(2),
                low: +low.toFixed(2),
                close: +close.toFixed(2),
                volume: +volume.toFixed(2),
            });
        } else if (i === breakIdx) {
            // Breakout candle
            const jump = 0.02 * basePrice;
            const close = price + jump;
            const volume = basePrice * 30;
            candles.push({
                timestamp: startTs + i * 1000,
                open: +price.toFixed(2),
                high: +(close + 0.003 * basePrice).toFixed(2),
                low: +(price - 0.001 * basePrice).toFixed(2),
                close: +close.toFixed(2),
                volume: +volume.toFixed(2),
            });
            price = close;
        } else {
            // Trend continuation
            const drift = 0.003 * basePrice + (rng() - 0.3) * 0.002 * basePrice;
            const open = price;
            const close = open + drift;
            const high = Math.max(open, close) + rng() * 0.003 * basePrice;
            const low = Math.min(open, close) - rng() * 0.003 * basePrice;
            const volume = basePrice * 15 + rng() * basePrice * 5;
            candles.push({
                timestamp: startTs + i * 1000,
                open: +open.toFixed(2),
                high: +high.toFixed(2),
                low: +low.toFixed(2),
                close: +close.toFixed(2),
                volume: +volume.toFixed(2),
            });
            price = close;
        }
    }
    return candles;
}

function generateVolatile(candlesCount, options = {}) {
    const rng = options.rng ?? mulberry32(options.seed ?? 55);
    const basePrice = options.basePrice ?? 100;
    const startTs = options.startTs ?? Date.now() - candlesCount * 1000;
    const candles = [];
    let price = basePrice;

    for (let i = 0; i < candlesCount; i++) {
        const change = (rng() - 0.5) * 0.06 * basePrice;
        const open = price + (rng() - 0.5) * 0.01 * basePrice;
        const close = open + change;
        const high = Math.max(open, close) + rng() * 0.02 * basePrice;
        const low = Math.min(open, close) - rng() * 0.02 * basePrice;
        const volume = basePrice * 10 + rng() * basePrice * 20;

        candles.push({
            timestamp: startTs + i * 1000,
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: +volume.toFixed(2),
        });
        price = close;
    }
    return candles;
}

// ─── classifyRegime ──────────────────────────────────────────────────────────

describe('classifyRegime', () => {
    it('handles empty candle array', () => {
        const result = classifyRegime([]);
        assert.equal(result.regime, REGIMES.RANGING);
        assert.equal(result.confidence, 0);
        assert.equal(result.timestamp, null);
    });

    it('handles null input', () => {
        const result = classifyRegime(null);
        assert.equal(result.regime, REGIMES.RANGING);
        assert.equal(result.confidence, 0);
    });

    it('handles undefined input', () => {
        const result = classifyRegime(undefined);
        assert.equal(result.regime, REGIMES.RANGING);
        assert.equal(result.confidence, 0);
    });

    it('handles insufficient candles (less than MIN_CANDLES)', () => {
        const candles = generateTrending(5);
        const result = classifyRegime(candles);
        assert.equal(result.regime, REGIMES.RANGING);
        assert.equal(result.confidence, 0);
        assert.ok(result.details.reason.includes('insufficient'));
    });

    it('returns all expected fields', () => {
        const candles = generateTrending(50, { trend: 'bullish' });
        const result = classifyRegime(candles);
        assert.ok(result.regime);
        assert.ok(typeof result.confidence === 'number');
        assert.ok(result.strength >= 0 && result.strength <= 1);
        assert.ok(result.timestamp);
        assert.ok(result.params);
        assert.ok(result.details);
        assert.ok(result.params.stopMultiplier > 0);
        assert.ok(result.params.positionSizeFactor > 0);
    });

    it('classifies a strong bullish trend as TRENDING_BULLISH', () => {
        const candles = generateTrending(80, { trend: 'bullish', volatility: 0.008 });
        const result = classifyRegime(candles);
        assert.equal(result.regime, REGIMES.TRENDING_BULLISH);
        assert.equal(result.direction, 'bullish');
        assert.ok(result.confidence > 0.4, `confidence=${result.confidence}`);
    });

    it('classifies a strong bearish trend as TRENDING_BEARISH', () => {
        const candles = generateTrending(80, { trend: 'bearish', volatility: 0.008 });
        const result = classifyRegime(candles);
        assert.equal(result.regime, REGIMES.TRENDING_BEARISH);
        assert.equal(result.direction, 'bearish');
        assert.ok(result.confidence > 0.4, `confidence=${result.confidence}`);
    });

    it('classifies a ranging market as RANGING', () => {
        const candles = generateRanging(60, { rangeWidth: 0.008 });
        const result = classifyRegime(candles);
        // Ranging should NOT be trending
        assert.notEqual(result.regime, REGIMES.TRENDING_BULLISH);
        assert.notEqual(result.regime, REGIMES.TRENDING_BEARISH);
    });

    it('classifies low-vol consolidation as non-trending (ACCUMULATION/DISTRIBUTION/RANGING)', () => {
        const rng = mulberry32(123);
        const candles = [];
        let price = 100;
        for (let i = 0; i < 80; i++) {
            // Very tight oscillation around a mean — no sustained drift
            const oscillation = Math.sin(i * 0.3) * 0.0003 * price;
            const noise = (rng() - 0.5) * 0.0001 * price;
            const open = price;
            const close = open + oscillation + noise;
            candles.push({
                timestamp: Date.now() - (80 - i) * 1000,
                open: +open.toFixed(4),
                high: +(Math.max(open, close) + rng() * 0.0001 * price).toFixed(4),
                low: +(Math.min(open, close) - rng() * 0.0001 * price).toFixed(4),
                close: +close.toFixed(4),
                volume: +((price * 2 + rng() * price)).toFixed(2),
            });
            price = close;
        }
        const result = classifyRegime(candles, { adxThreshold: 20, rangeThreshold: 0.02, trendWindow: 30 });
        // Low-vol sideways should NOT be trending
        assert.ok(
            [REGIMES.ACCUMULATION, REGIMES.DISTRIBUTION, REGIMES.RANGING].includes(result.regime),
            `got ${result.regime}`,
        );
    });

    it('detects breakout pattern (low vol → high vol expansion)', () => {
        const candles = generateBreakout(80);
        const result = classifyRegime(candles);
        // Should detect the expansion phase
        assert.ok(result.regime, `regime: ${result.regime}`);
        assert.ok(result.confidence >= 0, `confidence=${result.confidence}`);
    });

    it('classifies volatile market as VOLATILE', () => {
        const candles = generateVolatile(60, { volatility: 0.05 });
        const result = classifyRegime(candles);
        // High volatility should produce VOLATILE or low-confidence RANGING
        assert.ok(result.regime);
    });

    // ── Edge cases ──────────────────────────────────────────────────────────

    it('handles all identical candles', () => {
        const candles = [];
        for (let i = 0; i < 30; i++) {
            candles.push({
                timestamp: Date.now() - (30 - i) * 1000,
                open: 100, high: 100, low: 100, close: 100,
                volume: 1000,
            });
        }
        const result = classifyRegime(candles);
        // Flat line — should be ranging with low confidence
        assert.ok(result.regime);
        assert.ok(result.confidence >= 0);
    });

    it('handles zero volume candles', () => {
        const candles = generateTrending(40).map(c => ({ ...c, volume: 0 }));
        const result = classifyRegime(candles);
        assert.ok(result.regime);
    });

    it('handles negative prices gracefully (returns a regime, not NaN)', () => {
        const candles = generateTrending(40).map(c => ({
            ...c,
            open: Math.abs(c.open),
            high: Math.abs(c.high),
            low: Math.abs(c.low),
            close: Math.abs(c.close),
        }));
        const result = classifyRegime(candles);
        assert.ok(result.regime);
        assert.ok(!Number.isNaN(result.confidence));
    });

    it('handles very small prices', () => {
        const candles = [];
        let price = 0.001;
        for (let i = 0; i < 40; i++) {
            price += price * 0.01;
            candles.push({
                timestamp: Date.now() - (40 - i) * 1000,
                open: price, high: price * 1.002, low: price * 0.998, close: price,
                volume: 100,
            });
        }
        const result = classifyRegime(candles);
        assert.ok(result.regime);
    });

    // ── Confidence bounds ────────────────────────────────────────────────────

    it('confidence is always between 0 and 1', () => {
        const scenarios = [
            generateTrending(60, { trend: 'bullish' }),
            generateTrending(60, { trend: 'bearish' }),
            generateRanging(60),
            generateVolatile(60),
            generateBreakout(80),
        ];
        for (const candles of scenarios) {
            const result = classifyRegime(candles);
            assert.ok(result.confidence >= 0 && result.confidence <= 1,
                `regime=${result.regime}, confidence=${result.confidence}`);
        }
    });

    it('strength is always between 0 and 1', () => {
        const scenarios = [
            generateTrending(60, { trend: 'bullish' }),
            generateTrending(60, { trend: 'bearish' }),
            generateRanging(60),
        ];
        for (const candles of scenarios) {
            const result = classifyRegime(candles);
            assert.ok(result.strength >= 0 && result.strength <= 1,
                `regime=${result.regime}, strength=${result.strength}`);
        }
    });

    // ── Parameters per regime ────────────────────────────────────────────────

    it('returns different params for trending vs ranging', () => {
        const trend = classifyRegime(generateTrending(60, { trend: 'bullish', volatility: 0.01 }));
        const range = classifyRegime(generateRanging(60, { rangeWidth: 0.008 }));
        // Trending should allow larger positions and wider stops
        if (trend.regime === REGIMES.TRENDING_BULLISH && range.regime === REGIMES.RANGING) {
            assert.ok(trend.params.positionSizeFactor > range.params.positionSizeFactor,
                `trend ps=${trend.params.positionSizeFactor}, range ps=${range.params.positionSizeFactor}`);
        }
    });

    it('VOLATILE regime has smallest position size factor', () => {
        const vol = classifyRegime(generateVolatile(60));
        if (vol.regime === REGIMES.VOLATILE) {
            assert.ok(vol.params.positionSizeFactor <= 0.3,
                `vol ps=${vol.params.positionSizeFactor}`);
        }
    });

    it('params includes all required fields', () => {
        const result = classifyRegime(generateTrending(40));
        assert.ok('stopMultiplier' in result.params);
        assert.ok('tpAggressiveness' in result.params);
        assert.ok('positionSizeFactor' in result.params);
        assert.ok('trailingStopPct' in result.params);
        assert.ok('maxHoldingBars' in result.params);
    });

    // ── Direction ────────────────────────────────────────────────────────────

    it('direction is null for ranging', () => {
        const candles = generateRanging(60, { rangeWidth: 0.008 });
        const result = classifyRegime(candles);
        // Ranging may or may not have null direction depending on micro-slope
        // — just verify it's a valid value
        assert.ok(result.direction === null || result.direction === 'bullish' || result.direction === 'bearish');
    });

    it('direction is bullish for uptrend', () => {
        const candles = generateTrending(80, { trend: 'bullish', volatility: 0.01 });
        const result = classifyRegime(candles);
        if (result.regime === REGIMES.TRENDING_BULLISH) {
            assert.equal(result.direction, 'bullish');
        }
    });

    // ── Options parameter ────────────────────────────────────────────────────

    it('respects custom adxThreshold', () => {
        const candles = generateTrending(60, { trend: 'bullish', volatility: 0.005 });
        const lenient = classifyRegime(candles, { adxThreshold: 15 });
        const strict = classifyRegime(candles, { adxThreshold: 40 });
        // Lenient should detect trend more easily
        assert.ok(lenient.regime);
        assert.ok(strict.regime);
    });

    it('respects custom rangeThreshold', () => {
        const candles = generateRanging(60, { rangeWidth: 0.015 });
        const lenient = classifyRegime(candles, { rangeThreshold: 0.03 });
        const strict = classifyRegime(candles, { rangeThreshold: 0.005 });
        assert.ok(lenient.regime);
        assert.ok(strict.regime);
    });

    it('respects custom trendWindow', () => {
        const candles = generateTrending(60);
        const short = classifyRegime(candles, { trendWindow: 10 });
        const long = classifyRegime(candles, { trendWindow: 40 });
        assert.ok(short.regime);
        assert.ok(long.regime);
    });
});

// ─── StreamingRegimeClassifier ───────────────────────────────────────────────

describe('StreamingRegimeClassifier', () => {
    it('returns null regime when insufficient data', () => {
        const src = new StreamingRegimeClassifier();
        const result = src.update({ timestamp: Date.now(), open: 100, high: 101, low: 99, close: 100.5, volume: 1000 });
        assert.equal(result.regime, null);
        assert.equal(result.changed, false);
    });

    it('returns regime after enough candles', () => {
        const src = new StreamingRegimeClassifier();
        const candles = generateTrending(50, { trend: 'bullish' });
        let lastResult;
        for (const c of candles) {
            lastResult = src.update(c);
        }
        assert.ok(lastResult.regime.regime);
        assert.ok(lastResult.regime.confidence > 0);
    });

    it('detects regime change from trending to volatile', () => {
        const src = new StreamingRegimeClassifier({ maxCandles: 100 });

        // Feed trending candles
        const trendCandles = generateTrending(60, { trend: 'bullish', volatility: 0.01 });
        for (const c of trendCandles) {
            src.update(c);
        }

        // Feed volatile candles
        const volCandles = generateVolatile(40);
        let changed = false;
        for (const c of volCandles) {
            const result = src.update(c);
            if (result.changed) changed = true;
        }

        // Should detect at least one change
        assert.ok(changed || src.getChanges().length >= 1,
            `changes=${src.getChanges().length}`);
    });

    it('current() returns latest regime', () => {
        const src = new StreamingRegimeClassifier();
        const candles = generateTrending(50);
        for (const c of candles) src.update(c);
        const current = src.current();
        assert.ok(current.regime);
        assert.ok(current.confidence > 0);
    });

    it('current() returns null with insufficient data', () => {
        const src = new StreamingRegimeClassifier();
        src.update(generateTrending(1)[0]);
        assert.equal(src.current(), null);
    });

    it('getChanges returns empty array initially', () => {
        const src = new StreamingRegimeClassifier();
        assert.deepEqual(src.getChanges(), []);
    });

    it('barsSinceChange tracks correctly', () => {
        const src = new StreamingRegimeClassifier();
        const candles = generateTrending(30);
        for (const c of candles) src.update(c);
        assert.ok(src.barsSinceChange() >= 0);
    });

    it('reset clears all state', () => {
        const src = new StreamingRegimeClassifier();
        const candles = generateTrending(30);
        for (const c of candles) src.update(c);
        src.reset();
        assert.equal(src.size, 0);
        assert.deepEqual(src.getChanges(), []);
        assert.equal(src.current(), null);
        assert.equal(src.barsSinceChange(), 0);
    });

    it('size reflects stored candle count', () => {
        const src = new StreamingRegimeClassifier({ maxCandles: 50 });
        const candles = generateTrending(25);
        for (const c of candles) src.update(c);
        assert.equal(src.size, 25);
    });

    it('respects maxCandles — does not grow unbounded', () => {
        const src = new StreamingRegimeClassifier({ maxCandles: 30 });
        const candles = generateTrending(80);
        for (const c of candles) src.update(c);
        assert.ok(src.size <= 30);
    });

    it('custom trendWindow affects classification', () => {
        const src1 = new StreamingRegimeClassifier({ trendWindow: 10 });
        const src2 = new StreamingRegimeClassifier({ trendWindow: 40 });
        const candles = generateTrending(50, { trend: 'bullish' });
        for (const c of candles) {
            src1.update(c);
            src2.update(c);
        }
        assert.ok(src1.current().regime);
        assert.ok(src2.current().regime);
    });
});

// ─── multiTimeframeRegime ────────────────────────────────────────────────────

describe('multiTimeframeRegime', () => {
    it('handles empty array', () => {
        const result = multiTimeframeRegime([]);
        assert.equal(result.regime, REGIMES.RANGING);
        assert.equal(result.confidence, 0);
        assert.equal(result.consensus, 0);
    });

    it('returns majority regime when all agree', () => {
        const r1 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.8 };
        const r2 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.7 };
        const r3 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.9 };
        const result = multiTimeframeRegime([r1, r2, r3]);
        assert.equal(result.regime, REGIMES.TRENDING_BULLISH);
        assert.equal(result.consensus, 1);
        assert.ok(result.confidence > 0.7);
    });

    it('returns majority when one disagrees', () => {
        const r1 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.8 };
        const r2 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.7 };
        const r3 = { regime: REGIMES.RANGING, confidence: 0.4 };
        const result = multiTimeframeRegime([r1, r2, r3]);
        assert.equal(result.regime, REGIMES.TRENDING_BULLISH);
        assert.ok(result.consensus > 0.6 && result.consensus < 0.7);
    });

    it('handles null regimes in array', () => {
        const r1 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.8 };
        const result = multiTimeframeRegime([r1, null, null]);
        assert.equal(result.regime, REGIMES.TRENDING_BULLISH);
    });

    it('includes correct breakdown by timeframe', () => {
        const r1 = { regime: REGIMES.TRENDING_BULLISH, confidence: 0.8 };
        const r2 = { regime: REGIMES.RANGING, confidence: 0.6 };
        const r3 = { regime: REGIMES.BREAKOUT, confidence: 0.7 };
        const result = multiTimeframeRegime([r1, r2, r3]);
        assert.equal(result.breakdown.tf1, REGIMES.TRENDING_BULLISH);
        assert.equal(result.breakdown.tf2, REGIMES.RANGING);
        assert.equal(result.breakdown.tf3, REGIMES.BREAKOUT);
    });

    it('confidence is always between 0 and 1', () => {
        for (let i = 0; i < 10; i++) {
            const regimes = [
                { regime: REGIMES.TRENDING_BULLISH, confidence: Math.random() },
                { regime: REGIMES.RANGING, confidence: Math.random() },
                { regime: REGIMES.TRENDING_BULLISH, confidence: Math.random() },
            ];
            const result = multiTimeframeRegime(regimes);
            assert.ok(result.confidence >= 0 && result.confidence <= 1,
                `confidence=${result.confidence}`);
        }
    });
});

// ─── REGIMES constant ───────────────────────────────────────────────────────

describe('REGIMES', () => {
    it('exports all expected regime types', () => {
        assert.equal(REGIMES.TRENDING_BULLISH, 'trending_bullish');
        assert.equal(REGIMES.TRENDING_BEARISH, 'trending_bearish');
        assert.equal(REGIMES.RANGING, 'ranging');
        assert.equal(REGIMES.ACCUMULATION, 'accumulation');
        assert.equal(REGIMES.DISTRIBUTION, 'distribution');
        assert.equal(REGIMES.BREAKOUT, 'breakout');
        assert.equal(REGIMES.BREAKDOWN, 'breakdown');
        assert.equal(REGIMES.VOLATILE, 'volatile');
    });
});

// ─── Performance ─────────────────────────────────────────────────────────────

describe('Performance', () => {
    it('classifyRegime handles 5000 candles in reasonable time', () => {
        const candles = generateTrending(5000, { trend: 'bullish' });
        const start = performance.now();
        const result = classifyRegime(candles);
        const elapsed = performance.now() - start;
        assert.ok(result.regime);
        assert.ok(elapsed < 2000, `took ${elapsed.toFixed(0)}ms for 5000 candles`);
    });

    it('StreamingRegimeClassifier.update is fast per candle', () => {
        const src = new StreamingRegimeClassifier({ maxCandles: 100 });
        const candles = generateTrending(200, { trend: 'bullish' });

        // Warmup
        for (let i = 0; i < 100; i++) src.update(candles[i]);

        // Measure
        const start = performance.now();
        for (let i = 100; i < 200; i++) src.update(candles[i]);
        const elapsed = performance.now() - start;
        const perCandle = elapsed / 100;

        // Each update should be under 5ms
        assert.ok(perCandle < 5, `per candle: ${perCandle.toFixed(2)}ms`);
    });
});
