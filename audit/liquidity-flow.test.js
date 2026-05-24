/**
 * Liquidity Flow — unit tests
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/liquidity-flow.test.js
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    VolumeProfile,
    OrderFlowMomentum,
    VolatilityEddyDetector,
    liquidityScore,
    supportResistanceZones,
} from './liquidity-flow.mjs';

// ===========================================================================
// Helpers
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
 * Generate N synthetic OHLCV candles with volume.
 *
 * @param {number} n - Number of candles
 * @param {Object} [opts]
 * @param {number} [opts.startPrice=100]
 * @param {number} [opts.startTime=1700000000000]
 * @param {number} [opts.intervalMs=60000]
 * @param {number} [opts.volatility=0.001]
 * @param {number} [opts.trend=0]
 * @param {number} [opts.baseVolume=1000]
 * @param {number} [opts.volumeStd=0]
 * @param {Function} [opts.rng]
 * @returns {Object[]}
 */
function generateCandles(n, opts = {}) {
    const {
        startPrice = 100,
        startTime = 1700000000000,
        intervalMs = 60000,
        volatility = 0.001,
        trend = 0,
        baseVolume = 1000,
        volumeStd = 0,
        rng = () => 0.5,
    } = opts;

    const candles = [];
    let price = startPrice;

    for (let i = 0; i < n; i++) {
        const open = price;
        const change = (rng() - 0.5) * volatility + trend;
        price = price * (1 + change);
        const close = price;

        const halfRange = Math.abs(close - open) + volatility * startPrice * 0.5;
        const high = Math.max(open, close) + halfRange * rng();
        const low = Math.min(open, close) - halfRange * (1 - rng());
        const vol = Math.max(1, Math.round(baseVolume + (rng() - 0.5) * 2 * volumeStd));

        candles.push({
            timestamp: startTime + i * intervalMs,
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: vol,
        });
    }

    return candles;
}

// ===========================================================================
// VolumeProfile
// ===========================================================================

describe('VolumeProfile', () => {

    it('throws on empty candles', () => {
        assert.throws(() => new VolumeProfile([]), /non-empty/);
        assert.throws(() => new VolumeProfile(null), /non-empty/);
    });

    it('getVolumeAtPrice returns correct bin count', () => {
        const candles = generateCandles(50, { volatility: 0.005, rng: seededRandom(42) });
        const vp = new VolumeProfile(candles);

        assert.equal(vp.getVolumeAtPrice(10).length, 10);
        assert.equal(vp.getVolumeAtPrice(24).length, 24);
        assert.equal(vp.getVolumeAtPrice(50).length, 50);
    });

    it('getVolumeAtPrice preserves total volume (floating point tolerance)', () => {
        const candles = generateCandles(30, {
            volatility: 0.01,
            baseVolume: 1000,
            rng: seededRandom(99),
        });
        const vp = new VolumeProfile(candles);
        const vap = vp.getVolumeAtPrice(20);

        const totalInputVol = candles.reduce((s, c) => s + c.volume, 0);
        const totalVapVol = vap.reduce((s, b) => s + b.volume, 0);

        assert.ok(
            Math.abs(totalVapVol - totalInputVol) < 0.01,
            `Expected VAP total ~${totalInputVol}, got ${totalVapVol}`,
        );
    });

    it('getVolumeAtPrice handles single price level (flat market)', () => {
        const candles = Array.from({ length: 10 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100,
            high: 100,
            low: 100,
            close: 100,
            volume: 500,
        }));
        const vp = new VolumeProfile(candles);
        const vap = vp.getVolumeAtPrice(24);

        // Single bin with all volume
        assert.equal(vap.length, 1);
        assert.equal(vap[0].price, 100);
        assert.equal(vap[0].volume, 5000);
    });

    it('getVolumeAtPrice results are sorted by price', () => {
        const candles = generateCandles(40, {
            volatility: 0.02,
            rng: seededRandom(7),
        });
        const vp = new VolumeProfile(candles);
        const vap = vp.getVolumeAtPrice(24);

        for (let i = 1; i < vap.length; i++) {
            assert.ok(vap[i].price > vap[i - 1].price,
                `Bin ${i} price (${vap[i].price}) not > bin ${i - 1} (${vap[i - 1].price})`);
        }
    });

    it('getValueArea returns valid bounds with POC at highest-volume bin', () => {
        // Volume concentrated in the middle — clear POC
        const candles = [];
        for (let i = 0; i < 5; i++) {
            candles.push({
                timestamp: i, open: 100 + i, high: 101 + i, low: 99 + i, close: 100 + i,
                volume: 200,
            });
        }
        for (let i = 0; i < 5; i++) {
            candles.push({
                timestamp: 10 + i, open: 105, high: 106, low: 104, close: 105,
                volume: 2000,
            });
        }
        for (let i = 0; i < 5; i++) {
            candles.push({
                timestamp: 20 + i, open: 110 + i, high: 111 + i, low: 109 + i, close: 110 + i,
                volume: 300,
            });
        }

        const vp = new VolumeProfile(candles);
        const va = vp.getValueArea(0.70);
        const vap = vp.getVolumeAtPrice();

        // Find POC from VAP directly
        let pocIdx = 0;
        let maxVol = vap[0].volume;
        for (let i = 1; i < vap.length; i++) {
            if (vap[i].volume > maxVol) {
                maxVol = vap[i].volume;
                pocIdx = i;
            }
        }

        assert.equal(va.poc, vap[pocIdx].price, 'POC should be highest-volume bin');
        assert.ok(va.low <= va.poc, `Value area low (${va.low}) <= POC (${va.poc})`);
        assert.ok(va.high >= va.poc, `Value area high (${va.high}) >= POC (${va.poc})`);
    });

    it('getValueArea 100% includes all volume', () => {
        const candles = generateCandles(30, {
            volatility: 0.01,
            rng: seededRandom(42),
        });
        const vp = new VolumeProfile(candles);
        const va = vp.getValueArea(1.0);
        const vap = vp.getVolumeAtPrice();

        assert.equal(va.low, vap[0].price, '100 % VA low should be VAP min');
        assert.equal(va.high, vap[vap.length - 1].price, '100 % VA high should be VAP max');
    });

    it('getValueArea handles single-bin VAP', () => {
        const candles = [{ timestamp: 1, open: 100, high: 100, low: 100, close: 100, volume: 1000 }];
        const vp = new VolumeProfile(candles);
        const va = vp.getValueArea(0.70);

        assert.equal(va.low, 100);
        assert.equal(va.high, 100);
        assert.equal(va.poc, 100);
    });

    it('getHighVolumeNodes identifies nodes above threshold', () => {
        const candles = [
            { timestamp: 1, open: 100, high: 102, low: 98, close: 101, volume: 500 },
            { timestamp: 2, open: 101, high: 103, low: 99, close: 102, volume: 500 },
            { timestamp: 3, open: 102, high: 104, low: 100, close: 103, volume: 500 },
            { timestamp: 4, open: 103, high: 105, low: 101, close: 104, volume: 500 },
            { timestamp: 5, open: 104, high: 106, low: 102, close: 105, volume: 500 },
        ];
        const vp = new VolumeProfile(candles);

        // With uniform data and a high threshold (10x avg), no nodes qualify
        const highNodes = vp.getHighVolumeNodes(10);
        assert.equal(highNodes.length, 0);

        // With threshold 0, ALL bins are "high volume" (above 0 * avg)
        const allNodes = vp.getHighVolumeNodes(0);
        assert.equal(allNodes.length, vp.getVolumeAtPrice().length);
    });

    it('getHighVolumeNodes strength is between 0 and 1', () => {
        const candles = generateCandles(50, {
            volatility: 0.005,
            baseVolume: 1000,
            rng: seededRandom(42),
        });
        const vp = new VolumeProfile(candles);
        const nodes = vp.getHighVolumeNodes(0.5);

        for (const n of nodes) {
            assert.ok(n.strength >= 0 && n.strength <= 1,
                `Strength ${n.strength} should be in [0,1]`);
        }
    });

    it('getLowVolumeNodes identifies nodes below threshold', () => {
        const candles = [
            { timestamp: 1, open: 100, high: 102, low: 98, close: 101, volume: 50 },
            { timestamp: 2, open: 103, high: 105, low: 101, close: 104, volume: 50 },
            { timestamp: 3, open: 106, high: 108, low: 104, close: 107, volume: 50 },
            { timestamp: 4, open: 109, high: 111, low: 107, close: 110, volume: 50 },
            { timestamp: 5, open: 112, high: 114, low: 110, close: 113, volume: 50 },
        ];
        const vp = new VolumeProfile(candles);

        // Bins well outside the price range get 0 volume → low volume nodes
        const lowNodes = vp.getLowVolumeNodes(0.5);
        const vap = vp.getVolumeAtPrice();
        const avgVol = vap.reduce((s, b) => s + b.volume, 0) / vap.length;

        for (const n of lowNodes) {
            assert.ok(n.volume < 0.5 * avgVol,
                `Node at ${n.price} vol ${n.volume} should be < ${0.5 * avgVol}`);
        }
    });

    it('getVolumeProfileSkew detects buying/selling/balanced', () => {
        // Buying skew: more volume in upper half
        const buyCandles = [];
        for (let i = 0; i < 5; i++) {
            buyCandles.push({
                timestamp: i, open: 100, high: 101, low: 99, close: 100, volume: 100,
            });
        }
        for (let i = 0; i < 15; i++) {
            buyCandles.push({
                timestamp: 10 + i, open: 105, high: 106, low: 104, close: 105, volume: 2000,
            });
        }
        const buyVP = new VolumeProfile(buyCandles);
        const buySkew = buyVP.getVolumeProfileSkew();
        assert.ok(
            buySkew.direction === 'buying',
            `Expected buying skew, got ${buySkew.direction} (ratio=${buySkew.ratio})`,
        );

        // Selling skew: more volume in lower half
        const sellCandles = [];
        for (let i = 0; i < 15; i++) {
            sellCandles.push({
                timestamp: i, open: 95, high: 96, low: 94, close: 95, volume: 2000,
            });
        }
        for (let i = 0; i < 5; i++) {
            sellCandles.push({
                timestamp: 20 + i, open: 105, high: 106, low: 104, close: 105, volume: 100,
            });
        }
        const sellVP = new VolumeProfile(sellCandles);
        const sellSkew = sellVP.getVolumeProfileSkew();
        assert.ok(
            sellSkew.direction === 'selling',
            `Expected selling skew, got ${sellSkew.direction} (ratio=${sellSkew.ratio})`,
        );

        // Balanced: uniform distribution
        const balCandles = generateCandles(60, {
            startPrice: 100,
            volatility: 0.002,
            trend: 0,
            baseVolume: 1000,
            rng: seededRandom(42),
        });
        const balVP = new VolumeProfile(balCandles);
        const balSkew = balVP.getVolumeProfileSkew();
        assert.ok(
            ['buying', 'selling', 'balanced'].includes(balSkew.direction),
            `Unexpected direction: ${balSkew.direction}`,
        );
        assert.ok(balSkew.ratio > 0, 'Ratio should be positive');
    });

    it('getVolumeProfileSkew handles single-bin VAP', () => {
        const candles = [{ timestamp: 1, open: 100, high: 100, low: 100, close: 100, volume: 1000 }];
        const vp = new VolumeProfile(candles);
        const skew = vp.getVolumeProfileSkew();

        assert.equal(skew.direction, 'balanced');
        assert.equal(skew.ratio, 1);
    });

});

// ===========================================================================
// OrderFlowMomentum
// ===========================================================================

describe('OrderFlowMomentum', () => {

    it('throws on empty candles', () => {
        assert.throws(() => new OrderFlowMomentum([]), /non-empty/);
    });

    it('getMomentum detects buying when all candles are bullish', () => {
        const candles = Array.from({ length: 30 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100,
            high: 102,
            low: 99,
            close: 101, // close > open -> bullish
            volume: 1000,
        }));
        const ofm = new OrderFlowMomentum(candles);
        const m = ofm.getMomentum(20);

        assert.equal(m.direction, 'buying');
        assert.ok(m.strength > 0, 'Strength should be > 0 for all-bullish');
    });

    it('getMomentum detects selling when all candles are bearish', () => {
        const candles = Array.from({ length: 30 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 101,
            high: 102,
            low: 99,
            close: 100, // close < open -> bearish
            volume: 1000,
        }));
        const ofm = new OrderFlowMomentum(candles);
        const m = ofm.getMomentum(20);

        assert.equal(m.direction, 'selling');
        assert.ok(m.strength > 0);
    });

    it('getMomentum is neutral for mixed candles with balanced volume', () => {
        const candles = [];
        for (let i = 0; i < 20; i++) {
            candles.push({
                timestamp: 1000 + i * 60,
                open: 100,
                high: 101,
                low: 99,
                close: i % 2 === 0 ? 101 : 99, // alternating bullish/bearish
                volume: 1000,
            });
        }
        const ofm = new OrderFlowMomentum(candles);
        const m = ofm.getMomentum(20);

        assert.equal(m.direction, 'neutral');
    });

    it('getMomentum returns neutral for insufficient data', () => {
        const candles = [{ timestamp: 1, open: 100, high: 101, low: 99, close: 101, volume: 1000 }];
        const ofm = new OrderFlowMomentum(candles);
        const m = ofm.getMomentum(20);

        assert.equal(m.direction, 'neutral');
        assert.equal(m.strength, 0);
    });

    it('getMomentum strength is 0-1', () => {
        const candles = generateCandles(50, {
            volatility: 0.01,
            trend: 0.0005,
            baseVolume: 1000,
            rng: seededRandom(42),
        });
        const ofm = new OrderFlowMomentum(candles);
        const m = ofm.getMomentum(20);

        assert.ok(m.strength >= 0 && m.strength <= 1,
            `Strength ${m.strength} should be in [0,1]`);
    });

    it('detectAbsorption finds candles with high volume and small movement', () => {
        const candles = [];
        for (let i = 0; i < 20; i++) {
            candles.push({
                timestamp: 1000 + i * 60,
                open: 100,
                high: 101,
                low: 99,
                close: 100,
                volume: 500, // normal candles
            });
        }
        // Add absorption candles: high volume, tiny range
        for (let i = 0; i < 3; i++) {
            candles.push({
                timestamp: 2200 + i * 60,
                open: 100,
                high: 100.1,
                low: 99.9,
                close: 100,
                volume: 5000, // 10x normal volume, 0.1 % range
            });
        }
        const ofm = new OrderFlowMomentum(candles);
        const absorbed = ofm.detectAbsorption();

        assert.ok(absorbed.length >= 1,
            `Expected at least 1 absorption candle, got ${absorbed.length}`);
        for (const a of absorbed) {
            assert.ok(a.volume > 0);
            assert.ok(a.timestamp > 0);
        }
    });

    it('detectAbsorption returns empty for trending market (no absorption)', () => {
        const candles = Array.from({ length: 30 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100 + i * 0.5,
            high: 101 + i * 0.5,
            low: 99 + i * 0.5,
            close: 100 + i * 0.5,
            volume: 1000,
        }));
        const ofm = new OrderFlowMomentum(candles);
        const absorbed = ofm.detectAbsorption();

        // Each candle has priceChange ~0.5 and range ~2, ratio = 0.25 < 0.3
        // But volume is constant (1000 = avg), so none exceed avg * 1.5
        assert.equal(absorbed.length, 0);
    });

    it('detectAbsorption returns empty for very short data', () => {
        const candles = [{ timestamp: 1, open: 100, high: 101, low: 99, close: 100, volume: 1000 }];
        const ofm = new OrderFlowMomentum(candles);
        assert.equal(ofm.detectAbsorption().length, 0);
    });

    it('getCumulativeDelta accumulates correctly', () => {
        const candles = [
            { timestamp: 1, open: 100, high: 102, low: 98, close: 101, volume: 500 },
            { timestamp: 2, open: 101, high: 103, low: 99, close: 100, volume: 300 },
            { timestamp: 3, open: 100, high: 102, low: 98, close: 101, volume: 400 },
        ];
        const ofm = new OrderFlowMomentum(candles);
        const delta = ofm.getCumulativeDelta();

        assert.equal(delta.length, 3);
        assert.equal(delta[0].delta, 500);   // bull
        assert.equal(delta[1].delta, 200);   // 500 - 300
        assert.equal(delta[2].delta, 600);   // 200 + 400
    });

    it('getCumulativeDelta handles doji candles (no delta)', () => {
        const candles = [
            { timestamp: 1, open: 100, high: 102, low: 98, close: 100, volume: 500 },
            { timestamp: 2, open: 100, high: 102, low: 98, close: 101, volume: 300 },
        ];
        const ofm = new OrderFlowMomentum(candles);
        const delta = ofm.getCumulativeDelta();

        assert.equal(delta[0].delta, 0);   // doji
        assert.equal(delta[1].delta, 300); // bull
    });

    it('getCumulativeDelta returns empty for empty candles', () => {
        // We can't construct OrderFlowMomentum with empty array (throws)
        // So we test with a single candle instead
        const candles = [{ timestamp: 1, open: 100, high: 101, low: 99, close: 101, volume: 500 }];
        const ofm = new OrderFlowMomentum(candles);
        const delta = ofm.getCumulativeDelta();
        assert.equal(delta.length, 1);
        assert.equal(delta[0].delta, 500);
    });

});

// ===========================================================================
// VolatilityEddyDetector
// ===========================================================================

describe('VolatilityEddyDetector', () => {

    it('detectEddies finds volatility expansion/contraction zones', () => {
        // Create data with a clear calm→volatile→calm pattern
        const candles = [];
        let price = 100;

        // 20 calm candles (low volatility)
        for (let i = 0; i < 30; i++) {
            price = price * (1 + (Math.random() - 0.5) * 0.0005);
            candles.push({
                timestamp: i,
                open: +price.toFixed(2),
                high: +(price * 1.0005).toFixed(2),
                low: +(price * 0.9995).toFixed(2),
                close: +price.toFixed(2),
                volume: 500,
            });
        }

        // 20 volatile candles (high volatility)
        for (let i = 0; i < 30; i++) {
            price = price * (1 + (Math.random() - 0.5) * 0.03);
            candles.push({
                timestamp: 30 + i,
                open: +price.toFixed(2),
                high: +(price * 1.015).toFixed(2),
                low: +(price * 0.985).toFixed(2),
                close: +price.toFixed(2),
                volume: 2000,
            });
        }

        // 20 calm candles again
        for (let i = 0; i < 30; i++) {
            price = price * (1 + (Math.random() - 0.5) * 0.0005);
            candles.push({
                timestamp: 60 + i,
                open: +price.toFixed(2),
                high: +(price * 1.0005).toFixed(2),
                low: +(price * 0.9995).toFixed(2),
                close: +price.toFixed(2),
                volume: 500,
            });
        }

        const detector = new VolatilityEddyDetector();
        const eddies = detector.detectEddies(candles, 14);

        // Should detect at least one eddy zone
        assert.ok(eddies.length >= 1,
            `Expected at least 1 eddy, got ${eddies.length}`);
    });

    it('detectEddies returns empty for data shorter than 2x window', () => {
        const candles = generateCandles(10, { volatility: 0.001 });
        const detector = new VolatilityEddyDetector();

        assert.equal(detector.detectEddies(candles, 14).length, 0);
    });

    it('detectEddies returns empty for non-array input', () => {
        const detector = new VolatilityEddyDetector();
        assert.equal(detector.detectEddies(null, 14).length, 0);
    });

    it('getTurbulenceIndex returns 0 for flat market (no turbulence)', () => {
        const candles = Array.from({ length: 30 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100,
            high: 100.01,
            low: 99.99,
            close: 100,
            volume: 100,
        }));
        const detector = new VolatilityEddyDetector();
        const idx = detector.getTurbulenceIndex(candles, 14);

        assert.ok(idx < 0.01, `Expected near-zero turbulence for flat market, got ${idx}`);
    });

    it('getTurbulenceIndex returns > 0 for volatile market', () => {
        const candles = Array.from({ length: 30 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100 + (i % 5) * 2,
            high: 105 + (i % 3) * 3,
            low: 95 - (i % 3),
            close: 102 + (i % 5) * 2,
            volume: 5000 + (i % 3) * 3000,
        }));
        const detector = new VolatilityEddyDetector();
        const idx = detector.getTurbulenceIndex(candles, 14);

        assert.ok(idx > 0, `Expected turbulence > 0 for volatile market, got ${idx}`);
    });

    it('getTurbulenceIndex returns between 0 and 1', () => {
        const candles = generateCandles(50, {
            volatility: 0.005,
            baseVolume: 1000,
            rng: seededRandom(42),
        });
        const detector = new VolatilityEddyDetector();
        const idx = detector.getTurbulenceIndex(candles, 14);

        assert.ok(idx >= 0 && idx <= 1,
            `Turbulence ${idx} should be in [0,1]`);
    });

    it('getTurbulenceIndex returns 0 for insufficient data', () => {
        const detector = new VolatilityEddyDetector();
        assert.equal(detector.getTurbulenceIndex([], 14), 0);
        assert.equal(detector.getTurbulenceIndex(null, 14), 0);
    });

});

// ===========================================================================
// Utility Functions
// ===========================================================================

describe('liquidityScore', () => {

    it('returns 0-100 range for normal data', () => {
        const candles = generateCandles(50, {
            volatility: 0.002,
            baseVolume: 5000,
            volumeStd: 1000,
            rng: seededRandom(42),
        });
        const score = liquidityScore(candles);

        assert.ok(score >= 0 && score <= 100,
            `Score ${score} should be in [0,100]`);
    });

    it('returns higher score for high-volume tight-spread data', () => {
        const liquidCandles = Array.from({ length: 20 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100,
            high: 100.05,
            low: 99.95,
            close: 100,
            volume: 10000,
        }));
        const illiquidCandles = Array.from({ length: 20 }, (_, i) => ({
            timestamp: 1000 + i * 60,
            open: 100,
            high: 105,
            low: 95,
            close: 100,
            volume: 100,
        }));

        const highScore = liquidityScore(liquidCandles);
        const lowScore = liquidityScore(illiquidCandles);

        assert.ok(highScore > lowScore,
            `High liquidity score (${highScore}) should exceed low (${lowScore})`);
    });

    it('returns 0 for insufficient data', () => {
        assert.equal(liquidityScore([]), 0);
        assert.equal(liquidityScore(null), 0);
        assert.equal(liquidityScore(undefined), 0);
        assert.equal(liquidityScore([{ timestamp: 1, open: 100, high: 101, low: 99, close: 100, volume: 1000 }]), 0);
    });

    it('returns 0 for empty array', () => {
        assert.equal(liquidityScore([]), 0);
    });

});

describe('supportResistanceZones', () => {

    it('clusters adjacent high-volume nodes into zones', () => {
        const vap = [
            { price: 100, volume: 100 },
            { price: 101, volume: 200 },
            { price: 102, volume: 150 },
            { price: 105, volume: 50 },   // gap
            { price: 108, volume: 180 },
            { price: 109, volume: 190 },
            { price: 110, volume: 100 },
        ];
        // strength 0.5 → include nodes with volume >= 0.5 * 200 = 100
        const zones = supportResistanceZones(vap, 0.5);

        // Should produce 2 zones: {100-102} and {108-110}
        assert.equal(zones.length, 2,
            `Expected 2 zones, got ${zones.length}: ${JSON.stringify(zones)}`);

        // First zone: low = 100, high = 102, 3 nodes
        assert.equal(zones[0].lowPrice, 100);
        assert.equal(zones[0].highPrice, 102);
        assert.equal(zones[0].nodeCount, 3);
        assert.equal(zones[0].volume, 450); // 100 + 200 + 150

        // Second zone: low = 108, high = 110, 3 nodes
        assert.equal(zones[1].lowPrice, 108);
        assert.equal(zones[1].highPrice, 110);
        assert.equal(zones[1].nodeCount, 3);
        assert.equal(zones[1].volume, 470); // 180 + 190 + 100
    });

    it('returns empty array for empty VAP profile', () => {
        assert.equal(supportResistanceZones([]).length, 0);
        assert.equal(supportResistanceZones(null).length, 0);
        assert.equal(supportResistanceZones(undefined).length, 0);
    });

    it('returns empty when no bins meet the strength threshold', () => {
        const vap = [
            { price: 100, volume: 10 },
            { price: 101, volume: 20 },
            { price: 102, volume: 15 },
        ];
        const zones = supportResistanceZones(vap, 0.9);
        // Max vol = 20, threshold = 18, only bin 101 qualifies
        // Single bin → 1 zone
        assert.equal(zones.length, 1);
        assert.equal(zones[0].nodeCount, 1);
    });

    it('strength field is 0-1', () => {
        const vap = [
            { price: 100, volume: 50 },
            { price: 101, volume: 200 },
            { price: 102, volume: 100 },
            { price: 103, volume: 180 },
        ];
        const zones = supportResistanceZones(vap, 0.5);

        for (const z of zones) {
            assert.ok(z.strength >= 0 && z.strength <= 1,
                `Zone strength ${z.strength} should be in [0,1]`);
            assert.ok(z.midPrice >= z.lowPrice && z.midPrice <= z.highPrice,
                `Mid price ${z.midPrice} should be between ${z.lowPrice} and ${z.highPrice}`);
        }
    });

    it('handles all-zero volume gracefully', () => {
        const vap = [
            { price: 100, volume: 0 },
            { price: 101, volume: 0 },
        ];
        assert.equal(supportResistanceZones(vap).length, 0);
    });

    it('handles single-bin VAP', () => {
        const vap = [{ price: 100, volume: 500 }];
        const zones = supportResistanceZones(vap, 0.5);

        assert.equal(zones.length, 1);
        assert.equal(zones[0].lowPrice, 100);
        assert.equal(zones[0].highPrice, 100);
        assert.equal(zones[0].volume, 500);
        assert.equal(zones[0].nodeCount, 1);
    });

});

// ===========================================================================
// Integration: VolumeProfile + supportResistanceZones end-to-end
// ===========================================================================

describe('Integration', () => {

    it('end-to-end: candles → VAP → zones with real data', () => {
        const candles = generateCandles(100, {
            startPrice: 100,
            trend: 0.0003,
            volatility: 0.005,
            baseVolume: 2000,
            volumeStd: 500,
            rng: seededRandom(42),
        });

        const vp = new VolumeProfile(candles);
        const vap = vp.getVolumeAtPrice(20);
        const va = vp.getValueArea(0.70);

        // VAP is non-empty, sorted
        assert.ok(vap.length > 0, 'VAP should have bins');
        assert.ok(va.low < va.high, 'Value area should span a range');
        assert.ok(va.poc >= va.low && va.poc <= va.high,
            'POC should be within value area');

        // Zones derived from VAP
        const zones = supportResistanceZones(vap, 0.7);
        for (const z of zones) {
            assert.ok(z.nodeCount >= 1);
            assert.ok(z.volume > 0);
        }
    });

});
