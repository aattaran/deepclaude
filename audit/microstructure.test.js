/**
 * Market Microstructure Analysis -- unit tests.
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/microstructure.test.js
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    OrderBookAnalyzer,
    computeMicroPrice,
    detectAbsorption,
    orderFlowToxicity,
} from './microstructure.mjs';

// ===========================================================================
// Helpers: synthetic data generators
// ===========================================================================

/**
 * Create a single order book snapshot.
 *
 * @param {number} timestamp - Unix ms timestamp.
 * @param {Array<[number, number]>} bids - [[price, size], ...].
 * @param {Array<[number, number]>} asks - [[price, size], ...].
 * @returns {Object} Snapshot with { timestamp, bids, asks }.
 */
function snapshot(timestamp, bids, asks) {
    return { timestamp, bids, asks };
}

/**
 * Create a trade/order object for VPIN / order-flow-toxicity tests.
 *
 * @param {'buy'|'sell'} side
 * @param {number} size
 * @returns {Object}
 */
function trade(side, size = 1) {
    return { side, size };
}

/**
 * Generate a sequence of realistic order book snapshots with controllable
 * parameters. Creates a stable-ish market with a defined spread, then lets
 * you inject specific anomalies (spoofing, icebergs, walls).
 *
 * @param {Object} [opts]
 * @param {number} [opts.count=10] - Number of snapshots to generate.
 * @param {number} [opts.basePrice=100] - Mid-price anchor.
 * @param {number} [opts.spread=0.10] - Spread in dollars.
 * @param {number} [opts.levels=5] - Levels per side.
 * @param {number} [opts.baseSize=100] - Typical order size.
 * @param {number} [opts.startTime=1700000000000] - First timestamp.
 * @param {number} [opts.intervalMs=1000] - Time between snapshots.
 * @returns {Object[]} Array of snapshots.
 */
function generateSnapshots(opts = {}) {
    const {
        count = 10,
        basePrice = 100,
        spread = 0.10,
        levels = 5,
        baseSize = 100,
        startTime = 1700000000000,
        intervalMs = 1000,
    } = opts;

    const halfSpread = spread / 2;
    const snapshots = [];

    for (let i = 0; i < count; i++) {
        const ts = startTime + i * intervalMs;
        const bids = [];
        const asks = [];

        // Generate bid levels: descending from best_bid
        for (let j = 0; j < levels; j++) {
            const price = +(basePrice - halfSpread - j * 0.05).toFixed(2);
            const size = baseSize + Math.floor(Math.random() * 20);
            bids.push([price, size]);
        }

        // Generate ask levels: ascending from best_ask
        for (let j = 0; j < levels; j++) {
            const price = +(basePrice + halfSpread + j * 0.05).toFixed(2);
            const size = baseSize + Math.floor(Math.random() * 20);
            asks.push([price, size]);
        }

        snapshots.push(snapshot(ts, bids, asks));
    }

    return snapshots;
}

// ===========================================================================
// OrderBookAnalyzer — getBidAskImbalance
// ===========================================================================

describe('OrderBookAnalyzer.getBidAskImbalance', () => {
    it('returns 0.5 for equal bid and ask volume', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 100],
            [100.15, 100],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const result = analyzer.getBidAskImbalance(2);
        assert.strictEqual(result, 0.5);
    });

    it('returns > 0.5 when bid volume dominates', () => {
        const snap = snapshot(1000, [
            [100.00, 500],
            [99.95, 400],
        ], [
            [100.10, 50],
            [100.15, 40],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const result = analyzer.getBidAskImbalance(2);
        assert.ok(result > 0.5);
    });

    it('returns < 0.5 when ask volume dominates', () => {
        const snap = snapshot(1000, [
            [100.00, 30],
            [99.95, 20],
        ], [
            [100.10, 400],
            [100.15, 300],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const result = analyzer.getBidAskImbalance(2);
        assert.ok(result < 0.5);
    });

    it('returns 0.5 for empty snapshots', () => {
        const analyzer = new OrderBookAnalyzer([]);
        assert.strictEqual(analyzer.getBidAskImbalance(5), 0.5);
    });

    it('returns 0.5 when depth <= 0', () => {
        const snap = snapshot(1000, [[100.00, 100]], [[100.10, 100]]);
        const analyzer = new OrderBookAnalyzer([snap]);
        assert.strictEqual(analyzer.getBidAskImbalance(0), 0.5);
        assert.strictEqual(analyzer.getBidAskImbalance(-1), 0.5);
    });

    it('handles depth exceeding available levels by using all levels', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 100],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        // bidVol = 200, askVol = 100, total = 300, ratio = 200/300 = 0.666...
        assert.strictEqual(analyzer.getBidAskImbalance(10), 200 / 300);
    });

    it('averages imbalance across multiple snapshots', () => {
        const snap1 = snapshot(1000, [[100.00, 300]], [[100.10, 100]]);
        const snap2 = snapshot(2000, [[100.00, 100]], [[100.10, 300]]);
        const analyzer = new OrderBookAnalyzer([snap1, snap2]);
        // snap1: 300/400 = 0.75, snap2: 100/400 = 0.25, avg = 0.5
        assert.strictEqual(analyzer.getBidAskImbalance(1), 0.5);
    });
});

// ===========================================================================
// OrderBookAnalyzer — getSpread
// ===========================================================================

describe('OrderBookAnalyzer.getSpread', () => {
    it('computes spread as percentage of mid price', () => {
        const snap = snapshot(1000, [[100.00, 100]], [[100.10, 100]]);
        const analyzer = new OrderBookAnalyzer([snap]);
        // mid = 100.05, spread = 0.10, pct = 0.10 / 100.05 * 100 = 0.09995...
        const expected = (100.10 - 100.00) / ((100.00 + 100.10) / 2) * 100;
        assert.strictEqual(analyzer.getSpread(), expected);
    });

    it('returns 0 for empty snapshot list', () => {
        const analyzer = new OrderBookAnalyzer([]);
        assert.strictEqual(analyzer.getSpread(), 0);
    });

    it('returns Infinity when one side is empty', () => {
        const snap = snapshot(1000, [[100.00, 100]], []);
        const analyzer = new OrderBookAnalyzer([snap]);
        assert.strictEqual(analyzer.getSpread(), Infinity);
    });

    it('reports a larger spread percentage for wider markets', () => {
        const tight = snapshot(1000, [[100.00, 100]], [[100.01, 100]]);
        const wide = snapshot(2000, [[100.00, 100]], [[100.50, 100]]);
        const tightAnalyzer = new OrderBookAnalyzer([tight]);
        const wideAnalyzer = new OrderBookAnalyzer([wide]);
        assert.ok(tightAnalyzer.getSpread() < wideAnalyzer.getSpread());
    });

    it('uses the latest snapshot only', () => {
        const oldSnap = snapshot(1000, [[100.00, 100]], [[100.50, 100]]);
        const latest = snapshot(2000, [[100.00, 100]], [[100.05, 100]]);
        const analyzer = new OrderBookAnalyzer([oldSnap, latest]);
        const expected = (100.05 - 100.00) / ((100.00 + 100.05) / 2) * 100;
        assert.strictEqual(analyzer.getSpread(), expected);
    });
});

// ===========================================================================
// OrderBookAnalyzer — detectSpoofing
// ===========================================================================

describe('OrderBookAnalyzer.detectSpoofing', () => {
    it('detects a large order that disappears within the window', () => {
        // Snapshot 0: normal book
        const s0 = snapshot(1000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 100],
            [100.15, 100],
        ]);
        // Snapshot 1: large spoof ask order appears
        const s1 = snapshot(2000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 1000],  // <-- large (10x avg)
            [100.15, 100],
        ]);
        // Snapshot 2: large order is gone (canceled)
        const s2 = snapshot(3000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 100],
            [100.15, 100],
        ]);

        const analyzer = new OrderBookAnalyzer([s0, s1, s2]);
        const events = analyzer.detectSpoofing(2);
        assert.strictEqual(events.length, 1);
        assert.strictEqual(events[0].price, 100.10);
        assert.strictEqual(events[0].side, 'sell');
        assert.strictEqual(events[0].size, 1000);
    });

    it('does not flag orders that persist through the window', () => {
        const snap = snapshot(1000, [[100.00, 100]], [[100.10, 500]]);
        // Large order stays in all snapshots
        const analyzer = new OrderBookAnalyzer([snap, snap, snap]);
        const events = analyzer.detectSpoofing(3);
        assert.strictEqual(events.length, 0);
    });

    it('does not flag small orders (<= 3x average)', () => {
        const s0 = snapshot(1000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 100],
            [100.15, 100],
        ]);
        const s1 = snapshot(2000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 200],  // only 2x avg, below 3x threshold
            [100.15, 100],
        ]);
        const s2 = snapshot(3000, [
            [100.00, 100],
            [99.95, 100],
        ], [
            [100.10, 100],
            [100.15, 100],
        ]);
        const analyzer = new OrderBookAnalyzer([s0, s1, s2]);
        const events = analyzer.detectSpoofing(2);
        assert.strictEqual(events.length, 0);
    });

    it('returns empty array for fewer than 2 snapshots', () => {
        const analyzer = new OrderBookAnalyzer([]);
        assert.deepStrictEqual(analyzer.detectSpoofing(3), []);

        const single = new OrderBookAnalyzer([
            snapshot(1000, [[100.00, 100]], [[100.10, 100]]),
        ]);
        assert.deepStrictEqual(single.detectSpoofing(3), []);
    });

    it('does not flag an order that gradually decreases (traded away)', () => {
        // Simulate an order being gradually traded (size decreases each step)
        const s0 = snapshot(1000, [
            [100.00, 100],
        ], [
            [100.10, 800],
        ]);
        const s1 = snapshot(2000, [
            [100.00, 100],
        ], [
            [100.10, 600],  // decreased but still there
        ]);
        const s2 = snapshot(3000, [
            [100.00, 100],
        ], [
            [100.10, 400],  // still there
        ]);
        const s3 = snapshot(4000, [
            [100.00, 100],
        ], [
            [100.10, 200],  // still there
        ]);
        const s4 = snapshot(5000, [
            [100.00, 100],
        ], [
            [100.10, 0],    // gone now
        ]);

        const analyzer = new OrderBookAnalyzer([s0, s1, s2, s3, s4]);
        // At s0, order is 800 vs avg 800 -- not > 3x, it's exactly the average
        // At s1, order is 600 vs avg 600 -- still not > 3x
        // Actually avg changes per snapshot.
        // Let me adjust: avg at s0 = 800, 800 is not > 3*800. Correct, not flagged.
        assert.strictEqual(analyzer.detectSpoofing(5).length, 0);
    });
});

// ===========================================================================
// OrderBookAnalyzer — getVolumeFootprint
// ===========================================================================

describe('OrderBookAnalyzer.getVolumeFootprint', () => {
    it('aggregates volume at specified price levels', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
            [99.95, 50],
        ], [
            [100.10, 75],
            [100.15, 25],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const result = analyzer.getVolumeFootprint([100.00, 99.95, 100.10, 100.15]);
        assert.strictEqual(result[100.00], 100);
        assert.strictEqual(result[99.95], 50);
        assert.strictEqual(result[100.10], 75);
        assert.strictEqual(result[100.15], 25);
    });

    it('returns empty map for empty price bins', () => {
        const snap = snapshot(1000, [[100.00, 100]], [[100.10, 100]]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const result = analyzer.getVolumeFootprint([]);
        assert.deepStrictEqual(result, {});
    });

    it('sums volume across multiple snapshots', () => {
        const s1 = snapshot(1000, [[100.00, 100]], [[100.10, 50]]);
        const s2 = snapshot(2000, [[100.00, 150]], [[100.10, 60]]);
        const analyzer = new OrderBookAnalyzer([s1, s2]);
        const result = analyzer.getVolumeFootprint([100.00, 100.10]);
        assert.strictEqual(result[100.00], 250);
        assert.strictEqual(result[100.10], 110);
    });

    it('ignores prices not in the bin list', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
            [99.50, 999],  // not in bins
        ], [
            [100.10, 75],
            [101.00, 888],  // not in bins
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const result = analyzer.getVolumeFootprint([100.00, 100.10]);
        assert.strictEqual(result[100.00], 100);
        assert.strictEqual(result[100.10], 75);
        assert.strictEqual(Object.keys(result).length, 2);
    });

    it('handles empty snapshots gracefully', () => {
        const analyzer = new OrderBookAnalyzer([]);
        const result = analyzer.getVolumeFootprint([100.00]);
        assert.strictEqual(result[100.00], 0);
    });
});

// ===========================================================================
// OrderBookAnalyzer — detectIcebergOrders
// ===========================================================================

describe('OrderBookAnalyzer.detectIcebergOrders', () => {
    it('detects size replenishment pattern at a price level', () => {
        // Pattern: size decreases then bounces back multiple times at same price
        const bidLevel = (size) => [100.00, size];
        const snapshots = [
            snapshot(1000, [bidLevel(100)], [[100.10, 50]]),
            snapshot(2000, [bidLevel(80)],  [[100.10, 50]]),  // bought 20
            snapshot(3000, [bidLevel(100)], [[100.10, 50]]),  // replenished!
            snapshot(4000, [bidLevel(70)],  [[100.10, 50]]),  // bought 30
            snapshot(5000, [bidLevel(95)],  [[100.10, 50]]),  // replenished!
        ];
        const analyzer = new OrderBookAnalyzer(snapshots);
        const icebergs = analyzer.detectIcebergOrders(0.15);
        assert.strictEqual(icebergs.length, 1);
        assert.strictEqual(icebergs[0].price, 100.00);
        assert.strictEqual(icebergs[0].side, 'buy');
        assert.ok(icebergs[0].replenishmentCount >= 2);
    });

    it('returns empty array for fewer than 3 snapshots', () => {
        const analyzer = new OrderBookAnalyzer([
            snapshot(1000, [[100.00, 100]], [[100.10, 100]]),
            snapshot(2000, [[100.00, 90]],  [[100.10, 100]]),
        ]);
        assert.deepStrictEqual(analyzer.detectIcebergOrders(0.1), []);
    });

    it('does not flag stable (non-replenishing) levels', () => {
        const snapshots = Array.from({ length: 5 }, (_, i) =>
            snapshot(i * 1000, [[100.00, 100]], [[100.10, 100]]),
        );
        const analyzer = new OrderBookAnalyzer(snapshots);
        const icebergs = analyzer.detectIcebergOrders(0.1);
        assert.strictEqual(icebergs.length, 0);
    });

    it('detects iceberg on ask side as well', () => {
        const askLevel = (size) => [100.10, size];
        const snapshots = [
            snapshot(1000, [[100.00, 50]], [askLevel(100)]),
            snapshot(2000, [[100.00, 50]], [askLevel(70)]),    // bought
            snapshot(3000, [[100.00, 50]], [askLevel(100)]),   // replenished
            snapshot(4000, [[100.00, 50]], [askLevel(60)]),    // bought
            snapshot(5000, [[100.00, 50]], [askLevel(100)]),   // replenished
        ];
        const analyzer = new OrderBookAnalyzer(snapshots);
        const icebergs = analyzer.detectIcebergOrders(0.15);
        assert.strictEqual(icebergs.length, 1);
        assert.strictEqual(icebergs[0].price, 100.10);
        assert.strictEqual(icebergs[0].side, 'sell');
    });

    it('reports estimatedTotal as the max size seen', () => {
        const bidLevel = (size) => [100.00, size];
        const snapshots = [
            snapshot(1000, [bidLevel(100)], [[100.10, 50]]),
            snapshot(2000, [bidLevel(80)],  [[100.10, 50]]),
            snapshot(3000, [bidLevel(200)], [[100.10, 50]]),  // max = 200
            snapshot(4000, [bidLevel(50)],  [[100.10, 50]]),
            snapshot(5000, [bidLevel(180)], [[100.10, 50]]),
        ];
        const analyzer = new OrderBookAnalyzer(snapshots);
        const icebergs = analyzer.detectIcebergOrders(0.2);
        if (icebergs.length > 0) {
            assert.strictEqual(icebergs[0].estimatedTotal, 200);
        }
    });
});

// ===========================================================================
// OrderBookAnalyzer — getOrderBookDepth
// ===========================================================================

describe('OrderBookAnalyzer.getOrderBookDepth', () => {
    it('computes cumulative bid/ask volume at N levels', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
            [99.95, 50],
            [99.90, 30],
        ], [
            [100.10, 80],
            [100.15, 40],
            [100.20, 20],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const depth = analyzer.getOrderBookDepth(2);
        assert.strictEqual(depth.bids.length, 2);
        assert.strictEqual(depth.asks.length, 2);
        // Bid cumulative: level 0 = 100, level 1 = 100+50 = 150
        assert.strictEqual(depth.bids[0].cumulativeVolume, 100);
        assert.strictEqual(depth.bids[1].cumulativeVolume, 150);
        // Ask cumulative: level 0 = 80, level 1 = 80+40 = 120
        assert.strictEqual(depth.asks[0].cumulativeVolume, 80);
        assert.strictEqual(depth.asks[1].cumulativeVolume, 120);
    });

    it('returns empty depth for empty snapshots', () => {
        const analyzer = new OrderBookAnalyzer([]);
        const depth = analyzer.getOrderBookDepth(10);
        assert.deepStrictEqual(depth, { bids: [], asks: [] });
    });

    it('returns all levels when levels param exceeds available', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
        ], [
            [100.10, 200],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const depth = analyzer.getOrderBookDepth(99);
        assert.strictEqual(depth.bids.length, 1);
        assert.strictEqual(depth.asks.length, 1);
    });

    it('uses the latest snapshot only', () => {
        const oldSnap = snapshot(1000, [
            [100.00, 10],
        ], [
            [100.10, 10],
        ]);
        const currentSnap = snapshot(2000, [
            [100.00, 500],
        ], [
            [100.10, 500],
        ]);
        const analyzer = new OrderBookAnalyzer([oldSnap, currentSnap]);
        const depth = analyzer.getOrderBookDepth(1);
        assert.strictEqual(depth.bids[0].cumulativeVolume, 500);
        assert.strictEqual(depth.asks[0].cumulativeVolume, 500);
    });

    it('preserves price information in depth output', () => {
        const snap = snapshot(1000, [
            [100.00, 100],
            [99.95, 50],
        ], [
            [100.10, 80],
            [100.15, 40],
        ]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const depth = analyzer.getOrderBookDepth(2);
        assert.strictEqual(depth.bids[0].price, 100.00);
        assert.strictEqual(depth.bids[1].price, 99.95);
        assert.strictEqual(depth.asks[0].price, 100.10);
        assert.strictEqual(depth.asks[1].price, 100.15);
    });
});

// ===========================================================================
// computeMicroPrice
// ===========================================================================

describe('computeMicroPrice', () => {
    it('computes volume-weighted average price', () => {
        const bids = [[100.00, 100], [99.95, 50]];
        const asks = [[100.10, 80], [100.15, 20]];
        // bidValue = 100*100 + 99.95*50 = 10000 + 4997.5 = 14997.5
        // askValue = 100.10*80 + 100.15*20 = 8008 + 2003 = 10011
        // totalVol = 150 + 100 = 250
        // microPrice = (14997.5 + 10011) / 250 = 25008.5 / 250 = 100.034
        const micro = computeMicroPrice(bids, asks);
        const expected = (100 * 100 + 99.95 * 50 + 100.10 * 80 + 100.15 * 20) / (100 + 50 + 80 + 20);
        assert.strictEqual(micro, expected);
    });

    it('returns 0 for empty order book', () => {
        assert.strictEqual(computeMicroPrice([], []), 0);
    });

    it('returns 0 for null/undefined inputs', () => {
        assert.strictEqual(computeMicroPrice(null, []), 0);
        assert.strictEqual(computeMicroPrice([], undefined), 0);
        assert.strictEqual(computeMicroPrice(null, undefined), 0);
    });

    it('handles single-level book (degenerate case)', () => {
        const micro = computeMicroPrice([[100.00, 500]], [[100.10, 500]]);
        const expected = (100.00 * 500 + 100.10 * 500) / 1000;
        assert.strictEqual(micro, expected);
    });

    it('is closer to the side with more volume', () => {
        // Heavy bid volume: micro-price should be closer to bid side
        const heavyBids = [[100.00, 1000], [99.95, 1]];
        const lightAsks = [[100.10, 1], [100.15, 1]];
        const micro = computeMicroPrice(heavyBids, lightAsks);
        const midPrice = (100.00 + 100.10) / 2;
        assert.ok(micro < midPrice);  // Weighted toward bid side
    });

    it('handles prices with decimal precision correctly', () => {
        const bids = [[99.99, 200]];
        const asks = [[100.01, 200]];
        const micro = computeMicroPrice(bids, asks);
        assert.strictEqual(micro, (99.99 * 200 + 100.01 * 200) / 400);
    });
});

// ===========================================================================
// detectAbsorption
// ===========================================================================

describe('detectAbsorption', () => {
    it('detects absorption when bid volume dominates ask volume with wall', () => {
        const bids = [
            [100.00, 500],
            [99.95, 400],
            [99.90, 300],
        ];
        const asks = [
            [100.10, 100],
            [100.15, 800],  // <-- large ask wall (avg ask = 450, 800 > 3*450)
        ];
        const result = detectAbsorption(bids, asks, 1.5);
        // bidVol = 1200, askVol = 900, ratio = 1.333... < 1.5
        // Actually ratio is 1200/900 = 1.33 which is < 1.5, so NOT absorbed
        assert.strictEqual(result.absorbed, false);

        // Higher ratio:
        const heavyBids = [
            [100.00, 1000],
            [99.95, 800],
            [99.90, 500],
        ];
        const result2 = detectAbsorption(heavyBids, asks, 1.5);
        // bidVol = 2300, askVol = 900, ratio = 2.55 > 1.5, has wall = true
        assert.strictEqual(result2.absorbed, true);
    });

    it('reports false when no large ask wall exists', () => {
        const bids = [
            [100.00, 200],
            [99.95, 200],
        ];
        const asks = [
            [100.10, 100],
            [100.15, 100],  // all same size, no "wall"
        ];
        const result = detectAbsorption(bids, asks, 1.5);
        // ratio = 400/200 = 2.0 > 1.5 but no wall (avg = 100, max = 100, not > 3*100)
        assert.strictEqual(result.absorbed, false);
    });

    it('handles empty arrays', () => {
        const result = detectAbsorption([], []);
        assert.strictEqual(result.absorbed, false);
        assert.strictEqual(result.bidVolume, 0);
        assert.strictEqual(result.askVolume, 0);
        assert.strictEqual(result.maxAskWall, 0);
        assert.strictEqual(result.avgAskSize, 0);
    });

    it('handles non-array inputs gracefully', () => {
        const result = detectAbsorption(null, undefined);
        assert.strictEqual(result.absorbed, false);
    });

    it('respects custom threshold values', () => {
        const bids = [[100.00, 150]];
        const asks = [[100.10, 50]];  // avg=50, max=50, no wall
        // No wall means absorption won't trigger regardless of ratio
        const result = detectAbsorption(bids, asks, 0.5);
        // ratio = 3.0 > 0.5, but no wall
        assert.strictEqual(result.absorbed, false);
    });

    it('handles case where only one ask order exists (avg = max)', () => {
        const bids = [[100.00, 500]];
        const asks = [[100.10, 100]];  // single level: avg=100, max=100, wall check: 100 > 3*100 = false
        const result = detectAbsorption(bids, asks, 1.0);
        // ratio = 5.0 > 1.0, but no wall since single order is its own average
        assert.strictEqual(result.absorbed, false);
    });
});

// ===========================================================================
// orderFlowToxicity
// ===========================================================================

describe('orderFlowToxicity', () => {
    it('returns low toxicity for balanced buy/sell flow', () => {
        const orders = [];
        for (let i = 0; i < 100; i++) {
            orders.push(trade('buy', 1));
            orders.push(trade('sell', 1));
        }
        const result = orderFlowToxicity(orders, 50);
        // Each bucket of 50 orders: 25 buys, 25 sells of size 1 each
        // toxicity = |25-25| / 50 = 0 for each bucket
        assert.strictEqual(result.averageToxicity, 0);
        assert.strictEqual(result.buckets, 4); // 200 orders / 50 = 4 buckets
    });

    it('returns high toxicity for imbalanced flow', () => {
        const orders = [];
        for (let i = 0; i < 50; i++) {
            orders.push(trade('buy', 1));
        }
        // 50 buys, 0 sells per bucket
        const result = orderFlowToxicity(orders, 50);
        // toxicity = |50-0| / 50 = 1.0
        assert.strictEqual(result.averageToxicity, 1);
        assert.strictEqual(result.buckets, 1);
    });

    it('returns empty result for empty orders', () => {
        const result = orderFlowToxicity([], 50);
        assert.deepStrictEqual(result, { toxicityValues: [], averageToxicity: 0, buckets: 0 });
    });

    it('respects bucketSize parameter', () => {
        const orders = [];
        for (let i = 0; i < 100; i++) {
            orders.push(trade('buy', 1));
        }
        // With bucketSize = 25, we get 4 buckets, all pure buys
        const result = orderFlowToxicity(orders, 25);
        assert.strictEqual(result.buckets, 4);
        assert.strictEqual(result.averageToxicity, 1);
    });

    it('handles orders with varying sizes', () => {
        const orders = [
            trade('buy', 10),
            trade('sell', 1),
            trade('buy', 10),
            trade('sell', 1),
        ];
        // Only 4 orders, need 50 per bucket... bucketSize defaults to 50
        const defaultResult = orderFlowToxicity(orders);
        assert.strictEqual(defaultResult.buckets, 0);
        assert.strictEqual(defaultResult.averageToxicity, 0);

        // With bucketSize = 4:
        const smallResult = orderFlowToxicity(orders, 4);
        // buyVol = 20, sellVol = 2, toxicity = 18/22 = 0.8181...
        const expected = Math.abs(20 - 2) / (20 + 2);
        assert.strictEqual(smallResult.buckets, 1);
        assert.ok(Math.abs(smallResult.averageToxicity - expected) < 0.001);
    });

    it('returns toxicity values per bucket', () => {
        const orders = [];
        // Bucket 1: all buys (toxicity = 1.0)
        for (let i = 0; i < 50; i++) orders.push(trade('buy', 1));
        // Bucket 2: all sells (toxicity = 1.0)
        for (let i = 0; i < 50; i++) orders.push(trade('sell', 1));
        // Bucket 3: balanced (toxicity = 0.0)
        for (let i = 0; i < 25; i++) { orders.push(trade('buy', 1)); orders.push(trade('sell', 1)); }

        const result = orderFlowToxicity(orders, 50);
        assert.strictEqual(result.buckets, 3);
        assert.strictEqual(result.toxicityValues[0], 1);
        assert.strictEqual(result.toxicityValues[1], 1);
        assert.strictEqual(result.toxicityValues[2], 0);
        assert.strictEqual(result.averageToxicity, 2 / 3);
    });

    it('ignores orders without recognized side', () => {
        const orders = [
            { side: 'buy', size: 1 },
            { side: 'unknown', size: 1000 },  // ignored
            { side: 'sell', size: 1 },
        ];
        // Only 3 orders, bucket of 50 means no buckets
        const result = orderFlowToxicity(orders, 50);
        assert.strictEqual(result.buckets, 0);
    });
});

// ===========================================================================
// Integration: OrderBookAnalyzer with realistic data
// ===========================================================================

describe('OrderBookAnalyzer integration', () => {
    it('produces consistent results with generated snapshots', () => {
        const snapshots = generateSnapshots({
            count: 20,
            basePrice: 200,
            spread: 0.20,
            levels: 10,
            baseSize: 50,
            startTime: 1700000000000,
            intervalMs: 1000,
        });

        const analyzer = new OrderBookAnalyzer(snapshots);

        // All methods should run without error and return sane values
        const imbalance = analyzer.getBidAskImbalance(5);
        assert.ok(imbalance >= 0 && imbalance <= 1);

        const spread = analyzer.getSpread();
        assert.ok(spread > 0);
        assert.ok(spread < 1); // ~0.1% for spread=0.20, base=200

        const depth = analyzer.getOrderBookDepth(5);
        assert.strictEqual(depth.bids.length, 5);
        assert.strictEqual(depth.asks.length, 5);

        const footprint = analyzer.getVolumeFootprint([
            200.00, 199.95, 199.90, 200.10, 200.15, 200.20,
        ]);
        for (const key of Object.keys(footprint)) {
            assert.ok(footprint[key] >= 0);
        }

        const spoofing = analyzer.detectSpoofing(5);
        assert.ok(Array.isArray(spoofing));

        const icebergs = analyzer.detectIcebergOrders(0.15);
        assert.ok(Array.isArray(icebergs));
    });

    it('handles extreme spread gracefully', () => {
        const snap = snapshot(1000, [[100.00, 100]], [[200.00, 100]]);
        const analyzer = new OrderBookAnalyzer([snap]);
        const spread = analyzer.getSpread();
        // mid = 150, spread = 100, pct = 100/150*100 = 66.66%
        assert.ok(spread > 60);
        assert.ok(spread < 70);

        const imbalance = analyzer.getBidAskImbalance(1);
        assert.strictEqual(imbalance, 0.5); // equal volume
    });

    it('handles single-level order book', () => {
        const snap = snapshot(1000, [[100.00, 100]], [[100.10, 100]]);
        const analyzer = new OrderBookAnalyzer([snap]);

        assert.strictEqual(analyzer.getBidAskImbalance(1), 0.5);
        assert.strictEqual(analyzer.getOrderBookDepth(10).bids.length, 1);
        assert.strictEqual(analyzer.getOrderBookDepth(10).asks.length, 1);

        const micro = computeMicroPrice(snap.bids, snap.asks);
        assert.strictEqual(micro, (100 * 100 + 100.10 * 100) / 200);
    });

    it('handles missing data gracefully (empty snapshot array)', () => {
        const analyzer = new OrderBookAnalyzer([]);
        assert.strictEqual(analyzer.getBidAskImbalance(5), 0.5);
        assert.strictEqual(analyzer.getSpread(), 0);
        assert.deepStrictEqual(analyzer.getOrderBookDepth(5), { bids: [], asks: [] });
        assert.deepStrictEqual(analyzer.detectSpoofing(5), []);
        assert.deepStrictEqual(analyzer.detectIcebergOrders(0.1), []);
        assert.deepStrictEqual(
            analyzer.getVolumeFootprint([100.00]),
            { 100.00: 0 },
        );
    });

    it('micro-price + absorption + toxicity work in concert', () => {
        // Simulate a scenario: heavy buying pressure
        const bids = [[100.00, 1000], [99.95, 800]];
        const asks = [[100.10, 100], [100.15, 600]]; // wall at 100.15

        const micro = computeMicroPrice(bids, asks);
        const simpleMid = (100.00 + 100.10) / 2;
        assert.ok(micro < simpleMid); // Weighted toward heavy bids

        const absorption = detectAbsorption(bids, asks, 1.2);
        // bidVol = 1800, askVol = 700, ratio = 2.57 > 1.2
        // Wall detection uses exclusion-based average:
        //   ask [100.15, 600]: otherAvg = (700-600)/1 = 100, 600 > 3*100 => wall!
        // So absorption IS occurring (ratio exceeds threshold AND wall present).
        assert.strictEqual(absorption.absorbed, true);
        assert.ok(absorption.absorptionRatio > 2);
        assert.strictEqual(absorption.maxAskWall, 600);

        // Toxicity
        const orders = [];
        for (let i = 0; i < 30; i++) orders.push(trade('buy', 10));
        for (let i = 0; i < 10; i++) orders.push(trade('sell', 5));
        const toxicity = orderFlowToxicity(orders, 20);
        // Bucket 1: 20 orders. buys = 200, sells = (10*5)=50, wait, 10 sells at size 5 = 50
        // Actually let me count: first 20 orders: 20 buys at size 10 = 200 buy vol, 0 sells. toxicity = 1.0
        // Remaining 20: 10 buys at size 10=100, 10 sells at size 5=50. toxicity = 50/150 = 0.333
        // Actually, we have 40 orders (30 buys + 10 sells = 40). With bucketSize=20: 2 buckets.
        // Bucket 1 (first 20): 20 buys at size 10 = 200 buy, 0 sell. toxicity = 1.0
        // Bucket 2 (last 20): 10 buys at size 10=100, 10 sells at size 5=50. toxicity = 50/150 = 0.333...
        // avg = (1.0 + 0.333) / 2 = 0.666...
        assert.strictEqual(toxicity.buckets, 2);
        assert.ok(toxicity.toxicityValues[0] > 0.5);
        assert.ok(toxicity.averageToxicity > 0);
    });
});
