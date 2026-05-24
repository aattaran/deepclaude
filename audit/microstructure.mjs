/**
 * Market Microstructure Analysis -- map the muscles and bones of order book
 * dynamics.
 *
 * Inspired by anatomical dissection of order books: this module provides the
 * "muscles and bones" of market structure analysis -- bid-ask dynamics,
 * spoofing detection, iceberg order detection, order flow toxicity (VPIN-like),
 * volume footprinting, absorption analysis, and micro-price computation.
 *
 * All imports are ESM. Zero npm dependencies.
 *
 * Usage:
 *   import {
 *     OrderBookAnalyzer,
 *     computeMicroPrice,
 *     detectAbsorption,
 *     orderFlowToxicity,
 *   } from './audit/microstructure.mjs';
 *
 *   const analyzer = new OrderBookAnalyzer(snapshots);
 *   const imbalance = analyzer.getBidAskImbalance(5);
 *   const spread    = analyzer.getSpread();
 *   const spoofing  = analyzer.detectSpoofing(3);
 *   const footprint = analyzer.getVolumeFootprint([100, 101, 102]);
 *   const icebergs  = analyzer.detectIcebergOrders(0.15);
 *   const depth     = analyzer.getOrderBookDepth(10);
 *
 *   const microPrice    = computeMicroPrice(bids, asks);
 *   const absorption    = detectAbsorption(bids, asks, 2.0);
 *   const toxicity      = orderFlowToxicity(orders, 50);
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Small epsilon for floating-point price comparison. */
const PRICE_EPSILON = 1e-9;

/** Default depth level for methods that accept a depth parameter. */
const DEFAULT_DEPTH = 5;

/** Default window size for spoofing detection. */
const DEFAULT_SPOOFING_WINDOW = 5;

/** Default threshold for iceberg detection (fractional size change). */
const DEFAULT_ICEBERG_THRESHOLD = 0.1;

/** Spoofing size multiplier: orders larger than N * average are flagged. */
const SPOOFING_SIZE_MULTIPLIER = 3;

/** Absorption wall multiplier: an order larger than N * average is a "wall". */
const ABSORPTION_WALL_MULTIPLIER = 3;

/** Minimum observations required before iceberg detection triggers. */
const MIN_ICEBERG_OBSERVATIONS = 3;

/** Minimum replenishment cycles to confirm an iceberg pattern. */
const MIN_ICEBERG_CYCLES = 2;

/** Default threshold for absorption ratio. */
const DEFAULT_ABSORPTION_THRESHOLD = 1.5;

/** Default VPIN bucket size (number of trades per bucket). */
const DEFAULT_VPIN_BUCKET_SIZE = 50;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Check if two prices are approximately equal within a tolerance.
 * @param {number} a
 * @param {number} b
 * @param {number} [eps=PRICE_EPSILON]
 * @returns {boolean}
 */
function priceEq(a, b, eps = PRICE_EPSILON) {
    return Math.abs(a - b) < eps;
}

/**
 * Sum an array of numbers. Returns 0 for empty arrays.
 * @param {number[]} arr
 * @returns {number}
 */
function sum(arr) {
    return arr.reduce((s, v) => s + v, 0);
}

/**
 * Compute the mean of an array of numbers. Returns 0 for empty arrays.
 * @param {number[]} arr
 * @returns {number}
 */
function mean(arr) {
    if (arr.length === 0) return 0;
    return sum(arr) / arr.length;
}

/**
 * Compute the average size of orders in a level-2 array [[price, size], ...].
 * Returns 0 for empty arrays.
 * @param {Array<[number, number]>} levels
 * @returns {number}
 */
function avgSize(levels) {
    if (levels.length === 0) return 0;
    return mean(levels.map(([, sz]) => sz));
}

/**
 * Sort order book levels: bids descending (best first), asks ascending.
 * Returns a shallow copy (does not mutate the input).
 * @param {Array<[number, number]>} levels
 * @param {'bid'|'ask'} side
 * @returns {Array<[number, number]>}
 */
function sortLevels(levels, side) {
    if (side === 'bid') {
        return [...levels].sort((a, b) => b[0] - a[0]);
    }
    return [...levels].sort((a, b) => a[0] - b[0]);
}

// ---------------------------------------------------------------------------
// OrderBookAnalyzer
// ---------------------------------------------------------------------------

/**
 * Analyzes a sequence of order book snapshots to detect market microstructure
 * patterns such as spoofing, iceberg orders, order book imbalance, and volume
 * footprints.
 *
 * Each snapshot must have the shape:
 *   { timestamp: number, bids: [[price, size], ...], asks: [[price, size], ...] }
 *
 * Bids and asks arrays may be in any order; the class sorts them internally.
 */
export class OrderBookAnalyzer {
    /**
     * @param {Object[]} [snapshots=[]] - Array of order book snapshots.
     */
    constructor(snapshots = []) {
        /** @type {Object[]} */
        this.snapshots = snapshots;
    }

    // -----------------------------------------------------------------------
    // 1. Bid-Ask Imbalance
    // -----------------------------------------------------------------------

    /**
     * Compute the average bid-to-total volume ratio at a given depth.
     *
     * For each snapshot, sums bid and ask volume across the top `depth` price
     * levels on each side, then computes bidVol / (bidVol + askVol).
     * Returns the average across all snapshots.
     *
     * A value > 0.5 indicates buying pressure (more bid volume).
     * A value < 0.5 indicates selling pressure (more ask volume).
     *
     * @param {number} [depth=5] - Number of price levels to include from the top.
     * @returns {number} Ratio from 0 to 1. Returns 0.5 for empty books.
     */
    getBidAskImbalance(depth = DEFAULT_DEPTH) {
        if (this.snapshots.length === 0) return 0.5;
        if (depth <= 0) return 0.5;

        const ratios = this.snapshots.map(snap => {
            const topBids = sortLevels(snap.bids, 'bid').slice(0, depth);
            const topAsks = sortLevels(snap.asks, 'ask').slice(0, depth);
            const bidVol = sum(topBids.map(([, sz]) => sz));
            const askVol = sum(topAsks.map(([, sz]) => sz));
            const total = bidVol + askVol;
            return total > 0 ? bidVol / total : 0.5;
        });

        return mean(ratios);
    }

    // -----------------------------------------------------------------------
    // 2. Spread
    // -----------------------------------------------------------------------

    /**
     * Compute the current bid-ask spread as a percentage of the mid price.
     *
     * Uses the latest snapshot in the sequence. If the order book is empty
     * on either side, returns Infinity.
     *
     * @returns {number} Spread as a percentage of mid price.
     */
    getSpread() {
        if (this.snapshots.length === 0) return 0;

        const latest = this.snapshots[this.snapshots.length - 1];
        const sortedBids = sortLevels(latest.bids, 'bid');
        const sortedAsks = sortLevels(latest.asks, 'ask');

        if (sortedBids.length === 0 || sortedAsks.length === 0) return Infinity;

        const bestBid = sortedBids[0][0];
        const bestAsk = sortedAsks[0][0];
        const mid = (bestBid + bestAsk) / 2;

        if (mid === 0) return 0;
        return ((bestAsk - bestBid) / mid) * 100;
    }

    // -----------------------------------------------------------------------
    // 3. Spoofing Detection
    // -----------------------------------------------------------------------

    /**
     * Detect potential spoofing orders across a sliding window of snapshots.
     *
     * Spoofing is identified as orders that are significantly larger than the
     * average at their level (> 3x) and that are placed then canceled within
     * `windowSize` snapshots (i.e., they disappear without being gradually
     * traded through).
     *
     * @param {number} [windowSize=5] - Number of snapshots to look ahead for
     *     cancellation.
     * @returns {Object[]} Array of spoofing events with shape:
     *     { price, size, side, placedAt, canceledAt }
     */
    detectSpoofing(windowSize = DEFAULT_SPOOFING_WINDOW) {
        const events = [];
        const seen = new Set();

        if (this.snapshots.length < 2) return events;

        for (let i = 0; i < this.snapshots.length - 1; i++) {
            const snap = this.snapshots[i];
            const sortedBids = sortLevels(snap.bids, 'bid');
            const sortedAsks = sortLevels(snap.asks, 'ask');

            const bidTotal = sum(sortedBids.map(([, s]) => s));
            const bidCount = sortedBids.length;
            const askTotal = sum(sortedAsks.map(([, s]) => s));
            const askCount = sortedAsks.length;

            const lookaheadEnd = Math.min(i + windowSize + 1, this.snapshots.length);

            // Check bid side for spoofing
            for (const [price, size] of sortedBids) {
                // Compare against the average of OTHER levels (exclude this one)
                // so that a single large order doesn't inflate the baseline.
                const otherAvg = bidCount > 1 ? (bidTotal - size) / (bidCount - 1) : 0;
                if (otherAvg > 0 && size > SPOOFING_SIZE_MULTIPLIER * otherAvg) {
                    for (let j = i + 1; j < lookaheadEnd; j++) {
                        const laterBids = sortLevels(this.snapshots[j].bids, 'bid');
                        const laterLevel = laterBids.find(([p]) => priceEq(p, price));
                        // The large order is "canceled" if the price level is gone
                        // OR the remaining size is no longer anomalously large.
                        const stillSpoofed = laterLevel &&
                            laterLevel[1] > SPOOFING_SIZE_MULTIPLIER * otherAvg;
                        if (!stillSpoofed) {
                            const key = `${price}-bid`;
                            if (!seen.has(key)) {
                                seen.add(key);
                                events.push({
                                    price,
                                    size,
                                    side: 'buy',
                                    placedAt: snap.timestamp,
                                    canceledAt: this.snapshots[j].timestamp,
                                });
                            }
                            break;
                        }
                    }
                }
            }

            // Check ask side for spoofing
            for (const [price, size] of sortedAsks) {
                const otherAvg = askCount > 1 ? (askTotal - size) / (askCount - 1) : 0;
                if (otherAvg > 0 && size > SPOOFING_SIZE_MULTIPLIER * otherAvg) {
                    for (let j = i + 1; j < lookaheadEnd; j++) {
                        const laterAsks = sortLevels(this.snapshots[j].asks, 'ask');
                        const laterLevel = laterAsks.find(([p]) => priceEq(p, price));
                        const stillSpoofed = laterLevel &&
                            laterLevel[1] > SPOOFING_SIZE_MULTIPLIER * otherAvg;
                        if (!stillSpoofed) {
                            const key = `${price}-ask`;
                            if (!seen.has(key)) {
                                seen.add(key);
                                events.push({
                                    price,
                                    size,
                                    side: 'sell',
                                    placedAt: snap.timestamp,
                                    canceledAt: this.snapshots[j].timestamp,
                                });
                            }
                            break;
                        }
                    }
                }
            }
        }

        return events;
    }

    // -----------------------------------------------------------------------
    // 4. Volume Footprint
    // -----------------------------------------------------------------------

    /**
     * Aggregate volume at each price level into a histogram.
     *
     * For each price bin, sums the sizes of all orders at that exact price
     * across all snapshots. The result shows where order book depth
     * concentrates — the "center of gravity" of liquidity.
     *
     * @param {number[]} priceBins - Array of price values to aggregate at.
     * @returns {Object<number, number>} Map of price -> total volume.
     */
    getVolumeFootprint(priceBins) {
        const footprint = {};
        for (const bin of priceBins) {
            footprint[bin] = 0;
        }

        if (!Array.isArray(priceBins) || priceBins.length === 0) {
            return footprint;
        }

        for (const snap of this.snapshots) {
            for (const [price, size] of snap.bids) {
                const match = priceBins.find(b => priceEq(b, price));
                if (match !== undefined) {
                    footprint[match] += size;
                }
            }
            for (const [price, size] of snap.asks) {
                const match = priceBins.find(b => priceEq(b, price));
                if (match !== undefined) {
                    footprint[match] += size;
                }
            }
        }

        return footprint;
    }

    // -----------------------------------------------------------------------
    // 5. Iceberg Order Detection
    // -----------------------------------------------------------------------

    /**
     * Detect potential iceberg (hidden) orders.
     *
     * Iceberg orders display a tell-tale pattern: the visible size at a given
     * price level decreases (a trade occurs), then is replenished back to a
     * similar level (the hidden portion is revealed). This function detects
     * price levels where this pattern repeats multiple times.
     *
     * @param {number} [threshold=0.1] - Minimum fractional size change to
     *     qualify as a replenishment event. 0.1 means a 10% change.
     * @returns {Object[]} Array of detected iceberg orders with shape:
     *     { price, side, estimatedTotal, replenishmentCount }
     */
    detectIcebergOrders(threshold = DEFAULT_ICEBERG_THRESHOLD) {
        if (this.snapshots.length < MIN_ICEBERG_OBSERVATIONS) return [];

        // Track size history at each (price, side) level
        /** @type {Map<string, number[]>} */
        const levelHistory = new Map();

        for (const snap of this.snapshots) {
            for (const [price, size] of snap.bids) {
                const key = `${price}:bid`;
                if (!levelHistory.has(key)) levelHistory.set(key, []);
                levelHistory.get(key).push(size);
            }
            for (const [price, size] of snap.asks) {
                const key = `${price}:ask`;
                if (!levelHistory.has(key)) levelHistory.set(key, []);
                levelHistory.get(key).push(size);
            }
        }

        const icebergs = [];

        for (const [key, sizes] of levelHistory) {
            if (sizes.length < MIN_ICEBERG_OBSERVATIONS) continue;

            const [priceStr, side] = key.split(':');
            const price = parseFloat(priceStr);
            let replenishmentCount = 0;

            // Look for the pattern: size drops then bounces back near previous
            for (let i = 1; i < sizes.length; i++) {
                const prev = sizes[i - 1];
                const curr = sizes[i];
                const drop = prev > 0 && curr < prev * (1 - threshold);

                if (!drop) continue;

                // After a drop, scan ahead for replenishment back toward prev
                for (let j = i + 1; j < sizes.length; j++) {
                    const recovered = sizes[j];
                    const bouncedBack =
                        recovered > curr * (1 + threshold) &&
                        Math.abs(recovered - prev) / prev <= threshold;

                    if (bouncedBack) {
                        replenishmentCount++;
                        i = j; // Skip ahead to avoid double-counting
                        break;
                    }
                }
            }

            if (replenishmentCount >= MIN_ICEBERG_CYCLES) {
                icebergs.push({
                    price,
                    side: side === 'bid' ? 'buy' : 'sell',
                    estimatedTotal: Math.max(...sizes),
                    replenishmentCount,
                });
            }
        }

        return icebergs;
    }

    // -----------------------------------------------------------------------
    // 6. Order Book Depth
    // -----------------------------------------------------------------------

    /**
     * Compute cumulative bid and ask volume at N price levels from the midpoint.
     *
     * Uses the latest snapshot. Walks from the best bid downward and from the
     * best ask upward, accumulating volume at each level.
     *
     * @param {number} [levels=10] - Number of price levels to include per side.
     * @returns {{bids: Object[], asks: Object[]}} Cumulative depth per side.
     *     Each entry: { price, cumulativeVolume }.
     */
    getOrderBookDepth(levels = 10) {
        if (this.snapshots.length === 0) {
            return { bids: [], asks: [] };
        }

        const latest = this.snapshots[this.snapshots.length - 1];
        const sortedBids = sortLevels(latest.bids, 'bid');
        const sortedAsks = sortLevels(latest.asks, 'ask');

        const bidDepth = [];
        let cumBid = 0;
        const bidCount = Math.min(levels, sortedBids.length);
        for (let i = 0; i < bidCount; i++) {
            cumBid += sortedBids[i][1];
            bidDepth.push({ price: sortedBids[i][0], cumulativeVolume: cumBid });
        }

        const askDepth = [];
        let cumAsk = 0;
        const askCount = Math.min(levels, sortedAsks.length);
        for (let i = 0; i < askCount; i++) {
            cumAsk += sortedAsks[i][1];
            askDepth.push({ price: sortedAsks[i][0], cumulativeVolume: cumAsk });
        }

        return { bids: bidDepth, asks: askDepth };
    }
}

// ============================================================================
// PURE UTILITY FUNCTIONS
// ============================================================================

/**
 * Compute the micro-price -- a volume-weighted mid-price.
 *
 * Unlike the simple mid-price (best bid + best ask) / 2, the micro-price
 * weights each price by its available size, giving a better estimate of the
 * "true" market price where liquidity actually sits.
 *
 * Formula:
 *   microPrice = (sum(bidPrice_i * bidSize_i) + sum(askPrice_j * askSize_j))
 *              / (sum(bidSize_i) + sum(askSize_j))
 *
 * @param {Array<[number, number]>} bids - Bid levels as [[price, size], ...].
 * @param {Array<[number, number]>} asks - Ask levels as [[price, size], ...].
 * @returns {number} The volume-weighted micro-price. Returns 0 if both sides
 *     are empty.
 */
export function computeMicroPrice(bids, asks) {
    // Need at least one level to compute anything meaningful
    if (!Array.isArray(bids) || !Array.isArray(asks)) return 0;
    if (bids.length === 0 && asks.length === 0) return 0;

    const bidValue = bids.reduce((s, [p, sz]) => s + p * sz, 0);
    const askValue = asks.reduce((s, [p, sz]) => s + p * sz, 0);
    const bidVol = sum(bids.map(([, sz]) => sz));
    const askVol = sum(asks.map(([, sz]) => sz));
    const totalVol = bidVol + askVol;

    if (totalVol === 0) return 0;
    return (bidValue + askValue) / totalVol;
}

/**
 * Detect whether large ask walls are being absorbed by bid-side volume.
 *
 * Absorption is a bullish signal: when significant bid volume exists relative
 * to ask volume AND there are outsized ask orders ("walls"), it suggests
 * buyers are willing to eat through the sell-side pressure.
 *
 * @param {Array<[number, number]>} bids - Bid levels as [[price, size], ...].
 * @param {Array<[number, number]>} asks - Ask levels as [[price, size], ...].
 * @param {number} [threshold=1.5] - Minimum bid/ask volume ratio to consider
 *     absorption significant.
 * @returns {Object} Absorption analysis:
 *     @property {boolean} absorbed - Whether absorption is occurring.
 *     @property {number} absorptionRatio - Bid volume / ask volume.
 *     @property {number} bidVolume - Total bid volume.
 *     @property {number} askVolume - Total ask volume.
 *     @property {number} maxAskWall - Size of the largest ask order.
 *     @property {number} avgAskSize - Average ask order size.
 */
export function detectAbsorption(
    bids,
    asks,
    threshold = DEFAULT_ABSORPTION_THRESHOLD,
) {
    if (!Array.isArray(bids) || !Array.isArray(asks)) {
        return {
            absorbed: false,
            absorptionRatio: 0,
            bidVolume: 0,
            askVolume: 0,
            maxAskWall: 0,
            avgAskSize: 0,
        };
    }

    const bidVolume = sum(bids.map(([, sz]) => sz));
    const askVolume = sum(asks.map(([, sz]) => sz));

    const sortedAsks = sortLevels(asks, 'ask');
    const maxAskWall = sortedAsks.length > 0
        ? Math.max(...sortedAsks.map(([, s]) => s))
        : 0;
    const avgAskSz = avgSize(sortedAsks);

    // Detect a "wall": any ask order > 3x the average of the OTHER ask levels.
    // Comparing against the exclusion-based average prevents the wall itself
    // from inflating the baseline.
    const askTotal = sum(sortedAsks.map(([, s]) => s));
    let hasWall = false;
    if (sortedAsks.length > 1) {
        for (const [, size] of sortedAsks) {
            const otherAvg = (askTotal - size) / (sortedAsks.length - 1);
            if (otherAvg > 0 && size > ABSORPTION_WALL_MULTIPLIER * otherAvg) {
                hasWall = true;
                break;
            }
        }
    }

    const absorptionRatio = askVolume > 0 ? bidVolume / askVolume : Infinity;

    return {
        absorbed: absorptionRatio > threshold && hasWall,
        absorptionRatio,
        bidVolume,
        askVolume,
        maxAskWall,
        avgAskSize: avgAskSz,
    };
}

/**
 * Estimate order flow toxicity using a VPIN-like (Volume-synchronized
 * Probability of Informed Trading) metric.
 *
 * The method divides trades into equal-volume buckets (by trade count), and
 * for each bucket computes:
 *   toxicity = |buyVolume - sellVolume| / (buyVolume + sellVolume)
 *
 * Higher values (closer to 1) indicate greater order flow imbalance,
 * suggesting a higher probability of informed (toxic) trading. Lower values
 * (closer to 0) suggest balanced, uninformed flow.
 *
 * @param {Object[]} orders - Array of order/trade objects.
 *     Each order must have: { side: 'buy'|'sell', size?: number }.
 *     If `size` is omitted, each order counts as 1 unit.
 * @param {number} [bucketSize=50] - Number of orders per VPIN bucket.
 * @returns {Object} Toxicity analysis:
 *     @property {number[]} toxicityValues - Toxicity per bucket (0-1).
 *     @property {number} averageToxicity - Mean toxicity across all buckets.
 *     @property {number} buckets - Number of buckets formed.
 */
export function orderFlowToxicity(orders, bucketSize = DEFAULT_VPIN_BUCKET_SIZE) {
    if (!Array.isArray(orders) || orders.length === 0 || bucketSize <= 0) {
        return { toxicityValues: [], averageToxicity: 0, buckets: 0 };
    }

    const toxicityValues = [];
    let i = 0;

    while (i + bucketSize <= orders.length) {
        const bucket = orders.slice(i, i + bucketSize);
        let buyVol = 0;
        let sellVol = 0;

        for (const order of bucket) {
            const vol = order.size || 1;
            if (order.side === 'buy') {
                buyVol += vol;
            } else if (order.side === 'sell') {
                sellVol += vol;
            }
        }

        const totalVol = buyVol + sellVol;
        const toxicity = totalVol > 0
            ? Math.abs(buyVol - sellVol) / totalVol
            : 0;
        toxicityValues.push(toxicity);

        i += bucketSize; // Non-overlapping buckets (standard VPIN approach)
    }

    const averageToxicity = toxicityValues.length > 0
        ? mean(toxicityValues)
        : 0;

    return {
        toxicityValues,
        averageToxicity,
        buckets: toxicityValues.length,
    };
}
