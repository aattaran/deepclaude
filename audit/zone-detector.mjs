/**
 * Zone Detector -- ultra-fast support/resistance/liquidity zone detection
 * optimised for 1-second timeframe data.
 *
 * Provides:
 *   - Order blocks        (last candle before an impulsive move)
 *   - Fair Value Gaps      (price gaps between consecutive wicks)
 *   - Breaker zones        (support that flips to resistance, or vice versa)
 *   - Liquidity voids      (price levels with minimal traded volume)
 *   - Zone clustering      (merge nearby zones into composite S/R clusters)
 *   - Streaming mode       (O(1) per candle via a ring buffer)
 *   - Utility functions    (strength at price, nearest zone, confluence)
 *
 * All imports are ESM. Zero npm dependencies.
 *
 * Usage:
 *   import {
 *     ZoneDetector,
 *     StreamingZoneDetector,
 *     zoneStrength,
 *     nearestZone,
 *     zoneConfluence,
 *   } from './audit/zone-detector.mjs';
 *
 *   const zd = new ZoneDetector(candles);
 *   const obs = zd.detectOrderBlocks(0.02);
 *   const fvgs = zd.detectFairValueGaps(0.01);
 *   const active = zd.getActiveZones(100.5);
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default impulsive-move threshold (price units). */
const DEFAULT_THRESHOLD = 0.01;

/** Default minimum gap width for fair value gap detection. */
const DEFAULT_MIN_GAP = 0.005;

/** Default volume threshold for liquidity void detection. */
const DEFAULT_VOLUME_THRESHOLD = 10;

/** Default distance for zone clustering (price units). */
const DEFAULT_CLUSTER_DISTANCE = 0.02;

/** Maximum age in milliseconds for a zone to be considered fresh. */
const MAX_ZONE_AGE_MS = 120_000;

/** Fraction of current price used as the proximity radius for active zones. */
const ACTIVE_PRICE_RADIUS_FRACTION = 0.003;

/** Number of bins for the volume-at-price histogram used by liquidity voids. */
const VAP_BINS = 200;

/** Small epsilon to avoid division by zero. */
const EPSILON = 1e-9;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Midpoint of two numbers.
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function mid(a, b) {
    return (a + b) / 2;
}

/**
 * Clamp a value between a min and max.
 * @param {number} v
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Round to 4 decimal places.
 * @param {number} v
 * @returns {number}
 */
function r4(v) {
    return +v.toFixed(4);
}

// ---------------------------------------------------------------------------
// Order-block detection
// ---------------------------------------------------------------------------

/**
 * Detect order blocks in a batch candle array.
 *
 * An order block is the candle immediately before a large impulsive move
 * (abs(close - open) > threshold).  The OB's direction matches the impulse.
 *
 * @param {Object[]} candles - OHLCV candles with { timestamp, open, high, low, close, volume }.
 * @param {number} threshold - Minimum body size to qualify as impulsive.
 * @returns {Object[]} Order blocks: { timestamp, price, direction, strength }.
 */
function detectOrderBlocks(candles, threshold) {
    const blocks = [];

    for (let i = 1; i < candles.length; i++) {
        const curr = candles[i];
        const prev = candles[i - 1];
        const body = Math.abs(curr.close - curr.open);

        if (body > threshold) {
            const direction = curr.close > curr.open ? 'bullish' : 'bearish';
            // For bullish OBs the relevant level is the OB candle's low (support);
            // for bearish OBs it is the high (resistance).
            const price = direction === 'bullish' ? prev.low : prev.high;
            const strength = r4(clamp(body / (threshold * 3), 0, 1));

            blocks.push({ timestamp: prev.timestamp, price, direction, strength });
        }
    }

    return blocks;
}

// ---------------------------------------------------------------------------
// Fair-value gap detection
// ---------------------------------------------------------------------------

/**
 * Detect fair value gaps between consecutive candle wicks.
 *
 * A bullish FVG forms when candle i+1's low > candle i's high (gap up).
 * A bearish FVG forms when candle i's low > candle i+1's high (gap down).
 *
 * @param {Object[]} candles
 * @param {number} minGap - Minimum gap size to report.
 * @returns {Object[]} FVGs: { timestamp, upperPrice, lowerPrice, direction }.
 */
function detectFairValueGaps(candles, minGap) {
    const gaps = [];

    for (let i = 0; i < candles.length - 1; i++) {
        const a = candles[i];
        const b = candles[i + 1];

        // Bullish gap: b opens above a's high
        const gapUp = b.low - a.high;
        if (gapUp > minGap) {
            gaps.push({
                timestamp: a.timestamp,
                upperPrice: r4(b.low),
                lowerPrice: r4(a.high),
                direction: 'bullish',
            });
        }

        // Bearish gap: b opens below a's low
        const gapDown = a.low - b.high;
        if (gapDown > minGap) {
            gaps.push({
                timestamp: a.timestamp,
                upperPrice: r4(a.low),
                lowerPrice: r4(b.high),
                direction: 'bearish',
            });
        }
    }

    return gaps;
}

// ---------------------------------------------------------------------------
// Breaker-zone detection
// ---------------------------------------------------------------------------

/**
 * Detect breaker zones -- where a previous support/resistance zone flips role.
 *
 * Scans forward from each order block.  If a bullish OB (support) is breached
 * to the downside, it flips to resistance.  If a bearish OB (resistance) is
 * breached to the upside, it flips to support.
 *
 * @param {Object[]} candles
 * @param {Object[]} orderBlocks - Pre-computed order blocks.
 * @returns {Object[]} Breakers: { timestamp, price, flipDirection, previousRole }.
 */
function detectBreakerZones(candles, orderBlocks) {
    if (candles.length < 2 || orderBlocks.length === 0) return [];

    const n = candles.length;
    const breakers = [];

    // Build a timestamp-to-index map for O(1) lookups in the hot loop.
    const posMap = new Map();
    for (let i = 0; i < n; i++) {
        posMap.set(candles[i].timestamp, i);
    }

    // Pre-compute suffix minimum low and suffix maximum high.
    // With these we can check IN AN O(1) LOOKUP whether an OB was ever
    // breached and skip the forward scan entirely for non-breached OBs.
    const suffixMinLow = new Float64Array(n);
    const suffixMaxHigh = new Float64Array(n);
    suffixMinLow[n - 1] = candles[n - 1].low;
    suffixMaxHigh[n - 1] = candles[n - 1].high;
    for (let i = n - 2; i >= 0; i--) {
        suffixMinLow[i] = Math.min(candles[i].low, suffixMinLow[i + 1]);
        suffixMaxHigh[i] = Math.max(candles[i].high, suffixMaxHigh[i + 1]);
    }

    for (const ob of orderBlocks) {
        const obIdx = posMap.get(ob.timestamp);
        if (obIdx == null || obIdx >= n - 1) continue;

        if (ob.direction === 'bullish' && suffixMinLow[obIdx + 1] < ob.price) {
            // Price eventually breached the support level; find the first candle.
            for (let j = obIdx + 1; j < n; j++) {
                if (candles[j].low < ob.price) {
                    breakers.push({
                        timestamp: candles[j].timestamp,
                        price: ob.price,
                        flipDirection: 'bearish',
                        previousRole: 'support',
                    });
                    break;
                }
            }
        } else if (ob.direction === 'bearish' && suffixMaxHigh[obIdx + 1] > ob.price) {
            for (let j = obIdx + 1; j < n; j++) {
                if (candles[j].high > ob.price) {
                    breakers.push({
                        timestamp: candles[j].timestamp,
                        price: ob.price,
                        flipDirection: 'bullish',
                        previousRole: 'resistance',
                    });
                    break;
                }
            }
        }
        // If the suffix check says no breach happened, skip entirely.
    }

    return breakers;
}

// ---------------------------------------------------------------------------
// Liquidity-void detection
// ---------------------------------------------------------------------------

/**
 * Detect liquidity voids -- price levels with minimal traded volume.
 *
 * Builds a VAP (Volume-at-Price) histogram over the candles and identifies
 * bins whose volume is at or below `volumeThreshold`.  Adjacent void bins
 * are merged into contiguous zones.
 *
 * @param {Object[]} candles
 * @param {number} volumeThreshold - Max total volume for a bin to be a void.
 * @returns {Object[]} Voids: { price, volume, zoneWidth }.
 */
function detectLiquidityVoids(candles, volumeThreshold) {
    if (candles.length === 0) return [];

    let minPrice = Infinity;
    let maxPrice = -Infinity;
    for (const c of candles) {
        if (c.low < minPrice) minPrice = c.low;
        if (c.high > maxPrice) maxPrice = c.high;
    }

    const range = maxPrice - minPrice;
    if (range < EPSILON) return [];

    const binWidth = range / VAP_BINS;
    const vols = new Float64Array(VAP_BINS);

    for (const c of candles) {
        const firstBin = Math.max(0, Math.floor((c.low - minPrice) / binWidth));
        const lastBin = Math.min(VAP_BINS - 1, Math.floor((c.high - minPrice) / binWidth));
        const volPerBin = c.volume / (lastBin - firstBin + 1);

        for (let b = firstBin; b <= lastBin; b++) {
            vols[b] += volPerBin;
        }
    }

    // Gather void bins (volume <= threshold)
    const voids = [];
    let i = 0;
    while (i < VAP_BINS) {
        if (vols[i] > volumeThreshold) {
            i++;
            continue;
        }

        const startIdx = i;
        let voidVol = 0;
        while (i < VAP_BINS && vols[i] <= volumeThreshold) {
            voidVol += vols[i];
            i++;
        }
        const endIdx = i - 1;

        const lowPrice = minPrice + startIdx * binWidth;
        const highPrice = minPrice + (endIdx + 1) * binWidth;
        voids.push({
            price: r4(mid(lowPrice, highPrice)),
            volume: +voidVol.toFixed(2),
            zoneWidth: r4(highPrice - lowPrice),
        });
    }

    return voids;
}

// ---------------------------------------------------------------------------
// Zone freshness
// ---------------------------------------------------------------------------

/**
 * Compute a freshness score (0-1) based on how recently the zone was formed.
 *
 * @param {number} zoneTimestamp
 * @param {number} latestTimestamp - Most recent candle timestamp.
 * @returns {number} 0 = stale, 1 = brand new.
 */
function freshness(zoneTimestamp, latestTimestamp) {
    const age = latestTimestamp - zoneTimestamp;
    if (age <= 0) return 1;
    return r4(clamp(1 - age / MAX_ZONE_AGE_MS, 0, 1));
}

/**
 * Check whether a zone's price is near a given reference price.
 *
 * @param {Object} zone - Zone object with a `price` field (or upperPrice/lowerPrice).
 * @param {number} currentPrice
 * @returns {boolean}
 */
function isNearPrice(zone, currentPrice) {
    const zPrice = zone.price;
    if (zPrice != null) {
        return Math.abs(zPrice - currentPrice) / (Math.abs(currentPrice) || EPSILON) <= ACTIVE_PRICE_RADIUS_FRACTION;
    }
    if (zone.upperPrice != null && zone.lowerPrice != null) {
        const zMid = mid(zone.upperPrice, zone.lowerPrice);
        return Math.abs(zMid - currentPrice) / (Math.abs(currentPrice) || EPSILON) <= ACTIVE_PRICE_RADIUS_FRACTION;
    }
    return false;
}

/**
 * Resolve a zone's representative price (prefer `.price`, fallback to mid of range).
 *
 * @param {Object} zone
 * @returns {number}
 */
function zonePrice(zone) {
    if (zone.price != null) return zone.price;
    if (zone.upperPrice != null && zone.lowerPrice != null) return mid(zone.upperPrice, zone.lowerPrice);
    return 0;
}

// ---------------------------------------------------------------------------
// ZoneDetector (batch)
// ---------------------------------------------------------------------------

/**
 * Batch zone detector -- processes a full array of 1s OHLCV candles and
 * detects support / resistance / liquidity zones.
 *
 * Detection methods (detectOrderBlocks, detectFairValueGaps, etc.) store
 * results internally.  Call them before getActiveZones / clusterZones.
 */
export class ZoneDetector {
    /**
     * @param {Object[]} [candles=[]] - Array of 1s OHLCV candles.
     *     Each candle: { timestamp, open, high, low, close, volume }.
     */
    constructor(candles = []) {
        /** @type {Object[]} */
        this.candles = Array.isArray(candles) ? candles : [];

        /** @private */ this._obs = [];
        /** @private */ this._fvgs = [];
        /** @private */ this._breakers = [];
        /** @private */ this._voids = [];
    }

    // -----------------------------------------------------------------------
    // Detection methods
    // -----------------------------------------------------------------------

    /**
     * Detect order blocks.
     *
     * @param {number} [threshold=DEFAULT_THRESHOLD] - Minimum body size for an
     *     impulsive move.
     * @returns {Object[]} { timestamp, price, direction, strength }.
     */
    detectOrderBlocks(threshold = DEFAULT_THRESHOLD) {
        this._obs = detectOrderBlocks(this.candles, threshold);
        return this._obs;
    }

    /**
     * Detect fair value gaps.
     *
     * @param {number} [minGap=DEFAULT_MIN_GAP] - Minimum gap width.
     * @returns {Object[]} { timestamp, upperPrice, lowerPrice, direction }.
     */
    detectFairValueGaps(minGap = DEFAULT_MIN_GAP) {
        this._fvgs = detectFairValueGaps(this.candles, minGap);
        return this._fvgs;
    }

    /**
     * Detect breaker zones (role flips).
     *
     * Relies on order blocks detected via detectOrderBlocks().  If none have
     * been detected yet, calls detectOrderBlocks(threshold) first.
     *
     * @param {number} [threshold=DEFAULT_THRESHOLD] - Threshold passed to
     *     detectOrderBlocks if no OBs exist yet.
     * @returns {Object[]} { timestamp, price, flipDirection, previousRole }.
     */
    detectBreakerZones(threshold = DEFAULT_THRESHOLD) {
        if (this._obs.length === 0) this.detectOrderBlocks(threshold);
        this._breakers = detectBreakerZones(this.candles, this._obs);
        return this._breakers;
    }

    /**
     * Detect liquidity voids.
     *
     * @param {number} [volumeThreshold=DEFAULT_VOLUME_THRESHOLD] - Max bin
     *     volume for a void.
     * @returns {Object[]} { price, volume, zoneWidth }.
     */
    detectLiquidityVoids(volumeThreshold = DEFAULT_VOLUME_THRESHOLD) {
        this._voids = detectLiquidityVoids(this.candles, volumeThreshold);
        return this._voids;
    }

    // -----------------------------------------------------------------------
    // Zone query / enrichment
    // -----------------------------------------------------------------------

    /**
     * Return all zones whose price level is near `currentPrice`, enriched with
     * a freshness score.
     *
     * Only returns zones that were previously detected via the detect* methods.
     * If no detection methods have been called, returns an empty array.
     *
     * @param {number} currentPrice - Current market price.
     * @returns {Object[]} Active zones with { type, price, strength, freshness,
     *     timestamp, ...typeSpecific }.
     */
    getActiveZones(currentPrice) {
        const latestTimestamp = this.candles.length > 0
            ? this.candles[this.candles.length - 1].timestamp
            : 0;

        const all = this._flattenZones();
        return all
            .filter(z => isNearPrice(z, currentPrice))
            .map(z => ({
                ...z,
                price: zonePrice(z),
                freshness: latestTimestamp > 0 ? freshness(z.timestamp, latestTimestamp) : 0,
            }));
    }

    /**
     * Group nearby zones into composite S/R clusters by price proximity.
     *
     * Uses all zones that have been detected so far.
     *
     * @param {number} [maxDistance=DEFAULT_CLUSTER_DISTANCE] - Max price
     *     distance between adjacent zones to be considered the same cluster.
     * @returns {Object[]} Clusters with { lowPrice, highPrice, midPrice,
     *     strength, zoneCount, types }.
     */
    clusterZones(maxDistance = DEFAULT_CLUSTER_DISTANCE) {
        const all = this._flattenZones();
        if (all.length === 0) return [];

        const withPrice = all
            .map(z => ({ ...z, price: zonePrice(z) }))
            .sort((a, b) => a.price - b.price);

        const clusters = [];
        let current = [withPrice[0]];

        for (let i = 1; i < withPrice.length; i++) {
            if (withPrice[i].price - withPrice[i - 1].price <= maxDistance) {
                current.push(withPrice[i]);
            } else {
                clusters.push(this._mergeCluster(current));
                current = [withPrice[i]];
            }
        }
        clusters.push(this._mergeCluster(current));

        return clusters;
    }

    // ---- Private helpers ------------------------------------------------

    /**
     * Flatten all detected zone types into a single array, tagging each with
     * its original type.
     *
     * @private
     * @returns {Object[]}
     */
    _flattenZones() {
        const all = [];

        for (const z of this._obs) {
            all.push({ type: 'orderBlock', ...z });
        }
        for (const z of this._fvgs) {
            all.push({ type: 'fvg', ...z });
        }
        for (const z of this._breakers) {
            all.push({ type: 'breaker', ...z });
        }
        for (const z of this._voids) {
            all.push({ type: 'liquidityVoid', ...z });
        }

        return all;
    }

    /**
     * Merge a group of zones into a single cluster descriptor.
     *
     * @private
     * @param {Object[]} zones - Zone objects with a `price` and optional `strength`.
     * @returns {Object}
     */
    _mergeCluster(zones) {
        const prices = zones.map(z => z.price);
        const lo = Math.min(...prices);
        const hi = Math.max(...prices);
        const avgStrength = zones.reduce((s, z) => s + (z.strength ?? 0.5), 0) / zones.length;

        return {
            lowPrice: r4(lo),
            highPrice: r4(hi),
            midPrice: r4(mid(lo, hi)),
            strength: r4(avgStrength),
            zoneCount: zones.length,
            types: [...new Set(zones.map(z => z.type))],
        };
    }
}

// ---------------------------------------------------------------------------
// StreamingZoneDetector
// ---------------------------------------------------------------------------

/**
 * Streaming zone detector -- O(1) per candle update using a ring buffer.
 *
 * Maintains a fixed-size rolling window of candles.  Each call to
 * `update(candle)` processes the new candle and returns any newly detected
 * zones (order blocks, fair value gaps, breaker flips).  Liquidity voids
 * are computed on demand when `getActiveZones` is called.
 *
 * Thresholds are supplied at construction time and remain fixed for the
 * lifetime of the detector.
 */
export class StreamingZoneDetector {
    /**
     * @param {Object} [opts]
     * @param {number} [opts.maxBars=200] - Ring buffer capacity.
     * @param {number} [opts.obThreshold=0.01] - Impulsive move threshold.
     * @param {number} [opts.minGap=0.005] - Minimum FVG gap.
     * @param {number} [opts.volumeThreshold=10] - Liquidity void threshold.
     * @param {number} [opts.breakerThreshold=0.01] - OB threshold for breakers.
     */
    constructor(opts = {}) {
        const {
            maxBars = 200,
            obThreshold = DEFAULT_THRESHOLD,
            minGap = DEFAULT_MIN_GAP,
            volumeThreshold = DEFAULT_VOLUME_THRESHOLD,
            breakerThreshold = DEFAULT_THRESHOLD,
        } = opts;

        /** @private */ this.maxBars = maxBars;
        /** @private */ this.obThreshold = obThreshold;
        /** @private */ this.minGap = minGap;
        /** @private */ this.volumeThreshold = volumeThreshold;
        /** @private */ this.breakerThreshold = breakerThreshold;

        /** @private */ this.buffer = [];
        /** @private */ this.orderBlocks = [];
        /** @private */ this.fairValueGaps = [];
        /** @private */ this.breakerZones = [];
        /** @private */ this.liquidityVoids = [];

        /** @private Cache: map from OB timestamp to the OB object for fast breaker checks. */
        this._obByTimestamp = new Map();
    }

    /**
     * Process a single new candle.  Returns any zones newly detected from this
     * candle (OB, FVG, breakers).
     *
     * @param {Object} candle - { timestamp, open, high, low, close, volume }.
     * @returns {{ orderBlocks: Object[], fairValueGaps: Object[], breakerZones: Object[] }}
     */
    update(candle) {
        const result = {
            orderBlocks: [],
            fairValueGaps: [],
            breakerZones: [],
        };

        // ---- Ring-buffer housekeeping ----
        this.buffer.push(candle);
        if (this.buffer.length > this.maxBars) {
            const removed = this.buffer.shift();
            this._expireZone(removed.timestamp);
        }

        const n = this.buffer.length;
        if (n < 2) return result;

        const prev = this.buffer[n - 2];

        // ---- 1. Order block detection (O(1)) ----
        const body = Math.abs(candle.close - candle.open);
        if (body > this.obThreshold) {
            const direction = candle.close > candle.open ? 'bullish' : 'bearish';
            const price = direction === 'bullish' ? prev.low : prev.high;
            const strength = r4(clamp(body / (this.obThreshold * 3), 0, 1));

            const ob = { timestamp: prev.timestamp, price, direction, strength };
            this.orderBlocks.push(ob);
            this._obByTimestamp.set(prev.timestamp, ob);
            result.orderBlocks.push(ob);
        }

        // ---- 2. Fair value gap detection (O(1)) ----
        // Bullish gap
        const gapUp = candle.low - prev.high;
        if (gapUp > this.minGap) {
            const fvg = {
                timestamp: prev.timestamp,
                upperPrice: r4(candle.low),
                lowerPrice: r4(prev.high),
                direction: 'bullish',
            };
            this.fairValueGaps.push(fvg);
            result.fairValueGaps.push(fvg);
        }

        // Bearish gap
        const gapDown = prev.low - candle.high;
        if (gapDown > this.minGap) {
            const fvg = {
                timestamp: prev.timestamp,
                upperPrice: r4(prev.low),
                lowerPrice: r4(candle.high),
                direction: 'bearish',
            };
            this.fairValueGaps.push(fvg);
            result.fairValueGaps.push(fvg);
        }

        // ---- 3. Breaker zone detection (O(num active OBs)) ----
        for (const [, ob] of this._obByTimestamp) {
            // Skip OBs that are the previous candle itself
            if (ob.timestamp === prev.timestamp) continue;

            if (ob.direction === 'bullish' && candle.low < ob.price) {
                const brk = {
                    timestamp: candle.timestamp,
                    price: ob.price,
                    flipDirection: 'bearish',
                    previousRole: 'support',
                };
                this.breakerZones.push(brk);
                result.breakerZones.push(brk);
                this._obByTimestamp.delete(ob.timestamp);
            } else if (ob.direction === 'bearish' && candle.high > ob.price) {
                const brk = {
                    timestamp: candle.timestamp,
                    price: ob.price,
                    flipDirection: 'bullish',
                    previousRole: 'resistance',
                };
                this.breakerZones.push(brk);
                result.breakerZones.push(brk);
                this._obByTimestamp.delete(ob.timestamp);
            }
        }

        return result;
    }

    /**
     * Return all zones that are near the given price level, with freshness
     * scores computed from the ring buffer's time range.
     *
     * Unlike the batch version, liquidity voids are computed on-the-fly from
     * the current ring buffer.  All other zones are accumulated since
     * construction or the last reset().
     *
     * @param {number} currentPrice
     * @returns {Object[]}
     */
    getActiveZones(currentPrice) {
        const latestTimestamp = this.buffer.length > 0
            ? this.buffer[this.buffer.length - 1].timestamp
            : 0;

        // Compute liquidity voids from the current buffer
        this.liquidityVoids = detectLiquidityVoids(this.buffer, this.volumeThreshold);

        const all = [];

        for (const z of this.orderBlocks) {
            all.push({ type: 'orderBlock', ...z });
        }
        for (const z of this.fairValueGaps) {
            all.push({ type: 'fvg', ...z });
        }
        for (const z of this.breakerZones) {
            all.push({ type: 'breaker', ...z });
        }
        for (const z of this.liquidityVoids) {
            all.push({ type: 'liquidityVoid', ...z });
        }

        return all
            .filter(z => isNearPrice(z, currentPrice))
            .map(z => ({
                ...z,
                price: zonePrice(z),
                freshness: latestTimestamp > 0 ? freshness(z.timestamp, latestTimestamp) : 0,
            }));
    }

    /**
     * Reset the ring buffer and clear all detected zones.
     */
    reset() {
        this.buffer.length = 0;
        this.orderBlocks.length = 0;
        this.fairValueGaps.length = 0;
        this.breakerZones.length = 0;
        this.liquidityVoids.length = 0;
        this._obByTimestamp.clear();
    }

    // ---- Private helpers ------------------------------------------------

    /**
     * Clean up zones associated with a candle that has been evicted from the
     * ring buffer, so the detector stays bounded.
     *
     * @private
     * @param {number} timestamp
     */
    _expireZone(timestamp) {
        const idx = this.orderBlocks.findIndex(z => z.timestamp === timestamp);
        if (idx !== -1) {
            this.orderBlocks.splice(idx, 1);
        }
        this._obByTimestamp.delete(timestamp);
    }
}

// ============================================================================
// PURE UTILITY FUNCTIONS
// ============================================================================

/**
 * Compute the composite strength (0-1) of all zones at a given price level.
 *
 * Each zone contributes its strength weighted by proximity.  A zone exactly
 * at the price contributes full strength; strength decays linearly to zero
 * at a distance of 0.5 % of the reference price.
 *
 * @param {Object[]} zones - Array of zone objects, each with a `price` and
 *     optional `strength` (defaults to 0.5).
 * @param {number} price - The price level to evaluate.
 * @returns {number} Composite strength between 0 and 1.
 */
export function zoneStrength(zones, price) {
    if (!Array.isArray(zones) || zones.length === 0 || price == null) return 0;

    const maxDist = Math.abs(price) * 0.005 + EPSILON;
    let total = 0;

    for (const z of zones) {
        const zPrice = zonePrice(z);
        const dist = Math.abs(zPrice - price);
        if (dist < maxDist) {
            const str = z.strength ?? 0.5;
            total += str * (1 - dist / maxDist);
        }
    }

    return r4(clamp(total, 0, 1));
}

/**
 * Find the nearest zone in a given direction from a price.
 *
 * - 'support': finds the zone with the highest price below `price` (nearest below).
 * - 'resistance': finds the zone with the lowest price above `price` (nearest above).
 *
 * @param {Object[]} zones - Array of zone objects (must have a resolvable price).
 * @param {number} price - Reference price.
 * @param {'support'|'resistance'} direction - Which side to search.
 * @returns {Object|null} The nearest zone, or null if none found.
 */
export function nearestZone(zones, price, direction) {
    if (!Array.isArray(zones) || zones.length === 0 || price == null) return null;

    let best = null;
    let bestDist = Infinity;

    for (const z of zones) {
        const zPrice = zonePrice(z);
        const diff = zPrice - price;

        if (direction === 'support' && diff < 0) {
            const absDist = Math.abs(diff);
            if (absDist < bestDist) {
                bestDist = absDist;
                best = z;
            }
        } else if (direction === 'resistance' && diff > 0) {
            if (diff < bestDist) {
                bestDist = diff;
                best = z;
            }
        }
    }

    return best;
}

/**
 * Count how many zones overlap within a price window of a given width.
 *
 * Uses a sliding-window scan over zones sorted by price to find the maximum
 * number of zones that fit within a window of width `priceWindow`.  Higher
 * counts mean stronger confluence (many zone types agree on a level).
 *
 * @param {Object[]} zones - Array of zone objects.
 * @param {number} priceWindow - Width of the price window.
 * @returns {number} Maximum number of overlapping zones (0 for empty input
 *     or non-positive priceWindow).
 */
export function zoneConfluence(zones, priceWindow) {
    if (!Array.isArray(zones) || zones.length === 0 || !priceWindow || priceWindow <= 0) return 0;

    const prices = zones
        .map(z => zonePrice(z))
        .sort((a, b) => a - b);

    let maxCount = 0;
    let left = 0;

    for (let right = 0; right < prices.length; right++) {
        while (prices[right] - prices[left] > priceWindow) {
            left++;
        }
        maxCount = Math.max(maxCount, right - left + 1);
    }

    return maxCount;
}
