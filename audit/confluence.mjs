/**
 * Confluence Engine — multi-timeframe support/resistance confluence detection.
 *
 * Detects price levels where zones from multiple timeframes cluster, producing
 * high-confluence support/resistance levels ("grail" levels when all major
 * timeframes agree).
 *
 * Zero npm dependencies. ESM module.
 *
 * Usage:
 *   import {
 *     confluenceLevels,
 *     multiTimeframeZones,
 *     ConfluenceHeatmap,
 *     zoneConfluenceScore,
 *     timeframeWeight,
 *   } from './confluence.mjs';
 *
 *   const levels = confluenceLevels(zonesByTimeframe, currentPrice, { minTimeframes: 3 });
 *   // => [{ price, type, score, timeframeCount, timeframes, zones, strength }]
 */

import { ZoneDetector } from './zone-detector.mjs';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Default options for confluence level detection. */
const DEFAULT_CONFLUENCE_OPTIONS = {
    priceWindow: 0.003,   // 0.3 % of current price
    minTimeframes: 2,     // minimum TFs to form a confluence level
    decayFactor: 0.8,     // freshness weight decay
};

/** Default options for multi-timeframe zone detection. */
const DEFAULT_MULTI_OPTIONS = {
    obThreshold: 0.01,
    minGap: 0.005,
    volumeThreshold: 10,
    breakerThreshold: 0.01,
};

/**
 * Timeframe weights for scoring.
 * Smaller timeframes => lower weight (more noise).
 */
const TIMEFRAME_WEIGHTS = Object.freeze({
    '1s': 0.3,
    '5s': 0.5,
    '15s': 0.7,
    '1m': 0.85,
    '5m': 1.0,
    '15m': 1.0,
});

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Round to 4 decimal places.
 * @param {number} v
 * @returns {number}
 */
function r4(v) {
    return +v.toFixed(4);
}

/**
 * Clamp a value between lo and hi.
 * @param {number} v
 * @param {number} lo
 * @param {number} hi
 * @returns {number}
 */
function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

// ---------------------------------------------------------------------------
// timeframeWeight
// ---------------------------------------------------------------------------

/**
 * Return the weight for a given timeframe.
 *
 * Smaller timeframes get less weight (more noise).
 *
 * @param {string} timeframe - e.g. '1s', '5s', '15s', '1m', '5m', '15m'.
 * @returns {number} Weight between 0 and 1. Unknown timeframes default to 0.5.
 */
export function timeframeWeight(timeframe) {
    return TIMEFRAME_WEIGHTS[timeframe] ?? 0.5;
}

// ---------------------------------------------------------------------------
// confluenceLevels
// ---------------------------------------------------------------------------

/**
 * Detect confluence levels by clustering zones from multiple timeframes.
 *
 * Collects all zones from all timeframes, clusters them by price proximity,
 * and scores each cluster based on timeframe diversity, average strength,
 * and freshness.
 *
 * @param {Object<string, Object[]>} zonesByTimeframe - Map of timeframe name
 *     to array of zone objects (as returned by ZoneDetector.getActiveZones).
 *     Each zone: { price, type, strength, freshness, timestamp, ... }.
 * @param {number} currentPrice - Current market price used for type
 *     classification and absolute price window calculation.
 * @param {Object} [options]
 * @param {number} [options.priceWindow=0.003] - Max relative distance (fraction
 *     of currentPrice) between adjacent zones to be considered same cluster.
 *     Default 0.3 %.
 * @param {number} [options.minTimeframes=2] - Minimum number of unique
 *     timeframes that must contribute for a level to be reported.
 * @param {number} [options.decayFactor=0.8] - Freshness weight decay. Higher
 *     values penalise stale zones more heavily.
 * @returns {Object[]} Confluence levels sorted by descending score.
 *     Each level: { price, type, score, timeframeCount, timeframes, zones, strength }.
 */
export function confluenceLevels(zonesByTimeframe, currentPrice, options = {}) {
    // ---- Validation ----
    if (!zonesByTimeframe || typeof zonesByTimeframe !== 'object' || Array.isArray(zonesByTimeframe)) {
        return [];
    }
    if (currentPrice == null || !Number.isFinite(currentPrice)) {
        return [];
    }

    const opts = { ...DEFAULT_CONFLUENCE_OPTIONS, ...options };
    const timeframes = Object.keys(zonesByTimeframe);
    const totalTimeframes = timeframes.length;

    if (totalTimeframes === 0) return [];

    const absPriceWindow = opts.priceWindow * Math.abs(currentPrice);
    const minTimeframes = Math.max(1, opts.minTimeframes);

    // ---- Flatten and tag all zones ----
    const allZones = [];
    for (const tf of timeframes) {
        const zones = zonesByTimeframe[tf];
        if (!Array.isArray(zones)) continue;
        for (const zone of zones) {
            if (zone.price == null || !Number.isFinite(zone.price)) continue;
            allZones.push({
                timeframe: tf,
                price: zone.price,
                strength: typeof zone.strength === 'number' && Number.isFinite(zone.strength)
                    ? clamp(zone.strength, 0, 1) : 0.5,
                freshness: typeof zone.freshness === 'number' && Number.isFinite(zone.freshness)
                    ? clamp(zone.freshness, 0, 1) : 0.5,
                zone,
            });
        }
    }

    if (allZones.length === 0) return [];

    // ---- Cluster by price proximity ----
    allZones.sort((a, b) => a.price - b.price);

    const clusters = [];
    let current = [allZones[0]];

    for (let i = 1; i < allZones.length; i++) {
        const gap = allZones[i].price - allZones[i - 1].price;
        if (gap <= absPriceWindow) {
            current.push(allZones[i]);
        } else {
            clusters.push(_finalizeCluster(current, timeframes, currentPrice, opts));
            current = [allZones[i]];
        }
    }
    clusters.push(_finalizeCluster(current, timeframes, currentPrice, opts));

    // ---- Filter and sort ----
    return clusters
        .filter(c => c && c.timeframeCount >= minTimeframes)
        .sort((a, b) => b.score - a.score);
}

/**
 * Finalise a zone cluster into a confluence level object.
 *
 * @private
 * @param {Object[]} taggedZones - Zone entries in the cluster.
 * @param {string[]} allTimeframes - All timeframe names in the input.
 * @param {number} currentPrice
 * @param {Object} opts - Merged confluence options.
 * @returns {Object|null} Confluence level or null if cluster is empty.
 */
function _finalizeCluster(taggedZones, allTimeframes, currentPrice, opts) {
    const n = taggedZones.length;
    if (n === 0) return null;

    // Cluster-wide metrics
    const avgPrice = taggedZones.reduce((s, z) => s + z.price, 0) / n;
    const avgStrength = taggedZones.reduce((s, z) => s + z.strength, 0) / n;
    const avgFreshness = taggedZones.reduce((s, z) => s + z.freshness, 0) / n;

    // Unique contributing timeframes
    const tfSet = new Set(taggedZones.map(z => z.timeframe));
    const timeframeCount = tfSet.size;
    const tfArray = Array.from(tfSet).sort();

    // Type: support when cluster lies below current price, resistance when above.
    // Zones at or above the price level are classified as resistance.
    const type = avgPrice < currentPrice ? 'support' : 'resistance';

    // ---- Score components ----
    // tfDiversity: fraction of possible timeframe diversity represented
    const totalTf = allTimeframes.length;
    const tfDiversity = totalTf > 1
        ? (timeframeCount - 1) / (totalTf - 1)
        : 0;

    // freshnessDecay: linear penalty for staleness scaled by decayFactor
    const freshnessDecay = 1 - opts.decayFactor * (1 - avgFreshness);

    const score = r4(clamp(tfDiversity * avgStrength * freshnessDecay, 0, 1));

    return {
        price: r4(avgPrice),
        type,
        score,
        timeframeCount,
        timeframes: tfArray,
        zones: taggedZones.map(z => z.zone),
        strength: r4(clamp(avgStrength, 0, 1)),
    };
}

// ---------------------------------------------------------------------------
// multiTimeframeZones
// ---------------------------------------------------------------------------

/**
 * Convenience function: detect zones across multiple timeframes and compute
 * confluence levels in one call.
 *
 * Creates a ZoneDetector for each timeframe's candles, runs all detection
 * methods, calls getActiveZones, then feeds the result into confluenceLevels.
 *
 * @param {Object<string, Object[]>} candlesByTimeframe - Map of timeframe name
 *     to array of OHLCV candles ({ timestamp, open, high, low, close, volume }).
 * @param {Object} [options] - Combined options.
 *     ZoneDetector keys: obThreshold, minGap, volumeThreshold, breakerThreshold.
 *     Confluence keys: priceWindow, minTimeframes, decayFactor.
 * @returns {Object[]} Confluence levels (same format as confluenceLevels).
 */
export function multiTimeframeZones(candlesByTimeframe, options = {}) {
    if (!candlesByTimeframe || typeof candlesByTimeframe !== 'object') return [];

    // ZoneDetector-specific options
    const zdOptions = {
        obThreshold: options.obThreshold ?? DEFAULT_MULTI_OPTIONS.obThreshold,
        minGap: options.minGap ?? DEFAULT_MULTI_OPTIONS.minGap,
        volumeThreshold: options.volumeThreshold ?? DEFAULT_MULTI_OPTIONS.volumeThreshold,
        breakerThreshold: options.breakerThreshold ?? DEFAULT_MULTI_OPTIONS.breakerThreshold,
    };

    // Everything else goes to confluenceLevels
    const zdKeys = new Set(['obThreshold', 'minGap', 'volumeThreshold', 'breakerThreshold']);
    const confluenceOptions = {};
    for (const [key, value] of Object.entries(options)) {
        if (!zdKeys.has(key)) {
            confluenceOptions[key] = value;
        }
    }

    const zonesByTimeframe = {};
    let globalCurrentPrice = 0;

    for (const [tf, candles] of Object.entries(candlesByTimeframe)) {
        if (!Array.isArray(candles) || candles.length === 0) {
            zonesByTimeframe[tf] = [];
            continue;
        }

        const zd = new ZoneDetector(candles);
        zd.detectOrderBlocks(zdOptions.obThreshold);
        zd.detectFairValueGaps(zdOptions.minGap);
        zd.detectBreakerZones(zdOptions.breakerThreshold);
        zd.detectLiquidityVoids(zdOptions.volumeThreshold);

        const currentPrice = candles[candles.length - 1].close;
        if (globalCurrentPrice === 0 && currentPrice > 0) {
            globalCurrentPrice = currentPrice;
        }
        zonesByTimeframe[tf] = zd.getActiveZones(currentPrice);
    }

    return confluenceLevels(zonesByTimeframe, globalCurrentPrice, confluenceOptions);
}

// ---------------------------------------------------------------------------
// ConfluenceHeatmap
// ---------------------------------------------------------------------------

/**
 * Tracks confluence levels over time to visualise how levels form and dissolve.
 *
 * Each call to `update()` records a snapshot of confluence levels. The heatmap
 * bins levels by price and tracks density, max confluence, and timeframe
 * participation across all snapshots.
 */
export class ConfluenceHeatmap {
    /**
     * @param {Object} [options]
     * @param {number} [options.priceBinSize=0.001] - Relative bin size for price
     *     clustering (fraction of average currentPrice). Default 0.1 %.
     * @param {number} [options.priceWindow] - Forwarded to confluenceLevels.
     * @param {number} [options.minTimeframes] - Forwarded to confluenceLevels.
     * @param {number} [options.decayFactor] - Forwarded to confluenceLevels.
     */
    constructor(options = {}) {
        const { priceBinSize = 0.001, ...confluenceOptions } = options;
        /** @type {number} Relative bin size for price clustering. */
        this.priceBinSize = priceBinSize;
        /** @type {Object} Options forwarded to confluenceLevels. */
        this.confluenceOptions = confluenceOptions;
        /** @type {Object[]} Historical snapshots. */
        this.history = [];
    }

    /**
     * Record a snapshot of confluence levels.
     *
     * @param {Object<string, Object[]>} zonesByTimeframe - Same format as
     *     confluenceLevels input.
     * @param {number} currentPrice - Current market price for this snapshot.
     */
    update(zonesByTimeframe, currentPrice) {
        const levels = confluenceLevels(zonesByTimeframe, currentPrice, this.confluenceOptions);
        this.history.push({
            timestamp: Date.now(),
            currentPrice,
            levels,
        });
    }

    /**
     * Return the current heatmap data, binned by price.
     *
     * Levels from all snapshots are grouped into price bins. Each bin reports
     * the density (fraction of snapshots with a level in this bin), the max
     * confluence score, and a breakdown of which timeframes contributed.
     *
     * @returns {Object[]} Binned confluence levels sorted by descending density.
     *     Each bin: { price, density, maxConfluence, timeframeBreakdown }.
     */
    getHeatmap() {
        if (this.history.length === 0) return [];

        // Determine a global bin width from the average currentPrice across snapshots
        const avgSnapshotPrice = this.history.reduce(
            (s, h) => s + h.currentPrice, 0,
        ) / this.history.length;
        const binWidth = this.priceBinSize * avgSnapshotPrice;

        if (binWidth === 0) return [];

        // Bin all levels across all snapshots
        /** @type {Map<number, { prices: number[], scores: number[], tfCount: Object<string, number> }>} */
        const bins = new Map();

        for (const snapshot of this.history) {
            for (const level of snapshot.levels) {
                const key = Math.round(level.price / binWidth);
                if (!bins.has(key)) {
                    bins.set(key, { prices: [], scores: [], tfCount: {} });
                }
                const bin = bins.get(key);
                bin.prices.push(level.price);
                bin.scores.push(level.score);
                for (const tf of level.timeframes) {
                    bin.tfCount[tf] = (bin.tfCount[tf] || 0) + 1;
                }
            }
        }

        const totalSnapshots = this.history.length;

        return Array.from(bins.entries())
            .map(([k, bin]) => ({
                price: bin.prices.reduce((s, p) => s + p, 0) / bin.prices.length,
                density: bin.prices.length / totalSnapshots,
                maxConfluence: Math.max(...bin.scores),
                timeframeBreakdown: { ...bin.tfCount },
            }))
            .sort((a, b) => b.density - a.density);
    }

    /**
     * Return levels that persistently show confluence above the given density
     * threshold.
     *
     * @param {number} minDensity - Minimum density (0-1) to qualify.
     * @returns {Object[]} Filtered heatmap bins.
     */
    getHighConfluenceLevels(minDensity) {
        return this.getHeatmap().filter(bin => bin.density >= minDensity);
    }

    /**
     * Clear all historical snapshots.
     */
    reset() {
        this.history = [];
    }
}

// ---------------------------------------------------------------------------
// zoneConfluenceScore
// ---------------------------------------------------------------------------

/**
 * Quick live scoring function: how much do the currently active zones agree?
 *
 * High score means zones cluster at nearby prices (strong signal).
 * Low score means zones are scattered (weaker signal).
 *
 * Uses 0.1 % price bins. Returns a value between 0 and 1.
 *
 * @param {Object[]} activeZones - Array of zone objects (must have a `price`
 *     field; `strength` and `freshness` are optional).
 * @param {number} currentPrice - Current market price.
 * @returns {number} Confluence score (0-1).
 */
export function zoneConfluenceScore(activeZones, currentPrice) {
    if (!Array.isArray(activeZones) || activeZones.length === 0) return 0;
    if (currentPrice == null || !Number.isFinite(currentPrice) || currentPrice <= 0) return 0;

    const binSize = Math.abs(currentPrice) * 0.001; // 0.1 % bins
    if (binSize === 0) return 0;

    // Count zones per price bin
    const bins = new Map();
    for (const zone of activeZones) {
        if (zone.price == null || !Number.isFinite(zone.price)) continue;
        const key = Math.round(zone.price / binSize);
        bins.set(key, (bins.get(key) || 0) + 1);
    }

    if (bins.size === 0) return 0;

    const totalZones = activeZones.length;
    const maxBinCount = Math.max(...bins.values());

    // Concentration: fraction of zones in the most crowded bin
    const concentration = maxBinCount / totalZones;

    // Diversity penalty: more bins means more scattered
    const binRatio = bins.size / Math.max(1, totalZones);
    const diversityFactor = 1 - binRatio * 0.5;

    return r4(clamp(concentration * 0.6 + diversityFactor * 0.4, 0, 1));
}
