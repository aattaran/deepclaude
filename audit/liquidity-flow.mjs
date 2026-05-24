/**
 * Liquidity Flow — physics-inspired liquidity analysis module.
 *
 * Models liquidity like Leonardo da Vinci modeled water:
 *   - Volume as current
 *   - Volatility as turbulence
 *   - Support/resistance as riverbanks
 *
 * Provides Volume Profile analysis (Market Profile theory), Order Flow
 * momentum tracking, volatility eddy detection, and composite liquidity
 * scoring for the deepclaude trading bot system.
 *
 * Usage:
 *   import {
 *     VolumeProfile,
 *     OrderFlowMomentum,
 *     VolatilityEddyDetector,
 *     liquidityScore,
 *     supportResistanceZones,
 *   } from './audit/liquidity-flow.mjs';
 *
 * Zero npm dependencies. ESM only.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum number of candles for any meaningful analysis. */
const MIN_CANDLES = 5;

// ---------------------------------------------------------------------------
// VolumeProfile Class
// ---------------------------------------------------------------------------

/**
 * Volume Profile analysis — constructs a Volume-at-Price (VAP) histogram from
 * OHLCV candle data and identifies key price levels (value area, POC,
 * high/low volume nodes).
 *
 * Inspired by Market Profile theory: price levels where high volume traded
 * act as "riverbanks" (support/resistance), while low-volume areas are
 * "rapids" where price moves quickly.
 */
export class VolumeProfile {
    /**
     * @param {Object[]} candles - Array of OHLCV candles
     * @param {number} candles[].timestamp - Unix timestamp (ms)
     * @param {number} candles[].open
     * @param {number} candles[].high
     * @param {number} candles[].low
     * @param {number} candles[].close
     * @param {number} candles[].volume
     */
    constructor(candles) {
        if (!Array.isArray(candles) || candles.length === 0) {
            throw new Error('VolumeProfile: candles must be a non-empty array');
        }
        this.candles = candles;
        /** @private */ this._cache = new Map();
    }

    /**
     * Build a Volume-at-Price (VAP) histogram.
     *
     * Splits the full price range into `bins` equal-width buckets and
     * distributes each candle's volume proportionally across the price
     * levels it spans (simplified "equal distribution" model).
     *
     * @param {number} [bins=24] - Number of price bins
     * @returns {{price: number, volume: number}[]} Sorted by price ascending
     */
    getVolumeAtPrice(bins = 24) {
        const key = `vap_${bins}`;
        if (this._cache.has(key)) return this._cache.get(key);

        const { candles } = this;

        // Find global price range
        let minPrice = Infinity;
        let maxPrice = -Infinity;
        for (const c of candles) {
            if (c.low < minPrice) minPrice = c.low;
            if (c.high > maxPrice) maxPrice = c.high;
        }

        // Single price level — one bin with all volume
        if (minPrice >= maxPrice) {
            const result = [{
                price: minPrice,
                volume: candles.reduce((s, c) => s + c.volume, 0),
            }];
            this._cache.set(key, result);
            return result;
        }

        const binWidth = (maxPrice - minPrice) / bins;
        const result = new Array(bins);

        // Initialize bins at their mid-prices
        for (let i = 0; i < bins; i++) {
            result[i] = { price: +(minPrice + binWidth * (i + 0.5)).toFixed(6), volume: 0 };
        }

        // Distribute each candle's volume across the bins it spans
        for (const c of candles) {
            if (c.volume <= 0 || c.high <= c.low) continue;

            const firstBin = Math.max(0, Math.floor((c.low - minPrice) / binWidth));
            const lastBin = Math.min(bins - 1, Math.floor((c.high - minPrice) / binWidth));
            const numBins = lastBin - firstBin + 1;
            const volPerBin = c.volume / numBins;

            for (let i = firstBin; i <= lastBin; i++) {
                result[i].volume += volPerBin;
            }
        }

        this._cache.set(key, result);
        return result;
    }

    /**
     * Find the Value Area — the price range containing `percentage` of total
     * traded volume.
     *
     * Starting from the Point of Control (POC = highest-volume price level),
     * expands outward one bin at a time, always adding the adjacent bin with
     * the higher volume, until the target percentage is reached.
     *
     * @param {number} [percentage=0.70] - Fraction of total volume (0-1)
     * @returns {{low: number, high: number, poc: number}}
     */
    getValueArea(percentage = 0.70) {
        const vap = this.getVolumeAtPrice();
        const totalVolume = vap.reduce((s, b) => s + b.volume, 0);

        if (totalVolume === 0) {
            return { low: 0, high: 0, poc: 0 };
        }

        if (vap.length === 1) {
            return { low: vap[0].price, high: vap[0].price, poc: vap[0].price };
        }

        // Locate Point of Control (POC) — bin with maximum volume
        let pocIdx = 0;
        let maxVol = vap[0].volume;
        for (let i = 1; i < vap.length; i++) {
            if (vap[i].volume > maxVol) {
                maxVol = vap[i].volume;
                pocIdx = i;
            }
        }

        const targetVolume = totalVolume * percentage;
        let accumulated = vap[pocIdx].volume;
        let lo = pocIdx;
        let hi = pocIdx;

        // Expand outward, always picking the denser neighbour
        while (accumulated < targetVolume) {
            const leftVol = lo > 0 ? vap[lo - 1].volume : -1;
            const rightVol = hi < vap.length - 1 ? vap[hi + 1].volume : -1;

            if (leftVol >= rightVol && leftVol >= 0) {
                lo--;
                accumulated += leftVol;
            } else if (rightVol >= 0) {
                hi++;
                accumulated += rightVol;
            } else {
                break; // Cannot expand further
            }
        }

        return {
            low: vap[lo].price,
            high: vap[hi].price,
            poc: vap[pocIdx].price,
        };
    }

    /**
     * Identify "riverbanks" — price levels with unusually high volume.
     *
     * These act as strong support/resistance zones where price tends to
     * respect or reverse. Strength is normalized 0-1 relative to the
     * maximum volume level.
     *
     * @param {number} [threshold=1.5] - Multiplier of average bin volume
     * @returns {{price: number, volume: number, strength: number}[]}
     */
    getHighVolumeNodes(threshold = 1.5) {
        const vap = this.getVolumeAtPrice();
        const avgVolume = vap.reduce((s, b) => s + b.volume, 0) / vap.length;
        const maxVolume = Math.max(...vap.map(b => b.volume));

        if (avgVolume === 0 || maxVolume === 0) return [];

        const cutoff = threshold * avgVolume;

        return vap
            .filter(b => b.volume > cutoff)
            .map(b => ({
                price: b.price,
                volume: +b.volume.toFixed(2),
                strength: +Math.min(1, b.volume / maxVolume).toFixed(4),
            }));
    }

    /**
     * Identify "gaps" — price levels with unusually low volume.
     *
     * Price tends to move quickly through these areas (like rapids),
     * offering minimal support or resistance.
     *
     * @param {number} [threshold=0.5] - Multiplier of average bin volume
     * @returns {{price: number, volume: number}[]}
     */
    getLowVolumeNodes(threshold = 0.5) {
        const vap = this.getVolumeAtPrice();
        const avgVolume = vap.reduce((s, b) => s + b.volume, 0) / vap.length;

        if (avgVolume === 0) return [];

        const cutoff = threshold * avgVolume;

        return vap
            .filter(b => b.volume < cutoff)
            .map(b => ({
                price: b.price,
                volume: +b.volume.toFixed(2),
            }));
    }

    /**
     * Determine volume profile skew.
     *
     * Compares total volume in the upper half of the price range vs the
     * lower half. When volume concentrates at higher price levels it
     * suggests buying dominance (and vice versa for lower levels).
     *
     * @returns {{direction: 'buying'|'selling'|'balanced', ratio: number}}
     */
    getVolumeProfileSkew() {
        const vap = this.getVolumeAtPrice();
        if (vap.length < 2) {
            return { direction: 'balanced', ratio: 1 };
        }

        const midIdx = Math.floor(vap.length / 2);
        const lowerVol = vap.slice(0, midIdx).reduce((s, b) => s + b.volume, 0);
        const upperVol = vap.slice(midIdx).reduce((s, b) => s + b.volume, 0);

        const totalVol = lowerVol + upperVol;
        if (totalVol === 0) return { direction: 'balanced', ratio: 1 };

        let ratio;
        if (lowerVol === 0) {
            ratio = upperVol > 0 ? 99 : 1;
        } else {
            ratio = upperVol / lowerVol;
        }

        let direction;
        if (ratio > 1.2) direction = 'buying';
        else if (ratio < 0.8) direction = 'selling';
        else direction = 'balanced';

        return { direction, ratio: +Math.min(ratio, 99).toFixed(4) };
    }
}

// ---------------------------------------------------------------------------
// OrderFlowMomentum Class
// ---------------------------------------------------------------------------

/**
 * Order Flow momentum analysis using OHLCV data.
 *
 * Uses a simple heuristic: compares close to open to classify each candle
 * as bullish (close > open) or bearish (close < open), weighted by volume.
 * This approximates order flow imbalance without tick-level data.
 */
export class OrderFlowMomentum {
    /**
     * @param {Object[]} candles - Array of OHLCV candles with volume
     */
    constructor(candles) {
        if (!Array.isArray(candles) || candles.length === 0) {
            throw new Error('OrderFlowMomentum: candles must be a non-empty array');
        }
        this.candles = candles;
    }

    /**
     * Calculate order-flow momentum over a sliding window.
     *
     * Compares total bullish volume to total bearish volume within the
     * window. Strength is normalised to 0-1 where 1 means all volume is
     * in one direction.
     *
     * @param {number} [windowSize=20] - Number of candles to consider
     * @returns {{direction: 'buying'|'selling'|'neutral', strength: number}}
     */
    getMomentum(windowSize = 20) {
        const { candles } = this;
        if (candles.length < 2) {
            return { direction: 'neutral', strength: 0 };
        }

        const window = candles.slice(-Math.min(windowSize, candles.length));
        let bullVol = 0;
        let bearVol = 0;

        for (const c of window) {
            if (c.close > c.open) {
                bullVol += c.volume;
            } else if (c.close < c.open) {
                bearVol += c.volume;
            } else {
                // Doji — split evenly
                bullVol += c.volume / 2;
                bearVol += c.volume / 2;
            }
        }

        const totalVol = bullVol + bearVol;
        if (totalVol === 0) return { direction: 'neutral', strength: 0 };

        const net = bullVol - bearVol;
        const strength = Math.abs(net) / totalVol;

        let direction;
        if (strength < 0.1) direction = 'neutral';
        else if (net > 0) direction = 'buying';
        else direction = 'selling';

        return { direction, strength: +strength.toFixed(4) };
    }

    /**
     * Detect absorption candles — where high volume meets minimal price
     * movement. This pattern often signals institutional accumulation or
     * distribution (large players absorbing the opposing flow).
     *
     * Criteria: volume > 1.5x average AND price-change / range < 0.3.
     *
     * @returns {{timestamp: number, volume: number, priceChange: number}[]}
     */
    detectAbsorption() {
        const { candles } = this;
        if (candles.length < MIN_CANDLES) return [];

        const avgVolume = candles.reduce((s, c) => s + c.volume, 0) / candles.length;
        const results = [];

        for (const c of candles) {
            const priceChange = Math.abs(c.close - c.open);
            const range = c.high - c.low;

            if (c.volume > avgVolume * 1.5 && range > 0 && priceChange / range < 0.3) {
                results.push({
                    timestamp: c.timestamp,
                    volume: c.volume,
                    priceChange: +priceChange.toFixed(4),
                });
            }
        }

        return results;
    }

    /**
     * Cumulative Delta — running sum of bullish minus bearish volume.
     *
     * Each candle contributes +volume when close > open (buying pressure),
     * -volume when close < open (selling pressure), and 0 for doji.
     *
     * @returns {{timestamp: number, delta: number}[]}
     */
    getCumulativeDelta() {
        const { candles } = this;
        if (candles.length === 0) return [];

        let runningDelta = 0;
        const result = [];

        for (const c of candles) {
            if (c.close > c.open) {
                runningDelta += c.volume;
            } else if (c.close < c.open) {
                runningDelta -= c.volume;
            }
            // Doji contributes 0 delta

            result.push({
                timestamp: c.timestamp,
                delta: runningDelta,
            });
        }

        return result;
    }
}

// ---------------------------------------------------------------------------
// VolatilityEddyDetector Class
// ---------------------------------------------------------------------------

/**
 * Detects volatility "eddies" — local zones where volatility spikes then
 * mean-reverts, analogous to eddy currents in fluid dynamics.
 *
 * Uses ATR (Average True Range) expansion / contraction patterns to locate
 * these zones and measure their intensity.
 */
export class VolatilityEddyDetector {
    /**
     * Detect volatility eddies — zones where ATR expands above a threshold
     * and subsequently contracts back below the mean.
     *
     * @param {Object[]} candles - OHLCV data
     * @param {number} [windowSize=14] - ATR calculation period
     * @returns {{startIdx: number, endIdx: number, intensity: number, meanPrice: number}[]}
     */
    detectEddies(candles, windowSize = 14) {
        if (!Array.isArray(candles) || candles.length < windowSize * 2) return [];

        const atrs = this._calcATR(candles, windowSize);
        if (atrs.length < 3) return [];

        // Threshold: mean ATR + one standard deviation
        const meanATR = atrs.reduce((s, v) => s + v, 0) / atrs.length;
        const atrStd = Math.sqrt(
            atrs.reduce((s, v) => s + (v - meanATR) ** 2, 0) / atrs.length,
        );
        const expansionThreshold = meanATR + atrStd;

        const eddies = [];
        let inEddy = false;
        let eddyStart = 0;
        let peakATR = 0;

        for (let i = 1; i < atrs.length; i++) {
            if (!inEddy && atrs[i] > expansionThreshold && atrs[i] > atrs[i - 1]) {
                // Eddy starts: ATR breaks above threshold while increasing
                inEddy = true;
                eddyStart = i;
                peakATR = atrs[i];
            } else if (inEddy) {
                if (atrs[i] > peakATR) peakATR = atrs[i];

                // Eddy ends: ATR falls back below the mean while decreasing
                if (atrs[i] < meanATR && atrs[i] < atrs[i - 1]) {
                    inEddy = false;
                    const intensity = meanATR > 0 ? (peakATR - meanATR) / meanATR : 0;
                    const startCandle = Math.max(0, eddyStart + windowSize - 1);
                    const endCandle = Math.min(candles.length - 1, i + windowSize - 1);
                    const meanPrice = this._meanMidPrice(candles, startCandle, endCandle);

                    eddies.push({
                        startIdx: startCandle,
                        endIdx: endCandle,
                        intensity: +intensity.toFixed(4),
                        meanPrice: +meanPrice.toFixed(4),
                    });
                }
            }
        }

        return eddies;
    }

    /**
     * Calculate a turbulence index — a 0-1 metric of how "turbulent"
     * current price action is.
     *
     * Combines three components:
     *   1. Relative volatility (ATR / price) — 40 % weight
     *   2. Volatility of volatility (rate of change of ATR) — 30 %
     *   3. Volume relative to average — 30 %
     *
     * @param {Object[]} candles - OHLCV data
     * @param {number} [windowSize=14] - Lookback window
     * @returns {number} 0-1 turbulence score
     */
    getTurbulenceIndex(candles, windowSize = 14) {
        if (!Array.isArray(candles) || candles.length < windowSize + 1) return 0;

        const atrs = this._calcATR(candles, windowSize);
        if (atrs.length < 2) return 0;

        const recent = candles.slice(-windowSize);
        const recentATRs = atrs.slice(-windowSize);

        // -- Component 1: relative volatility --
        const avgPrice = recent.reduce(
            (s, c) => s + (c.high + c.low) / 2, 0,
        ) / windowSize;
        const currentATR = recentATRs[recentATRs.length - 1];
        const relVol = avgPrice > 0 ? currentATR / avgPrice : 0;
        const normRelVol = Math.min(1, relVol * 10);

        // -- Component 2: volatility of volatility (ATR rate of change) --
        let avgATRChange = 0;
        let changeCount = 0;
        for (let i = 1; i < recentATRs.length; i++) {
            const prev = recentATRs[i - 1];
            if (prev > 0) {
                avgATRChange += Math.abs(recentATRs[i] - prev) / prev;
                changeCount++;
            }
        }
        avgATRChange = changeCount > 0 ? avgATRChange / changeCount : 0;
        const normATRChange = Math.min(1, avgATRChange * 5);

        // -- Component 3: volume spike --
        const avgVolume = recent.reduce((s, c) => s + c.volume, 0) / windowSize;
        const currentVolume = recent[recent.length - 1].volume;
        const volRatio = avgVolume > 0 ? currentVolume / avgVolume : 1;
        const normVolSpike = Math.min(1, Math.max(0, (volRatio - 1) / 2));

        // -- Weighted combination --
        const turbulence = normRelVol * 0.4 + normATRChange * 0.3 + normVolSpike * 0.3;

        return +Math.min(1, Math.max(0, turbulence)).toFixed(4);
    }

    // ---- Internal helpers ------------------------------------------------

    /**
     * Calculate Average True Range using Wilder's smoothed method.
     *
     * @private
     * @param {Object[]} candles
     * @param {number} period
     * @returns {number[]} ATR values; length = candles.length - period
     */
    _calcATR(candles, period) {
        const trs = [];
        for (let i = 1; i < candles.length; i++) {
            const h = candles[i].high;
            const l = candles[i].low;
            const pc = candles[i - 1].close;
            trs.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
        }

        if (trs.length < period) return [];

        const atrs = [];
        // First ATR: simple average of first 'period' TRs
        let atr = trs.slice(0, period).reduce((s, v) => s + v, 0) / period;
        atrs.push(atr);

        // Subsequent ATRs: Wilder's smoothed average
        for (let i = period; i < trs.length; i++) {
            atr = (atr * (period - 1) + trs[i]) / period;
            atrs.push(atr);
        }

        return atrs; // length = candles.length - period
    }

    /**
     * Compute mean mid-price over a candle range.
     *
     * @private
     * @param {Object[]} candles
     * @param {number} start - inclusive
     * @param {number} end - inclusive
     * @returns {number}
     */
    _meanMidPrice(candles, start, end) {
        const lo = Math.max(0, start);
        const hi = Math.min(candles.length - 1, end);
        if (lo > hi) return 0;

        let sum = 0;
        let count = 0;
        for (let i = lo; i <= hi; i++) {
            sum += (candles[i].high + candles[i].low) / 2;
            count++;
        }
        return count > 0 ? sum / count : 0;
    }
}

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/**
 * Composite liquidity score for a set of candles.
 *
 * Factors: average volume (0-40), spread tightness (0-30), volume
 * consistency (0-30). Returns a 0-100 score where higher = more liquid.
 *
 * @param {Object[]} candles - OHLCV data with volume
 * @returns {number} 0-100 liquidity score
 */
export function liquidityScore(candles) {
    if (!Array.isArray(candles) || candles.length < MIN_CANDLES) return 0;

    const n = candles.length;

    // -- Volume component (0-40) --
    const avgVolume = candles.reduce((s, c) => s + c.volume, 0) / n;
    // Scale so ~2500 volume = 40 pts (calibrated for common crypto/USD pairs)
    const volumeScore = Math.min(40, (avgVolume / 2500) * 40);

    // -- Spread component (0-30) --
    // Tighter spreads = more liquid
    const avgRange = candles.reduce((s, c) => s + (c.high - c.low), 0) / n;
    const avgMidPrice = candles.reduce(
        (s, c) => s + (c.high + c.low) / 2, 0,
    ) / n;
    const spreadRatio = avgMidPrice > 0 ? avgRange / avgMidPrice : 1;
    // A spread ratio of 0.002 (0.2 %) or less → full points
    const spreadScore = Math.max(0, 30 * (1 - Math.min(1, spreadRatio * 500)));

    // -- Consistency component (0-30) --
    // Low coefficient of variation = consistent liquidity
    const volStd = Math.sqrt(
        candles.reduce((s, c) => s + (c.volume - avgVolume) ** 2, 0) / n,
    );
    const cv = avgVolume > 0 ? volStd / avgVolume : 1;
    const consistencyScore = 30 * Math.max(0, 1 - Math.min(1, cv));

    return Math.min(100, Math.max(0, Math.round(volumeScore + spreadScore + consistencyScore)));
}

/**
 * Cluster high-volume VAP nodes into support / resistance zones.
 *
 * Adjacent high-volume price levels are grouped into a single zone.
 * A node is included if its volume is at least `strength` × maxVolume.
 *
 * @param {{price: number, volume: number}[]} vapProfile - Output from
 *     VolumeProfile.getVolumeAtPrice() (sorted by price)
 * @param {number} [strength=0.7] - Minimum node strength (0-1) for inclusion
 * @returns {{
 *   lowPrice: number,
 *   highPrice: number,
 *   midPrice: number,
 *   volume: number,
 *   strength: number,
 *   nodeCount: number,
 * }[]}
 */
export function supportResistanceZones(vapProfile, strength = 0.7) {
    if (!Array.isArray(vapProfile) || vapProfile.length === 0) return [];

    const maxVol = Math.max(...vapProfile.map(b => b.volume));
    if (maxVol === 0) return [];

    const cutoff = strength * maxVol;

    // Filter to significant nodes and sort by price
    const significant = vapProfile
        .filter(b => b.volume >= cutoff)
        .sort((a, b) => a.price - b.price);

    if (significant.length === 0) return [];

    // Infer typical bin width from first two bins
    const binWidth = vapProfile.length > 1
        ? Math.max(vapProfile[1].price - vapProfile[0].price, 1e-10)
        : 1;

    const zones = [];
    let current = {
        lowPrice: significant[0].price,
        highPrice: significant[0].price,
        totalVolume: significant[0].volume,
        count: 1,
    };

    for (let i = 1; i < significant.length; i++) {
        const gap = significant[i].price - significant[i - 1].price;

        if (gap <= binWidth * 1.5) {
            // Adjacent or nearly adjacent — merge into current zone
            current.highPrice = significant[i].price;
            current.totalVolume += significant[i].volume;
            current.count++;
        } else {
            // Gap detected — finalise current zone
            zones.push(_finaliseZone(current, maxVol));
            current = {
                lowPrice: significant[i].price,
                highPrice: significant[i].price,
                totalVolume: significant[i].volume,
                count: 1,
            };
        }
    }

    // Finalise the last zone
    zones.push(_finaliseZone(current, maxVol));

    return zones;
}

/**
 * Convert a raw zone accumulator into the output shape.
 *
 * @private
 * @param {{lowPrice: number, highPrice: number, totalVolume: number, count: number}} acc
 * @param {number} maxVol - Maximum single-bin volume across the VAP
 * @returns {Object}
 */
function _finaliseZone(acc, maxVol) {
    const avgBinVol = acc.totalVolume / acc.count;
    return {
        lowPrice: acc.lowPrice,
        highPrice: acc.highPrice,
        midPrice: +((acc.lowPrice + acc.highPrice) / 2).toFixed(6),
        volume: +acc.totalVolume.toFixed(2),
        strength: +Math.min(1, avgBinVol / maxVol).toFixed(4),
        nodeCount: acc.count,
    };
}
