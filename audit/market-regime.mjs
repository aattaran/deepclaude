/**
 * Market Regime Classifier — determines the market's current behavioral state
 *
 * Regimes detected:
 *   TRENDING_BULLISH / TRENDING_BEARISH — directional momentum
 *   RANGING — sideways, bounded by support/resistance
 *   ACCUMULATION — quiet low-vol compressing before a markup
 *   DISTRIBUTION — quiet low-vol after a rally, before markdown
 *   BREAKOUT — breaking out of a range with momentum
 *   BREAKDOWN — breaking down from a range with momentum
 *   VOLATILE — high turbulence, no clear regime
 *
 * Each regime comes with a confidence score and actionable parameters
 * (stop distance multiplier, take profit aggressiveness, position sizing factor).
 *
 * ES module. Zero npm dependencies. Uses Node built-ins only.
 */

const { min, max, abs, sqrt, pow } = Math;

// ─── Constants ───────────────────────────────────────────────────────────────

const REGIMES = {
    TRENDING_BULLISH: 'trending_bullish',
    TRENDING_BEARISH: 'trending_bearish',
    RANGING: 'ranging',
    ACCUMULATION: 'accumulation',
    DISTRIBUTION: 'distribution',
    BREAKOUT: 'breakout',
    BREAKDOWN: 'breakdown',
    VOLATILE: 'volatile',
};

const DEFAULT_WINDOW = 20;
const MIN_CANDLES = 10;

// ─── 1. Technical Indicators (pure functions) ────────────────────────────────

/**
 * Simple Moving Average over the last N closes.
 * Returns an array of same length as candles (early entries are null).
 */
function sma(candles, period) {
    const out = new Array(candles.length).fill(null);
    if (candles.length < period) return out;
    let sum = 0;
    for (let i = 0; i < period; i++) sum += candles[i].close;
    out[period - 1] = sum / period;
    for (let i = period; i < candles.length; i++) {
        sum += candles[i].close - candles[i - period].close;
        out[i] = sum / period;
    }
    return out;
}

/**
 * Exponential Moving Average.
 */
function ema(candles, period) {
    const out = new Array(candles.length).fill(null);
    if (candles.length === 0) return out;
    const k = 2 / (period + 1);
    // Seed with SMA for the first value
    let sum = 0;
    for (let i = 0; i < Math.min(period, candles.length); i++) sum += candles[i].close;
    out[period - 1] = sum / period;
    for (let i = period; i < candles.length; i++) {
        out[i] = candles[i].close * k + out[i - 1] * (1 - k);
    }
    return out;
}

/**
 * Average True Range — measures volatility.
 */
function atr(candles, period = 14) {
    const out = new Array(candles.length).fill(null);
    if (candles.length < 2) return out;

    const tr = new Array(candles.length).fill(0);
    for (let i = 1; i < candles.length; i++) {
        tr[i] = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        );
    }

    // Seed first ATR
    let sum = 0;
    const start = Math.min(period + 1, candles.length);
    for (let i = 1; i < start; i++) sum += tr[i];
    out[start - 1] = sum / (start - 1);

    for (let i = start; i < candles.length; i++) {
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period;
    }

    return out;
}

/**
 * Average Directional Index — trend strength (0-100).
 * Returns an array parallel to candles.
 */
function adx(candles, period = 14) {
    const out = new Array(candles.length).fill(null);
    if (candles.length < period + 1) return out;

    const dmPlus = new Array(candles.length).fill(0);
    const dmMinus = new Array(candles.length).fill(0);
    const tr = new Array(candles.length).fill(0);

    for (let i = 1; i < candles.length; i++) {
        const up = candles[i].high - candles[i - 1].high;
        const down = candles[i - 1].low - candles[i].low;

        if (up > down && up > 0) dmPlus[i] = up;
        if (down > up && down > 0) dmMinus[i] = down;

        tr[i] = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        );
    }

    // Wilder's smoothing
    const smoothedTr = new Array(candles.length).fill(null);
    const smoothedDmPlus = new Array(candles.length).fill(null);
    const smoothedDmMinus = new Array(candles.length).fill(null);

    // Seed with sum over first `period` bars
    let trSum = 0, dpSum = 0, dmSum = 0;
    const seedEnd = Math.min(period, candles.length - 1);
    for (let i = 1; i <= seedEnd; i++) {
        trSum += tr[i];
        dpSum += dmPlus[i];
        dmSum += dmMinus[i];
    }

    smoothedTr[seedEnd] = trSum;
    smoothedDmPlus[seedEnd] = dpSum;
    smoothedDmMinus[seedEnd] = dmSum;

    for (let i = seedEnd + 1; i < candles.length; i++) {
        smoothedTr[i] = smoothedTr[i - 1] - smoothedTr[i - 1] / period + tr[i];
        smoothedDmPlus[i] = smoothedDmPlus[i - 1] - smoothedDmPlus[i - 1] / period + dmPlus[i];
        smoothedDmMinus[i] = smoothedDmMinus[i - 1] - smoothedDmMinus[i - 1] / period + dmMinus[i];
    }

    const diPlus = new Array(candles.length).fill(null);
    const diMinus = new Array(candles.length).fill(null);
    const dx = new Array(candles.length).fill(null);

    for (let i = seedEnd; i < candles.length; i++) {
        if (smoothedTr[i] === 0) continue;
        diPlus[i] = (smoothedDmPlus[i] / smoothedTr[i]) * 100;
        diMinus[i] = (smoothedDmMinus[i] / smoothedTr[i]) * 100;
        const sumDi = diPlus[i] + diMinus[i];
        if (sumDi > 0) {
            dx[i] = (abs(diPlus[i] - diMinus[i]) / sumDi) * 100;
        }
    }

    // Smoothed DX = ADX
    for (let i = seedEnd; i < candles.length; i++) {
        if (i === seedEnd && dx[seedEnd] !== null) {
            let dxSum = 0, dxCount = 0;
            for (let j = seedEnd; j < Math.min(seedEnd + period, candles.length) && dx[j] !== null; j++) {
                dxSum += dx[j];
                dxCount++;
            }
            out[i] = dxCount > 0 ? dxSum / dxCount : null;
        } else if (out[i - 1] !== null && dx[i] !== null) {
            out[i] = (out[i - 1] * (period - 1) + dx[i]) / period;
        }
    }

    return out;
}

/**
 * Linear Regression slope over `period` closes.
 * Returns slope (price change per bar) and r-squared for each bar.
 */
function linearRegressionSlope(candles, period) {
    const out = new Array(candles.length).fill(null);
    if (candles.length < period) return out;

    // Precompute x (indices)
    const meanX = (period - 1) / 2;

    for (let i = period - 1; i < candles.length; i++) {
        let sumXY = 0, sumXX = 0, sumY = 0;
        for (let j = 0; j < period; j++) {
            const idx = i - period + 1 + j;
            const price = candles[idx].close;
            sumXY += j * price;
            sumXX += j * j;
            sumY += price;
        }
        const meanY = sumY / period;
        const slope = (sumXY - period * meanX * meanY) / (sumXX - period * meanX * meanX);

        // R-squared
        let ssRes = 0, ssTot = 0;
        for (let j = 0; j < period; j++) {
            const idx = i - period + 1 + j;
            const predicted = slope * (j - meanX) + meanY;
            ssRes += pow(candles[idx].close - predicted, 2);
            ssTot += pow(candles[idx].close - meanY, 2);
        }
        const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 0;

        out[i] = { slope, r2 };
    }

    return out;
}

/**
 * Bollinger Bands.
 * Returns arrays of { middle, upper, lower, bandwidth, percentB }.
 * bandwidth = (upper - lower) / middle — measures volatility relative to price.
 * percentB = (price - lower) / (upper - lower) — where price is in the band.
 */
function bollingerBands(candles, period = 20, multiplier = 2) {
    const out = new Array(candles.length).fill(null);
    const ma = sma(candles, period);

    for (let i = period - 1; i < candles.length; i++) {
        let variance = 0;
        for (let j = i - period + 1; j <= i; j++) {
            variance += pow(candles[j].close - ma[i], 2);
        }
        const std = sqrt(variance / period);
        const upper = ma[i] + multiplier * std;
        const lower = ma[i] - multiplier * std;
        out[i] = {
            middle: ma[i],
            upper,
            lower,
            bandwidth: ma[i] > 0 ? (upper - lower) / ma[i] : 0,
            percentB: (upper - lower) > 0 ? (candles[i].close - lower) / (upper - lower) : 0.5,
        };
    }

    return out;
}

/**
 * Volume trend — average volume over a period.
 */
function avgVolume(candles, period) {
    const out = new Array(candles.length).fill(null);
    if (candles.length < period) return out;
    let sum = 0;
    for (let i = 0; i < period; i++) sum += candles[i].volume;
    out[period - 1] = sum / period;
    for (let i = period; i < candles.length; i++) {
        sum += candles[i].volume - candles[i - period].volume;
        out[i] = sum / period;
    }
    return out;
}

// ─── 2. Support / Resistance Detection ───────────────────────────────────────

/**
 * Find local minima/maxima for S/R level detection.
 */
function findPivots(candles, lookback = 5) {
    const highs = [];
    const lows = [];

    for (let i = lookback; i < candles.length - lookback; i++) {
        let isHigh = true, isLow = true;
        for (let j = i - lookback; j <= i + lookback; j++) {
            if (j === i) continue;
            if (candles[j].high >= candles[i].high) isHigh = false;
            if (candles[j].low <= candles[i].low) isLow = false;
        }
        if (isHigh) highs.push({ index: i, price: candles[i].high, timestamp: candles[i].timestamp });
        if (isLow) lows.push({ index: i, price: candles[i].low, timestamp: candles[i].timestamp });
    }

    return { highs, lows };
}

/**
 * Cluster nearby price levels.
 */
function clusterLevels(levels, maxDistancePct = 0.005) {
    if (!levels.length) return [];
    const sorted = [...levels].sort((a, b) => a.price - b.price);
    const clusters = [];
    let current = { price: sorted[0].price, count: 1, timestamps: [sorted[0].timestamp] };

    for (let i = 1; i < sorted.length; i++) {
        const pctDiff = abs(sorted[i].price - current.price) / current.price;
        if (pctDiff <= maxDistancePct) {
            current.price = (current.price * current.count + sorted[i].price) / (current.count + 1);
            current.count++;
            current.timestamps.push(sorted[i].timestamp);
        } else {
            clusters.push(current);
            current = { price: sorted[i].price, count: 1, timestamps: [sorted[i].timestamp] };
        }
    }
    clusters.push(current);

    return clusters.sort((a, b) => b.count - a.count);
}

// ─── 3. Regime Classification ────────────────────────────────────────────────

/**
 * Classify the current market regime from an array of OHLCV candles.
 *
 * Uses:
 *   - ADX for trend strength
 *   - Linear regression for trend direction
 *   - Bollinger Bands for volatility
 *   - Pivot S/R for range detection
 *   - Volume profile for accumulation/distribution
 *
 * @param {Array<{ timestamp, open, high, low, close, volume }>} candles
 * @param {object} [options]
 * @param {number} [options.trendWindow=20] — candles for trend detection
 * @param {number} [options.volWindow=14] — candles for volatility (ATR)
 * @param {number} [options.adxThreshold=25] — ADX above this = trending
 * @param {number} [options.rangeThreshold=0.01] — max range width as fraction of price for ranging
 * @param {number} [options.volExpansionFactor=1.5] — vol expansion for breakout detection
 * @returns {RegimeResult}
 */
export function classifyRegime(candles, options = {}) {
    if (!candles || candles.length < MIN_CANDLES) {
        return {
            regime: REGIMES.RANGING,
            confidence: 0,
            direction: null,
            strength: 0,
            timestamp: candles?.[candles.length - 1]?.timestamp ?? null,
            params: defaultParams(),
            details: { reason: 'insufficient data' },
        };
    }

    const trendWindow = options.trendWindow ?? DEFAULT_WINDOW;
    const volWindow = options.volWindow ?? 14;
    const adxThreshold = options.adxThreshold ?? 25;
    const rangeThreshold = options.rangeThreshold ?? 0.01;
    const volExpansionFactor = options.volExpansionFactor ?? 1.5;

    const lastIdx = candles.length - 1;
    const currentPrice = candles[lastIdx].close;

    // Compute indicators
    const adxArr = adx(candles, volWindow);
    const atrArr = atr(candles, volWindow);
    const bb = bollingerBands(candles, trendWindow);
    const regressReg = linearRegressionSlope(candles, trendWindow);
    const volAvg = avgVolume(candles, trendWindow);

    const currentADX = adxArr[lastIdx] ?? 25;
    const currentATR = atrArr[lastIdx] ?? 0;
    const currentBB = bb[lastIdx];
    const currentReg = regressReg[lastIdx];
    const currentVolAvg = volAvg[lastIdx] ?? 0;

    // ATR as % of price
    const atrPct = currentPrice > 0 ? currentATR / currentPrice : 0;

    // Trend direction from regression slope
    const slopeNormalized = currentReg
        ? currentReg.slope / currentPrice
        : 0;
    const r2 = currentReg?.r2 ?? 0;
    const direction = slopeNormalized > 0.0001 ? 'bullish'
        : slopeNormalized < -0.0001 ? 'bearish'
        : null;

    // Trend strength (ADX normalized to 0-1)
    const trendStrength = min(currentADX / 50, 1); // ADX > 50 = very strong trend

    // Volatility regime
    const bandwidth = currentBB?.bandwidth ?? 0;
    const isLowVol = bandwidth < 0.02; // Bollinger Band width < 2%
    const isHighVol = bandwidth > 0.08; // Bollinger Band width > 8%

    // Volume analysis
    const recentVol = candles.slice(max(0, lastIdx - 5), lastIdx + 1)
        .reduce((s, c) => s + c.volume, 0) / min(5, lastIdx + 1);
    const volRatio = currentVolAvg > 0 ? recentVol / currentVolAvg : 1;

    // Pivots and S/R
    const pivots = findPivots(candles, 5);
    const resistantClusters = clusterLevels(pivots.highs, 0.005);
    const supportClusters = clusterLevels(pivots.lows, 0.005);

    // Find nearest support and resistance
    const nearestResistance = resistantClusters
        .filter(r => r.price > currentPrice)
        .sort((a, b) => a.price - b.price)[0] ?? null;
    const nearestSupport = supportClusters
        .filter(s => s.price < currentPrice)
        .sort((a, b) => b.price - a.price)[0] ?? null;

    // Range width
    const rangeWidth = nearestResistance && nearestSupport
        ? (nearestResistance.price - nearestSupport.price) / currentPrice
        : Infinity;

    let regime, confidence, details = {};

    // ── Classification Logic ──

    // 1. Check for trending regime
    if (currentADX > adxThreshold && r2 > 0.5 && direction) {
        confidence = min(trendStrength * 0.8 + r2 * 0.2, 1);
        regime = direction === 'bullish' ? REGIMES.TRENDING_BULLISH : REGIMES.TRENDING_BEARISH;
        details = {
            reason: `${direction} trend: ADX=${currentADX.toFixed(1)}, R²=${r2.toFixed(3)}, slope=${(slopeNormalized * 100).toFixed(4)}%`,
            adx: +currentADX.toFixed(1),
            r2: +r2.toFixed(3),
            slope: slopeNormalized,
        };
    }
    // 2. Check for breakout/breakdown (low vol → high vol expansion with direction)
    else if (isLowVol && volRatio > volExpansionFactor) {
        const breakDirection = slopeNormalized > 0.00005 ? 'bullish' : 'bearish';
        regime = breakDirection === 'bullish' ? REGIMES.BREAKOUT : REGIMES.BREAKDOWN;
        confidence = min(volRatio / 3, 1);
        details = {
            reason: `${breakDirection} ${regime}: vol expanded ${volRatio.toFixed(1)}x from low BB width ${(bandwidth * 100).toFixed(2)}%`,
            bandwidth: +bandwidth.toFixed(4),
            volumeRatio: +volRatio.toFixed(1),
        };
    }
    // 3. Check for ranging (clear S/R, moderate vol, low trend)
    else if (rangeWidth < rangeThreshold && !isHighVol && currentADX < adxThreshold) {
        regime = REGIMES.RANGING;
        confidence = min(1 - rangeWidth / rangeThreshold, 1 - currentADX / 50, 1);
        details = {
            reason: `range width=${(rangeWidth * 100).toFixed(2)}%, ADX=${currentADX.toFixed(1)}`,
            support: nearestSupport?.price ?? null,
            resistance: nearestResistance?.price ?? null,
            rangeWidth: +rangeWidth.toFixed(4),
            adx: +currentADX.toFixed(1),
        };
    }
    // 4. Check for accumulation (low vol, flat/drifting slightly, below mid-range, low ADX)
    else if (isLowVol && currentADX < adxThreshold && !direction && rangeWidth < 0.05) {
        // Accumulation: price near bottom of range, low vol
        const posInRange = nearestResistance && nearestSupport
            ? (currentPrice - nearestSupport.price) / (nearestResistance.price - nearestSupport.price)
            : 0.5;
        if (posInRange < 0.4) {
            regime = REGIMES.ACCUMULATION;
            confidence = min((1 - posInRange) * 1.5, 1);
        } else {
            regime = REGIMES.DISTRIBUTION;
            confidence = min(posInRange * 1.2, 1);
        }
        details = {
            reason: `${regime}: low vol (BBw=${(bandwidth * 100).toFixed(2)}%), pos=${(posInRange * 100).toFixed(0)}% in range`,
            bandwidth: +bandwidth.toFixed(4),
            positionInRange: +posInRange.toFixed(3),
        };
    }
    // 5. Check for distribution (low vol after a rally, near top of range)
    else if (isLowVol && currentADX < 30 && slopeNormalized < 0 && rangeWidth < 0.05) {
        const posInRange = nearestResistance && nearestSupport
            ? (currentPrice - nearestSupport.price) / (nearestResistance.price - nearestSupport.price)
            : 0.5;
        regime = posInRange > 0.6 ? REGIMES.DISTRIBUTION : REGIMES.RANGING;
        confidence = posInRange > 0.6 ? min(posInRange * 0.8, 0.7) : 0.5;
        details = {
            reason: `${regime}: near top of range, vol normal`,
            positionInRange: +posInRange.toFixed(3),
        };
    }
    // 6. High volatility = no clear regime
    else if (isHighVol) {
        regime = REGIMES.VOLATILE;
        confidence = min(bandwidth / 0.1, 0.8);
        details = {
            reason: `high volatility: BB width=${(bandwidth * 100).toFixed(2)}%`,
            bandwidth: +bandwidth.toFixed(4),
        };
    }
    // 7. Default to ranging with low confidence
    else {
        regime = REGIMES.RANGING;
        confidence = 0.4;
        details = { reason: 'no clear regime pattern detected' };
    }

    return {
        regime,
        confidence: +confidence.toFixed(3),
        direction,
        strength: +trendStrength.toFixed(3),
        timestamp: candles[lastIdx].timestamp,
        params: regimeParams(regime, atrPct, rangeWidth, confidence),
        details,
    };
}

/**
 * Get actionable parameters for each regime.
 */
function regimeParams(regime, atrPct, rangeWidth, confidence) {
    const base = {
        stopMultiplier: 1.0,
        tpAggressiveness: 0.5,
        positionSizeFactor: 1.0,
        trailingStopPct: atrPct * 2,
        maxHoldingBars: 100,
    };

    switch (regime) {
        case REGIMES.TRENDING_BULLISH:
            return {
                ...base,
                stopMultiplier: 1.5,
                tpAggressiveness: 0.3,
                positionSizeFactor: 1.0,
                trailingStopPct: atrPct * 2.5,
                maxHoldingBars: 200,
            };
        case REGIMES.TRENDING_BEARISH:
            return {
                ...base,
                stopMultiplier: 1.5,
                tpAggressiveness: 0.3,
                positionSizeFactor: 0.7,
                trailingStopPct: atrPct * 2.5,
                maxHoldingBars: 200,
            };
        case REGIMES.RANGING:
            return {
                ...base,
                stopMultiplier: 0.8,
                tpAggressiveness: 0.8,
                positionSizeFactor: 0.6,
                trailingStopPct: rangeWidth * 0.5,
                maxHoldingBars: 50,
            };
        case REGIMES.ACCUMULATION:
            return {
                ...base,
                stopMultiplier: 0.5,
                tpAggressiveness: 0.9,
                positionSizeFactor: 0.4,
                trailingStopPct: atrPct * 1.5,
                maxHoldingBars: 150,
            };
        case REGIMES.DISTRIBUTION:
            return {
                ...base,
                stopMultiplier: 1.2,
                tpAggressiveness: 0.7,
                positionSizeFactor: 0.3,
                trailingStopPct: atrPct * 2,
                maxHoldingBars: 60,
            };
        case REGIMES.BREAKOUT:
            return {
                ...base,
                stopMultiplier: 0.7,
                tpAggressiveness: 0.4,
                positionSizeFactor: 0.8,
                trailingStopPct: atrPct * 2,
                maxHoldingBars: 150,
            };
        case REGIMES.BREAKDOWN:
            return {
                ...base,
                stopMultiplier: 0.7,
                tpAggressiveness: 0.5,
                positionSizeFactor: 0.5,
                trailingStopPct: atrPct * 1.5,
                maxHoldingBars: 100,
            };
        case REGIMES.VOLATILE:
            return {
                ...base,
                stopMultiplier: 2.0,
                tpAggressiveness: 0.5,
                positionSizeFactor: 0.2,
                trailingStopPct: atrPct * 3,
                maxHoldingBars: 30,
            };
        default:
            return base;
    }
}

function defaultParams() {
    return regimeParams(REGIMES.RANGING, 0.01, 0.02, 0);
}

// ─── 4. Regime Change Detection ──────────────────────────────────────────────

/**
 * Streaming regime classifier — efficiently maintains state across candle updates.
 *
 * Stores a ring buffer of the last N candles. On each update(), classifies
 * the current regime and detects when it changes.
 */
export class StreamingRegimeClassifier {
    constructor(options = {}) {
        this._maxCandles = options.maxCandles ?? 200;
        this._trendWindow = options.trendWindow ?? DEFAULT_WINDOW;
        this._volWindow = options.volWindow ?? 14;
        this._adxThreshold = options.adxThreshold ?? 25;
        this._candles = [];
        this._currentRegime = null;
        this._regimeHistory = [];
        this._lastChangeIdx = 0;
    }

    /**
     * Feed a new candle. Returns `{ regime, changed }` — changed is true
     * if the regime switched on this candle.
     */
    update(candle) {
        this._candles.push(candle);
        if (this._candles.length > this._maxCandles) {
            this._candles.shift();
        }

        if (this._candles.length < MIN_CANDLES) {
            return { regime: null, changed: false };
        }

        const result = classifyRegime(this._candles, {
            trendWindow: this._trendWindow,
            volWindow: this._volWindow,
            adxThreshold: this._adxThreshold,
        });

        const changed = this._currentRegime !== null && this._currentRegime !== result.regime;
        if (changed || this._currentRegime === null) {
            this._regimeHistory.push({
                ts: candle.timestamp,
                from: this._currentRegime,
                to: result.regime,
                confidence: result.confidence,
                index: this._candles.length - 1,
            });
            this._currentRegime = result.regime;
            this._lastChangeIdx = this._candles.length - 1;
        }

        return { regime: result, changed };
    }

    /** Get the current regime result (null if insufficient data). */
    current() {
        if (this._candles.length < MIN_CANDLES) return null;
        return classifyRegime(this._candles, {
            trendWindow: this._trendWindow,
            volWindow: this._volWindow,
            adxThreshold: this._adxThreshold,
        });
    }

    /** Get all regime changes detected. */
    getChanges() {
        return [...this._regimeHistory];
    }

    /** Bars since last regime change. */
    barsSinceChange() {
        if (this._candles.length === 0) return 0;
        return this._candles.length - 1 - this._lastChangeIdx;
    }

    /** Reset classifier state. */
    reset() {
        this._candles = [];
        this._currentRegime = null;
        this._regimeHistory = [];
        this._lastChangeIdx = 0;
    }

    /** Number of candles currently stored. */
    get size() { return this._candles.length; }
}

// ─── 5. Multi-Timeframe Regime Confluence ────────────────────────────────────

/**
 * Compare regime classifications across multiple timeframes.
 * Returns a confluent view — when multiple timeframes agree, confidence is higher.
 *
 * @param {RegimeResult[]} regimes — array of results from classifyRegime on different timeframes
 * @returns {MultiTimeframeResult}
 */
export function multiTimeframeRegime(regimes) {
    if (!regimes.length) return { regime: REGIMES.RANGING, confidence: 0, consensus: 0, breakdown: {} };

    const counts = {};
    for (const r of regimes) {
        if (!r) continue;
        counts[r.regime] = (counts[r.regime] || 0) + 1;
    }

    // Find majority regime
    let maxCount = 0, majorityRegime = REGIMES.RANGING;
    for (const [regime, count] of Object.entries(counts)) {
        if (count > maxCount) {
            maxCount = count;
            majorityRegime = regime;
        }
    }

    const consensus = maxCount / regimes.length;
    const avgConfidence = regimes.reduce((s, r) => s + (r?.confidence ?? 0), 0) / regimes.length;
    const combinedConfidence = consensus * 0.7 + avgConfidence * 0.3;

    const breakdown = {};
    for (let i = 0; i < regimes.length; i++) {
        if (regimes[i]) {
            breakdown[`tf${i + 1}`] = regimes[i].regime;
        }
    }

    return {
        regime: majorityRegime,
        confidence: +combinedConfidence.toFixed(3),
        consensus: +consensus.toFixed(3),
        breakdown,
    };
}

// ─── Exports ─────────────────────────────────────────────────────────────────

export { REGIMES };
