/**
 * Smart Money Concepts (SMC/ICT) Detector
 *
 * Detects advanced market structure patterns from OHLCV candle data.
 * Zero dependencies. ESM module.
 *
 * Concepts:
 *   - Swing Points: Local highs/lows with configurable lookback
 *   - Break of Structure (BOS): Price breaking a swing point
 *   - Change of Character (CHoCH): First break against prevailing trend
 *   - Order Blocks (OB): Last candle before an impulsive move
 *   - Mitigation Blocks: Price returning to and bouncing from an OB
 *   - Breaker Blocks: S/R level flip after a breakout
 *   - Imbalance (FVG): Price gaps / volume spikes
 *   - Inducement: False breakout that traps traders
 */

const { isArray } = Array;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Clamp a value between min and max. */
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/** Fractional difference between two prices. */
const fracDiff = (a, b) => (a === 0 ? 0 : Math.abs(a - b) / a);

/** Default volume for fixture generation. */
const DEFAULT_VOLUME = 1000;

// ---------------------------------------------------------------------------
// SMCDetector
// ---------------------------------------------------------------------------

export class SMCDetector {
  /**
   * @param {Array}  candles  - Array of { timestamp, open, high, low, close, volume }
   * @param {Object} options
   * @param {number} [options.swingLookback=5]  - Bars to look left/right for swing points
   * @param {number} [options.bosThreshold=0.0005]  - BOS threshold as fraction of price
   * @param {number} [options.chochThreshold=0.0003]  - CHoCH threshold
   */
  constructor(candles = [], options = {}) {
    this.candles = candles || [];
    this.swingLookback = options.swingLookback ?? 5;
    this.bosThreshold = options.bosThreshold ?? 0.0005;
    this.chochThreshold = options.chochThreshold ?? 0.0003;

    // result caches
    this._swingPoints = undefined;
    this._bos = undefined;
    this._choch = undefined;
    this._orderBlocks = undefined;
    this._mitigations = undefined;
    this._breakers = undefined;
    this._imbalances = undefined;
    this._inducements = undefined;
  }

  // -----------------------------------------------------------------------
  // Swing Points
  // -----------------------------------------------------------------------

  /**
   * Detect swing high and low points.
   * A swing high is a bar whose high is higher than `swingLookback` bars
   * before and after it.  Swing low is the inverse.
   */
  detectSwingPoints() {
    if (this._swingPoints !== undefined) return this._swingPoints;

    const { candles, swingLookback } = this;
    const swingHighs = [];
    const swingLows = [];

    if (candles && candles.length >= swingLookback * 2 + 1) {
      for (let i = swingLookback; i < candles.length - swingLookback; i++) {
        const c = candles[i];

        // ---- swing high ----
        let isHigh = true;
        for (let j = i - swingLookback; j <= i + swingLookback; j++) {
          if (j === i) continue;
          if (candles[j].high >= c.high) { isHigh = false; break; }
        }
        if (isHigh) {
          swingHighs.push({ index: i, price: c.high, timestamp: c.timestamp });
        }

        // ---- swing low ----
        let isLow = true;
        for (let j = i - swingLookback; j <= i + swingLookback; j++) {
          if (j === i) continue;
          if (candles[j].low <= c.low) { isLow = false; break; }
        }
        if (isLow) {
          swingLows.push({ index: i, price: c.low, timestamp: c.timestamp });
        }
      }
    }

    this._swingPoints = { swingHighs, swingLows };
    return this._swingPoints;
  }

  // -----------------------------------------------------------------------
  // Break of Structure
  // -----------------------------------------------------------------------

  /**
   * Detect Breaks of Structure (BOS).
   * Bullish BOS: price breaks above the most recent unbroken swing high.
   * Bearish BOS:  price breaks below the most recent unbroken swing low.
   */
  detectBOS() {
    if (this._bos !== undefined) return this._bos;

    const bos = [];
    const { candles } = this;
    if (candles.length < 3) { this._bos = bos; return bos; }

    const { swingHighs, swingLows } = this.detectSwingPoints();
    const brokenHighs = new Set();
    const brokenLows = new Set();

    for (let i = 1; i < candles.length; i++) {
      const c = candles[i];

      // ---- bullish BOS: break most recent unbroken swing high ----
      for (let j = swingHighs.length - 1; j >= 0; j--) {
        const sh = swingHighs[j];
        if (sh.index >= i || brokenHighs.has(sh.index)) continue;
        const threshold = sh.price * (1 + this.bosThreshold);
        if (c.high > threshold && c.close > sh.price) {
          bos.push({
            index: i,
            timestamp: c.timestamp,
            direction: 'bullish',
            brokenSwingIndex: sh.index,
            brokenSwingPrice: sh.price
          });
          brokenHighs.add(sh.index);
        }
        break; // only the most recent unbroken swing high matters
      }

      // ---- bearish BOS: break most recent unbroken swing low ----
      for (let j = swingLows.length - 1; j >= 0; j--) {
        const sl = swingLows[j];
        if (sl.index >= i || brokenLows.has(sl.index)) continue;
        const threshold = sl.price * (1 - this.bosThreshold);
        if (c.low < threshold && c.close < sl.price) {
          bos.push({
            index: i,
            timestamp: c.timestamp,
            direction: 'bearish',
            brokenSwingIndex: sl.index,
            brokenSwingPrice: sl.price
          });
          brokenLows.add(sl.index);
        }
        break;
      }
    }

    this._bos = bos;
    return bos;
  }

  // -----------------------------------------------------------------------
  // Change of Character
  // -----------------------------------------------------------------------

  /**
   * Detect Change of Character (CHoCH).
   *
   * A CHoCH is the first break against the prevailing trend direction.
   * - In a prior uptrend, the first break below a swing low → bearish CHoCH.
   * - In a prior downtrend, the first break above a swing high → bullish CHoCH.
   *
   * Returns an array with at most one entry per direction (first of each).
   */
  detectCHoCH() {
    if (this._choch !== undefined) return this._choch;

    const choch = [];
    const { candles } = this;
    if (candles.length < this.swingLookback * 2 + 1) { this._choch = choch; return choch; }

    const { swingHighs, swingLows } = this.detectSwingPoints();

    let foundBearish = false;
    let foundBullish = false;

    // Helper: check whether the swing structure BEFORE a given swing point
    // showed an uptrend (last 2 highs rising AND last 2 lows rising).
    const wasPriorUptrend = (beforeIndex) => {
      const ph = swingHighs.filter(sh => sh.index < beforeIndex);
      const pl = swingLows.filter(sl => sl.index < beforeIndex);
      if (ph.length < 2 || pl.length < 2) return false;
      return (
        ph[ph.length - 1].price > ph[ph.length - 2].price &&
        pl[pl.length - 1].price > pl[pl.length - 2].price
      );
    };

    // Helper: check whether the swing structure BEFORE a given swing point
    // showed a downtrend (last 2 highs falling AND last 2 lows falling).
    const wasPriorDowntrend = (beforeIndex) => {
      const ph = swingHighs.filter(sh => sh.index < beforeIndex);
      const pl = swingLows.filter(sl => sl.index < beforeIndex);
      if (ph.length < 2 || pl.length < 2) return false;
      return (
        ph[ph.length - 1].price < ph[ph.length - 2].price &&
        pl[pl.length - 1].price < pl[pl.length - 2].price
      );
    };

    const brokenLows = new Set();
    const brokenHighs = new Set();

    for (let i = 1; i < candles.length; i++) {
      if (foundBearish && foundBullish) break;
      const c = candles[i];

      // ---- bearish CHoCH ----
      if (!foundBearish) {
        const mruLow = ((arr) => {
          for (let k = arr.length - 1; k >= 0; k--) {
            if (arr[k].index < i && !brokenLows.has(arr[k].index)) return arr[k];
          }
          return null;
        })(swingLows);

        if (mruLow) {
          const threshold = mruLow.price * (1 - this.chochThreshold);
          if (c.low < threshold && c.close < mruLow.price) {
            if (wasPriorUptrend(mruLow.index)) {
              choch.push({
                index: i,
                timestamp: c.timestamp,
                direction: 'bearish',
                brokenSwingIndex: mruLow.index
              });
              foundBearish = true;
            }
            brokenLows.add(mruLow.index);
          }
        }
      }

      // ---- bullish CHoCH ----
      if (!foundBullish) {
        const mruHigh = ((arr) => {
          for (let k = arr.length - 1; k >= 0; k--) {
            if (arr[k].index < i && !brokenHighs.has(arr[k].index)) return arr[k];
          }
          return null;
        })(swingHighs);

        if (mruHigh) {
          const threshold = mruHigh.price * (1 + this.chochThreshold);
          if (c.high > threshold && c.close > mruHigh.price) {
            if (wasPriorDowntrend(mruHigh.index)) {
              choch.push({
                index: i,
                timestamp: c.timestamp,
                direction: 'bullish',
                brokenSwingIndex: mruHigh.index
              });
              foundBullish = true;
            }
            brokenHighs.add(mruHigh.index);
          }
        }
      }
    }

    this._choch = choch;
    return choch;
  }

  // -----------------------------------------------------------------------
  // Order Blocks
  // -----------------------------------------------------------------------

  /**
   * Detect SMC Order Blocks.
   *
   * Bullish OB: the last bearish candle immediately before a strong bullish
   *   impulsive move (the OB is the bearish candle, its range is the zone).
   * Bearish OB: the last bullish candle immediately before a strong bearish
   *   impulsive move.
   *
   * Returns array of { index, timestamp, price, type, range: {high, low}, strength }
   */
  detectOrderBlocks() {
    if (this._orderBlocks !== undefined) return this._orderBlocks;

    const obs = [];
    const { candles } = this;
    if (candles.length < 5) { this._orderBlocks = obs; return obs; }

    for (let i = 2; i < candles.length; i++) {
      const prev = candles[i - 1];
      const curr = candles[i];

      // ---- Bullish OB: prev is bearish, curr is strongly bullish ----
      if (prev.close < prev.open && curr.close > curr.open) {
        const movePct = (curr.close - curr.open) / curr.open;
        const prevMovePct = (prev.open - prev.close) / prev.open;
        if (prevMovePct > 0 && movePct > prevMovePct * 1.5) {
          obs.push({
            index: i - 1,             // the bearish candle is the OB
            timestamp: prev.timestamp,
            price: prev.open,         // top of OB body (open of bearish candle)
            type: 'bullish',
            range: {
              high: Math.max(prev.high, prev.open),
              low: Math.min(prev.low, prev.close)
            },
            strength: movePct / prevMovePct
          });
        }
      }

      // ---- Bearish OB: prev is bullish, curr is strongly bearish ----
      if (prev.close > prev.open && curr.close < curr.open) {
        const movePct = (curr.open - curr.close) / curr.open;
        const prevMovePct = (prev.close - prev.open) / prev.open;
        if (prevMovePct > 0 && movePct > prevMovePct * 1.5) {
          obs.push({
            index: i - 1,             // the bullish candle is the OB
            timestamp: prev.timestamp,
            price: prev.close,        // bottom of OB body (close of bullish candle)
            type: 'bearish',
            range: {
              high: Math.max(prev.high, prev.close),
              low: Math.min(prev.low, prev.open)
            },
            strength: movePct / prevMovePct
          });
        }
      }
    }

    this._orderBlocks = obs;
    return obs;
  }

  // -----------------------------------------------------------------------
  // Mitigation Blocks
  // -----------------------------------------------------------------------

  /**
   * Detect Mitigation Blocks — price returns to an OB zone and bounces
   * (the OB "mitigates" the move).
   */
  detectMitigationBlocks() {
    if (this._mitigations !== undefined) return this._mitigations;

    const mitigations = [];
    const { candles } = this;
    if (candles.length < 5) { this._mitigations = mitigations; return mitigations; }

    const obs = this.detectOrderBlocks();

    for (const ob of obs) {
      for (let i = ob.index + 2; i < candles.length; i++) {
        const c = candles[i];
        const touchesZone = c.high >= ob.range.low && c.low <= ob.range.high;
        if (!touchesZone) continue;

        const mitigated =
          ob.type === 'bullish'
            ? c.low <= ob.range.high && c.close > ob.range.high
            : c.high >= ob.range.low && c.close < ob.range.low;

        if (mitigated) {
          mitigations.push({
            index: i,
            timestamp: c.timestamp,
            price: ob.type === 'bullish' ? ob.range.high : ob.range.low,
            type: ob.type,
            mitigatedOBIndex: ob.index
          });
          break; // first mitigation per OB only
        }
      }
    }

    this._mitigations = mitigations;
    return mitigations;
  }

  // -----------------------------------------------------------------------
  // Breaker Blocks
  // -----------------------------------------------------------------------

  /**
   * Detect Breaker Blocks — a previous S/R level that was broken and then
   * acts as the opposite (support becomes resistance or vice versa).
   */
  detectBreakerBlocks() {
    if (this._breakers !== undefined) return this._breakers;

    const breakers = [];
    const { candles } = this;
    if (candles.length < 10) { this._breakers = breakers; return breakers; }

    const { swingHighs, swingLows } = this.detectSwingPoints();
    const BREAKER_TOLERANCE = 0.005; // 0.5 % tolerance for touching level

    // ---- Swing highs: broken above → now acts as support (bullish breaker) ----
    for (const sh of swingHighs) {
      let brokeAbove = false;
      let breakIdx = -1;

      for (let i = sh.index + 2; i < candles.length; i++) {
        if (!brokeAbove && candles[i].close > sh.price) {
          brokeAbove = true;
          breakIdx = i;
          continue;
        }
        if (brokeAbove && i > breakIdx + 1) {
          const c = candles[i];
          const nearLevel = fracDiff(c.low, sh.price) <= BREAKER_TOLERANCE;
          if (nearLevel && c.close > sh.price) {
            breakers.push({
              index: i,
              timestamp: c.timestamp,
              price: sh.price,
              type: 'bullish',
              failedLevel: sh.price
            });
            break;
          }
        }
      }
    }

    // ---- Swing lows: broken below → now acts as resistance (bearish breaker) ----
    for (const sl of swingLows) {
      let brokeBelow = false;
      let breakIdx = -1;

      for (let i = sl.index + 2; i < candles.length; i++) {
        if (!brokeBelow && candles[i].close < sl.price) {
          brokeBelow = true;
          breakIdx = i;
          continue;
        }
        if (brokeBelow && i > breakIdx + 1) {
          const c = candles[i];
          const nearLevel = fracDiff(c.high, sl.price) <= BREAKER_TOLERANCE;
          if (nearLevel && c.close < sl.price) {
            breakers.push({
              index: i,
              timestamp: c.timestamp,
              price: sl.price,
              type: 'bearish',
              failedLevel: sl.price
            });
            break;
          }
        }
      }
    }

    this._breakers = breakers;
    return breakers;
  }

  // -----------------------------------------------------------------------
  // Imbalance (FVG + Volume Imbalance)
  // -----------------------------------------------------------------------

  /**
   * Detect imbalances: Fair Value Gaps (wick non-overlap) and volume
   * imbalances (volume spike with a directional move).
   */
  detectImbalance() {
    if (this._imbalances !== undefined) return this._imbalances;

    const imbalances = [];
    const { candles } = this;
    if (candles.length < 3) { this._imbalances = imbalances; return imbalances; }

    for (let i = 0; i < candles.length - 2; i++) {
      const c1 = candles[i];
      const c2 = candles[i + 1];
      const c3 = candles[i + 2];

      // ---- Bullish FVG: c3.low > c1.high (gap up) ----
      if (c3.low > c1.high) {
        const avgRange = ((c1.high - c1.low) + (c2.high - c2.low) + (c3.high - c3.low)) / 3;
        const gap = c3.low - c1.high;
        imbalances.push({
          index: i + 2,
          timestamp: c3.timestamp,
          type: 'FVG',
          upperPrice: c3.low,
          lowerPrice: c1.high,
          direction: 'bullish',
          fillPercent: avgRange > 0 ? clamp(gap / avgRange, 0, 1) : 0
        });
      }

      // ---- Bearish FVG: c3.high < c1.low (gap down) ----
      if (c3.high < c1.low) {
        const avgRange = ((c1.high - c1.low) + (c2.high - c2.low) + (c3.high - c3.low)) / 3;
        const gap = c1.low - c3.high;
        imbalances.push({
          index: i + 2,
          timestamp: c3.timestamp,
          type: 'FVG',
          upperPrice: c1.low,
          lowerPrice: c3.high,
          direction: 'bearish',
          fillPercent: avgRange > 0 ? clamp(gap / avgRange, 0, 1) : 0
        });
      }

      // ---- Volume imbalance ----
      if (c2.volume > c1.volume * 1.5 && c2.volume > c3.volume * 1.5) {
        const volGap = Math.abs(c2.close - c2.open) / (c2.open || 1);
        if (volGap > 0.002) { // ≥ 0.2 % move
          imbalances.push({
            index: i + 1,
            timestamp: c2.timestamp,
            type: 'volume_imbalance',
            upperPrice: Math.max(c2.open, c2.close),
            lowerPrice: Math.min(c2.open, c2.close),
            direction: c2.close > c2.open ? 'bullish' : 'bearish',
            fillPercent: 0
          });
        }
      }
    }

    this._imbalances = imbalances;
    return imbalances;
  }

  // -----------------------------------------------------------------------
  // Inducement (False Breakout)
  // -----------------------------------------------------------------------

  /**
   * Detect Inducement — a false move to trap traders before reversing.
   * Detected as a swing-point break that immediately reverses within a few bars.
   */
  detectInducement() {
    if (this._inducements !== undefined) return this._inducements;

    const inducements = [];
    const { candles } = this;
    if (candles.length < this.swingLookback * 2 + 3) {
      this._inducements = inducements;
      return inducements;
    }

    const { swingHighs, swingLows } = this.detectSwingPoints();
    const REVERSAL_BARS = 3; // must reverse within this many bars

    // ---- False break above swing high (bearish inducement) ----
    for (const sh of swingHighs) {
      let breakoutIdx = -1;
      const start = sh.index + 1;
      if (start + REVERSAL_BARS >= candles.length) continue;

      for (let i = start; i < candles.length - 2; i++) {
        const c = candles[i];

        if (breakoutIdx === -1 && c.high > sh.price) {
          breakoutIdx = i;
        }

        if (breakoutIdx !== -1 && i > breakoutIdx && i <= breakoutIdx + REVERSAL_BARS) {
          if (c.close < sh.price) {
            const next = candles[i + 1];
            if (next && next.close < c.close) {
              inducements.push({
                index: breakoutIdx,
                timestamp: candles[breakoutIdx].timestamp,
                price: candles[breakoutIdx].high,
                direction: 'bearish',
                targetLevel: sh.price
              });
              break;
            }
          }
        }
      }
    }

    // ---- False break below swing low (bullish inducement) ----
    for (const sl of swingLows) {
      let breakoutIdx = -1;
      const start = sl.index + 1;
      if (start + REVERSAL_BARS >= candles.length) continue;

      for (let i = start; i < candles.length - 2; i++) {
        const c = candles[i];

        if (breakoutIdx === -1 && c.low < sl.price) {
          breakoutIdx = i;
        }

        if (breakoutIdx !== -1 && i > breakoutIdx && i <= breakoutIdx + REVERSAL_BARS) {
          if (c.close > sl.price) {
            const next = candles[i + 1];
            if (next && next.close > c.close) {
              inducements.push({
                index: breakoutIdx,
                timestamp: candles[breakoutIdx].timestamp,
                price: candles[breakoutIdx].low,
                direction: 'bullish',
                targetLevel: sl.price
              });
              break;
            }
          }
        }
      }
    }

    this._inducements = inducements;
    return inducements;
  }

  // -----------------------------------------------------------------------
  // getSMCMap — combined view near currentPrice
  // -----------------------------------------------------------------------

  /**
   * Return a combined map of all SMC levels near `currentPrice` (within 5 %).
   *
   * @param {number} currentPrice
   * @returns {{ supports: Array, resistances: Array, activeOBs: Array,
   *             activeFVGs: Array, inducementZones: Array }}
   */
  getSMCMap(currentPrice) {
    const supports = [];
    const resistances = [];
    const activeOBs = [];
    const activeFVGs = [];
    const inducementZones = [];

    const withinRange = (price) => fracDiff(price, currentPrice) <= 0.05;

    const { swingHighs, swingLows } = this.detectSwingPoints();

    for (const sh of swingHighs) {
      if (withinRange(sh.price)) resistances.push({ type: 'swing_high', price: sh.price, index: sh.index });
    }
    for (const sl of swingLows) {
      if (withinRange(sl.price)) supports.push({ type: 'swing_low', price: sl.price, index: sl.index });
    }

    for (const ob of this.detectOrderBlocks()) {
      if (withinRange(ob.price)) activeOBs.push({ type: 'order_block', price: ob.price, range: ob.range, obType: ob.type, index: ob.index });
    }

    for (const im of this.detectImbalance()) {
      if (im.type === 'FVG' && withinRange(im.upperPrice)) {
        activeFVGs.push({ type: 'FVG', upperPrice: im.upperPrice, lowerPrice: im.lowerPrice, direction: im.direction, index: im.index });
      }
    }

    for (const ind of this.detectInducement()) {
      if (withinRange(ind.price)) inducementZones.push({ type: 'inducement', price: ind.price, direction: ind.direction, targetLevel: ind.targetLevel, index: ind.index });
    }

    return { supports, resistances, activeOBs, activeFVGs, inducementZones };
  }

  // -----------------------------------------------------------------------
  // Internal: Trend estimation
  // -----------------------------------------------------------------------

  /**
   * Estimate the prevailing trend from the last few swing points.
   * @returns {'bullish'|'bearish'|'ranging'}
   */
  _determineTrend() {
    const { swingHighs, swingLows } = this.detectSwingPoints();

    if (swingHighs.length < 2 || swingLows.length < 2) return 'ranging';

    const rh = swingHighs.slice(-3);
    const rl = swingLows.slice(-3);

    let higherHighs = 0;
    let lowerHighs = 0;
    let higherLows = 0;
    let lowerLows = 0;

    for (let i = 1; i < rh.length; i++) {
      if (rh[i].price > rh[i - 1].price) higherHighs++;
      else if (rh[i].price < rh[i - 1].price) lowerHighs++;
    }
    for (let i = 1; i < rl.length; i++) {
      if (rl[i].price > rl[i - 1].price) higherLows++;
      else if (rl[i].price < rl[i - 1].price) lowerLows++;
    }

    if (higherHighs >= 1 && higherLows >= 1) return 'bullish';
    if (lowerHighs >= 1 && lowerLows >= 1) return 'bearish';
    return 'ranging';
  }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/**
 * Determine trend direction from a candle series.
 *
 * @param {Array}  candles
 * @param {number} [lookback=20]  Number of candles to analyse
 * @returns {'bullish'|'bearish'|'ranging'}
 */
export function smcTrend(candles, lookback = 20) {
  if (!candles || !isArray(candles) || candles.length < lookback) return 'ranging';

  const slice = candles.slice(-lookback);
  const detector = new SMCDetector(slice, {
    swingLookback: Math.min(5, Math.max(2, Math.floor(lookback / 4)))
  });
  return detector._determineTrend();
}

/**
 * Find liquidity clusters — groups of candles with very similar
 * highs or lows (within 0.1 %).
 *
 * @param {Array}  candles
 * @param {string} [type='both']  'highs' | 'lows' | 'both'
 * @returns {Array<{ price: number, type: string, indices: number[],
 *                    count: number, avgPrice: number }>}
 */
export function liquidityLevels(candles, type = 'both') {
  if (!candles || !isArray(candles) || candles.length < 3) return [];

  const THRESHOLD = 0.001; // 0.1 %
  const checkHighs = type === 'highs' || type === 'both';
  const checkLows = type === 'lows' || type === 'both';

  const values = [];
  for (let i = 0; i < candles.length; i++) {
    if (checkHighs) values.push({ index: i, price: candles[i].high, kind: 'high' });
    if (checkLows) values.push({ index: i, price: candles[i].low, kind: 'low' });
  }

  const used = new Set();
  const clusters = [];

  for (let i = 0; i < values.length; i++) {
    if (used.has(i)) continue;
    const cluster = {
      price: values[i].price,
      type: values[i].kind,
      indices: [values[i].index],
      count: 1,
      avgPrice: values[i].price
    };

    for (let j = i + 1; j < values.length; j++) {
      if (used.has(j) || values[j].kind !== values[i].kind) continue;
      if (fracDiff(values[j].price, cluster.avgPrice) <= THRESHOLD) {
        cluster.indices.push(values[j].index);
        cluster.count++;
        cluster.avgPrice =
          (cluster.avgPrice * (cluster.count - 1) + values[j].price) / cluster.count;
        used.add(j);
      }
    }

    if (cluster.count >= 2) {
      cluster.price = cluster.avgPrice;
      clusters.push(cluster);
    }
    used.add(i);
  }

  return clusters.sort((a, b) => b.count - a.count);
}

/**
 * Detect the FIRST sign of a market structure shift (trend change).
 * Combines overall trend analysis with CHoCH detection.
 *
 * @param {Array} candles
 * @returns {{ detected: boolean, direction: 'bullish'|'bearish'|null,
 *             price: number, timestamp: (number|null) }}
 */
export function marketStructureShift(candles) {
  if (!candles || !isArray(candles) || candles.length < 20) {
    return { detected: false, direction: null, price: 0, timestamp: null };
  }

  const detector = new SMCDetector(candles);
  const trend = detector._determineTrend();
  const choch = detector.detectCHoCH();

  if (choch.length > 0) {
    const first = choch[0];
    return {
      detected: true,
      direction: first.direction,
      price: candles[first.index].close,
      timestamp: first.timestamp
    };
  }

  return { detected: false, direction: null, price: 0, timestamp: null };
}
