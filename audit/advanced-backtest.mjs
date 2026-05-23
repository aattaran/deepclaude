/**
 * Advanced Backtest Engine — quantitative finance rigor for strategy evaluation.
 *
 * Provides Monte Carlo simulation, walk-forward optimization, multi-strategy
 * comparison, deep drawdown analysis, and equity curve metric calculations.
 *
 * This is a companion to the basic backtest.mjs — it adds statistical rigor
 * without modifying existing code.
 *
 * Usage:
 *   import {
 *     monteCarloSim,
 *     walkForward,
 *     compareStrategies,
 *     analyzeDrawdowns,
 *     equityCurveMetrics,
 *   } from './audit/advanced-backtest.mjs';
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TRADING_DAYS_PER_YEAR = 252;
const TRADING_MINUTES_PER_YEAR = TRADING_DAYS_PER_YEAR * 6.5 * 60; // ~98,280

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Fisher-Yates shuffle of an array (in-place).
 * @param {Array} arr
 * @returns {Array}
 */
function shuffle(arr) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

/**
 * Bootstrap sample (sample WITH replacement) of an array.
 * @param {number[]} arr
 * @returns {number[]}
 */
function bootstrapSample(arr) {
    const n = arr.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = arr[Math.floor(Math.random() * n)];
    }
    return out;
}

/**
 * Compute the mean of an array of numbers.
 * @param {number[]} arr
 * @returns {number}
 */
function mean(arr) {
    if (arr.length === 0) return 0;
    return arr.reduce((s, v) => s + v, 0) / arr.length;
}

/**
 * Compute the standard deviation (sample) of an array of numbers.
 * @param {number[]} arr
 * @returns {number}
 */
function stdDev(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    const variance = arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1);
    return Math.sqrt(variance);
}

/**
 * Compute a percentile value from a sorted array using linear interpolation.
 * @param {number[]} sorted - must be sorted ascending
 * @param {number} p - percentile 0-1
 * @returns {number}
 */
function percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0];
    const index = p * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return sorted[lower];
    const frac = index - lower;
    return sorted[lower] * (1 - frac) + sorted[upper] * frac;
}

/**
 * Compute the Sharpe ratio (annualized, assuming daily returns).
 * @param {number[]} dailyReturns
 * @param {number} [riskFreeRate=0.05]
 * @returns {number}
 */
function calcSharpe(dailyReturns, riskFreeRate = 0.05) {
    if (dailyReturns.length < 2) return 0;
    const m = mean(dailyReturns);
    const sd = stdDev(dailyReturns);
    if (sd === 0) return m > 0 ? Infinity : 0;
    const dailyRfr = riskFreeRate / TRADING_DAYS_PER_YEAR;
    return ((m - dailyRfr) / sd) * Math.sqrt(TRADING_DAYS_PER_YEAR);
}

/**
 * Compute the Sortino ratio (annualized).
 * @param {number[]} dailyReturns
 * @param {number} [riskFreeRate=0.05]
 * @returns {number}
 */
function calcSortino(dailyReturns, riskFreeRate = 0.05) {
    if (dailyReturns.length < 2) return 0;
    const m = mean(dailyReturns);
    const downReturns = dailyReturns.filter(r => r < 0);
    if (downReturns.length === 0) return m > 0 ? Infinity : 0;
    const downVariance = downReturns.reduce((s, r) => s + r ** 2, 0) / downReturns.length;
    const downSd = Math.sqrt(downVariance);
    if (downSd === 0) return 0;
    const dailyRfr = riskFreeRate / TRADING_DAYS_PER_YEAR;
    return ((m - dailyRfr) / downSd) * Math.sqrt(TRADING_DAYS_PER_YEAR);
}

/**
 * Compute max drawdown from an equity curve.
 * @param {{equity: number}[]} curve
 * @returns {{maxDrawdown: number, maxDrawdownUsd: number}}
 */
function calcMaxDrawdown(curve) {
    if (curve.length < 2) return { maxDrawdown: 0, maxDrawdownUsd: 0 };
    let peak = curve[0].equity;
    let maxDD = 0;
    let maxDDUsd = 0;
    for (const pt of curve) {
        if (pt.equity > peak) peak = pt.equity;
        const dd = peak > 0 ? (peak - pt.equity) / peak : 0;
        if (dd > maxDD) {
            maxDD = dd;
            maxDDUsd = peak - pt.equity;
        }
    }
    return { maxDrawdown: maxDD, maxDrawdownUsd: maxDDUsd };
}

/**
 * Compute daily returns from an equity curve.
 * @param {{timestamp: number, equity: number}[]} curve
 * @returns {{date: string, return: number}[]}
 */
function calcDailyReturns(curve) {
    if (curve.length < 2) return [];
    const days = new Map();
    for (const pt of curve) {
        const date = new Date(pt.timestamp).toISOString().slice(0, 10);
        // Keep the last equity value for each day
        days.set(date, pt.equity);
    }
    const sortedDates = [...days.keys()].sort();
    const returns = [];
    for (let i = 1; i < sortedDates.length; i++) {
        const prevEq = days.get(sortedDates[i - 1]);
        const currEq = days.get(sortedDates[i]);
        if (prevEq !== 0) {
            returns.push({
                date: sortedDates[i],
                return: (currEq - prevEq) / prevEq,
            });
        }
    }
    return returns;
}

/**
 * Compute the t-statistic and approximate p-value for two independent samples
 * (Welch's t-test).
 * @param {number[]} sampleA
 * @param {number[]} sampleB
 * @returns {{tStatistic: number, df: number, pValue: number}}
 */
function welchTTest(sampleA, sampleB) {
    const n1 = sampleA.length;
    const n2 = sampleB.length;
    if (n1 < 2 || n2 < 2) return { tStatistic: 0, df: 0, pValue: 1 };

    const m1 = mean(sampleA);
    const m2 = mean(sampleB);
    const v1 = stdDev(sampleA) ** 2;
    const v2 = stdDev(sampleB) ** 2;
    const se = Math.sqrt(v1 / n1 + v2 / n2);

    if (se === 0) return { tStatistic: 0, df: 0, pValue: 1 };

    const t = (m1 - m2) / se;

    // Welch-Satterthwaite degrees of freedom
    const num = (v1 / n1 + v2 / n2) ** 2;
    const denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1);
    const df = denom > 0 ? num / denom : 0;

    // Approximate two-tailed p-value using the t-distribution
    const pValue = 2 * (1 - tCdf(Math.abs(t), df));

    return { tStatistic: t, df, pValue };
}

/**
 * Approximation of the t-distribution CDF using a regularized incomplete beta
 * function (Abramowitz and Stegun 26.7.1 / AS 63).
 *
 * P(T <= t) for t >= 0.
 *
 * @param {number} t - t-statistic (>= 0)
 * @param {number} df - degrees of freedom
 * @returns {number}
 */
function tCdf(t, df) {
    if (t <= 0) return 0.5;
    if (df <= 0) return 0.5;
    if (!isFinite(t)) return 1;

    // Use the relationship to the regularized incomplete beta function:
    // P(T <= t) = 1 - 0.5 * I(df/(df+t^2), df/2, 0.5)
    const x = df / (df + t * t);
    const a = df / 2;
    const b = 0.5;
    const ib = regIncompleteBeta(x, a, b);
    return 1 - 0.5 * ib;
}

/**
 * Regularized incomplete beta function I_x(a, b).
 * Uses the continued fraction representation (Lentz's method).
 *
 * @param {number} x - value 0..1
 * @param {number} a - shape parameter > 0
 * @param {number} b - shape parameter > 0
 * @returns {number}
 */
function regIncompleteBeta(x, a, b) {
    if (x < 0 || x > 1) return 0;
    if (x === 0 || x === 1) return x === 0 ? 0 : 1;

    // Use symmetry if it helps convergence
    if (x > (a + 1) / (a + b + 2)) {
        return 1 - regIncompleteBeta(1 - x, b, a);
    }

    // Lentz's continued fraction method for I_x(a,b)
    // Uses the modified Lentz algorithm from Numerical Recipes.
    const lbeta = logBeta(a, b);
    const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta - Math.log(a));

    const qab = a + b;
    const qap = a + 1;
    const qam = a - 1;
    let c = 1;
    let d = 1 - qab * x / qap;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    let h = d;

    for (let m = 1; m <= 200; m++) {
        const m2 = 2 * m;

        // Even step: a_{2m}
        const aaEven = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aaEven * d;
        if (Math.abs(d) < 1e-30) d = 1e-30;
        c = 1 + aaEven / c;
        if (Math.abs(c) < 1e-30) c = 1e-30;
        d = 1 / d;
        h *= c * d;

        // Odd step: a_{2m+1}
        const aaOdd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aaOdd * d;
        if (Math.abs(d) < 1e-30) d = 1e-30;
        c = 1 + aaOdd / c;
        if (Math.abs(c) < 1e-30) c = 1e-30;
        d = 1 / d;
        const delta = c * d;
        h *= delta;

        if (Math.abs(delta - 1) < 1e-10) break;
    }

    return front * h;
}

/**
 * Log of the beta function B(a, b).
 * @param {number} a
 * @param {number} b
 * @returns {number}
 */
function logBeta(a, b) {
    return logGamma(a) + logGamma(b) - logGamma(a + b);
}

/**
 * Log-gamma function using the Lanczos approximation.
 * @param {number} z
 * @returns {number}
 */
function logGamma(z) {
    if (z < 0.5) {
        // Reflection formula
        return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    z -= 1;
    const g = 7;
    const c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = c[0];
    for (let i = 1; i < g + 2; i++) {
        x += c[i] / (z + i);
    }
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Run a simple backtest for a strategy against candles, returning trades.
 * This is a lightweight internal runner (does not depend on backtest.mjs).
 *
 * @param {import('./backtest.mjs').Candle[]} candles
 * @param {Function} strategyFn - (candle, history) => Signal | null
 * @param {Object} [opts]
 * @param {string} [opts.symbol='BT']
 * @param {number} [opts.startingEquity=100000]
 * @param {number} [opts.contractSize=2]
 * @param {number} [opts.slippagePts=0.25]
 * @param {number} [opts.commissionPerTrade=1.24]
 * @returns {{trades: Object[], equityCurve: {timestamp:number, equity:number}[]}}
 */
function runMiniBacktest(candles, strategyFn, opts = {}) {
    const symbol = opts.symbol || 'BT';
    const startingEquity = opts.startingEquity || 100000;
    const contractSize = opts.contractSize || 2;
    const slippage = opts.slippagePts || 0.25;
    const commission = opts.commissionPerTrade || 1.24;

    const trades = [];
    let equity = startingEquity;
    let inPosition = null;

    for (let i = 20; i < candles.length; i++) {
        const currentCandle = candles[i];
        const history = candles.slice(0, i);

        // Exit logic for open position
        if (inPosition) {
            const exit = getSimpleExit(inPosition, currentCandle, candles.slice(i), history);
            if (exit !== null) {
                const pnl = calcPnl(inPosition, exit.price, contractSize, commission);
                trades.push({
                    symbol,
                    direction: inPosition.direction,
                    entry_price: inPosition.entryPrice,
                    exit_price: exit.price,
                    pnl_usd: pnl,
                    hold_sec: (currentCandle.timestamp - inPosition.entryTime) / 1000,
                    entry_time: inPosition.entryTime,
                    exit_time: currentCandle.timestamp,
                    exit_reason: exit.reason,
                });
                equity += pnl;
                inPosition = null;
            }
        }

        // Entry logic
        if (!inPosition) {
            const signal = strategyFn(currentCandle, history);
            if (signal) {
                const entryPrice = signal.direction === 'long'
                    ? currentCandle.close + slippage
                    : currentCandle.close - slippage;
                inPosition = {
                    direction: signal.direction,
                    entryPrice,
                    entryTime: signal.timestamp || currentCandle.timestamp,
                    signal,
                };
            }
        }
    }

    // Close any open position at last candle
    if (inPosition) {
        const lastCandle = candles[candles.length - 1];
        const pnl = calcPnl(inPosition, lastCandle.close, contractSize, commission);
        trades.push({
            symbol,
            direction: inPosition.direction,
            entry_price: inPosition.entryPrice,
            exit_price: lastCandle.close,
            pnl_usd: pnl,
            hold_sec: (lastCandle.timestamp - inPosition.entryTime) / 1000,
            entry_time: inPosition.entryTime,
            exit_time: lastCandle.timestamp,
            exit_reason: 'EOD',
        });
        equity += pnl;
    }

    // Build equity curve
    const equityCurve = buildEquityCurveFromTrades(trades, startingEquity);
    return { trades, equityCurve };
}

/**
 * Simple exit logic (stop-loss, take-profit, SMA crossover).
 * Mirrors the logic in backtest.mjs to keep this module self-contained.
 */
function getSimpleExit(position, candle, futureCandles, history,
    stopLossPct = 0.02, takeProfitPct = 0.04) {
    const entry = position.entryPrice;

    if (position.direction === 'long') {
        const stopPrice = entry * (1 - stopLossPct);
        if (candle.low <= stopPrice) return { price: stopPrice, reason: 'stop_loss' };
        const targetPrice = entry * (1 + takeProfitPct);
        if (candle.high >= targetPrice) return { price: targetPrice, reason: 'take_profit' };
    } else {
        const stopPrice = entry * (1 + stopLossPct);
        if (candle.high >= stopPrice) return { price: stopPrice, reason: 'stop_loss' };
        const targetPrice = entry * (1 - takeProfitPct);
        if (candle.low <= targetPrice) return { price: targetPrice, reason: 'take_profit' };
    }

    // SMA crossover exit
    if (history.length >= 30) {
        const fastPeriod = 10;
        const slowPeriod = 30;
        const slowSlice = history.slice(-slowPeriod);
        const slowSma = avgClose(slowSlice);
        const fastSma = avgClose(slowSlice.slice(-fastPeriod));
        const prevSlowSlice = history.slice(-slowPeriod - 1, -1);
        const prevSlowSma = avgClose(prevSlowSlice);
        const prevFastSma = avgClose(prevSlowSlice.slice(-fastPeriod));

        if (position.direction === 'long' && prevFastSma >= prevSlowSma && fastSma < slowSma) {
            return { price: candle.close, reason: 'sma_crossover' };
        }
        if (position.direction === 'short' && prevFastSma <= prevSlowSma && fastSma > slowSma) {
            return { price: candle.close, reason: 'sma_crossover' };
        }
    }

    if (futureCandles.length < 2) {
        return { price: candle.close, reason: 'end_of_data' };
    }

    return null;
}

function calcPnl(position, exitPrice, contractSize, commission) {
    const rawPnl = position.direction === 'long'
        ? (exitPrice - position.entryPrice) * contractSize
        : (position.entryPrice - exitPrice) * contractSize;
    return rawPnl - (commission * 2);
}

function avgClose(candles) {
    return candles.reduce((s, c) => s + c.close, 0) / candles.length;
}

function buildEquityCurveFromTrades(trades, startingEquity = 100000) {
    const sorted = [...trades].sort((a, b) => a.exit_time - b.exit_time);
    let equity = startingEquity;
    const curve = [];
    if (sorted.length > 0) {
        curve.push({ timestamp: sorted[0].entry_time, equity: startingEquity });
    }
    for (const t of sorted) {
        equity += t.pnl_usd;
        curve.push({ timestamp: t.exit_time, equity });
    }
    return curve;
}

// ============================================================================
// 1. MONTE CARLO SIMULATION
// ============================================================================

/**
 * Monte Carlo simulation for a strategy's trade PnL sequence.
 *
 * Uses bootstrap resampling (with replacement) to generate N simulated equity
 * curves from the observed trade PnLs.  This preserves the return distribution
 * while breaking temporal dependencies, giving a distribution of possible
 * outcomes.
 *
 * @param {number[]} tradePnLs - array of realized P&L values per trade
 * @param {Object} [opts]
 * @param {number} [opts.simulations=10000] - number of Monte Carlo trials
 * @param {number} [opts.confidence=0.95] - confidence level for bands (0-1)
 * @param {number} [opts.startingEquity=100000] - starting account equity
 * @returns {Object} simulation results
 * @property {number} simulations - number of trials run
 * @property {number} confidence - confidence level used
 * @property {{step: number, equity: number}[]} medianEquityCurve - pointwise median
 * @property {{step: number, lower: number, upper: number}[]} confidenceBands
 * @property {number} probabilityOfProfit - fraction of simulations ending positive
 * @property {number} expectedFinalEquity - mean final equity across simulations
 * @property {number} expectedReturn - mean total return across simulations
 * @property {number} cvar - Conditional Value at Risk (expected shortfall at tail)
 * @property {{mean: number, stdDev: number, min: number, p25: number, p50: number, p75: number, max: number}} finalEquityDistribution
 * @property {{mean: number, stdDev: number, min: number, p25: number, p50: number, p75: number, max: number}} maxDrawdownDistribution
 */
export function monteCarloSim(tradePnLs, opts = {}) {
    const simulations = opts.simulations || 10000;
    const confidence = opts.confidence || 0.95;
    const startingEquity = opts.startingEquity || 100000;

    if (!Array.isArray(tradePnLs) || tradePnLs.length === 0) {
        return {
            simulations: 0,
            confidence,
            medianEquityCurve: [],
            confidenceBands: [],
            probabilityOfProfit: 0,
            expectedFinalEquity: startingEquity,
            expectedReturn: 0,
            cvar: 0,
            finalEquityDistribution: { mean: startingEquity, stdDev: 0, min: startingEquity, p25: startingEquity, p50: startingEquity, p75: startingEquity, max: startingEquity },
            maxDrawdownDistribution: { mean: 0, stdDev: 0, min: 0, p25: 0, p50: 0, p75: 0, max: 0 },
        };
    }

    const n = tradePnLs.length;
    const allFinalEquities = new Array(simulations);
    const allMaxDDs = new Array(simulations);
    const allEquityCurves = new Array(simulations);

    for (let sim = 0; sim < simulations; sim++) {
        const sampled = bootstrapSample(tradePnLs);
        let eq = startingEquity;
        let peak = eq;
        let maxDD = 0;
        const curve = new Array(n);

        for (let step = 0; step < n; step++) {
            eq += sampled[step];
            if (eq > peak) peak = eq;
            const dd = peak > 0 ? (peak - eq) / peak : 0;
            if (dd > maxDD) maxDD = dd;
            curve[step] = eq;
        }

        allFinalEquities[sim] = eq;
        allMaxDDs[sim] = maxDD;
        allEquityCurves[sim] = curve;
    }

    // -- Aggregate results --

    // Probability of profit
    const profitCount = allFinalEquities.filter(eq => eq > startingEquity).length;
    const probabilityOfProfit = profitCount / simulations;

    // Expected final equity
    const expectedFinalEquity = mean(allFinalEquities);

    // Expected return
    const expectedReturn = (expectedFinalEquity - startingEquity) / startingEquity;

    // CVaR (expected shortfall) — mean of worst (1-confidence) tail
    const sortedFinals = [...allFinalEquities].sort((a, b) => a - b);
    const tailIndex = Math.floor(simulations * (1 - confidence));
    const tailEquities = sortedFinals.slice(0, Math.max(tailIndex, 1));
    const cvar = startingEquity - mean(tailEquities); // loss relative to starting equity

    // Final equity distribution summary
    const finalEquityDistribution = {
        mean: +expectedFinalEquity.toFixed(2),
        stdDev: +stdDev(allFinalEquities).toFixed(2),
        min: +sortedFinals[0].toFixed(2),
        p25: +percentile(sortedFinals, 0.25).toFixed(2),
        p50: +percentile(sortedFinals, 0.50).toFixed(2),
        p75: +percentile(sortedFinals, 0.75).toFixed(2),
        max: +sortedFinals[sortedFinals.length - 1].toFixed(2),
    };

    // Max drawdown distribution
    const sortedDDs = [...allMaxDDs].sort((a, b) => a - b);
    const maxDrawdownDistribution = {
        mean: +mean(allMaxDDs).toFixed(4),
        stdDev: +stdDev(allMaxDDs).toFixed(4),
        min: +sortedDDs[0].toFixed(4),
        p25: +percentile(sortedDDs, 0.25).toFixed(4),
        p50: +percentile(sortedDDs, 0.50).toFixed(4),
        p75: +percentile(sortedDDs, 0.75).toFixed(4),
        max: +sortedDDs[sortedDDs.length - 1].toFixed(4),
    };

    // Median equity curve (pointwise median across simulations)
    const medianEquityCurve = [];
    const confidenceBands = [];
    const alpha = 1 - confidence;
    const lowerP = alpha / 2;
    const upperP = 1 - alpha / 2;

    for (let step = 0; step < n; step++) {
        const stepEquities = allEquityCurves.map(curve => curve[step]);
        const sortedStep = stepEquities.sort((a, b) => a - b);
        medianEquityCurve.push({
            step,
            equity: +percentile(sortedStep, 0.50).toFixed(2),
        });
        confidenceBands.push({
            step,
            lower: +percentile(sortedStep, lowerP).toFixed(2),
            upper: +percentile(sortedStep, upperP).toFixed(2),
        });
    }

    return {
        simulations,
        confidence,
        medianEquityCurve,
        confidenceBands,
        probabilityOfProfit: +probabilityOfProfit.toFixed(4),
        expectedFinalEquity: +expectedFinalEquity.toFixed(2),
        expectedReturn: +expectedReturn.toFixed(4),
        cvar: +cvar.toFixed(2),
        finalEquityDistribution,
        maxDrawdownDistribution,
    };
}

// ============================================================================
// 2. WALK-FORWARD OPTIMIZATION
// ============================================================================

/**
 * Walk-forward optimization to test strategy robustness.
 *
 * Splits historical data into rolling in-sample (IS) / out-of-sample (OOS)
 * windows.  For each window, the strategy is optimized over the IS portion by
 * grid-searching `paramCandidates`, then the best-performing parameter set is
 * tested on unseen OOS data.  Reports per-window metrics, aggregate OOS
 * performance, and a parameter stability score.
 *
 * @param {import('./backtest.mjs').Candle[]} candles - historical OHLCV data
 * @param {Function} strategyFactory - (params) => (candle, history) => Signal|null
 * @param {Object[]} paramCandidates - array of parameter objects to grid-search
 * @param {Object} [opts]
 * @param {number} [opts.inSamplePct=0.6] - fraction of each window for training
 * @param {number} [opts.outOfSamplePct=0.4] - fraction of each window for testing
 * @param {number} [opts.stepCount=5] - number of rolling windows
 * @param {string} [opts.symbol='WF']
 * @param {Object} [opts.backtestOpts] - extra opts passed to mini backtest
 * @param {Function} [opts.objectiveFn] - (backtestResult) => number to maximize (default: Sharpe)
 * @returns {Object} walk-forward results
 * @property {Object[]} windows - per-window results
 * @property {Object} aggregate - aggregated OOS metrics
 * @property {number} paramStabilityScore - 0-1, how often the same params were selected
 */
export function walkForward(candles, strategyFactory, paramCandidates, opts = {}) {
    const inSamplePct = opts.inSamplePct || 0.6;
    const outOfSamplePct = opts.outOfSamplePct || 0.4;
    const stepCount = opts.stepCount || 5;
    const symbol = opts.symbol || 'WF';
    const backtestOpts = Object.assign({ symbol }, opts.backtestOpts);
    const objectiveFn = opts.objectiveFn || (res => res.sharpe);

    if (!Array.isArray(candles) || candles.length < 100) {
        return {
            windows: [],
            aggregate: {
                totalOosTrades: 0,
                totalOosPnL: 0,
                avgOosSharpe: 0,
                avgOosSortino: 0,
                avgOosMaxDD: 0,
                avgOosWinRate: 0,
                avgOosProfitFactor: 0,
            },
            paramStabilityScore: 0,
        };
    }

    if (!Array.isArray(paramCandidates) || paramCandidates.length === 0) {
        return {
            windows: [],
            aggregate: {
                totalOosTrades: 0,
                totalOosPnL: 0,
                avgOosSharpe: 0,
                avgOosSortino: 0,
                avgOosMaxDD: 0,
                avgOosWinRate: 0,
                avgOosProfitFactor: 0,
            },
            paramStabilityScore: 0,
        };
    }

    const totalLen = candles.length;
    const windowIsLen = Math.floor(totalLen * inSamplePct);
    const windowOosLen = Math.floor(totalLen * outOfSamplePct);
    const windowLen = windowIsLen + windowOosLen;
    const step = Math.max(1, Math.floor((totalLen - windowLen) / (stepCount - 1)));

    const windows = [];
    const bestParamIndices = [];

    for (let w = 0; w < stepCount; w++) {
        const start = w * step;
        const isStart = start;
        const isEnd = start + windowIsLen;
        const oosStart = isEnd;
        const oosEnd = Math.min(start + windowLen, totalLen);

        if (oosEnd > totalLen || oosEnd - oosStart < 20) break;

        const inSampleCandles = candles.slice(isStart, isEnd);
        const outOfSampleCandles = candles.slice(oosStart, oosEnd);

        // Grid-search parameters on in-sample
        let bestScore = -Infinity;
        let bestParams = paramCandidates[0];
        let bestIsResult = null;

        for (const params of paramCandidates) {
            const strategyFn = strategyFactory(params);
            const result = runMiniBacktest(inSampleCandles, strategyFn, backtestOpts);
            const metrics = computeStrategyMetrics(result.trades, result.equityCurve);
            const score = objectiveFn(metrics);
            if (score > bestScore) {
                bestScore = score;
                bestParams = params;
                bestIsResult = metrics;
            }
        }

        // Test best params on out-of-sample
        const oosStrategyFn = strategyFactory(bestParams);
        const oosResult = runMiniBacktest(outOfSampleCandles, oosStrategyFn, backtestOpts);
        const oosMetrics = computeStrategyMetrics(oosResult.trades, oosResult.equityCurve);

        windows.push({
            windowIndex: w,
            candleRange: {
                inSample: { start: isStart, end: isEnd },
                outOfSample: { start: oosStart, end: oosEnd },
            },
            bestParams,
            inSample: {
                trades: bestIsResult.totalTrades,
                sharpe: +bestIsResult.sharpe.toFixed(4),
                sortino: +bestIsResult.sortino.toFixed(4),
                maxDrawdown: +bestIsResult.maxDrawdown.toFixed(4),
                totalPnL: +bestIsResult.totalPnL.toFixed(2),
                winRate: +bestIsResult.winRate.toFixed(4),
                profitFactor: bestIsResult.profitFactor === Infinity ? 999 : +bestIsResult.profitFactor.toFixed(4),
            },
            outOfSample: {
                trades: oosMetrics.totalTrades,
                sharpe: +oosMetrics.sharpe.toFixed(4),
                sortino: +oosMetrics.sortino.toFixed(4),
                maxDrawdown: +oosMetrics.maxDrawdown.toFixed(4),
                totalPnL: +oosMetrics.totalPnL.toFixed(2),
                winRate: +oosMetrics.winRate.toFixed(4),
                profitFactor: oosMetrics.profitFactor === Infinity ? 999 : +oosMetrics.profitFactor.toFixed(4),
            },
        });

        bestParamIndices.push(
            paramCandidates.findIndex(
                p => JSON.stringify(p) === JSON.stringify(bestParams),
            ),
        );
    }

    // Aggregate out-of-sample metrics
    const oosSharpeValues = windows.map(w => w.outOfSample.sharpe).filter(v => isFinite(v));
    const oosSortinoValues = windows.map(w => w.outOfSample.sortino).filter(v => isFinite(v));
    const oosDDValues = windows.map(w => w.outOfSample.maxDrawdown);
    const oosWinRateValues = windows.map(w => w.outOfSample.winRate);
    const oosProfitFactorValues = windows.map(w => w.outOfSample.profitFactor === 999 ? Infinity : w.outOfSample.profitFactor);

    const aggregate = {
        totalOosTrades: windows.reduce((s, w) => s + w.outOfSample.trades, 0),
        totalOosPnL: +windows.reduce((s, w) => s + w.outOfSample.totalPnL, 0).toFixed(2),
        avgOosSharpe: oosSharpeValues.length > 0 ? +mean(oosSharpeValues).toFixed(4) : 0,
        avgOosSortino: oosSortinoValues.length > 0 ? +mean(oosSortinoValues).toFixed(4) : 0,
        avgOosMaxDD: +mean(oosDDValues).toFixed(4),
        avgOosWinRate: +mean(oosWinRateValues).toFixed(4),
        avgOosProfitFactor: oosProfitFactorValues.length > 0
            ? +mean(oosProfitFactorValues.filter(v => isFinite(v))).toFixed(4)
            : 0,
    };

    // Parameter stability score: fraction of adjacent windows that chose the same params
    let stableCount = 0;
    for (let i = 1; i < bestParamIndices.length; i++) {
        if (bestParamIndices[i] === bestParamIndices[i - 1]) stableCount++;
    }
    const paramStabilityScore = bestParamIndices.length > 1
        ? stableCount / (bestParamIndices.length - 1)
        : 1;

    return { windows, aggregate, paramStabilityScore: +paramStabilityScore.toFixed(4) };
}

// ============================================================================
// 3. MULTI-STRATEGY COMPARISON
// ============================================================================

/**
 * Compare multiple strategies on the same historical data.
 *
 * Runs each strategy function against the same candle set, produces a ranked
 * comparison table with key performance metrics, and performs a Welch's t-test
 * between the top 2 strategies by Sharpe ratio to assess statistical
 * significance of the performance difference.
 *
 * @param {import('./backtest.mjs').Candle[]} candles - historical OHLCV data
 * @param {Object<string, Function>} strategies - map of {name: strategyFn}
 * @param {Object} [opts]
 * @param {string} [opts.symbol='CMP']
 * @param {number} [opts.startingEquity=100000]
 * @returns {Object} comparison results
 * @property {Object[]} ranking - sorted by Sharpe descending
 * @property {Object|null} significanceTest - Welch's t-test between top 2
 */
export function compareStrategies(candles, strategies, opts = {}) {
    const symbol = opts.symbol || 'CMP';
    const startingEquity = opts.startingEquity || 100000;
    const backtestOpts = { symbol, startingEquity, ...opts };

    if (!Array.isArray(candles) || candles.length < 20) {
        return { ranking: [], significanceTest: null };
    }

    const names = Object.keys(strategies);
    if (names.length === 0) {
        return { ranking: [], significanceTest: null };
    }

    const results = [];

    for (const name of names) {
        const strategyFn = strategies[name];
        const { trades, equityCurve } = runMiniBacktest(candles, strategyFn, backtestOpts);
        const metrics = computeStrategyMetrics(trades, equityCurve);

        // Compute additional per-trade statistics
        const avgTradeDuration = trades.length > 0
            ? mean(trades.map(t => t.hold_sec))
            : 0;
        const avgPnLPerTrade = trades.length > 0
            ? mean(trades.map(t => t.pnl_usd))
            : 0;

        // Daily returns for Sharpe/Sortino
        const dailyReturns = calcDailyReturns(equityCurve).map(d => d.return);
        const sharpe = calcSharpe(dailyReturns);
        const sortino = calcSortino(dailyReturns);

        results.push({
            name,
            totalTrades: trades.length,
            totalPnL: +metrics.totalPnL.toFixed(2),
            sharpe: +sharpe.toFixed(4),
            sortino: +sortino.toFixed(4),
            maxDrawdown: +metrics.maxDrawdown.toFixed(4),
            winRate: +metrics.winRate.toFixed(4),
            profitFactor: metrics.profitFactor === Infinity
                ? 999
                : +metrics.profitFactor.toFixed(4),
            avgTradeDurationSec: +avgTradeDuration.toFixed(1),
            avgPnLPerTrade: +avgPnLPerTrade.toFixed(2),
            tradePnLs: trades.map(t => t.pnl_usd),
        });
    }

    // Rank by Sharpe descending
    results.sort((a, b) => {
        // Handle Infinity
        if (a.sharpe === Infinity && b.sharpe === Infinity) return 0;
        if (a.sharpe === Infinity) return -1;
        if (b.sharpe === Infinity) return 1;
        return b.sharpe - a.sharpe;
    });

    // Build ranking table (without raw trade PnLs)
    const ranking = results.map(r => ({
        name: r.name,
        rank: results.indexOf(r) + 1,
        totalTrades: r.totalTrades,
        totalPnL: r.totalPnL,
        sharpe: r.sharpe,
        sortino: r.sortino,
        maxDrawdown: r.maxDrawdown,
        winRate: r.winRate,
        profitFactor: r.profitFactor,
        avgTradeDurationSec: r.avgTradeDurationSec,
        avgPnLPerTrade: r.avgPnLPerTrade,
    }));

    // Welch's t-test between top 2
    let significanceTest = null;
    if (results.length >= 2) {
        const top = results[0];
        const second = results[1];

        if (top.tradePnLs.length >= 2 && second.tradePnLs.length >= 2) {
            const test = welchTTest(top.tradePnLs, second.tradePnLs);
            significanceTest = {
                winner: top.name,
                runnerUp: second.name,
                tStatistic: +test.tStatistic.toFixed(4),
                degreesOfFreedom: +test.df.toFixed(2),
                pValue: +test.pValue.toFixed(6),
                significantAt95: test.pValue < 0.05,
            };
        }
    }

    return { ranking, significanceTest };
}

// ============================================================================
// 4. DRAWDOWN ANALYSIS
// ============================================================================

/**
 * Deep drawdown analysis of an equity curve.
 *
 * Identifies every distinct drawdown period from peak to trough to recovery,
 * calculates depth and duration for each, and returns the most severe
 * drawdowns plus summary statistics.
 *
 * @param {{timestamp: number, equity: number}[]} equityCurve
 * @returns {Object} drawdown analysis
 * @property {Object[]} topDrawdowns - 5 worst drawdowns by depth
 * @property {number} averageDrawdown - mean depth of all drawdowns
 * @property {number} averageDrawdownDuration - mean duration in ms
 * @property {number} averageRecoveryTime - mean recovery duration in ms
 * @property {number} maxDrawdownDepth - deepest drawdown overall (0-1)
 * @property {number} maxDrawdownDuration - longest drawdown duration in ms
 * @property {number} drawdownCount - total number of distinct drawdowns
 * @property {{timestamp: number, equity: number, peak: number, underwater: number}[]} underwaterChart - equity below ATH
 */
export function analyzeDrawdowns(equityCurve) {
    if (!Array.isArray(equityCurve) || equityCurve.length < 2) {
        return {
            topDrawdowns: [],
            averageDrawdown: 0,
            averageDrawdownDuration: 0,
            averageRecoveryTime: 0,
            maxDrawdownDepth: 0,
            maxDrawdownDuration: 0,
            drawdownCount: 0,
            underwaterChart: [],
        };
    }

    const sorted = [...equityCurve].sort((a, b) => a.timestamp - b.timestamp);

    let peak = sorted[0].equity;
    let peakTimestamp = sorted[0].timestamp;
    let troughEq = sorted[0].equity;
    let troughTimestamp = sorted[0].timestamp;

    const drawdowns = [];
    const underwaterChart = [];
    let inDrawdown = false;
    let currentDrawdown = null;

    for (const pt of sorted) {
        if (pt.equity > peak) {
            // If we were in a drawdown, it's now recovered
            if (inDrawdown && currentDrawdown) {
                currentDrawdown.recovery = {
                    timestamp: pt.timestamp,
                    equity: pt.equity,
                };
                currentDrawdown.recoveryTime = pt.timestamp - currentDrawdown.trough.timestamp;
                currentDrawdown.duration = pt.timestamp - currentDrawdown.start.timestamp;
                drawdowns.push(currentDrawdown);
                currentDrawdown = null;
                inDrawdown = false;
            }
            peak = pt.equity;
            peakTimestamp = pt.timestamp;
            troughEq = pt.equity;
        }

        const ddFromPeak = peak > 0 ? (peak - pt.equity) / peak : 0;

        if (ddFromPeak > 0) {
            if (!inDrawdown) {
                inDrawdown = true;
                currentDrawdown = {
                    start: { timestamp: peakTimestamp, equity: peak },
                    trough: { timestamp: pt.timestamp, equity: pt.equity },
                    depth: 0,
                    duration: 0,
                    recovery: null,
                    recoveryTime: 0,
                };
            }
            // Update trough if deeper
            if (pt.equity < troughEq) {
                troughEq = pt.equity;
                if (currentDrawdown) {
                    currentDrawdown.trough = { timestamp: pt.timestamp, equity: pt.equity };
                }
            }
            if (currentDrawdown) {
                currentDrawdown.depth = (currentDrawdown.start.equity - pt.equity) / currentDrawdown.start.equity;
            }
        }

        // Build underwater chart data
        underwaterChart.push({
            timestamp: pt.timestamp,
            equity: pt.equity,
            peak,
            underwater: +ddFromPeak.toFixed(6),
        });
    }

    // If still in a drawdown at end, record it (no recovery)
    if (inDrawdown && currentDrawdown) {
        currentDrawdown.duration = sorted[sorted.length - 1].timestamp - currentDrawdown.start.timestamp;
        drawdowns.push(currentDrawdown);
    }

    if (drawdowns.length === 0) {
        return {
            topDrawdowns: [],
            averageDrawdown: 0,
            averageDrawdownDuration: 0,
            averageRecoveryTime: 0,
            maxDrawdownDepth: 0,
            maxDrawdownDuration: 0,
            drawdownCount: 0,
            underwaterChart,
        };
    }

    // Sort by depth descending, take top 5
    drawdowns.sort((a, b) => b.depth - a.depth);
    const topDrawdowns = drawdowns.slice(0, 5).map(d => ({
        startTimestamp: d.start.timestamp,
        startEquity: +d.start.equity.toFixed(2),
        troughTimestamp: d.trough.timestamp,
        troughEquity: +d.trough.equity.toFixed(2),
        recoveryTimestamp: d.recovery ? d.recovery.timestamp : null,
        depthPct: +(d.depth * 100).toFixed(2),
        durationMs: d.duration,
        durationDays: +(d.duration / 86400000).toFixed(2),
        recoveryTimeMs: d.recoveryTime || null,
        recoveryTimeDays: d.recoveryTime ? +(d.recoveryTime / 86400000).toFixed(2) : null,
    }));

    const depths = drawdowns.map(d => d.depth);
    const durations = drawdowns.map(d => d.duration);
    const recoveryTimes = drawdowns
        .filter(d => d.recoveryTime > 0)
        .map(d => d.recoveryTime);

    return {
        topDrawdowns,
        averageDrawdown: +mean(depths).toFixed(4),
        averageDrawdownDuration: +mean(durations).toFixed(0),
        averageRecoveryTime: recoveryTimes.length > 0
            ? +mean(recoveryTimes).toFixed(0)
            : 0,
        maxDrawdownDepth: +Math.max(...depths).toFixed(4),
        maxDrawdownDuration: +Math.max(...durations).toFixed(0),
        drawdownCount: drawdowns.length,
        underwaterChart,
    };
}

// ============================================================================
// 5. EQUITY CURVE METRICS
// ============================================================================

/**
 * Calculate a comprehensive set of equity curve metrics.
 *
 * Includes CAGR, annualized volatility, Calmar ratio, stability (R-squared of
 * linear fit), ulcer index, and pain index — all essential quantitative
 * measures for evaluating strategy quality beyond simple return.
 *
 * @param {{timestamp: number, equity: number}[]} equityCurve
 * @param {Object} [opts]
 * @param {number} [opts.startingEquity] - if not provided, inferred from first point
 * @returns {Object} equity metrics
 * @property {number} cagr - Compound Annual Growth Rate
 * @property {number} volatility - annualized standard deviation of daily returns
 * @property {number} calmarRatio - CAGR / max drawdown
 * @property {number} stability - R² of linear fit to equity curve (0-1)
 * @property {number} ulcerIndex - root-mean-square drawdown from peak
 * @property {number} painIndex - mean drawdown from peak
 * @property {number} totalReturn - total percentage return
 * @property {number} totalReturnUsd - total dollar return
 * @property {number} yearsElapsed - time span in years
 * @property {number} avgDailyReturn - mean daily return
 * @property {number} dailyReturnStd - standard deviation of daily returns
 * @property {number} bestDay - best single-day return
 * @property {number} worstDay - worst single-day return
 * @property {number} positiveDayRatio - fraction of days with positive return
 */
export function equityCurveMetrics(equityCurve, opts = {}) {
    if (!Array.isArray(equityCurve) || equityCurve.length < 2) {
        return {
            cagr: 0,
            volatility: 0,
            calmarRatio: 0,
            stability: 0,
            ulcerIndex: 0,
            painIndex: 0,
            totalReturn: 0,
            totalReturnUsd: 0,
            yearsElapsed: 0,
            avgDailyReturn: 0,
            dailyReturnStd: 0,
            bestDay: 0,
            worstDay: 0,
            positiveDayRatio: 0,
        };
    }

    const sorted = [...equityCurve].sort((a, b) => a.timestamp - b.timestamp);
    const startEq = opts.startingEquity || sorted[0].equity;
    const endEq = sorted[sorted.length - 1].equity;

    // Time span in years
    const startTs = sorted[0].timestamp;
    const endTs = sorted[sorted.length - 1].timestamp;
    const yearsElapsed = Math.max((endTs - startTs) / (365.25 * 86400000), 1 / TRADING_DAYS_PER_YEAR);

    // Total return
    const totalReturn = startEq > 0 ? (endEq - startEq) / startEq : 0;
    const totalReturnUsd = endEq - startEq;

    // CAGR
    const cagr = Math.pow(1 + totalReturn, 1 / yearsElapsed) - 1;

    // Daily returns
    const dailyReturns = calcDailyReturns(sorted).map(d => d.return);

    if (dailyReturns.length < 2) {
        return {
            cagr: +cagr.toFixed(6),
            volatility: 0,
            calmarRatio: 0,
            stability: 0,
            ulcerIndex: 0,
            painIndex: 0,
            totalReturn: +totalReturn.toFixed(6),
            totalReturnUsd: +totalReturnUsd.toFixed(2),
            yearsElapsed: +yearsElapsed.toFixed(4),
            avgDailyReturn: dailyReturns.length > 0 ? +mean(dailyReturns).toFixed(6) : 0,
            dailyReturnStd: 0,
            bestDay: dailyReturns.length > 0 ? +Math.max(...dailyReturns).toFixed(6) : 0,
            worstDay: dailyReturns.length > 0 ? +Math.min(...dailyReturns).toFixed(6) : 0,
            positiveDayRatio: dailyReturns.length > 0
                ? +(dailyReturns.filter(r => r > 0).length / dailyReturns.length).toFixed(4)
                : 0,
        };
    }

    // Annualized volatility
    const dailyVol = stdDev(dailyReturns);
    const volatility = dailyVol * Math.sqrt(TRADING_DAYS_PER_YEAR);

    // Max drawdown
    const { maxDrawdown } = calcMaxDrawdown(sorted);

    // Calmar ratio
    const calmarRatio = maxDrawdown > 0 ? cagr / maxDrawdown : (cagr > 0 ? Infinity : 0);

    // Stability: R² of linear regression of equity curve values against time
    // Higher R² means more consistent growth (less wobble)
    const n = sorted.length;
    const eqValues = sorted.map(p => p.equity);
    const timeValues = sorted.map(p => p.timestamp);

    const meanEq = mean(eqValues);
    const meanTime = mean(timeValues);

    let ssRes = 0;
    let ssTot = 0;
    let slopeNum = 0;
    let slopeDen = 0;

    for (let i = 0; i < n; i++) {
        const dt = timeValues[i] - meanTime;
        const de = eqValues[i] - meanEq;
        slopeNum += dt * de;
        slopeDen += dt * dt;
    }

    const slope = slopeDen > 0 ? slopeNum / slopeDen : 0;
    const intercept = meanEq - slope * meanTime;

    for (let i = 0; i < n; i++) {
        const predicted = intercept + slope * timeValues[i];
        ssRes += (eqValues[i] - predicted) ** 2;
        ssTot += (eqValues[i] - meanEq) ** 2;
    }

    const stability = ssTot > 0 ? Math.max(0, Math.min(1, 1 - ssRes / ssTot)) : 0;

    // Ulcer Index: sqrt(mean of squared drawdowns from peak)
    let runningPeak = sorted[0].equity;
    let sumSquaredDD = 0;
    let sumDD = 0;

    for (const pt of sorted) {
        if (pt.equity > runningPeak) runningPeak = pt.equity;
        const dd = runningPeak > 0 ? (runningPeak - pt.equity) / runningPeak : 0;
        sumSquaredDD += dd * dd;
        sumDD += dd;
    }

    const ulcerIndex = Math.sqrt(sumSquaredDD / n);
    const painIndex = sumDD / n;

    // Summary stats
    const bestDay = Math.max(...dailyReturns);
    const worstDay = Math.min(...dailyReturns);
    const positiveDayRatio = dailyReturns.filter(r => r > 0).length / dailyReturns.length;

    return {
        cagr: +cagr.toFixed(6),
        volatility: +volatility.toFixed(6),
        calmarRatio: calmarRatio === Infinity ? 999 : +calmarRatio.toFixed(4),
        stability: +stability.toFixed(6),
        ulcerIndex: +ulcerIndex.toFixed(6),
        painIndex: +painIndex.toFixed(6),
        totalReturn: +totalReturn.toFixed(6),
        totalReturnUsd: +totalReturnUsd.toFixed(2),
        yearsElapsed: +yearsElapsed.toFixed(4),
        avgDailyReturn: +mean(dailyReturns).toFixed(6),
        dailyReturnStd: +dailyVol.toFixed(6),
        bestDay: +bestDay.toFixed(6),
        worstDay: +worstDay.toFixed(6),
        positiveDayRatio: +positiveDayRatio.toFixed(4),
    };
}

// ============================================================================
// Internal: compute standard strategy metrics from trades + equity curve
// ============================================================================

/**
 * Compute a bundle of standard metrics from a set of trades and equity curve.
 * @param {Object[]} trades
 * @param {{timestamp: number, equity: number}[]} equityCurve
 * @returns {Object}
 */
function computeStrategyMetrics(trades, equityCurve) {
    const totalTrades = trades.length;
    if (totalTrades === 0) {
        return {
            totalTrades: 0,
            totalPnL: 0,
            sharpe: 0,
            sortino: 0,
            maxDrawdown: 0,
            winRate: 0,
            profitFactor: 0,
        };
    }

    const wins = trades.filter(t => t.pnl_usd > 0);
    const losses = trades.filter(t => t.pnl_usd < 0);
    const winRate = wins.length / totalTrades;
    const totalPnL = trades.reduce((s, t) => s + t.pnl_usd, 0);
    const grossProfit = wins.reduce((s, t) => s + t.pnl_usd, 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnl_usd, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

    const { maxDrawdown } = calcMaxDrawdown(equityCurve);

    return {
        totalTrades,
        totalPnL,
        sharpe: 0, // Will be computed by caller with daily returns if needed
        sortino: 0,
        maxDrawdown,
        winRate,
        profitFactor,
    };
}
