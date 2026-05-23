/**
 * Backtest Engine — historical strategy simulation.
 *
 * Replays trade signals against historical candle data, calculates
 * walk-forward performance, and detects regime-specific patterns.
 *
 * Usage:
 *   import { runBacktest, walkForward } from './audit/backtest.mjs';
 *   const result = runBacktest(candles, strategyFn, { symbol: 'MNQ' });
 */

import { generateId, AUDIT_STATUS } from './schemas.mjs';

/**
 * @typedef {Object} Candle
 * @property {number} timestamp - unix ms
 * @property {number} open
 * @property {number} high
 * @property {number} low
 * @property {number} close
 * @property {number} [volume]
 */

/**
 * @typedef {Object} Signal
 * @property {string} symbol
 * @property {'long'|'short'} direction
 * @property {number} timestamp - unix ms
 * @property {number} price
 * @property {number} [score]
 * @property {Object} [metadata]
 */

/**
 * @typedef {Object} BacktestResult
 * @property {string} backtest_id
 * @property {string} timestamp - ISO 8601
 * @property {string} symbol
 * @property {number} total_signals
 * @property {number} total_trades
 * @property {number} win_rate
 * @property {number} total_pnl_usd
 * @property {number} sharpe_ratio
 * @property {number} max_drawdown
 * @property {number} profit_factor
 * @property {Object[]} regime_breakdown - performance by regime
 * @property {Object} simulated_vs_realized - edge comparison
 */

/**
 * Run a backtest of a strategy function against historical candles.
 *
 * @param {Candle[]} candles - historical OHLCV data
 * @param {function(Candle, Candle[]): Signal|null} strategyFn - signal generator
 * @param {Object} opts
 * @param {string} [opts.symbol='MNQ']
 * @param {number} [opts.startingEquity=50000]
 * @param {number} [opts.contractSize=2] - $2/point for MNQ
 * @param {number} [opts.slippagePts=0.25]
 * @param {number} [opts.commissionPerTrade=1.24] - per side
 * @returns {BacktestResult}
 */
export function runBacktest(candles, strategyFn, opts = {}) {
    const symbol = opts.symbol || 'MNQ';
    const startingEquity = opts.startingEquity || 50000;
    const contractSize = opts.contractSize || 2; // $2/pt for MNQ
    const slippage = opts.slippagePts || 0.25;
    const commission = opts.commissionPerTrade || 1.24;

    if (candles.length < 20) {
        return emptyBacktest(symbol, startingEquity);
    }

    const signals = [];
    const trades = [];
    let equity = startingEquity;
    let peak = equity;
    let maxDD = 0;
    let inPosition = null;

    // Walk-forward: generate signals candle by candle
    for (let i = 20; i < candles.length; i++) {
        const currentCandle = candles[i];
        const history = candles.slice(0, i);

        // Check for exit if in position
        if (inPosition) {
            const exitPrice = getExitPrice(inPosition, currentCandle, candles.slice(i), history);
            if (exitPrice !== null) {
                // Close position
                const pnl = calculatePnl(inPosition, exitPrice, contractSize, slippage, commission);
                trades.push({
                    symbol,
                    direction: inPosition.direction,
                    entry_price: inPosition.entryPrice,
                    exit_price: exitPrice,
                    pnl_usd: pnl,
                    hold_sec: (currentCandle.timestamp - inPosition.entryTime) / 1000,
                    entry_time: inPosition.entryTime,
                    exit_time: currentCandle.timestamp,
                    exit_reason: inPosition.exitReason || 'SIGNAL',
                });
                equity += pnl;
                if (equity > peak) peak = equity;
                const dd = (peak - equity) / peak;
                if (dd > maxDD) maxDD = dd;
                inPosition = null;
            }
        }

        // Generate new signal if flat
        if (!inPosition) {
            const signal = strategyFn(currentCandle, history);
            if (signal) {
                signals.push(signal);
                // Enter position
                const entryPrice = signal.direction === 'long'
                    ? currentCandle.close + slippage
                    : currentCandle.close - slippage;
                inPosition = {
                    direction: signal.direction,
                    entryPrice,
                    entryTime: signal.timestamp || currentCandle.timestamp,
                    exitReason: signal.metadata?.exitReason || null,
                    signal,
                };
            }
        }
    }

    // Close any open position at last candle
    if (inPosition) {
        const lastCandle = candles[candles.length - 1];
        const exitPrice = lastCandle.close;
        const pnl = calculatePnl(inPosition, exitPrice, contractSize, 0, commission);
        trades.push({
            symbol,
            direction: inPosition.direction,
            entry_price: inPosition.entryPrice,
            exit_price: exitPrice,
            pnl_usd: pnl,
            hold_sec: (lastCandle.timestamp - inPosition.entryTime) / 1000,
            entry_time: inPosition.entryTime,
            exit_time: lastCandle.timestamp,
            exit_reason: 'EOD',
        });
        equity += pnl;
    }

    // Calculate metrics
    const wins = trades.filter(t => t.pnl_usd > 0);
    const losses = trades.filter(t => t.pnl_usd < 0);
    const winRate = trades.length > 0 ? wins.length / trades.length : 0;
    const totalPnl = trades.reduce((s, t) => s + t.pnl_usd, 0);
    const grossProfit = wins.reduce((s, t) => s + t.pnl_usd, 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnl_usd, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

    // Sharpe from trade returns
    const returns = trades.map(t => t.pnl_usd / startingEquity);
    const sharpe = calcSharpeFromReturns(returns);

    // Regime breakdown
    const regimeBreakdown = analyzeRegimes(candles, trades);

    // Simulated vs realized edge
    const avgSignalScore = signals.length > 0
        ? signals.reduce((s, sig) => s + (sig.score || 0), 0) / signals.length
        : 0;
    const simulatedVsRealized = {
        avg_signal_score: +avgSignalScore.toFixed(2),
        signal_count: signals.length,
        trade_count: trades.length,
        signal_to_trade_ratio: signals.length > 0
            ? +(trades.length / signals.length).toFixed(2)
            : 0,
        edge_per_trade: trades.length > 0
            ? +(totalPnl / trades.length / contractSize).toFixed(2)
            : 0,
    };

    let status = AUDIT_STATUS.GREEN;
    if (maxDD >= 0.15) status = AUDIT_STATUS.RED;
    else if (maxDD >= 0.10 || sharpe < 0.5) status = AUDIT_STATUS.YELLOW;

    return {
        backtest_id: generateId('bt'),
        timestamp: new Date().toISOString(),
        symbol,
        total_signals: signals.length,
        total_trades: trades.length,
        win_rate: +winRate.toFixed(4),
        total_pnl_usd: +totalPnl.toFixed(2),
        sharpe_ratio: +sharpe.toFixed(4),
        max_drawdown: +maxDD.toFixed(4),
        profit_factor: profitFactor === Infinity ? 999 : +profitFactor.toFixed(4),
        regime_breakdown: regimeBreakdown,
        simulated_vs_realized: simulatedVsRealized,
        status,
    };
}

function getExitPrice(position, candle, futureCandles, history = [], stopLossPercent = 0.02, takeProfitPercent = 0.04) {
    const nextCandle = futureCandles[1];
    if (!nextCandle) return candle.close;

    const entry = position.entryPrice;

    if (position.direction === 'long') {
        // Stop-loss: exit if price drops below entry * (1 - stopLossPercent)
        const stopPrice = entry * (1 - stopLossPercent);
        if (candle.low <= stopPrice) return stopPrice;

        // Take-profit: exit if price rises above entry * (1 + takeProfitPercent)
        const targetPrice = entry * (1 + takeProfitPercent);
        if (candle.high >= targetPrice) return targetPrice;
    } else {
        // Stop-loss: exit if price rises above entry * (1 + stopLossPercent)
        const stopPrice = entry * (1 + stopLossPercent);
        if (candle.high >= stopPrice) return stopPrice;

        // Take-profit: exit if price drops below entry * (1 - takeProfitPercent)
        const targetPrice = entry * (1 - takeProfitPercent);
        if (candle.low <= targetPrice) return targetPrice;
    }

    // SMA crossover exit — third exit condition
    // For longs, exit when fast SMA crosses below slow SMA (bearish signal)
    // For shorts, exit when fast SMA crosses above slow SMA (bullish signal)
    if (history.length >= 30) {
        const fastPeriod = 10;
        const slowPeriod = 30;
        const fastSma = avgClose(history.slice(-fastPeriod));
        const slowSma = avgClose(history.slice(-slowPeriod));
        const prevFastSma = avgClose(history.slice(-fastPeriod - 1, -1));
        const prevSlowSma = avgClose(history.slice(-slowPeriod - 1, -1));

        if (position.direction === 'long' && prevFastSma >= prevSlowSma && fastSma < slowSma) {
            return candle.close; // Bearish crossover
        }
        if (position.direction === 'short' && prevFastSma <= prevSlowSma && fastSma > slowSma) {
            return candle.close; // Bullish crossover
        }
    }

    return null; // hold
}

function calculatePnl(position, exitPrice, contractSize, slippage, commission) {
    const rawPnl = position.direction === 'long'
        ? (exitPrice - position.entryPrice) * contractSize
        : (position.entryPrice - exitPrice) * contractSize;
    return rawPnl - (slippage * contractSize * 2) - (commission * 2);
}

function calcSharpeFromReturns(returns) {
    if (returns.length < 2) return 0;
    const mean = returns.reduce((s, r) => s + r, 0) / returns.length;
    const variance = returns.reduce((s, r) => s + (r - mean) ** 2, 0) / (returns.length - 1);
    const stdDev = Math.sqrt(variance);
    if (stdDev === 0) return mean > 0 ? 999 : 0;
    return mean / stdDev;
}

/**
 * Break down performance by market regime.
 */
function analyzeRegimes(candles, trades) {
    const regimes = {
        trending_up: { trades: 0, wins: 0, pnl: 0 },
        trending_down: { trades: 0, wins: 0, pnl: 0 },
        ranging: { trades: 0, wins: 0, pnl: 0 },
        volatile: { trades: 0, wins: 0, pnl: 0 },
    };

    // Classify each 20-candle window
    const windowSize = 20;
    for (let i = windowSize; i < candles.length; i += windowSize) {
        const window = candles.slice(i - windowSize, i);
        const regime = classifyRegime(window);

        // Find trades that occurred in this window
        const windowStart = window[0].timestamp;
        const windowEnd = window[window.length - 1].timestamp;
        const windowTrades = trades.filter(
            t => t.exit_time >= windowStart && t.exit_time <= windowEnd,
        );

        for (const t of windowTrades) {
            regimes[regime].trades++;
            if (t.pnl_usd > 0) regimes[regime].wins++;
            regimes[regime].pnl += t.pnl_usd;
        }
    }

    // Format output
    const breakdown = [];
    for (const [regime, stats] of Object.entries(regimes)) {
        if (stats.trades === 0) continue;
        breakdown.push({
            regime,
            trades: stats.trades,
            win_rate: +(stats.wins / stats.trades).toFixed(4),
            total_pnl: +stats.pnl.toFixed(2),
            avg_pnl: +(stats.pnl / stats.trades).toFixed(2),
        });
    }

    return breakdown;
}

function classifyRegime(window) {
    const first = window[0].close;
    const last = window[window.length - 1].close;
    const pctChange = (last - first) / first;

    const highs = window.map(c => c.high);
    const lows = window.map(c => c.low);
    const range = Math.max(...highs) - Math.min(...lows);
    const avgClose = window.reduce((s, c) => s + c.close, 0) / window.length;
    const rangePct = range / avgClose;

    if (rangePct > 0.03) return 'volatile';
    if (rangePct < 0.005) return 'ranging';
    if (pctChange > 0.01) return 'trending_up';
    if (pctChange < -0.01) return 'trending_down';
    return 'ranging';
}

function emptyBacktest(symbol, equity) {
    return {
        backtest_id: generateId('bt'),
        timestamp: new Date().toISOString(),
        symbol,
        total_signals: 0,
        total_trades: 0,
        win_rate: 0,
        total_pnl_usd: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        profit_factor: 0,
        regime_breakdown: [],
        simulated_vs_realized: { avg_signal_score: 0, signal_count: 0, trade_count: 0, signal_to_trade_ratio: 0, edge_per_trade: 0 },
        status: AUDIT_STATUS.YELLOW,
    };
}

/**
 * Generate sample candles for backtest validation.
 */
export function generateSampleCandles(count = 500, trend = 'random') {
    const candles = [];
    let price = 19500;
    const baseTs = 1716150000000;

    for (let i = 0; i < count; i++) {
        let drift;
        switch (trend) {
            case 'bull': drift = 2 + Math.random() * 4; break;
            case 'bear': drift = -2 - Math.random() * 4; break;
            case 'sideways': drift = (Math.random() - 0.5) * 2; break;
            default: drift = (Math.random() - 0.48) * 6; break; // slight upward bias
        }

        const open = price;
        const close = open + drift;
        const high = Math.max(open, close) + Math.random() * 10;
        const low = Math.min(open, close) - Math.random() * 10;

        candles.push({
            timestamp: baseTs + i * 60000, // 1-minute candles
            open: +open.toFixed(2),
            high: +high.toFixed(2),
            low: +low.toFixed(2),
            close: +close.toFixed(2),
            volume: Math.round(100 + Math.random() * 900),
        });

        price = close;
    }

    return candles;
}

/**
 * Simple moving average crossover strategy for testing.
 */
export function smaCrossStrategy(candle, history, fastPeriod = 10, slowPeriod = 30) {
    if (history.length < slowPeriod) return null;

    const fastSma = avgClose(history.slice(-fastPeriod));
    const slowSma = avgClose(history.slice(-slowPeriod));
    const prevFastSma = avgClose(history.slice(-fastPeriod - 1, -1));
    const prevSlowSma = avgClose(history.slice(-slowPeriod - 1, -1));

    if (prevFastSma <= prevSlowSma && fastSma > slowSma) {
        return { symbol: 'MNQ', direction: 'long', timestamp: candle.timestamp, price: candle.close, score: 0.6, metadata: { fastSma, slowSma } };
    }
    if (prevFastSma >= prevSlowSma && fastSma < slowSma) {
        return { symbol: 'MNQ', direction: 'short', timestamp: candle.timestamp, price: candle.close, score: 0.6, metadata: { fastSma, slowSma } };
    }
    return null;
}

function avgClose(candles) {
    return candles.reduce((s, c) => s + c.close, 0) / candles.length;
}
