/**
 * Performance Audit Hook — strategy health monitoring.
 *
 * Calculates Sharpe Ratio, Max Drawdown, Profit Factor, Win Rate, and
 * Sortino Ratio from trade records. Emits GREEN/YELLOW/RED status based
 * on configurable thresholds.
 *
 * Usage:
 *   import { audit } from './audit/performance-audit-hook.mjs';
 *   const result = audit(trades, { timeframe: '30d', symbol: 'MNQ' });
 */

import { generateId, AUDIT_STATUS, THRESHOLDS } from './schemas.mjs';

const TRADING_DAYS_PER_YEAR = 252;
const RISK_FREE_RATE = 0.05; // 5% annual

/**
 * Run a full performance audit on a set of trade records.
 *
 * @param {import('./schemas.mjs').TradeRecord[]} trades
 * @param {Object} opts
 * @param {string} [opts.timeframe='30d']
 * @param {string} [opts.symbol] - filter to specific symbol
 * @param {number} [opts.startTime] - unix ms, filter trades after this
 * @param {number} [opts.endTime] - unix ms, filter trades before this
 * @returns {import('./schemas.mjs').AuditResult}
 */
export function audit(trades, opts = {}) {
    const timeframe = opts.timeframe || '30d';
    const symbol = opts.symbol || null;
    const startingEquity = opts.startingEquity || 50000;

    // Filter
    let filtered = trades;
    if (symbol) filtered = filtered.filter(t => t.symbol === symbol);
    if (opts.startTime) filtered = filtered.filter(t => t.exit_time >= opts.startTime);
    if (opts.endTime) filtered = filtered.filter(t => t.exit_time <= opts.endTime);

    if (filtered.length === 0) {
        return emptyResult(timeframe, symbol || 'ALL');
    }

    const totalTrades = filtered.length;
    const wins = filtered.filter(t => t.pnl_usd > 0);
    const losses = filtered.filter(t => t.pnl_usd < 0);
    const winRate = wins.length / totalTrades;
    const totalPnl = filtered.reduce((s, t) => s + t.pnl_usd, 0);
    const avgPnl = totalPnl / totalTrades;
    const avgHold = filtered.reduce((s, t) => s + t.hold_sec, 0) / totalTrades;

    // Profit Factor
    const grossProfit = wins.reduce((s, t) => s + t.pnl_usd, 0);
    const grossLoss = Math.abs(losses.reduce((s, t) => s + t.pnl_usd, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

    // Equity curve from cumulative P&L
    const equityCurve = buildEquityCurve(filtered, startingEquity);

    // Max Drawdown
    const { maxDrawdown, maxDrawdownUsd } = calcMaxDrawdown(equityCurve, filtered);

    // Sharpe Ratio (annualized)
    const returns = calcPeriodReturns(equityCurve, filtered, startingEquity);
    const sharpe = calcSharpe(returns);

    // Sortino Ratio (annualized)
    const sortino = calcSortino(returns);

    // Status + warnings
    const warnings = [];
    let status = AUDIT_STATUS.GREEN;

    if (maxDrawdown >= THRESHOLDS.MAX_DRAWDOWN_CRIT) {
        status = AUDIT_STATUS.RED;
        warnings.push(`CRITICAL: Max drawdown ${(maxDrawdown * 100).toFixed(1)}% exceeds ${(THRESHOLDS.MAX_DRAWDOWN_CRIT * 100).toFixed(0)}% threshold. Diversification may be premature — fix strategy first.`);
    } else if (maxDrawdown >= THRESHOLDS.MAX_DRAWDOWN_WARN) {
        status = status === AUDIT_STATUS.GREEN ? AUDIT_STATUS.YELLOW : status;
        warnings.push(`WARNING: Max drawdown ${(maxDrawdown * 100).toFixed(1)}% exceeds ${(THRESHOLDS.MAX_DRAWDOWN_WARN * 100).toFixed(0)}% warning level.`);
    }

    if (sharpe < THRESHOLDS.SHARPE_CRITICAL) {
        status = AUDIT_STATUS.RED;
        warnings.push(`CRITICAL: Sharpe ratio ${sharpe.toFixed(2)} is negative — strategy is destroying value.`);
    } else if (sharpe < THRESHOLDS.SHARPE_MINIMUM) {
        status = status === AUDIT_STATUS.GREEN ? AUDIT_STATUS.YELLOW : status;
        warnings.push(`WARNING: Sharpe ratio ${sharpe.toFixed(2)} below ${THRESHOLDS.SHARPE_MINIMUM} minimum.`);
    }

    if (winRate < THRESHOLDS.WIN_RATE_MINIMUM) {
        status = status === AUDIT_STATUS.GREEN ? AUDIT_STATUS.YELLOW : status;
        warnings.push(`WARNING: Win rate ${(winRate * 100).toFixed(1)}% below ${(THRESHOLDS.WIN_RATE_MINIMUM * 100).toFixed(0)}% minimum.`);
    }

    if (profitFactor < THRESHOLDS.PROFIT_FACTOR_MIN && profitFactor !== Infinity) {
        status = status === AUDIT_STATUS.GREEN ? AUDIT_STATUS.YELLOW : status;
        warnings.push(`WARNING: Profit factor ${profitFactor.toFixed(2)} below ${THRESHOLDS.PROFIT_FACTOR_MIN} minimum.`);
    }

    return {
        audit_id: generateId('audit'),
        timestamp: new Date().toISOString(),
        timeframe,
        symbol: symbol || 'ALL',
        total_trades: totalTrades,
        win_rate: +winRate.toFixed(4),
        total_pnl_usd: +totalPnl.toFixed(2),
        avg_pnl_per_trade: +avgPnl.toFixed(4),
        sharpe_ratio: +sharpe.toFixed(4),
        sortino_ratio: +sortino.toFixed(4),
        max_drawdown: +maxDrawdown.toFixed(4),
        max_drawdown_usd: +maxDrawdownUsd.toFixed(2),
        profit_factor: profitFactor === Infinity ? 999 : +profitFactor.toFixed(4),
        avg_hold_sec: +avgHold.toFixed(1),
        status,
        warnings,
    };
}

function buildEquityCurve(trades, startingEquity = 50000) {
    const sorted = [...trades].sort((a, b) => a.exit_time - b.exit_time);
    let equity = startingEquity;
    const curve = [{ timestamp: sorted[0]?.entry_time || Date.now(), equity }];
    for (const t of sorted) {
        equity += t.pnl_usd;
        curve.push({ timestamp: t.exit_time, equity });
    }
    return curve;
}

function calcMaxDrawdown(equityCurve, _trades) {
    let peak = equityCurve[0]?.equity || 0;
    let maxDD = 0;
    let maxDDUsd = 0;

    for (const point of equityCurve) {
        if (point.equity > peak) {
            peak = point.equity;
        }
        const dd = (peak - point.equity) / peak;
        if (dd > maxDD) {
            maxDD = dd;
            maxDDUsd = peak - point.equity;
        }
    }
    return { maxDrawdown: maxDD, maxDrawdownUsd: maxDDUsd };
}

function calcPeriodReturns(equityCurve, trades, startingEquity = 50000) {
    if (trades.length < 2) return [];
    // Group P&L by day for daily returns
    const dailyPnl = new Map();
    for (const t of trades) {
        const day = new Date(t.exit_time).toISOString().slice(0, 10);
        dailyPnl.set(day, (dailyPnl.get(day) || 0) + t.pnl_usd);
    }
    const days = [...dailyPnl.values()];
    const typicalEquity = equityCurve.length > 0 ? equityCurve[0].equity : startingEquity;

    // If only 1 day of data, use trade-level returns instead
    if (days.length < 2) {
        return trades.map(t => t.pnl_usd / typicalEquity);
    }
    return days.map(pnl => pnl / typicalEquity);
}

function calcSharpe(dailyReturns) {
    if (dailyReturns.length < 2) return 0;
    const mean = dailyReturns.reduce((s, r) => s + r, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((s, r) => s + (r - mean) ** 2, 0) / (dailyReturns.length - 1);
    const stdDev = Math.sqrt(variance);
    if (stdDev === 0) return 0;
    const dailySharpe = (mean - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR) / stdDev;
    return dailySharpe * Math.sqrt(TRADING_DAYS_PER_YEAR);
}

function calcSortino(dailyReturns) {
    if (dailyReturns.length < 2) return 0;
    const mean = dailyReturns.reduce((s, r) => s + r, 0) / dailyReturns.length;
    const downReturns = dailyReturns.filter(r => r < 0);
    if (downReturns.length === 0) return mean > 0 ? 10 : 0; // no downside = excellent
    const downVariance = downReturns.reduce((s, r) => s + r ** 2, 0) / downReturns.length;
    const downStdDev = Math.sqrt(downVariance);
    if (downStdDev === 0) return 0;
    const dailySortino = (mean - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR) / downStdDev;
    return dailySortino * Math.sqrt(TRADING_DAYS_PER_YEAR);
}

function emptyResult(timeframe, symbol) {
    return {
        audit_id: generateId('audit'),
        timestamp: new Date().toISOString(),
        timeframe,
        symbol,
        total_trades: 0,
        win_rate: 0,
        total_pnl_usd: 0,
        avg_pnl_per_trade: 0,
        sharpe_ratio: 0,
        sortino_ratio: 0,
        max_drawdown: 0,
        max_drawdown_usd: 0,
        profit_factor: 0,
        avg_hold_sec: 0,
        status: AUDIT_STATUS.YELLOW,
        warnings: ['No trades found for the specified period.'],
    };
}

/**
 * Convenience: run audit and log results.
 */
export function auditAndLog(trades, opts = {}) {
    const result = audit(trades, opts);
    const icon = result.status === 'GREEN' ? '✓' : result.status === 'RED' ? '✗' : '⚠';

    console.log(`\n${icon} AUDIT [${result.status}] ${result.symbol} ${result.timeframe}`);
    console.log(`   Trades: ${result.total_trades} | Win Rate: ${(result.win_rate * 100).toFixed(1)}%`);
    console.log(`   P&L: $${result.total_pnl_usd} | Avg: $${result.avg_pnl_per_trade}`);
    console.log(`   Sharpe: ${result.sharpe_ratio} | Sortino: ${result.sortino_ratio}`);
    console.log(`   Max DD: ${(result.max_drawdown * 100).toFixed(2)}% ($${result.max_drawdown_usd})`);
    console.log(`   Profit Factor: ${result.profit_factor === 999 ? '∞' : result.profit_factor.toFixed(2)}`);

    if (result.warnings.length > 0) {
        for (const w of result.warnings) {
            console.log(`   ${w}`);
        }
    }
    console.log('');
    return result;
}
