/**
 * Audit data schemas — canonical shapes for all audit records.
 * Used by performance-audit-hook.mjs and stress-test.mjs.
 */

/**
 * @typedef {Object} TradeRecord
 * @property {string} symbol - e.g. "MNQ", "NQ"
 * @property {'long'|'short'} direction
 * @property {number} entry_price
 * @property {number} exit_price
 * @property {number} quantity
 * @property {number} pnl_usd - realized P&L in USD
 * @property {number} hold_sec - hold duration in seconds
 * @property {number} entry_time - unix ms
 * @property {number} exit_time - unix ms
 * @property {string} [exit_reason] - e.g. "BROKER-FLATTEN", "TARGET", "STOP", "SIGNAL"
 */

/**
 * @typedef {Object} EquityPoint
 * @property {number} timestamp - unix ms
 * @property {number} equity - account equity in USD
 */

/**
 * @typedef {Object} AuditResult
 * @property {string} audit_id
 * @property {string} timestamp - ISO 8601
 * @property {string} timeframe - e.g. "30d", "7d", "48h"
 * @property {string} symbol
 * @property {number} total_trades
 * @property {number} win_rate - 0-1
 * @property {number} total_pnl_usd
 * @property {number} avg_pnl_per_trade
 * @property {number} sharpe_ratio - annualized
 * @property {number} sortino_ratio - annualized
 * @property {number} max_drawdown - 0-1 fraction
 * @property {number} max_drawdown_usd
 * @property {number} profit_factor
 * @property {number} avg_hold_sec
 * @property {'GREEN'|'YELLOW'|'RED'} status
 * @property {string[]} warnings
 */

/**
 * @typedef {Object} StressTestResult
 * @property {string} test_id
 * @property {string} timestamp - ISO 8601
 * @property {string} scenario - e.g. "48h-sideways", "volatility-spike"
 * @property {string} symbol
 * @property {number} starting_equity
 * @property {number} projected_ending_equity
 * @property {number} projected_max_drawdown
 * @property {number} projected_sharpe
 * @property {string} regime_vulnerability - e.g. "mean-reversion", "trend", "neutral"
 * @property {string} recommended_hedge - e.g. "FX", "Commodities", "None"
 * @property {'GREEN'|'YELLOW'|'RED'} status
 */

export const AUDIT_STATUS = {
    GREEN:  'GREEN',   // All thresholds passed
    YELLOW: 'YELLOW',  // Warning thresholds breached
    RED:    'RED',     // Critical thresholds breached
};

export const REGIME_VULNERABILITY = {
    TREND:         'trend',          // Needs trending markets
    MEAN_REVERSION: 'mean-reversion', // Needs mean-reverting markets
    NEUTRAL:        'neutral',        // Works in any regime
};

export const RECOMMENDED_HEDGE = {
    FX:          'FX',           // Mean-reversion: look at FX pairs
    COMMODITIES: 'Commodities',  // Trend-following: look at Commodities
    NONE:        'None',         // No hedge needed
};

export const THRESHOLDS = {
    MAX_DRAWDOWN_WARN:  0.10,  // 10% — yellow
    MAX_DRAWDOWN_CRIT:  0.15,  // 15% — red
    SHARPE_MINIMUM:     0.5,   // Below this = yellow
    SHARPE_CRITICAL:    0.0,   // Below this = red
    WIN_RATE_MINIMUM:   0.35,  // Below this = yellow
    PROFIT_FACTOR_MIN:  1.2,   // Below this = yellow
    SIDEWAYS_RANGE_PCT: 0.005, // 0.5% range = sideways
};

/**
 * Generate a unique audit ID.
 */
export function generateId(prefix = 'audit') {
    const ts = Date.now();
    const rand = Math.random().toString(36).slice(2, 8);
    return `${prefix}_${ts}_${rand}`;
}

/**
 * Validate a trade record shape.
 */
export function validateTradeRecord(record) {
    const errors = [];
    if (!record.symbol) errors.push('symbol required');
    if (!['long', 'short'].includes(record.direction)) errors.push('direction must be long|short');
    if (typeof record.pnl_usd !== 'number') errors.push('pnl_usd must be number');
    if (typeof record.entry_price !== 'number') errors.push('entry_price must be number');
    if (typeof record.exit_price !== 'number') errors.push('exit_price must be number');
    if (typeof record.hold_sec !== 'number') errors.push('hold_sec must be number');
    return { valid: errors.length === 0, errors };
}

// ---------------------------------------------------------------------------
// Real 2026 cross-asset correlation matrix (sourced May 2026)
// Values are rolling 60-day correlation to NQ futures
// ---------------------------------------------------------------------------

export const CORRELATION_TO_NQ = {
    ES:    { normal: 0.88, stress: 0.94, label: 'S&P 500 futures — leverage, not diversification' },
    MGC:   { normal: -0.08, stress: 0.42, label: 'Micro Gold — fails as safe haven during margin cascades' },
    MCL:   { normal: -0.22, stress: -0.35, label: 'Micro Crude — negative during supply shocks, best diversifier' },
    SIL:   { normal: -0.05, stress: 0.30, label: 'Micro Silver — industrial + precious, mixed signal' },
    EURUSD:{ normal: -0.35, stress: -0.28, label: 'Euro FX — consistent negative correlation, mean-reversion' },
    USDJPY:{ normal:  0.25, stress:  0.40, label: 'Yen FX — risk-on correlated, weak diversifier' },
    ZB:    { normal: -0.30, stress:  0.15, label: '30Y Treasuries — correlation flips positive in inflation stress' },
};

export const CURRENT_MARKET_REGIME = {
    date: '2026-05-22',
    nq_price: 29600,
    nq_rsi_daily: 80,
    vix: 18,
    volume_vs_avg: -0.11,
    oil_wti: 102,
    gold: 4550,
    regime: 'overbought_rally',
    risks: [
        'Narrow leadership — 10 stocks drive 69% of S&P gains',
        'Volume confirmation failure — rally on declining volume',
        'Elliott Wave completion — 8-12% correction expected',
        'MGC correlation flips positive during stress (margin cascades)',
        'Iran geopolitical premium in oil — supply shock risk',
    ],
};
