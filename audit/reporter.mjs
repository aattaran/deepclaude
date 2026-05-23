/**
 * Audit Reporter — generates formatted reports from audit + stress test results.
 *
 * Output formats: markdown (for PRs/commits), text (for terminal), JSON (for pipelines).
 *
 * Usage:
 *   import { generateReport } from './audit/reporter.mjs';
 *   const md = generateReport(auditResult, stressResults, 'markdown');
 */

import { AUDIT_STATUS, CORRELATION_TO_NQ, CURRENT_MARKET_REGIME } from './schemas.mjs';

const DIVIDER = '═'.repeat(62);
const THIN =   '─'.repeat(62);

/**
 * Generate a formatted report.
 *
 * @param {import('./schemas.mjs').AuditResult} auditResult
 * @param {Object} stressResults - output of runFullStressSuite()
 * @param {'markdown'|'text'|'json'} [format='text']
 * @returns {string}
 */
export function generateReport(auditResult, stressResults, format = 'text') {
    switch (format) {
        case 'markdown': return generateMarkdown(auditResult, stressResults);
        case 'json':     return generateJson(auditResult, stressResults);
        case 'text':
        default:         return generateText(auditResult, stressResults);
    }
}

function generateText(audit, stress) {
    const statusIcon = audit.status === 'GREEN' ? '✓' : audit.status === 'RED' ? '✗' : '⚠';
    const lines = [];

    lines.push('');
    lines.push(`  ${DIVIDER}`);
    lines.push(`  ${statusIcon} STRATEGY AUDIT — ${audit.symbol} (${audit.timeframe})`);
    lines.push(`  ${DIVIDER}`);
    lines.push(`  Generated: ${audit.timestamp}`);
    lines.push(`  Status:    ${audit.status}`);
    lines.push('');

    // Performance metrics
    lines.push(`  ── Performance Metrics ${THIN.slice(0, 40)}`);
    lines.push(`  Total Trades:      ${audit.total_trades}`);
    lines.push(`  Win Rate:          ${(audit.win_rate * 100).toFixed(1)}%`);
    lines.push(`  Total P&L:         $${audit.total_pnl_usd.toLocaleString()}`);
    lines.push(`  Avg P&L/Trade:     $${audit.avg_pnl_per_trade.toFixed(2)}`);
    lines.push(`  Profit Factor:     ${audit.profit_factor === 999 ? '∞' : audit.profit_factor.toFixed(2)}`);
    lines.push(`  Avg Hold:          ${audit.avg_hold_sec.toFixed(0)}s`);
    lines.push('');

    // Risk metrics
    lines.push(`  ── Risk Metrics ${THIN.slice(0, 47)}`);
    lines.push(`  Sharpe Ratio:      ${audit.sharpe_ratio.toFixed(4)}${audit.sharpe_ratio < 0 ? ' ⚠ NEGATIVE' : ''}`);
    lines.push(`  Sortino Ratio:     ${audit.sortino_ratio.toFixed(4)}`);
    lines.push(`  Max Drawdown:      ${(audit.max_drawdown * 100).toFixed(2)}%`);
    lines.push(`  Max Drawdown USD:  $${audit.max_drawdown_usd.toLocaleString()}`);
    lines.push('');

    // Warnings
    if (audit.warnings.length > 0) {
        lines.push(`  ── Warnings ${THIN.slice(0, 52)}`);
        for (const w of audit.warnings) {
            const prefix = w.startsWith('CRITICAL') ? '✗' : '⚠';
            lines.push(`  ${prefix} ${w}`);
        }
        lines.push('');
    }

    // Stress test results
    if (stress && stress.results) {
        lines.push(`  ── Stress Test Results ${THIN.slice(0, 41)}`);
        lines.push(`  Scenario             End Equity      Max DD     Sharpe  Status`);
        lines.push(`  -------------------  --------------  ---------  ------  ------`);
        for (const r of stress.results) {
            const eq = `$${r.projected_ending_equity.toLocaleString()}`.padStart(14);
            const dd = `${(r.projected_max_drawdown * 100).toFixed(2)}%`.padStart(9);
            const sh = r.projected_sharpe.toFixed(2).padStart(6);
            const st = r.status;
            const sc = r.scenario.padEnd(19);
            lines.push(`  ${sc}  ${eq}  ${dd}  ${sh}  ${st}`);
        }
        lines.push('');

        // Worst case
        const w = stress.worst;
        lines.push(`  ── Worst Case: ${w.scenario} ${THIN.slice(0, 33)}`);
        lines.push(`  Max Drawdown:      ${(w.projected_max_drawdown * 100).toFixed(2)}%`);
        lines.push(`  Regime Dependency: ${w.regime_vulnerability}`);
        lines.push(`  Recommended Hedge: ${w.recommended_hedge}`);
        lines.push('');
    }

    // Recommendations
    lines.push(`  ── Recommendations ${THIN.slice(0, 46)}`);
    lines.push(...buildRecommendations(audit, stress));
    lines.push('');
    lines.push(`  ${DIVIDER}`);
    lines.push('');

    return lines.join('\n');
}

function generateMarkdown(audit, stress) {
    const statusIcon = audit.status === 'GREEN' ? '🟢' : audit.status === 'RED' ? '🔴' : '🟡';
    const lines = [];

    lines.push(`# ${statusIcon} Strategy Audit — ${audit.symbol} (${audit.timeframe})`);
    lines.push('');
    lines.push(`**Generated:** ${audit.timestamp}`);
    lines.push(`**Status:** ${audit.status}`);
    lines.push('');

    lines.push('## Performance Metrics');
    lines.push('');
    lines.push('| Metric | Value |');
    lines.push('|--------|-------|');
    lines.push(`| Total Trades | ${audit.total_trades} |`);
    lines.push(`| Win Rate | ${(audit.win_rate * 100).toFixed(1)}% |`);
    lines.push(`| Total P&L | $${audit.total_pnl_usd.toLocaleString()} |`);
    lines.push(`| Avg P&L/Trade | $${audit.avg_pnl_per_trade.toFixed(2)} |`);
    lines.push(`| Profit Factor | ${audit.profit_factor === 999 ? '∞' : audit.profit_factor.toFixed(2)} |`);
    lines.push(`| Avg Hold | ${audit.avg_hold_sec.toFixed(0)}s |`);
    lines.push('');

    lines.push('## Risk Metrics');
    lines.push('');
    lines.push('| Metric | Value |');
    lines.push('|--------|-------|');
    lines.push(`| Sharpe Ratio | ${audit.sharpe_ratio.toFixed(4)} |`);
    lines.push(`| Sortino Ratio | ${audit.sortino_ratio.toFixed(4)} |`);
    lines.push(`| Max Drawdown | ${(audit.max_drawdown * 100).toFixed(2)}% |`);
    lines.push(`| Max Drawdown USD | $${audit.max_drawdown_usd.toLocaleString()} |`);
    lines.push('');

    if (audit.warnings.length > 0) {
        lines.push('## Warnings');
        lines.push('');
        for (const w of audit.warnings) {
            lines.push(`- ${w.startsWith('CRITICAL') ? '🔴' : '🟡'} ${w}`);
        }
        lines.push('');
    }

    if (stress && stress.results) {
        lines.push('## Stress Test Results');
        lines.push('');
        lines.push('| Scenario | End Equity | Max DD | Sharpe | Status |');
        lines.push('|----------|------------|--------|--------|--------|');
        for (const r of stress.results) {
            lines.push(`| ${r.scenario} | $${r.projected_ending_equity.toLocaleString()} | ${(r.projected_max_drawdown * 100).toFixed(2)}% | ${r.projected_sharpe.toFixed(2)} | ${r.status} |`);
        }
        lines.push('');
        lines.push(`**Worst case:** ${stress.worst.scenario} — ${(stress.worst.projected_max_drawdown * 100).toFixed(2)}% DD`);
        lines.push(`**Regime dependency:** ${stress.worst.regime_vulnerability}`);
        lines.push(`**Recommended hedge:** ${stress.worst.recommended_hedge}`);
        lines.push('');
    }

    lines.push('## Recommendations');
    lines.push('');
    for (const r of buildRecommendations(audit, stress)) {
        lines.push(`- ${r}`);
    }
    lines.push('');

    return lines.join('\n');
}

function generateJson(audit, stress) {
    const output = {
        audit: {
            ...audit,
            warnings: undefined, // included in recommendations
        },
        stress_test: stress || null,
        recommendations: buildRecommendations(audit, stress),
        warnings: audit.warnings,
    };
    return JSON.stringify(output, null, 2);
}

function buildRecommendations(audit, stress) {
    const recs = [];

    // Drawdown-based recommendations
    if (audit.max_drawdown >= 0.15) {
        recs.push('CRITICAL: Drawdown >15%. PAUSE trading. Debug strategy before adding any new instruments.');
    } else if (audit.max_drawdown >= 0.10) {
        recs.push('WARNING: Drawdown >10%. Reduce position size 50% until DD recovers below 7.5%.');
    }

    // Sharpe-based
    if (audit.sharpe_ratio < 0) {
        recs.push('Negative Sharpe: Strategy is destroying risk-adjusted value. Re-evaluate entry/exit logic.');
    } else if (audit.sharpe_ratio < 0.5) {
        recs.push('Low Sharpe (<0.5): Consider adding a non-correlated hedge rather than more directional exposure.');
    }

    // Stress test-based
    if (stress && stress.worst) {
        const w = stress.worst;

        if (w.regime_vulnerability === 'trend' && w.projected_max_drawdown > 0.10) {
            recs.push(`Regime dependency detected (${w.scenario}): Strategy bleeds in sideways markets. Consider FX pairs (mean-reversion) or Gold (safe-haven) as hedges.`);
        }

        if (w.regime_vulnerability === 'mean-reversion' && w.projected_max_drawdown > 0.10) {
            recs.push(`Regime dependency detected (${w.scenario}): Strategy struggles in trending markets. Consider Commodities or trend-following instruments as diversifiers.`);
        }

        if (w.recommended_hedge === 'FX') {
            recs.push('Hedge candidate: FX pairs (EUR/USD, USD/JPY) — low correlation to equities, mean-reversion characteristics.');
        } else if (w.recommended_hedge === 'Commodities') {
            recs.push('Hedge candidate: Commodities (GC/MGC, CL/MCL) — different risk drivers than equity indices.');
        }
    }

    // Instrument-specific with real 2026 correlation data
    if (audit.symbol === 'NQ' || audit.symbol === 'MNQ') {
        const es = CORRELATION_TO_NQ.ES;
        const mgc = CORRELATION_TO_NQ.MGC;
        const mcl = CORRELATION_TO_NQ.MCL;

        recs.push(`ES correlation to NQ: ${(es.normal * 100).toFixed(0)}% normal / ${(es.stress * 100).toFixed(0)}% stress — leverage, not diversification.`);
        recs.push(`MGC (Gold) correlation: ${(mgc.normal * 100).toFixed(0)}% normal but flips to +${(mgc.stress * 100).toFixed(0)}% during stress (margin cascades). ${mgc.label}.`);
        recs.push(`MCL (Crude) correlation: ${(mcl.normal * 100).toFixed(0)}% normal / ${(mcl.stress * 100).toFixed(0)}% stress — ${mcl.label}. Strongest diversifier in current regime.`);

        // Current regime context
        if (CURRENT_MARKET_REGIME.regime === 'overbought_rally') {
            recs.push(`CURRENT: NQ at ~${CURRENT_MARKET_REGIME.nq_price}, RSI ${CURRENT_MARKET_REGIME.nq_rsi_daily} (overbought), VIX ${CURRENT_MARKET_REGIME.vix} (deceptive calm). Volume ${Math.abs(CURRENT_MARKET_REGIME.volume_vs_avg) * 100}% below avg — correction risk elevated.`);
        }
    }

    // Profit factor
    if (audit.profit_factor < 1.0 && audit.total_trades > 10) {
        recs.push('Profit factor < 1.0: Strategy is losing money. Fix core edge before diversifying.');
    }

    // Default
    if (recs.length === 0) {
        recs.push('All metrics within healthy ranges. Strategy is stable — diversification is optional but not urgent.');
    }

    return recs;
}

/**
 * Convenience: run full audit pipeline and print report.
 */
export function runFullAudit(doAudit, doStress, opts = {}) {
    const format = opts.format || 'text';

    // Run audit
    const auditResult = doAudit();

    // Run stress tests
    let stressResults = null;
    if (doStress) {
        stressResults = doStress();
    }

    // Generate and print report
    const report = generateReport(auditResult, stressResults, format);
    console.log(report);

    return { audit: auditResult, stress: stressResults, report };
}
