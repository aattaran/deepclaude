/**
 * Stress Test Engine — market regime simulation.
 *
 * Simulates what happens to the strategy under different market conditions:
 * - 48-hour sideways (range-bound) market
 * - Volatility spike (2x normal ATR)
 * - Trend continuation (bull/bear)
 *
 * Usage:
 *   import { runStressTest } from './audit/stress-test.mjs';
 *   const result = runStressTest(trades, { scenario: '48h-sideways' });
 */

import { generateId, AUDIT_STATUS, THRESHOLDS } from './schemas.mjs';

const SCENARIOS = {
    SIDEWAYS_48H:    '48h-sideways',
    VOL_SPIKE:       'volatility-spike',
    TREND_BULL:      'trend-bull',
    TREND_BEAR:      'trend-bear',
};

/**
 * Run a stress test against historical trade data.
 *
 * @param {import('./schemas.mjs').TradeRecord[]} trades
 * @param {Object} opts
 * @param {string} [opts.scenario='48h-sideways']
 * @param {string} [opts.symbol]
 * @param {number} [opts.startingEquity=50000]
 * @param {number} [opts.sidewaysRangePct=0.005] - 0.5% range
 * @returns {import('./schemas.mjs').StressTestResult}
 */
export function runStressTest(trades, opts = {}) {
    const scenario = opts.scenario || SCENARIOS.SIDEWAYS_48H;
    const symbol = opts.symbol || null;
    const startingEquity = opts.startingEquity || 50000;
    const rangePct = opts.sidewaysRangePct || THRESHOLDS.SIDEWAYS_RANGE_PCT;

    let filtered = trades;
    if (symbol) filtered = filtered.filter(t => t.symbol === symbol);

    if (filtered.length === 0) {
        return emptyStressResult(scenario, symbol || 'ALL', startingEquity);
    }

    // Calculate baseline metrics from actual trades
    const baseline = analyzeTradePatterns(filtered);

    // Project equity under the stress scenario
    const projection = simulateScenario(filtered, baseline, scenario, startingEquity, rangePct);

    // Determine regime vulnerability
    const regimeVuln = determineRegimeVulnerability(baseline, projection, scenario);

    // Determine recommended hedge
    const hedge = recommendHedge(regimeVuln, projection);

    // Status
    let status = AUDIT_STATUS.GREEN;
    if (projection.maxDD >= THRESHOLDS.MAX_DRAWDOWN_CRIT) {
        status = AUDIT_STATUS.RED;
    } else if (projection.maxDD >= THRESHOLDS.MAX_DRAWDOWN_WARN || projection.endEquity < startingEquity * 0.95) {
        status = AUDIT_STATUS.YELLOW;
    }

    return {
        test_id: generateId('stress'),
        timestamp: new Date().toISOString(),
        scenario,
        symbol: symbol || 'ALL',
        starting_equity: startingEquity,
        projected_ending_equity: +projection.endEquity.toFixed(2),
        projected_max_drawdown: +projection.maxDD.toFixed(4),
        projected_sharpe: +projection.sharpe.toFixed(4),
        regime_vulnerability: regimeVuln,
        recommended_hedge: hedge,
        status,
    };
}

function analyzeTradePatterns(trades) {
    const wins = trades.filter(t => t.pnl_usd > 0);
    const losses = trades.filter(t => t.pnl_usd < 0);

    const avgWin = wins.length > 0
        ? wins.reduce((s, t) => s + t.pnl_usd, 0) / wins.length
        : 0;
    const avgLoss = losses.length > 0
        ? Math.abs(losses.reduce((s, t) => s + t.pnl_usd, 0) / losses.length)
        : 0;

    const winRate = trades.length > 0 ? wins.length / trades.length : 0;

    // Detect if strategy is trend-following or mean-reversion
    // Trend-following: wins come from large directional moves
    // Mean-reversion: wins come from small reversals
    const avgHold = trades.reduce((s, t) => s + t.hold_sec, 0) / trades.length;

    // Calculate trade frequency (trades per hour)
    const durationHours = trades.length > 1
        ? (trades[trades.length - 1].exit_time - trades[0].entry_time) / 3600000
        : 1;
    const tradesPerHour = trades.length / Math.max(durationHours, 0.01);

    // Detect regime preference from P&L distribution relative to hold time
    const quickWins = wins.filter(t => t.hold_sec < 300); // sub-5min wins
    const slowWins = wins.filter(t => t.hold_sec >= 300);
    const quickWinRatio = wins.length > 0 ? quickWins.length / wins.length : 0;

    // Quick wins + short holds → likely mean-reversion
    // Long holds + trending P&L → likely trend-following
    const styleBias = quickWinRatio > 0.6 ? 'mean-reversion' : 'trend';

    return {
        avgWin,
        avgLoss,
        winRate,
        avgHold,
        tradesPerHour,
        styleBias,
        quickWinRatio,
    };
}

function simulateScenario(trades, baseline, scenario, startingEquity, rangePct) {
    switch (scenario) {
        case SCENARIOS.SIDEWAYS_48H:
            return simulateSideways(trades, baseline, startingEquity, rangePct);
        case SCENARIOS.VOL_SPIKE:
            return simulateVolSpike(trades, baseline, startingEquity);
        case SCENARIOS.TREND_BULL:
            return simulateTrend(trades, baseline, startingEquity, 'bull');
        case SCENARIOS.TREND_BEAR:
            return simulateTrend(trades, baseline, startingEquity, 'bear');
        default:
            return simulateSideways(trades, baseline, startingEquity, rangePct);
    }
}

/**
 * Simulate 48 hours of sideways (range-bound) market.
 *
 * In a sideways market:
 * - Trend-following strategies bleed: wins become small, losses stay same
 * - Mean-reversion strategies thrive: wins stay same, losses shrink
 * - Win rate drops for trend-followers, stays stable for mean-reversion
 */
function simulateSideways(trades, baseline, startingEquity, rangePct) {
    const hours = 48;
    const projectedTrades = Math.round(baseline.tradesPerHour * hours);

    // In sideways: price oscillates in a tight range
    // Trend-followers get chopped — win rate drops, avg win shrinks
    let adjWinRate, adjAvgWin, adjAvgLoss;

    if (baseline.styleBias === 'trend') {
        // Trend-following in sideways = death by a thousand cuts
        adjWinRate = baseline.winRate * 0.4;      // Win rate drops 60%
        adjAvgWin = baseline.avgWin * 0.3;          // Wins are 70% smaller
        adjAvgLoss = baseline.avgLoss * 0.9;         // Losses stay similar
    } else {
        // Mean-reversion in sideways = more opportunities
        adjWinRate = baseline.winRate * 1.1;         // Win rate up slightly
        adjAvgWin = baseline.avgWin * 0.8;           // Wins modestly smaller
        adjAvgLoss = baseline.avgLoss * 0.7;          // Losses shrink
    }

    const wins = Math.round(projectedTrades * adjWinRate);
    const losses = projectedTrades - wins;
    const totalPnl = (wins * adjAvgWin) - (losses * adjAvgLoss);
    const endEquity = startingEquity + totalPnl;

    // Project drawdown from loss streaks
    const maxConsecutiveLosses = Math.ceil(losses * 0.3); // worst 30% of losses consecutive
    const maxDD = (maxConsecutiveLosses * adjAvgLoss) / startingEquity;

    // Project Sharpe — lower in sideways
    const dailyReturn = totalPnl / 2 / startingEquity; // 2 days
    const projectedSharpe = dailyReturn > 0 ? dailyReturn * Math.sqrt(252) * 0.5 : dailyReturn * Math.sqrt(252);

    return { endEquity, maxDD, sharpe: projectedSharpe, projectedTrades };
}

/**
 * Simulate a volatility spike (2x normal ATR).
 */
function simulateVolSpike(trades, baseline, startingEquity) {
    const hours = 48;
    const projectedTrades = Math.round(baseline.tradesPerHour * hours * 1.5); // 50% more signals

    // In high vol: both wins and losses are larger
    const adjAvgWin = baseline.avgWin * 1.5;
    const adjAvgLoss = baseline.avgLoss * 1.8; // losses amplified more than wins (slippage)
    const adjWinRate = baseline.winRate * 0.85; // harder to predict

    const wins = Math.round(projectedTrades * adjWinRate);
    const losses = projectedTrades - wins;
    const totalPnl = (wins * adjAvgWin) - (losses * adjAvgLoss);
    const endEquity = startingEquity + totalPnl;

    const maxDD = (losses > 0 ? (losses * adjAvgLoss * 0.4) / startingEquity : 0);
    const dailyReturn = totalPnl / 2 / startingEquity;
    const projectedSharpe = dailyReturn * Math.sqrt(252) * 0.6;

    return { endEquity, maxDD, sharpe: projectedSharpe, projectedTrades };
}

/**
 * Simulate a trending market.
 */
function simulateTrend(trades, baseline, startingEquity, direction) {
    const hours = 48;
    const projectedTrades = Math.round(baseline.tradesPerHour * hours);

    const trendTailwind = direction === 'bull' ? 1.3 : 0.7;
    let adjWinRate, adjAvgWin;

    if (baseline.styleBias === 'trend') {
        adjWinRate = Math.min(baseline.winRate * 1.4, 0.75);
        adjAvgWin = baseline.avgWin * trendTailwind;
    } else {
        adjWinRate = baseline.winRate * 1.1;
        adjAvgWin = baseline.avgWin * trendTailwind;
    }

    const adjAvgLoss = baseline.avgLoss * (direction === 'bull' ? 0.8 : 1.2);
    const wins = Math.round(projectedTrades * adjWinRate);
    const losses = projectedTrades - wins;
    const totalPnl = (wins * adjAvgWin) - (losses * adjAvgLoss);
    const endEquity = startingEquity + totalPnl;

    const maxDD = losses > 0 ? (losses * adjAvgLoss * 0.3) / startingEquity : 0;
    const dailyReturn = totalPnl / 2 / startingEquity;
    const projectedSharpe = dailyReturn * Math.sqrt(252);

    return { endEquity, maxDD, sharpe: projectedSharpe, projectedTrades };
}

function determineRegimeVulnerability(baseline, projection, scenario) {
    if (scenario === SCENARIOS.SIDEWAYS_48H) {
        // If equity drops >5% in sideways, strategy needs trending markets
        if (projection.endEquity < projection.startingEquity * 0.95) {
            return 'trend';
        }
        // If equity stays flat or up in sideways, it may be mean-reversion
        if (projection.maxDD < 0.05) {
            return 'mean-reversion';
        }
        return 'neutral';
    }

    if (scenario === SCENARIOS.VOL_SPIKE) {
        if (projection.maxDD > 0.15) return 'trend'; // needs calm markets
    }

    return 'neutral';
}

function recommendHedge(regimeVuln, projection) {
    if (regimeVuln === 'trend') {
        return projection.maxDD > 0.10 ? 'FX' : 'Commodities';
    }
    if (regimeVuln === 'mean-reversion') {
        return 'Commodities';
    }
    return 'None';
}

function emptyStressResult(scenario, symbol, equity) {
    return {
        test_id: generateId('stress'),
        timestamp: new Date().toISOString(),
        scenario,
        symbol,
        starting_equity: equity,
        projected_ending_equity: equity,
        projected_max_drawdown: 0,
        projected_sharpe: 0,
        regime_vulnerability: 'neutral',
        recommended_hedge: 'None',
        status: AUDIT_STATUS.YELLOW,
    };
}

/**
 * Run all stress scenarios and return a comparison.
 */
export function runFullStressSuite(trades, opts = {}) {
    const startingEquity = opts.startingEquity || 50000;
    const symbol = opts.symbol || null;

    const scenarios = [
        SCENARIOS.SIDEWAYS_48H,
        SCENARIOS.VOL_SPIKE,
        SCENARIOS.TREND_BULL,
        SCENARIOS.TREND_BEAR,
    ];

    const results = scenarios.map(s => runStressTest(trades, { ...opts, scenario: s }));

    // Find worst scenario
    let worst = results[0];
    for (const r of results) {
        if (r.projected_max_drawdown > worst.projected_max_drawdown) worst = r;
    }

    console.log(`\n⚡ STRESS TEST SUITE — ${symbol || 'ALL'} (starting $${startingEquity.toLocaleString()})`);
    console.log('   Scenario          | End Equity | Max DD  | Sharpe | Status');
    console.log('   ------------------|------------|---------|--------|-------');
    for (const r of results) {
        const eqStr = `$${r.projected_ending_equity.toLocaleString()}`.padStart(10);
        const ddStr = `${(r.projected_max_drawdown * 100).toFixed(2)}%`.padStart(7);
        const shStr = r.projected_sharpe.toFixed(2).padStart(6);
        const stStr = r.status;
        console.log(`   ${r.scenario.padEnd(18)} | ${eqStr} | ${ddStr} | ${shStr} | ${stStr}`);
    }
    console.log(`\n   Worst case: ${worst.scenario} — ${(worst.projected_max_drawdown * 100).toFixed(2)}% DD`);
    console.log(`   Regime vulnerability: ${worst.regime_vulnerability}`);
    console.log(`   Recommended hedge: ${worst.recommended_hedge}\n`);

    return { results, worst };
}

export { SCENARIOS };
