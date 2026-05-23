/**
 * Grail Demo — full end-to-end trading system demonstration.
 *
 * Generates data → optimizes parameters → runs backtest → renders dashboard.
 * One command produces a complete trading system report with actionable parameters.
 *
 * Usage:
 *   node audit/grail.mjs --symbol BTC-USD --days 90
 *   node audit/grail.mjs --symbol MBT --days 30 --tf 5m --optimize
 *   node audit/grail.mjs --symbol SOL-USD --days 60 --full
 *
 * Modes:
 *   --quick     Fast demo: fixed params, no optimization (default)
 *   --optimize  Grid search + hill climb for best params
 *   --full      Full pipeline: optimize → backtest → per-regime → dashboard
 *
 * ES module. Zero npm dependencies.
 */

import { generateSyntheticCandles } from './trade.mjs';
import { optimize, optimizePerRegime, gridSearch } from './optimizer.mjs';
import { createOrchestrator, runBacktest } from './orchestrator.mjs';
import { renderDashboard, colorize } from './dashboard.mjs';
import { classifyRegime, REGIMES } from './market-regime.mjs';

// ─── ANSI ───────────────────────────────────────────────────────────────────────

const C = {
    reset: '\x1b[0m', bold: '\x1b[1m', dim: '\x1b[2m',
    red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m',
    blue: '\x1b[34m', magenta: '\x1b[35m', cyan: '\x1b[36m', white: '\x1b[37m',
    bgGreen: '\x1b[42m', bgRed: '\x1b[41m', bgYellow: '\x1b[43m',
};

// ─── Argument Parsing ───────────────────────────────────────────────────────────

function parseArgs(argv) {
    const args = {
        mode: 'quick',
        symbol: 'BTC-USD',
        days: 60,
        timeframe: '15m',
        seed: 42,
        volatility: 0.012,
        basePrice: 100,
        output: 'color',
    };

    for (let i = 0; i < argv.length; i++) {
        switch (argv[i]) {
            case '--quick': args.mode = 'quick'; break;
            case '--optimize': args.mode = 'optimize'; break;
            case '--full': args.mode = 'full'; break;
            case '--symbol': args.symbol = argv[++i]; break;
            case '--days': args.days = +argv[++i]; break;
            case '--tf':
            case '--timeframe': args.timeframe = argv[++i]; break;
            case '--seed': args.seed = +argv[++i]; break;
            case '--volatility': args.volatility = +argv[++i]; break;
            case '--base-price': args.basePrice = +argv[++i]; break;
            case '--output': args.output = argv[++i]; break;
            case '--help':
            case '-h':
                console.log(`
${C.bold}deepclaude Grail Demo${C.reset} — end-to-end trading system demonstration

${C.bold}MODES:${C.reset}
  ${C.cyan}--quick${C.reset}      Fast demo with fixed params (default)
  ${C.cyan}--optimize${C.reset}   Grid search + hill climbing for optimal params
  ${C.cyan}--full${C.reset}       Full pipeline: optimize → backtest → per-regime → dashboard

${C.bold}OPTIONS:${C.reset}
  ${C.yellow}--symbol${C.reset} <sym>      Trading symbol (default: BTC-USD)
  ${C.yellow}--days${C.reset} <n>          Days of synthetic data (default: 60)
  ${C.yellow}--tf${C.reset} <timeframe>    1m|5m|15m|1h|4h|1d (default: 15m)
  ${C.yellow}--seed${C.reset} <n>          PRNG seed for reproducibility (default: 42)
  ${C.yellow}--volatility${C.reset} <pct>  Volatility for synthetic data (default: 0.012)

${C.bold}EXAMPLES:${C.reset}
  ${C.green}node audit/grail.mjs --quick --symbol BTC-USD${C.reset}
  ${C.green}node audit/grail.mjs --optimize --symbol MBT --days 30 --tf 5m${C.reset}
  ${C.green}node audit/grail.mjs --full --symbol SOL-USD --days 90${C.reset}
`);
                process.exit(0);
        }
    }
    return args;
}

// ─── Timeframe Bars ─────────────────────────────────────────────────────────────

const TF_BARS = { '1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1 };

// ─── Report Sections ────────────────────────────────────────────────────────────

function printHeader(args) {
    console.log(`\n${C.bold}${C.cyan}╔══════════════════════════════════════════════════════════════╗${C.reset}`);
    console.log(`${C.bold}${C.cyan}║${C.reset}  ${C.bold}deepclaude GRAIL DEMO${C.reset}  │  ${C.white}${args.symbol.padEnd(20)}${C.reset}  ${C.dim}${new Date().toISOString().slice(0, 19)}${C.reset}  ${C.bold}${C.cyan}║${C.reset}`);
    console.log(`${C.bold}${C.cyan}╚══════════════════════════════════════════════════════════════╝${C.reset}\n`);
}

function printSeparator(title) {
    console.log(`\n${C.bold}${C.cyan}── ${title} ${C.dim}${'─'.repeat(56)}${C.reset}\n`);
}

function regimeBar(regimeCounts, total) {
    const colors = {
        trending_bullish: C.bgGreen,
        trending_bearish: C.bgRed,
        ranging: C.bgYellow,
        accumulation: C.cyan,
        distribution: C.magenta,
        breakout: C.green + C.bold,
        breakdown: C.red + C.bold,
        volatile: C.magenta + C.bold,
    };
    const labels = {
        trending_bullish: 'BULL', trending_bearish: 'BEAR', ranging: 'RANGE',
        accumulation: 'ACCUM', distribution: 'DIST', breakout: 'BRKOUT',
        breakdown: 'BRKDWN', volatile: 'VOL',
    };

    let bar = '';
    for (const [regime, count] of Object.entries(regimeCounts)) {
        const pct = count / total;
        const width = Math.max(1, Math.round(pct * 50));
        const color = colors[regime] ?? C.dim;
        const label = labels[regime] ?? regime.slice(0, 5);
        bar += `${color}${'█'.repeat(width)}${C.reset} ${label} ${(pct * 100).toFixed(0)}%  `;
    }
    return bar;
}

// ─── Quick Mode ─────────────────────────────────────────────────────────────────

function runQuick(args, candles) {
    // Run single-shot analysis
    const orch = createOrchestrator({ minConfidence: 0.15 });
    const decision = orch.run(candles, { symbol: args.symbol });

    // Run backtest with reasonable defaults
    const btResult = runBacktest(candles, {
        symbol: args.symbol,
        stopLossPct: 0.02,
        takeProfitPct: 0.04,
        cooldownBars: 10,
        warmupBars: 50,
        minConfidence: 0.15,
    });

    // Regime distribution
    const regimeCounts = {};
    for (const c of candles) {
        const r = c._regime ?? 'unknown';
        regimeCounts[r] = (regimeCounts[r] ?? 0) + 1;
    }

    // Output
    const { bold, dim, reset, green, red, yellow, white, cyan } = C;

    console.log(`${bold}Data Summary${reset}`);
    console.log(`  ${dim}Candles:${reset} ${candles.length}  ${dim}Timeframe:${reset} ${args.timeframe}  ${dim}Days:${reset} ${args.days}  ${dim}Seed:${reset} ${args.seed}`);
    if (candles.length > 0) {
        console.log(`  ${dim}First:${reset} ${candles[0].close.toFixed(4)}  ${dim}Last:${reset} ${candles[candles.length - 1].close.toFixed(4)}  ${dim}Vol:${reset} ${(args.volatility * 100).toFixed(1)}%`);
    }
    if (candles.length > 0) {
        console.log(`  ${dim}Regime Mix:${reset} ${regimeBar(regimeCounts, candles.length)}`);
    }

    printSeparator('Live Analysis');
    console.log(`  ${dim}Price:${reset}  ${white}${decision.price?.toFixed(4) ?? 'N/A'}${reset}`);
    console.log(`  ${dim}Action:${reset} ${decision.action === 'BUY' ? green + 'BUY ▲' : decision.action === 'SELL' ? red + 'SELL ▼' : yellow + 'HOLD ―'}${reset}  ${dim}Confidence:${reset} ${(decision.confidence * 100).toFixed(1)}%  ${dim}Score:${reset} ${decision.compositeScore.toFixed(3)}`);
    console.log(`  ${dim}Regime:${reset} ${decision.regime} (${(decision.regimeConfidence * 100).toFixed(0)}%)  ${dim}Direction:${reset} ${decision.direction ?? 'none'}`);
    console.log(`  ${dim}Support:${reset} ${decision.support ? decision.support.price.toFixed(4) + ' (str: ' + decision.support.strength.toFixed(2) + ')' : 'none'}`);
    console.log(`  ${dim}Resistance:${reset} ${decision.resistance ? decision.resistance.price.toFixed(4) + ' (str: ' + decision.resistance.strength.toFixed(2) + ')' : 'none'}`);

    printSeparator('Backtest Results');
    const { stats } = btResult;
    console.log(`  ${dim}Trades:${reset}      ${stats.totalTrades} (${stats.winningTrades}W / ${stats.losingTrades}L)  ${dim}Win Rate:${reset} ${(stats.winRate * 100).toFixed(1)}%`);
    console.log(`  ${dim}Total PnL:${reset}    ${stats.totalPnl >= 0 ? green : red}${(stats.totalPnl * 100).toFixed(3)}%${reset}  ${dim}Profit Factor:${reset} ${stats.profitFactor.toFixed(2)}`);
    console.log(`  ${dim}Avg Win:${reset}      ${green}+${(stats.avgWin * 100).toFixed(3)}%${reset}  ${dim}Avg Loss:${reset} ${red}${(stats.avgLoss * 100).toFixed(3)}%${reset}`);

    if (Object.keys(stats.perRegime).length > 0) {
        console.log(`\n  ${bold}Per-Regime:${reset}`);
        for (const [regime, r] of Object.entries(stats.perRegime).sort((a, b) => b[1].count - a[1].count)) {
            const wr = (r.winRate * 100).toFixed(0);
            const wrColor = r.winRate >= 0.5 ? green : red;
            const pnlColor = r.totalPnl >= 0 ? green : red;
            console.log(`    ${regime.padEnd(20)} ${String(r.count).padStart(3)} trades  ${wrColor}${wr.padStart(3)}% WR${reset}  ${pnlColor}${(r.totalPnl * 100).toFixed(2).padStart(8)}% PnL${reset}`);
        }
    }

    if (btResult.trades.length > 0) {
        printSeparator('Recent Trades');
        console.log(`  ${dim}ENTRY  EXIT   DIR    PNL%      REASON      REGIME${reset}`);
        for (const t of btResult.trades.slice(-5)) {
            const dir = t.direction === 'long' ? green + 'LONG ' : red + 'SHORT';
            const pnl = t.pnl > 0 ? green + '+' + (t.pnl * 100).toFixed(3) + '%' : red + (t.pnl * 100).toFixed(3) + '%';
            console.log(`  ${String(t.entryBar).padEnd(5)} ${String(t.exitBar).padEnd(5)} ${dir}${reset}  ${pnl}${reset}  ${dim}${(t.exitReason ?? '?').padEnd(10)}${reset}  ${t.regime ?? 'unknown'}`);
        }
    }

    printSeparator('Signals');
    for (const s of decision.signals) {
        const sign = s.value >= 0 ? green + '+' : red;
        console.log(`  ${cyan}${s.source}/${s.name}${reset} → ${sign}${s.value.toFixed(3)}${reset} ${dim}(conf: ${s.confidence.toFixed(2)})${reset}`);
    }

    if (decision.params && Object.keys(decision.params).length > 0) {
        console.log(`\n  ${bold}Regime Params:${reset}`);
        for (const [k, v] of Object.entries(decision.params)) {
            console.log(`    ${dim}${k}:${reset} ${typeof v === 'number' ? v.toFixed(3) : v}`);
        }
    }

    console.log('');
}

// ─── Optimize Mode ──────────────────────────────────────────────────────────────

function runOptimize(args, candles) {
    const { bold, dim, reset, green, red, yellow, white, cyan } = C;

    printSeparator('Phase 1: Grid Search');
    const result = optimize(candles, {
        symbol: args.symbol,
        objective: 'profitFactor',
        coarseSpace: {
            stopLossPct: [0.01, 0.02, 0.03, 0.05],
            takeProfitPct: [0.02, 0.04, 0.06, 0.10],
            cooldownBars: [5, 10, 20],
            warmupBars: [50],
            minConfidence: [0.1, 0.15, 0.2],
            fusionMethod: ['weighted'],
        },
        maxIterations: 10,
        neighborsPerIteration: 6,
        verbose: true,
    });

    console.log(`\n  ${bold}Grid Best:${reset} ${white}${result.objective}=${result.bestScore.toFixed(4)}${reset}`);
    console.log(`  ${dim}Params:${reset} SL=${(result.bestParams.stopLossPct * 100).toFixed(1)}% TP=${(result.bestParams.takeProfitPct * 100).toFixed(1)}% cooldown=${result.bestParams.cooldownBars} minConf=${result.bestParams.minConfidence}`);
    console.log(`  ${dim}Trades:${reset} ${result.bestStats.totalTrades} WR=${(result.bestStats.winRate * 100).toFixed(1)}% PF=${result.bestStats.profitFactor.toFixed(2)} PnL=${(result.bestStats.totalPnl * 100).toFixed(3)}%`);

    printSeparator('Top 5 Parameter Sets');
    const topN = result.gridResult?.topN ?? result.gridResult?.allResults?.slice(0, 5) ?? [];
    for (let i = 0; i < Math.min(5, topN.length); i++) {
        const r = topN[i];
        const marker = i === 0 ? ` ${yellow}★${reset}` : '  ';
        console.log(`${marker} ${dim}#${i + 1}${reset} SL=${(r.params.stopLossPct * 100).toFixed(1)}% TP=${(r.params.takeProfitPct * 100).toFixed(1)}% cool=${r.params.cooldownBars} conf=${r.params.minConfidence} → ${white}score=${r.score.toFixed(4)}${reset} ${dim}(WR=${(r.stats.winRate * 100).toFixed(0)}% PnL=${(r.stats.totalPnl * 100).toFixed(3)}%)${reset}`);
    }

    printSeparator('Phase 2: Hill Climb');
    console.log(`  ${dim}Starting from grid best, climbing...${reset}`);
    console.log(`  ${white}Final score:${reset} ${result.bestScore.toFixed(4)} ${dim}(improvement: ${((result.hillResult.bestScore - result.gridResult.bestScore) / Math.max(Math.abs(result.gridResult.bestScore), 0.001) * 100).toFixed(1)}%)${reset}`);

    // Per-regime optimization
    printSeparator('Per-Regime Optimization');
    const perRegime = optimizePerRegime(candles, {
        symbol: args.symbol,
        objective: 'profitFactor',
        coarseSpace: {
            stopLossPct: [0.02, 0.04],
            takeProfitPct: [0.04, 0.08],
            cooldownBars: [10],
            warmupBars: [50],
            minConfidence: [0.15],
            fusionMethod: ['weighted'],
        },
        maxIterations: 5,
        neighborsPerIteration: 3,
        verbose: false,
    });

    for (const [regime, r] of Object.entries(perRegime).sort((a, b) => b[1].bestScore - a[1].bestScore)) {
        const pfColor = r.bestScore >= 1.5 ? green : r.bestScore >= 1.0 ? yellow : red;
        console.log(`  ${regime.padEnd(20)} SL=${(r.bestParams.stopLossPct * 100).toFixed(1)}% TP=${(r.bestParams.takeProfitPct * 100).toFixed(1)}% → ${pfColor}PF=${r.bestScore.toFixed(2)}${reset} ${dim}(WR=${(r.bestStats.winRate * 100).toFixed(0)}% Trades=${r.bestStats.totalTrades})${reset}`);
    }

    console.log(`\n${bold}${green}✓ Optimization complete.${reset} Use --full for complete pipeline with dashboard.\n`);
}

// ─── Full Mode ──────────────────────────────────────────────────────────────────

function runFull(args, candles) {
    const { bold, dim, reset, green, red, yellow, white, cyan } = C;

    // Optimize
    console.log(`${dim}Running optimizer...${reset}`);
    const optResult = optimize(candles, {
        symbol: args.symbol,
        objective: 'profitFactor',
        coarseSpace: {
            stopLossPct: [0.01, 0.02, 0.03, 0.05],
            takeProfitPct: [0.02, 0.04, 0.06, 0.10],
            cooldownBars: [5, 10, 20],
            warmupBars: [50],
            minConfidence: [0.1, 0.15, 0.2],
            fusionMethod: ['weighted'],
        },
        maxIterations: 10,
        neighborsPerIteration: 6,
        verbose: true,
    });

    // Run best backtest
    console.log(`\n${dim}Running final backtest with best params...${reset}`);
    const btResult = runBacktest(candles, {
        symbol: args.symbol,
        ...optResult.bestParams,
    });

    printHeader(args);

    // Dashboard
    const dashboardData = {
        symbol: args.symbol,
        price: candles[candles.length - 1].close,
        action: btResult.stats.totalPnl > 0 ? 'BUY' : btResult.stats.winRate > 0.4 ? 'BUY' : 'HOLD',
        confidence: Math.min(btResult.stats.winRate, 1),
        compositeScore: btResult.stats.totalPnl > 0 ? 0.5 : btResult.stats.totalPnl < 0 ? -0.3 : 0,
        regime: classifyRegime(candles).regime,
        regimeConfidence: classifyRegime(candles).confidence,
        support: null,
        resistance: null,
        activeZones: [],
        signals: [
            { source: 'backtest', name: 'win_rate', value: btResult.stats.winRate, confidence: 0.8 },
            { source: 'backtest', name: 'profit_factor', value: Math.min(btResult.stats.profitFactor / 3, 1), confidence: 0.7 },
            { source: 'optimizer', name: 'optimized_params', value: btResult.stats.totalPnl > 0 ? 0.6 : 0, confidence: 0.6 },
        ],
        params: optResult.bestParams,
    };

    const dashboard = renderDashboard(dashboardData, { width: 61 });
    console.log(colorize(dashboard));

    // Performance summary
    printSeparator('Optimized Parameters');
    const bp = optResult.bestParams;
    console.log(`  SL=${(bp.stopLossPct * 100).toFixed(1)}%  TP=${(bp.takeProfitPct * 100).toFixed(1)}%  Cooldown=${bp.cooldownBars}  Warmup=${bp.warmupBars}  MinConf=${bp.minConfidence}  Fusion=${bp.fusionMethod}`);

    printSeparator('Backtest Performance');
    const { stats } = btResult;
    console.log(`  ${bold}Trades:${reset} ${stats.totalTrades} (${stats.winningTrades}W / ${stats.losingTrades}L)  ${bold}Win Rate:${reset} ${(stats.winRate * 100).toFixed(1)}%`);
    console.log(`  ${bold}Total PnL:${reset} ${stats.totalPnl >= 0 ? green : red}${(stats.totalPnl * 100).toFixed(3)}%${reset}  ${bold}Profit Factor:${reset} ${stats.profitFactor.toFixed(2)}`);
    console.log(`  ${bold}Avg Win:${reset} ${green}+${(stats.avgWin * 100).toFixed(3)}%${reset}  ${bold}Avg Loss:${reset} ${red}${(stats.avgLoss * 100).toFixed(3)}%${reset}`);

    if (Object.keys(stats.perRegime).length > 0) {
        console.log(`\n  ${bold}Per-Regime Breakdown:${reset}`);
        for (const [regime, r] of Object.entries(stats.perRegime).sort((a, b) => b[1].count - a[1].count)) {
            const pf = r.count > 0 ? (r.avgPnl > 0 ? r.avgPnl / Math.abs(r.avgPnl) * r.winRate : 0) : 0;
            const pfColor = pf >= 1.5 ? green : pf >= 1 ? yellow : red;
            console.log(`    ${regime.padEnd(20)} ${String(r.count).padStart(3)} trades  WR=${(r.winRate * 100).toFixed(0)}%  Avg=${(r.avgPnl * 100).toFixed(3)}%  Total=${(r.totalPnl * 100).toFixed(2)}%`);
        }
    }

    // Verdict
    printSeparator('Verdict');
    const isProfitable = btResult.stats.profitFactor >= 1.3 && btResult.stats.winRate >= 0.4;
    const isGrail = btResult.stats.profitFactor >= 2.0 && btResult.stats.winRate >= 0.5 && btResult.stats.totalTrades >= 10;

    if (isGrail) {
        console.log(`  ${bold}${green}★★★ GRAIL STATUS: VIABLE ★★★${reset}`);
        console.log(`  ${green}This parameter set meets the grail threshold:${reset}`);
        console.log(`  ${green}• Profit Factor ≥ 2.0${reset}`);
        console.log(`  ${green}• Win Rate ≥ 50%${reset}`);
        console.log(`  ${green}• Sufficient trade volume${reset}`);
        console.log(`\n  ${white}Ready for paper trading on Topstep.${reset}`);
    } else if (isProfitable) {
        console.log(`  ${yellow}★ PROFITABLE — needs tuning for grail status${reset}`);
        console.log(`  ${dim}Consider: more data, tighter spreads, or regime-specific params.${reset}`);
    } else {
        console.log(`  ${red}✗ NOT YIELDING — adjust volatility, timeframe, or symbol${reset}`);
        console.log(`  ${dim}Try: --volatility 0.02 --days 120 --tf 5m for more action.${reset}`);
    }

    console.log('');
}

// ─── Main ───────────────────────────────────────────────────────────────────────

function main() {
    const args = parseArgs(process.argv.slice(2));

    if (!TF_BARS[args.timeframe]) {
        console.error(`Error: unknown timeframe '${args.timeframe}'`);
        process.exit(1);
    }

    const barCount = args.days * TF_BARS[args.timeframe];
    console.error(`Generating ${barCount} ${args.timeframe} candles for ${args.symbol} (${args.days} days, seed=${args.seed})...`);

    const candles = generateSyntheticCandles(barCount, {
        seed: args.seed,
        basePrice: args.basePrice,
        volatility: args.volatility,
    });

    printHeader(args);

    switch (args.mode) {
        case 'quick':
            runQuick(args, candles);
            break;
        case 'optimize':
            runOptimize(args, candles);
            break;
        case 'full':
            runFull(args, candles);
            break;
        default:
            console.error(`Unknown mode: ${args.mode}`);
            process.exit(1);
    }
}

// Run if called directly
if (process.argv[1]?.endsWith('grail.mjs')) {
    main();
}

export { runQuick, runOptimize, runFull };
