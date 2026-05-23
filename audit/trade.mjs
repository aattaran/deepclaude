/**
 * Unified Trade CLI — single-command pipeline runner
 *
 * Usage:
 *   node audit/trade.mjs --analyze --symbol SOL-USD --candles data.json
 *   node audit/trade.mjs --backtest --symbol SOL-USD --days 30
 *   node audit/trade.mjs --live --symbol SOL-USD
 *
 * Wires the full stack: ZoneDetector → Market Regime → Signal Fusion → Order Book → Decision
 *
 * ES module. Zero npm dependencies. Node built-ins only.
 */

import { readFileSync } from 'node:fs';
import { createOrchestrator, runBacktest } from './orchestrator.mjs';
import { classifyRegime, REGIMES } from './market-regime.mjs';
import { ZoneDetector } from './zone-detector.mjs';

// ─── Constants ──────────────────────────────────────────────────────────────────

const TIMEFRAMES = {
    '1m': { barsPerDay: 1440, label: '1-minute' },
    '5m': { barsPerDay: 288, label: '5-minute' },
    '15m': { barsPerDay: 96, label: '15-minute' },
    '1h': { barsPerDay: 24, label: '1-hour' },
    '4h': { barsPerDay: 6, label: '4-hour' },
    '1d': { barsPerDay: 1, label: 'daily' },
};

const ANSI = {
    reset: '\x1b[0m',
    bold: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
    bgRed: '\x1b[41m',
    bgGreen: '\x1b[42m',
    bgYellow: '\x1b[43m',
};

// ─── Argument Parsing ───────────────────────────────────────────────────────────

function parseArgs(argv) {
    const args = {
        mode: null,
        symbol: 'BTC-USD',
        candlesPath: null,
        days: 30,
        timeframe: '15m',
        output: 'color',
        seed: null,
        warmup: 50,
        cooldown: 10,
        stopLoss: 0.02,
        takeProfit: 0.04,
        basePrice: 100,
        volatility: 0.01,
        maxHold: 100,
        fusionMethod: 'weighted',
        minConfidence: 0.3,
        zoneThreshold: 0.015,
    };

    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        switch (arg) {
            case '--analyze': args.mode = 'analyze'; break;
            case '--backtest': args.mode = 'backtest'; break;
            case '--live': args.mode = 'live'; break;
            case '--help':
            case '-h':
                printHelp();
                process.exit(0);
            case '--symbol': args.symbol = argv[++i]; break;
            case '--candles': args.candlesPath = argv[++i]; break;
            case '--days': args.days = +argv[++i]; break;
            case '--tf':
            case '--timeframe': args.timeframe = argv[++i]; break;
            case '--output': args.output = argv[++i]; break;
            case '--seed': args.seed = +argv[++i]; break;
            case '--warmup': args.warmup = +argv[++i]; break;
            case '--cooldown': args.cooldown = +argv[++i]; break;
            case '--sl': args.stopLoss = +argv[++i]; break;
            case '--tp': args.takeProfit = +argv[++i]; break;
            case '--base-price': args.basePrice = +argv[++i]; break;
            case '--volatility': args.volatility = +argv[++i]; break;
            case '--max-hold': args.maxHold = +argv[++i]; break;
            case '--fusion': args.fusionMethod = argv[++i]; break;
            case '--min-confidence': args.minConfidence = +argv[++i]; break;
            case '--zone-threshold': args.zoneThreshold = +argv[++i]; break;
        }
    }

    return args;
}

function printHelp() {
    const { bold, dim, reset, cyan, yellow, green } = ANSI;
    console.log(`
${bold}deepclaude Trade CLI${reset} — unified trading pipeline runner
${dim}Wires ZoneDetector → Market Regime → Signal Fusion → Order Book → Decision${reset}

${bold}MODES:${reset}
  ${cyan}--analyze${reset}     Single-shot analysis of candle data
  ${cyan}--backtest${reset}    Walk-forward backtest with SL/TP/cooldown
  ${cyan}--live${reset}        Continuous streaming simulation

${bold}OPTIONS:${reset}
  ${yellow}--symbol${reset} <sym>      Trading symbol (default: BTC-USD)
  ${yellow}--candles${reset} <file>    Path to JSON candle file
  ${yellow}--days${reset} <n>          Days of synthetic data for backtest (default: 30)
  ${yellow}--tf${reset} <timeframe>    1m|5m|15m|1h|4h|1d (default: 15m)
  ${yellow}--output${reset} <fmt>      json|table|color (default: color)
  ${yellow}--seed${reset} <n>          PRNG seed for reproducible backtests
  ${yellow}--warmup${reset} <n>        Warmup bars before first trade (default: 50)
  ${yellow}--cooldown${reset} <n>      Cooldown bars between trades (default: 10)
  ${yellow}--sl${reset} <pct>          Stop loss % (default: 0.02)
  ${yellow}--tp${reset} <pct>          Take profit % (default: 0.04)
  ${yellow}--max-hold${reset} <n>      Max holding bars (default: 100)
  ${yellow}--base-price${reset} <n>    Starting price for synthetic data (default: 100)
  ${yellow}--volatility${reset} <pct>  Volatility for synthetic data (default: 0.01)
  ${yellow}--fusion${reset} <method>   weighted|bayesian|voting (default: weighted)
  ${yellow}--min-confidence${reset} <n> Minimum confidence threshold (default: 0.3)

${bold}EXAMPLES:${reset}
  ${green}node audit/trade.mjs --analyze --symbol SOL-USD --candles data.json${reset}
  ${green}node audit/trade.mjs --backtest --symbol BTC-USD --days 60 --seed 42${reset}
  ${green}node audit/trade.mjs --live --symbol MBT --tf 5m --volatility 0.008${reset}
`);
}

// ─── Data Generation ────────────────────────────────────────────────────────────

function mulberry32(seed) {
    return function () {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

/**
 * Generate synthetic candles with a realistic regime mix:
 * trending (bull/bear), ranging/chop, breakouts, and volatile spikes.
 */
function generateSyntheticCandles(count, options = {}) {
    const rng = mulberry32(options.seed ?? Math.floor(Math.random() * 2147483647));
    const basePrice = options.basePrice ?? 100;
    const volatility = options.volatility ?? 0.01;
    const startTs = options.startTs ?? Date.now() - count * 60000;

    const candles = [];
    let price = basePrice;

    // Regime switching: cycle through trending → ranging → breakout → volatile
    const regimeLength = Math.max(20, Math.floor(count / 5));
    let regime = 'trending_bullish';
    let regimeBars = 0;
    let trendDir = 1;

    for (let i = 0; i < count; i++) {
        // Switch regime periodically
        if (regimeBars >= regimeLength) {
            regimeBars = 0;
            const roll = rng();
            if (roll < 0.35) {
                trendDir = trendDir > 0 ? -1 : 1;
                regime = trendDir > 0 ? 'trending_bullish' : 'trending_bearish';
            } else if (roll < 0.6) {
                regime = 'ranging';
            } else if (roll < 0.8) {
                regime = trendDir > 0 ? 'breakout' : 'breakdown';
            } else {
                regime = 'volatile';
            }
        }
        regimeBars++;

        let drift;
        const volFactor = regime === 'volatile' ? 2.5 : regime === 'breakout' || regime === 'breakdown' ? 1.8 : 1.0;

        switch (regime) {
            case 'trending_bullish':
                drift = volatility * 0.15 + rng() * volatility * 0.2;
                break;
            case 'trending_bearish':
                drift = -volatility * 0.15 - rng() * volatility * 0.2;
                break;
            case 'ranging':
                drift = (rng() - 0.5) * volatility * 0.3;
                break;
            case 'breakout':
                drift = volatility * 0.4 + rng() * volatility * 0.3;
                break;
            case 'breakdown':
                drift = -volatility * 0.4 - rng() * volatility * 0.3;
                break;
            case 'volatile':
                drift = (rng() - 0.5) * volatility * 2;
                break;
            default:
                drift = (rng() - 0.5) * volatility * 0.3;
        }

        const wickFactor = regime === 'volatile' ? 0.6 : 0.3;
        const open = price;
        const close = open + drift * basePrice * volFactor;
        const high = Math.max(open, close) + rng() * volatility * wickFactor * basePrice * volFactor;
        const low = Math.min(open, close) - rng() * volatility * wickFactor * basePrice * volFactor;
        const volume = basePrice * 10 + rng() * basePrice * 5 * volFactor;

        candles.push({
            timestamp: startTs + i * 60000,
            open: +open.toFixed(4),
            high: +high.toFixed(4),
            low: +low.toFixed(4),
            close: +close.toFixed(4),
            volume: +volume.toFixed(2),
            _regime: regime,
        });

        price = close;
    }

    return candles;
}

// ─── Load Candles ──────────────────────────────────────────────────────────────

function loadCandles(path) {
    let raw;
    try {
        raw = readFileSync(path, 'utf-8');
    } catch (err) {
        if (err.code === 'ENOENT') {
            throw new Error(`file not found: ${path}`);
        }
        throw new Error(`error reading ${path}: ${err.message}`);
    }

    let data;
    try {
        data = JSON.parse(raw);
    } catch (err) {
        throw new Error(`invalid JSON in ${path}: ${err.message}`);
    }

    // Support both array-of-candles and { candles: [...] } shapes
    const candles = Array.isArray(data) ? data : (data.candles ?? data.data ?? data.ohlcv ?? null);
    if (!candles || !Array.isArray(candles) || candles.length === 0) {
        throw new Error(`no valid candle array found in ${path}`);
    }

    // Validate minimal candle shape
    const sample = candles[0];
    if (typeof sample.close !== 'number') {
        // Try to normalize: { o, h, l, c, v } → { open, high, low, close, volume }
        if (typeof sample.c === 'number') {
            return candles.map(c => ({
                timestamp: c.t ?? c.timestamp ?? c.time ?? 0,
                open: c.o ?? c.open,
                high: c.h ?? c.high,
                low: c.l ?? c.low,
                close: c.c ?? c.close,
                volume: c.v ?? c.volume ?? 0,
            }));
        }
        throw new Error(`candles in ${path} must have 'close' or 'c' fields`);
    }

    return candles;
}

// ─── Formatters ─────────────────────────────────────────────────────────────────

function colorAction(action) {
    switch (action) {
        case 'BUY': return `${ANSI.bgGreen}${ANSI.bold} BUY  ${ANSI.reset}`;
        case 'SELL': return `${ANSI.bgRed}${ANSI.bold} SELL ${ANSI.reset}`;
        case 'HOLD': return `${ANSI.bgYellow}${ANSI.bold} HOLD ${ANSI.reset}`;
        default: return action;
    }
}

function colorPnl(pnl) {
    if (pnl > 0) return `${ANSI.green}+${pnl.toFixed(2)}%${ANSI.reset}`;
    if (pnl < 0) return `${ANSI.red}${pnl.toFixed(2)}%${ANSI.reset}`;
    return `${ANSI.dim}0.00%${ANSI.reset}`;
}

function colorRegime(regime) {
    const map = {
        trending_bullish: ANSI.green,
        trending_bearish: ANSI.red,
        ranging: ANSI.yellow,
        accumulation: ANSI.cyan,
        distribution: ANSI.magenta,
        breakout: ANSI.bold + ANSI.green,
        breakdown: ANSI.bold + ANSI.red,
        volatile: ANSI.bold + ANSI.magenta,
    };
    const c = map[regime] ?? ANSI.dim;
    return `${c}${regime}${ANSI.reset}`;
}

// ─── Analyze Mode ───────────────────────────────────────────────────────────────

function runAnalyze(candles, args) {
    const orch = createOrchestrator({
        zoneThreshold: args.zoneThreshold,
        fusionMethod: args.fusionMethod,
        minConfidence: args.minConfidence,
    });

    const decision = orch.run(candles, { symbol: args.symbol });

    if (args.output === 'json') {
        console.log(JSON.stringify(decision, null, 2));
        return;
    }

    // Color terminal output
    const { bold, dim, reset, cyan, blue, white, green, red } = ANSI;

    console.log(`\n${bold}╔══════════════════════════════════════════════╗${reset}`);
    console.log(`${bold}║  deepclaude ANALYZE  │  ${args.symbol.padEnd(14)} ║${reset}`);
    console.log(`${bold}╚══════════════════════════════════════════════╝${reset}\n`);

    console.log(`${dim}Price:${reset} ${white}${decision.price?.toFixed(4) ?? 'N/A'}${reset}  ${dim}Regime:${reset} ${colorRegime(decision.regime)} (${(decision.regimeConfidence * 100).toFixed(0)}%)`);
    console.log(`${dim}Action:${reset} ${colorAction(decision.action)}  ${dim}Confidence:${reset} ${(decision.confidence * 100).toFixed(1)}%  ${dim}Score:${reset} ${decision.compositeScore.toFixed(3)}`);
    console.log(`${dim}Direction:${reset} ${decision.direction ?? 'none'}  ${dim}Reasoning:${reset} ${decision.reasoning}`);

    if (decision.support) {
        console.log(`${dim}Support:${reset} ${green}${decision.support.price.toFixed(4)}${reset} (strength: ${decision.support.strength.toFixed(2)})`);
    }
    if (decision.resistance) {
        console.log(`${dim}Resistance:${reset} ${red}${decision.resistance.price.toFixed(4)}${reset} (strength: ${decision.resistance.strength.toFixed(2)})`);
    }

    console.log(`\n${bold}Active Zones:${reset}`);
    for (const z of decision.activeZones) {
        const typeColor = z.type === 'support' ? ANSI.green : z.type === 'resistance' ? ANSI.red : ANSI.cyan;
        console.log(`  ${typeColor}${z.price.toFixed(4)}${reset} ${dim}${z.type}${reset} strength=${z.strength.toFixed(2)} fresh=${z.freshness?.toFixed(1) ?? 'N/A'}`);
    }

    console.log(`\n${bold}Signals:${reset}`);
    for (const s of decision.signals) {
        const sign = s.value >= 0 ? ANSI.green + '+' : ANSI.red;
        console.log(`  ${cyan}${s.source}/${s.name}${reset} → ${sign}${s.value.toFixed(3)}${reset} (conf: ${s.confidence.toFixed(2)})`);
    }

    if (decision.params && Object.keys(decision.params).length > 0) {
        console.log(`\n${bold}Regime Params:${reset}`);
        for (const [k, v] of Object.entries(decision.params)) {
            console.log(`  ${dim}${k}:${reset} ${typeof v === 'number' ? v.toFixed(3) : v}`);
        }
    }

    console.log('');
}

// ─── Backtest Mode ──────────────────────────────────────────────────────────────

function runBacktestMode(candles, args) {
    const result = runBacktest(candles, {
        symbol: args.symbol,
        warmupBars: args.warmup,
        cooldownBars: args.cooldown,
        maxHoldingBars: args.maxHold,
        stopLossPct: args.stopLoss,
        takeProfitPct: args.takeProfit,
        zoneThreshold: args.zoneThreshold,
        fusionMethod: args.fusionMethod,
        minConfidence: args.minConfidence,
    });

    if (args.output === 'json') {
        console.log(JSON.stringify(result, null, 2));
        return;
    }

    const { stats } = result;
    const { bold, dim, reset, cyan, green, red, yellow, white, magenta } = ANSI;

    console.log(`\n${bold}╔══════════════════════════════════════════════╗${reset}`);
    console.log(`${bold}║  deepclaude BACKTEST  │  ${args.symbol.padEnd(14)} ║${reset}`);
    console.log(`${bold}╚══════════════════════════════════════════════╝${reset}\n`);

    console.log(`${dim}Candles:${reset} ${candles.length}  ${dim}Warmup:${reset} ${args.warmup}  ${dim}Cooldown:${reset} ${args.cooldown}`);
    console.log(`${dim}SL:${reset} ${(args.stopLoss * 100).toFixed(1)}%  ${dim}TP:${reset} ${(args.takeProfit * 100).toFixed(1)}%  ${dim}Max Hold:${reset} ${args.maxHold}b\n`);

    // Summary card
    const winRatePct = (stats.winRate * 100).toFixed(1);
    const winRateColor = stats.winRate >= 0.5 ? green : stats.winRate >= 0.4 ? yellow : red;

    console.log(`${bold}── Performance ──${reset}`);
    console.log(`  ${dim}Trades:${reset}      ${stats.totalTrades} (${stats.winningTrades}W / ${stats.losingTrades}L)`);
    console.log(`  ${dim}Win Rate:${reset}     ${winRateColor}${winRatePct}%${reset}`);
    console.log(`  ${dim}Total PnL:${reset}    ${stats.totalPnl >= 0 ? green : red}${(stats.totalPnl * 100).toFixed(3)}%${reset}`);
    console.log(`  ${dim}Avg Win:${reset}      ${green}+${(stats.avgWin * 100).toFixed(3)}%${reset}`);
    console.log(`  ${dim}Avg Loss:${reset}     ${red}${(stats.avgLoss * 100).toFixed(3)}%${reset}`);
    console.log(`  ${dim}Profit Factor:${reset} ${stats.profitFactor >= 1.5 ? green : stats.profitFactor >= 1 ? yellow : red}${stats.profitFactor}${reset}`);

    // Per-regime breakdown
    const regimes = Object.entries(stats.perRegime);
    if (regimes.length > 0) {
        console.log(`\n${bold}── Per-Regime Breakdown ──${reset}`);
        console.log(`  ${dim}${'REGIME'.padEnd(22)} TRADES   WIN%     AVG PNL     TOTAL${reset}`);
        for (const [regime, r] of regimes.sort((a, b) => b[1].count - a[1].count)) {
            const wr = (r.winRate * 100).toFixed(0);
            const wp = wr >= 50 ? green : red;
            const totalColor = r.totalPnl >= 0 ? green : red;
            console.log(`  ${colorRegime(regime).padEnd(42)} ${String(r.count).padEnd(8)} ${wp}${wr.padEnd(6)}%${reset}   ${(r.avgPnl * 100).toFixed(3).padStart(8)}%  ${totalColor}${(r.totalPnl * 100).toFixed(2).padStart(8)}%${reset}`);
        }
    }

    // Trade log (last 10)
    if (result.trades.length > 0) {
        console.log(`\n${bold}── Recent Trades (last ${Math.min(10, result.trades.length)}) ──${reset}`);
        console.log(`  ${dim}ENTRY  EXIT   DIR    PNL%      REASON      REGIME${reset}`);
        const recent = result.trades.slice(-10);
        for (const t of recent) {
            const dir = t.direction === 'long' ? green + 'LONG' : red + 'SHORT';
            const pnlStr = t.pnl > 0 ? green + '+' + (t.pnl * 100).toFixed(3) + '%' : red + (t.pnl * 100).toFixed(3) + '%';
            console.log(`  ${String(t.entryBar).padEnd(5)} ${String(t.exitBar).padEnd(5)} ${dir}${reset}  ${pnlStr}${reset}  ${dim}${(t.exitReason ?? '?').padEnd(10)}${reset}  ${colorRegime(t.regime ?? 'unknown')}`);
        }
    }

    console.log('');
}

// ─── Live Mode ──────────────────────────────────────────────────────────────────

let liveRunning = false;

function runLive(args) {
    liveRunning = true;
    const orch = createOrchestrator({
        zoneThreshold: args.zoneThreshold,
        fusionMethod: args.fusionMethod,
        minConfidence: args.minConfidence,
    });

    const tf = TIMEFRAMES[args.timeframe] ?? TIMEFRAMES['15m'];
    const barsPerDay = tf.barsPerDay;
    const barMs = Math.floor((24 * 60 * 60 * 1000) / barsPerDay);

    console.log(`${ANSI.bold}╔══════════════════════════════════════════════╗${ANSI.reset}`);
    console.log(`${ANSI.bold}║  deepclaude LIVE  │  ${args.symbol.padEnd(14)} ║${ANSI.reset}`);
    console.log(`${ANSI.bold}╚══════════════════════════════════════════════╝${ANSI.reset}`);
    console.log(`${ANSI.dim}Timeframe: ${tf.label}  |  Ctrl+C to stop${ANSI.reset}\n`);

    // Print header
    console.log(`${ANSI.dim}${'BAR'.padEnd(5)} ${'PRICE'.padEnd(10)} ${'ACTION'.padEnd(8)} ${'CONF'.padEnd(6)} ${'SCORE'.padEnd(8)} ${'REGIME'.padEnd(22)} ${'REASONING'}${ANSI.reset}`);
    console.log(`${ANSI.dim}${'─'.repeat(80)}${ANSI.reset}`);

    const rng = mulberry32(args.seed ?? Math.floor(Math.random() * 2147483647));
    const initialCandles = generateSyntheticCandles(args.warmup + 1, {
        seed: args.seed ?? Math.floor(Math.random() * 2147483647),
        basePrice: args.basePrice,
        volatility: args.volatility,
    });
    const candles = initialCandles;
    let barIndex = candles.length;

    // Run first analysis
    const firstDecision = orch.run(candles, { symbol: args.symbol });
    printLiveRow(barIndex, candles[candles.length - 1].close, firstDecision);

    // Stream new bars
    let nextBarTime = Date.now() + 2000;

    const interval = setInterval(() => {
        if (!liveRunning) {
            clearInterval(interval);
            return;
        }

        // Generate next candle from last price
        const lastPrice = candles[candles.length - 1].close;
        const drift = (rng() - 0.48) * args.volatility * lastPrice;
        const open = lastPrice;
        const close = open + drift;
        const high = Math.max(open, close) + rng() * args.volatility * 0.2 * lastPrice;
        const low = Math.min(open, close) - rng() * args.volatility * 0.2 * lastPrice;

        candles.push({
            timestamp: Date.now(),
            open: +open.toFixed(4),
            high: +high.toFixed(4),
            low: +low.toFixed(4),
            close: +close.toFixed(4),
            volume: lastPrice * 10 + rng() * lastPrice * 5,
        });

        barIndex++;

        const decision = orch.run(candles, { symbol: args.symbol });
        printLiveRow(barIndex, close, decision);

        if (candles.length > 1000) {
            candles.splice(0, candles.length - 500);
        }
    }, 2000);

    process.on('SIGINT', () => {
        liveRunning = false;
        clearInterval(interval);
        console.log(`\n${ANSI.dim}Live session ended. ${barIndex} bars processed.${ANSI.reset}\n`);
        process.exit(0);
    });
}

function printLiveRow(bar, price, decision) {
    const { dim, reset, green, red, cyan } = ANSI;
    const scoreColor = decision.compositeScore > 0.1 ? green : decision.compositeScore < -0.1 ? red : dim;
    const confColor = decision.confidence >= 0.6 ? green : decision.confidence >= 0.3 ? cyan : dim;

    console.log(
        `${String(bar).padEnd(5)} ` +
        `${price.toFixed(4).padEnd(10)} ` +
        `${colorAction(decision.action)} ${ANSI.reset} ` +
        `${confColor}${(decision.confidence * 100).toFixed(0).padStart(3)}%${ANSI.reset}  ` +
        `${scoreColor}${decision.compositeScore.toFixed(3).padStart(7)}${ANSI.reset} ` +
        `${colorRegime(decision.regime).padEnd(39)} ` +
        `${dim}${decision.reasoning ?? ''}${ANSI.reset}`
    );
}

// ─── Main ───────────────────────────────────────────────────────────────────────

function main() {
    const args = parseArgs(process.argv.slice(2));

    if (!args.mode) {
        console.error('Error: specify a mode: --analyze, --backtest, or --live');
        console.error('Use --help for usage information.');
        process.exit(1);
    }

    if (!TIMEFRAMES[args.timeframe]) {
        console.error(`Error: unknown timeframe '${args.timeframe}'. Use: ${Object.keys(TIMEFRAMES).join(', ')}`);
        process.exit(1);
    }

    // Load or generate candles
    let candles;
    if (args.candlesPath) {
        try {
            candles = loadCandles(args.candlesPath);
            console.error(`Loaded ${candles.length} candles from ${args.candlesPath}`);
        } catch (err) {
            console.error(`Error: ${err.message}`);
            process.exit(1);
        }
    } else {
        const tf = TIMEFRAMES[args.timeframe];
        const barCount = args.days * tf.barsPerDay;
        candles = generateSyntheticCandles(barCount, {
            seed: args.seed ?? undefined,
            basePrice: args.basePrice,
            volatility: args.volatility,
        });
        console.error(`Generated ${candles.length} synthetic ${tf.label} candles (${args.days} days, seed=${args.seed ?? 'random'})`);
    }

    switch (args.mode) {
        case 'analyze':
            runAnalyze(candles, args);
            break;
        case 'backtest':
            runBacktestMode(candles, args);
            break;
        case 'live':
            runLive(args);
            break;
        default:
            console.error(`Unknown mode: ${args.mode}`);
            process.exit(1);
    }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]?.replace(/\\/g, '/')}` || process.argv[1]?.endsWith('trade.mjs')) {
    main();
}

export { generateSyntheticCandles, loadCandles, parseArgs };
