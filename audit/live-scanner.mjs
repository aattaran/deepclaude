/**
 * Live Scanner — end-to-end daily pipeline runner.
 *
 * Fetches real/synthetic data → scans multiple symbols with optimizer →
 * ranks by viability → runs grail verdict on top picks.
 *
 * Usage:
 *   node audit/live-scanner.mjs
 *   node audit/live-scanner.mjs --symbols BTC-USD,ETH-USD,SOL-USD --tf 1h --days 30
 *   node audit/live-scanner.mjs --top 5 --min-score 1.2 --full
 *   node audit/live-scanner.mjs --live  # uses real Binance data when available
 *
 * ES module. Zero npm dependencies.
 */

import { loadOrFetch } from './datafeed.mjs';
import { optimize } from './optimizer.mjs';
import { getPreset, applyPreset } from './presets.mjs';
import { renderDashboard, colorize } from './dashboard.mjs';
import { runFull } from './grail.mjs';
import { generateSyntheticCandles } from './trade.mjs';

// ─── ANSI ───────────────────────────────────────────────────────────────────────

const C = {
    reset: '\x1b[0m', bold: '\x1b[1m', dim: '\x1b[2m',
    red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m',
    cyan: '\x1b[36m', blue: '\x1b[34m', white: '\x1b[37m',
    magenta: '\x1b[35m',
};

// ─── Constants ──────────────────────────────────────────────────────────────────

const DEFAULT_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'MBT', 'XRP-USD', 'ADA-USD', 'DOGE-USD'];
const BARS_PER_TF = { '1m': 1440, '5m': 288, '15m': 96, '1h': 24, '4h': 6, '1d': 1 };

// Efficient coarse grid for scanner
const SCAN_PARAM_SPACE = {
    stopLossPct: [0.02, 0.04],
    takeProfitPct: [0.04, 0.08],
    cooldownBars: [10],
    warmupBars: [50],
    minConfidence: [0.15, 0.25],
    fusionMethod: ['weighted'],
};

// ─── 1. Argument Parsing ────────────────────────────────────────────────────────

function parseArgs(argv) {
    const args = {
        symbols: DEFAULT_SYMBOLS,
        timeframe: '1h',
        days: 30,
        top: 3,
        minScore: 1.0,
        full: false,
        live: false,
        verbose: false,
        seed: 42,
        cacheDir: null,
        preset: true,
    };

    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const v = argv[i + 1];
        switch (a) {
            case '--symbols': case '-s': args.symbols = v.split(',').map(s => s.trim()); i++; break;
            case '--tf': case '--timeframe': args.timeframe = v; i++; break;
            case '--days': case '-d': args.days = parseInt(v, 10); i++; break;
            case '--top': case '-n': args.top = parseInt(v, 10); i++; break;
            case '--min-score': args.minScore = parseFloat(v); i++; break;
            case '--full': args.full = true; break;
            case '--live': args.live = true; break;
            case '--verbose': case '-v': args.verbose = true; break;
            case '--seed': args.seed = parseInt(v, 10); i++; break;
            case '--no-preset': args.preset = false; break;
            case '--cache-dir': args.cacheDir = v; i++; break;
            case '--help': case '-h': printHelp(); process.exit(0);
        }
    }

    return args;
}

function printHelp() {
    console.log(`
${C.bold}Live Scanner${C.reset} — Daily trading pipeline

${C.cyan}Usage:${C.reset}
  node audit/live-scanner.mjs [options]

${C.cyan}Options:${C.reset}
  -s, --symbols      Comma-separated symbols (default: top 7)
  --tf, --timeframe   Candle timeframe (default: 1h)
  -d, --days          Days of data (default: 30)
  -n, --top           Top N to detail (default: 3)
  --min-score         Minimum score to recommend (default: 1.0)
  --live              Use real Binance data when available
  --full              Run grail full verdict on top picks
  -v, --verbose       Show per-symbol progress
  --no-preset         Skip preset-based defaults
  --seed              PRNG seed (default: 42)
  --cache-dir         Custom cache directory
`);
}

// ─── 2. Core Pipeline ───────────────────────────────────────────────────────────

/**
 * Fetch or generate candles for a single symbol.
 */
async function fetchSymbolData(symbol, args) {
    const barsPerDay = BARS_PER_TF[args.timeframe] ?? 24;
    const neededBars = Math.min(args.days * barsPerDay, 1000);

    if (args.live) {
        try {
            return await loadOrFetch(symbol, args.timeframe, args.days, {
                cacheDir: args.cacheDir,
                allowSynthetic: true,
                seed: args.seed + DEFAULT_SYMBOLS.indexOf(symbol),
            });
        } catch (_) {
            // Fall through to synthetic
        }
    }

    // Synthetic with deterministic per-symbol seed
    const symIdx = DEFAULT_SYMBOLS.indexOf(symbol);
    const seed = args.seed + (symIdx >= 0 ? symIdx : symbol.length);
    const basePrice = estimateBasePrice(symbol);
    return generateSyntheticCandles(Math.max(neededBars, 100), {
        seed,
        basePrice,
        volatility: 0.012,
    });
}

function estimateBasePrice(symbol) {
    const upper = symbol.toUpperCase();
    if (upper.includes('BTC') || upper === 'MBT') return 75000;
    if (upper.includes('ETH')) return 3500;
    if (upper.includes('SOL')) return 150;
    if (upper.includes('XRP')) return 2.5;
    if (upper.includes('ADA')) return 0.6;
    if (upper.includes('DOGE')) return 0.15;
    if (upper.includes('AVAX')) return 30;
    if (upper.includes('DOT')) return 7;
    if (upper.includes('LINK')) return 15;
    return 100;
}

// ─── 3. Scan & Rank ─────────────────────────────────────────────────────────────

async function scanSymbol(symbol, candles, args) {
    const optConfig = {
        coarseSpace: SCAN_PARAM_SPACE,
        objective: 'profitFactor',
        maxIterations: 3,
        verbose: false,
    };

    if (args.preset) {
        const preset = getPreset(symbol);
        optConfig.coarseSpace = {
            ...SCAN_PARAM_SPACE,
            stopLossPct: [preset.stopLossPct * 0.8, preset.stopLossPct, preset.stopLossPct * 1.2],
            takeProfitPct: [preset.takeProfitPct * 0.8, preset.takeProfitPct, preset.takeProfitPct * 1.2],
        };
    }

    const result = await optimize(candles, optConfig);

    return {
        symbol,
        bestScore: result.bestScore,
        bestStats: result.bestStats,
        bestParams: result.bestParams,
        candleCount: candles.length,
    };
}

// ─── 4. Display ──────────────────────────────────────────────────────────────────

function renderScanTable(results, topN) {
    if (!results || !Array.isArray(results)) {
        return '\n  (no results)\n';
    }
    const sorted = [...results].sort((a, b) => b.bestScore - a.bestScore);
    const show = sorted.slice(0, Math.min(topN, results.length));

    const lines = [];
    const w = { rank: 5, symbol: 12, score: 10, trades: 8, win: 8, pf: 8, pnl: 10, sl: 10, tp: 10 };
    const sep = '─'.repeat(Object.values(w).reduce((a, b) => a + b, 0) + Object.keys(w).length * 3 + 1);

    lines.push('');
    lines.push(`${C.bold}${C.cyan}╔${'═'.repeat(70)}╗${C.reset}`);
    lines.push(`${C.bold}${C.cyan}║${C.reset}${C.bold}  SCAN RESULTS${' '.repeat(57)}${C.cyan}║${C.reset}`);
    lines.push(`${C.bold}${C.cyan}╚${'═'.repeat(70)}╝${C.reset}`);
    lines.push('');

    // Header
    const header = `${'RANK'.padStart(w.rank)}  ${'SYMBOL'.padEnd(w.symbol)}  ${'SCORE'.padStart(w.score)}  ${'TRADES'.padStart(w.trades)}  ${'WIN%'.padStart(w.win)}  ${'PF'.padStart(w.pf)}  ${'PNL%'.padStart(w.pnl)}  ${'BEST SL'.padStart(w.sl)}  ${'BEST TP'.padStart(w.tp)}`;
    lines.push(`${C.bold}${header}${C.reset}`);
    lines.push(C.dim + sep + C.reset);

    for (let i = 0; i < show.length; i++) {
        const r = show[i];
        const stats = r.bestStats || {};
        const rank = `${i + 1}`;
        const score = r.bestScore?.toFixed(2) ?? 'N/A';
        const trades = stats.totalTrades ?? 'N/A';
        const winRate = stats.winRate != null ? `${(stats.winRate * 100).toFixed(0)}%` : 'N/A';
        const pf = stats.profitFactor?.toFixed(2) ?? 'N/A';
        const pnl = stats.totalReturnPct != null ? `${stats.totalReturnPct.toFixed(1)}%` : 'N/A';
        const sl = r.bestParams?.stopLossPct != null ? `${(r.bestParams.stopLossPct * 100).toFixed(1)}%` : 'N/A';
        const tp = r.bestParams?.takeProfitPct != null ? `${(r.bestParams.takeProfitPct * 100).toFixed(1)}%` : 'N/A';

        const color = r.bestScore >= 1.5 ? C.green : r.bestScore >= 1.0 ? C.yellow : C.red;
        lines.push(`${color}${rank.padStart(w.rank)}${C.reset}  ${r.symbol.padEnd(w.symbol)}  ${score.padStart(w.score)}  ${String(trades).padStart(w.trades)}  ${winRate.padStart(w.win)}  ${pf.padStart(w.pf)}  ${pnl.padStart(w.pnl)}  ${sl.padStart(w.sl)}  ${tp.padStart(w.tp)}`);
    }

    lines.push(C.dim + sep + C.reset);
    lines.push(`${C.dim}Showing ${show.length} of ${results.length} symbols${C.reset}`);
    lines.push('');

    return lines.join('\n');
}

function renderVerdicts(verdicts) {
    if (!verdicts || verdicts.length === 0) return '';

    const lines = [];
    lines.push(`${C.bold}${C.magenta}╔${'═'.repeat(70)}╗${C.reset}`);
    lines.push(`${C.bold}${C.magenta}║${C.reset}${C.bold}  GRAIL VERDICTS${' '.repeat(55)}${C.magenta}║${C.reset}`);
    lines.push(`${C.bold}${C.magenta}╚${'═'.repeat(70)}╝${C.reset}`);
    lines.push('');

    for (const v of verdicts) {
        const verdictColor = v.verdict === 'GRAIL' ? C.green + C.bold :
                             v.verdict === 'PROFITABLE' ? C.green :
                             C.red;
        lines.push(`  ${C.bold}${v.symbol}${C.reset} → ${verdictColor}${v.verdict}${C.reset}  (score: ${v.score?.toFixed(2) ?? 'N/A'})`);
    }

    lines.push('');
    return lines.join('\n');
}

// ─── 5. Main ────────────────────────────────────────────────────────────────────

async function main() {
    const startTime = Date.now();
    const args = parseArgs(process.argv.slice(2));

    console.log(`${C.bold}${C.cyan}╔${'═'.repeat(70)}╗${C.reset}`);
    console.log(`${C.bold}${C.cyan}║${C.reset}${C.bold}  LIVE SCANNER — Trading System Pipeline${' '.repeat(31)}${C.cyan}║${C.reset}`);
    console.log(`${C.bold}${C.cyan}╚${'═'.repeat(70)}╝${C.reset}`);
    console.log(`  ${C.dim}${args.symbols.length} symbols | ${args.timeframe} | ${args.days} days | top ${args.top}${args.live ? ' | LIVE data' : ''}${args.full ? ' | full grail' : ''}${C.reset}`);
    console.log('');

    // Phase 1: Fetch data for all symbols
    console.log(`${C.bold}[1/3]${C.reset} Fetching data for ${args.symbols.length} symbols...`);
    const symbolData = {};
    const fetchStart = Date.now();

    for (const sym of args.symbols) {
        if (args.verbose) process.stdout.write(`  ${sym}... `);
        try {
            symbolData[sym] = await fetchSymbolData(sym, args);
            if (args.verbose) console.log(`${C.green}${symbolData[sym].length} candles${C.reset}`);
        } catch (err) {
            if (args.verbose) console.log(`${C.red}FAILED: ${err.message}${C.reset}`);
            symbolData[sym] = [];
        }
    }
    console.log(`  ${C.dim}Done in ${((Date.now() - fetchStart) / 1000).toFixed(1)}s${C.reset}`);
    console.log('');

    // Phase 2: Scan (optimize) each symbol
    console.log(`${C.bold}[2/3]${C.reset} Scanning ${Object.keys(symbolData).length} symbols...`);
    const scanStart = Date.now();
    const results = [];

    for (const sym of args.symbols) {
        const candles = symbolData[sym];
        if (!candles || candles.length < 50) {
            if (args.verbose) console.log(`  ${sym} ${C.dim}— insufficient data, skipping${C.reset}`);
            continue;
        }

        if (args.verbose) process.stdout.write(`  ${sym}... `);
        try {
            const result = await scanSymbol(sym, candles, args);
            results.push(result);
            const verdict = result.bestScore >= 2 ? 'GRAIL' :
                           result.bestScore >= 1.2 ? 'PROFITABLE' :
                           result.bestScore >= 1.0 ? 'MARGINAL' : 'SKIP';
            const vc = verdict === 'GRAIL' ? C.green + C.bold :
                       verdict === 'PROFITABLE' ? C.green :
                       verdict === 'MARGINAL' ? C.yellow : C.red;
            if (args.verbose) console.log(`${vc}${verdict}${C.reset} ${C.dim}(PF: ${result.bestScore?.toFixed(2)})${C.reset}`);
        } catch (err) {
            if (args.verbose) console.log(`${C.red}FAILED: ${err.message}${C.reset}`);
        }
    }
    console.log(`  ${C.dim}Done in ${((Date.now() - scanStart) / 1000).toFixed(1)}s${C.reset}`);
    console.log('');

    // Phase 3: Display results
    console.log(`${C.bold}[3/3]${C.reset} Results`);

    // Rank and display table
    results.sort((a, b) => b.bestScore - a.bestScore);
    console.log(renderScanTable(results, args.top));

    // Grail verdicts on top picks
    const topPicks = results.slice(0, args.top).filter(r => r.bestScore >= args.minScore);
    if (args.full && topPicks.length > 0) {
        console.log(`${C.bold}Running grail full on top ${topPicks.length} pick(s)...${C.reset}\n`);
        const verdicts = [];

        for (const pick of topPicks) {
            const candles = symbolData[pick.symbol];
            if (!candles || candles.length < 100) continue;

            try {
                const grailArgs = {
                    symbol: pick.symbol,
                    days: args.days,
                    timeframe: args.timeframe,
                    seed: args.seed,
                    volatility: 0.012,
                    basePrice: estimateBasePrice(pick.symbol),
                };

                // Capture grail output and extract verdict
                const oldLog = console.log;
                let captured = '';
                console.log = (...a) => { captured += a.join(' ') + '\n'; };
                try {
                    await runFull(grailArgs, candles);
                } finally {
                    console.log = oldLog;
                }

                // Extract verdict from captured output
                const isGrail = captured.includes('GRAIL');
                const isProfitable = captured.includes('PROFITABLE');
                const verdict = isGrail ? 'GRAIL' : isProfitable ? 'PROFITABLE' : 'NOT YIELDING';
                verdicts.push({ symbol: pick.symbol, verdict, score: pick.bestScore });

                // Print captured grail output directly
                process.stdout.write(colorize(captured));
            } catch (err) {
                if (args.verbose) console.error(`${C.red}Grail failed for ${pick.symbol}: ${err.message}${C.reset}`);
            }
        }

        if (verdicts.length > 0) console.log(renderVerdicts(verdicts));
    } else if (topPicks.length > 0) {
        // Quick verdicts based on scores
        const verdicts = topPicks.map(p => ({
            symbol: p.symbol,
            verdict: p.bestScore >= 2 ? 'GRAIL' : p.bestScore >= 1.2 ? 'PROFITABLE' : 'MARGINAL',
            score: p.bestScore,
        }));
        console.log(renderVerdicts(verdicts));
    }

    // Summary
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const grails = results.filter(r => r.bestScore >= 2).length;
    const profitable = results.filter(r => r.bestScore >= 1.2 && r.bestScore < 2).length;
    const marginals = results.filter(r => r.bestScore >= 1.0 && r.bestScore < 1.2).length;

    console.log(`${C.bold}${C.cyan}╔${'═'.repeat(70)}╗${C.reset}`);
    console.log(`${C.bold}${C.cyan}║${C.reset}${C.bold}  SUMMARY${' '.repeat(62)}${C.cyan}║${C.reset}`);
    console.log(`${C.bold}${C.cyan}╚${'═'.repeat(70)}╝${C.reset}`);
    console.log(`  ${C.green}${C.bold}GRAIL:${C.reset} ${grails}  |  ${C.green}PROFITABLE:${C.reset} ${profitable}  |  ${C.yellow}MARGINAL:${C.reset} ${marginals}  |  ${C.dim}Total: ${results.length} symbols${C.reset}`);
    console.log(`  ${C.dim}Pipeline completed in ${elapsed}s${C.reset}`);
    console.log('');
}

// ─── Entry Point ────────────────────────────────────────────────────────────────

const isMain = process.argv[1]?.endsWith('live-scanner.mjs') || process.argv[1]?.endsWith('live-scanner');
if (isMain) {
    main().catch(err => {
        console.error(`${C.red}FATAL: ${err.message}${C.reset}`);
        process.exit(1);
    });
}

export { parseArgs, fetchSymbolData, scanSymbol, renderScanTable, renderVerdicts, estimateBasePrice };
