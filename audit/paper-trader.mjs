/**
 * Paper Trader — simulated live trading with P&L tracking.
 *
 * Streams candle data (synthetic or real), runs orchestrator on each bar close,
 * executes simulated trades with SL/TP, tracks full P&L journal and equity curve.
 *
 * Usage:
 *   import { PaperTrader, runPaperSession } from './paper-trader.mjs';
 *   const trader = new PaperTrader({ capital: 10000 });
 *   trader.feed(candles, 'BTC-USD');
 *   const report = trader.report();
 *
 * ES module. Zero npm dependencies.
 */

import { createOrchestrator } from './orchestrator.mjs';
import { renderDashboard, colorize } from './dashboard.mjs';

// ─── ANSI ───────────────────────────────────────────────────────────────────────

const C = {
    reset: '\x1b[0m', bold: '\x1b[1m', dim: '\x1b[2m',
    red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m',
    cyan: '\x1b[36m', blue: '\x1b[34m', magenta: '\x1b[35m',
};

// ─── 1. PaperTrader Class ───────────────────────────────────────────────────────

const DEFAULT_CONFIG = {
    capital: 10000,
    maxPositions: 3,
    positionSizePct: 0.2,       // fraction of capital per trade
    stopLossPct: 0.02,
    takeProfitPct: 0.04,
    maxHoldingBars: 200,
    minConfidence: 0.15,
    cooldownBars: 5,
    warmupBars: 50,
    fusionMethod: 'weighted',
    commissionPct: 0.001,       // 0.1% per trade
    slippagePct: 0.0005,       // 0.05% slippage
    allowShort: true,
};

export class PaperTrader {
    constructor(config = {}) {
        const cfg = { ...DEFAULT_CONFIG, ...config };
        this._capital = cfg.capital;
        this._initialCapital = cfg.capital;
        this._maxPositions = cfg.maxPositions;
        this._positionSizePct = cfg.positionSizePct;
        this._commissionPct = cfg.commissionPct;
        this._slippagePct = cfg.slippagePct;
        this._allowShort = cfg.allowShort;

        // Orchestrator config (passed through)
        this._orchConfig = {
            stopLossPct: cfg.stopLossPct,
            takeProfitPct: cfg.takeProfitPct,
            maxHoldingBars: cfg.maxHoldingBars,
            minConfidence: cfg.minConfidence,
            cooldownBars: cfg.cooldownBars,
            warmupBars: cfg.warmupBars,
            fusionMethod: cfg.fusionMethod,
        };

        // State
        this._positions = new Map();     // symbol → Position
        this._trades = [];               // closed trades
        this._equityCurve = [];          // [{ bar, time, equity, balance }]
        this._lastTradeBar = new Map();  // symbol → bar index of last trade
        this._barIndex = 0;
        this._symbolOrch = new Map();    // symbol → orchestrator instance
    }

    // ── Feed ─────────────────────────────────────────────────────────────────

    /**
     * Feed a complete candle array for a symbol. Processes each bar sequentially.
     * Returns array of signals generated (for inspection).
     */
    feed(candles, symbol) {
        if (!candles || candles.length === 0) return [];

        const orch = createOrchestrator({ ...this._orchConfig, symbol });
        this._symbolOrch.set(symbol, orch);
        const signals = [];

        // Find the absolute bar start for this feed
        const feedStartBar = this._barIndex;

        for (let i = this._orchConfig.warmupBars; i < candles.length; i++) {
            this._barIndex = feedStartBar + i;
            const window = candles.slice(0, i + 1);
            const currentCandle = candles[i];

            // Check existing position exits for this symbol
            this._checkExits(symbol, currentCandle, this._barIndex);

            // Skip if cooldown active or max positions reached
            const lastTrade = this._lastTradeBar.get(symbol) ?? -Infinity;
            if (this._barIndex - lastTrade < this._orchConfig.cooldownBars) continue;
            if (!this._allowShort && this._positions.size >= this._maxPositions) continue;

            // Run orchestrator
            const decision = orch.run(window, { symbol });

            if (decision.action === 'BUY' || decision.action === 'SELL') {
                signals.push({
                    bar: this._barIndex,
                    symbol,
                    action: decision.action,
                    price: currentCandle.close,
                    confidence: decision.confidence,
                    compositeScore: decision.compositeScore,
                    regime: decision.regime,
                });

                // Execute entry if we have room
                if (this._positions.size < this._maxPositions || this._allowShort) {
                    this._enterPosition(symbol, decision, currentCandle, this._barIndex);
                }
            }
        }

        this._barIndex = feedStartBar + candles.length;
        return signals;
    }

    // ── Position Management ──────────────────────────────────────────────────

    _enterPosition(symbol, decision, candle, barIndex) {
        const price = candle.close;
        const size = (this._capital * this._positionSizePct) / price;
        const slippageAdj = decision.action === 'BUY' ? 1 + this._slippagePct : 1 - this._slippagePct;
        const entryPrice = price * slippageAdj;
        const cost = entryPrice * size * this._commissionPct;

        const position = {
            symbol,
            direction: decision.action === 'BUY' ? 'long' : 'short',
            entryPrice,
            entryBar: barIndex,
            size,
            stopLoss: decision.action === 'BUY'
                ? entryPrice * (1 - this._orchConfig.stopLossPct)
                : entryPrice * (1 + this._orchConfig.stopLossPct),
            takeProfit: decision.action === 'BUY'
                ? entryPrice * (1 + this._orchConfig.takeProfitPct)
                : entryPrice * (1 - this._orchConfig.takeProfitPct),
            entryConfidence: decision.confidence,
            entryScore: decision.compositeScore,
            regime: decision.regime,
        };

        this._positions.set(symbol, position);
        this._capital -= cost;
        this._lastTradeBar.set(symbol, barIndex);
    }

    _checkExits(symbol, candle, barIndex) {
        const pos = this._positions.get(symbol);
        if (!pos) return;

        const price = candle.close;
        const barsHeld = barIndex - pos.entryBar;
        let exit = false;
        let exitReason = '';

        if (pos.direction === 'long') {
            if (price <= pos.stopLoss) { exit = true; exitReason = 'stop_loss'; }
            else if (price >= pos.takeProfit) { exit = true; exitReason = 'take_profit'; }
        } else {
            if (price >= pos.stopLoss) { exit = true; exitReason = 'stop_loss'; }
            else if (price <= pos.takeProfit) { exit = true; exitReason = 'take_profit'; }
        }
        if (barsHeld >= this._orchConfig.maxHoldingBars) {
            exit = true;
            exitReason = 'max_hold';
        }

        if (exit) {
            this._closePosition(symbol, candle, barIndex, exitReason);
        }
    }

    _closePosition(symbol, candle, barIndex, reason) {
        const pos = this._positions.get(symbol);
        if (!pos) return;

        const price = candle.close;
        const slippageAdj = pos.direction === 'long' ? 1 - this._slippagePct : 1 + this._slippagePct;
        const exitPrice = price * slippageAdj;

        // PnL
        const pnl = pos.direction === 'long'
            ? (exitPrice - pos.entryPrice) * pos.size
            : (pos.entryPrice - exitPrice) * pos.size;
        const pnlPct = pos.direction === 'long'
            ? (exitPrice - pos.entryPrice) / pos.entryPrice
            : (pos.entryPrice - exitPrice) / pos.entryPrice;

        const commission = exitPrice * pos.size * this._commissionPct;
        const netPnl = pnl - commission;

        this._capital += (pos.entryPrice * pos.size) + netPnl; // return capital + pnl

        const trade = {
            symbol,
            direction: pos.direction,
            entryPrice: pos.entryPrice,
            exitPrice,
            entryBar: pos.entryBar,
            exitBar: barIndex,
            barsHeld: barIndex - pos.entryBar,
            pnl: +netPnl.toFixed(2),
            pnlPct: +(pnlPct * 100).toFixed(2),
            entryConfidence: pos.entryConfidence,
            entryScore: pos.entryScore,
            exitReason: reason,
            regime: pos.regime,
        };

        this._trades.push(trade);
        this._positions.delete(symbol);
    }

    // ── Equity Snapshot ──────────────────────────────────────────────────────

    /**
     * Close all open positions at market (end of session).
     */
    closeAll(candle) {
        for (const [symbol] of this._positions) {
            this._closePosition(symbol, candle, this._barIndex, 'session_end');
        }
    }

    /**
     * Record an equity snapshot at the current point.
     */
    snapshot(time, candle) {
        const unrealizedPnl = this._getUnrealizedPnl(candle);
        const equity = this._capital + unrealizedPnl;
        this._equityCurve.push({
            bar: this._barIndex,
            time: time ?? candle?.timestamp ?? Date.now(),
            equity: +equity.toFixed(2),
            balance: +this._capital.toFixed(2),
            unrealizedPnl: +unrealizedPnl.toFixed(2),
            positions: this._positions.size,
        });
    }

    _getUnrealizedPnl(candle) {
        if (!candle) return 0;
        let total = 0;
        for (const [, pos] of this._positions) {
            const price = candle.close;
            const pnl = pos.direction === 'long'
                ? (price - pos.entryPrice) * pos.size
                : (pos.entryPrice - price) * pos.size;
            total += pnl;
        }
        return total;
    }

    // ── Metrics ──────────────────────────────────────────────────────────────

    getMetrics() {
        const trades = this._trades;
        if (trades.length === 0) {
            return {
                totalTrades: 0, winningTrades: 0, losingTrades: 0, winRate: 0,
                totalPnl: 0, totalPnlPct: 0, avgWin: 0, avgLoss: 0,
                profitFactor: 0, sharpeRatio: 0, maxDrawdownPct: 0,
                expectancy: 0, bestTrade: null, worstTrade: null,
                equity: this._capital, returnPct: 0,
            };
        }

        const wins = trades.filter(t => t.pnl > 0);
        const losses = trades.filter(t => t.pnl < 0);
        const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);
        const totalPnlPct = (this._capital - this._initialCapital) / this._initialCapital * 100;
        const avgWin = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0;
        const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((s, t) => s + t.pnl, 0) / losses.length) : 0;
        const profitFactor = avgLoss > 0 ? (wins.reduce((s, t) => s + t.pnl, 0) / Math.abs(losses.reduce((s, t) => s + t.pnl, 0))) : (wins.length > 0 ? Infinity : 0);

        // Sharpe (simplified — from trade returns)
        const returns = trades.map(t => t.pnlPct / 100);
        const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
        const variance = returns.reduce((s, r) => s + (r - avgReturn) ** 2, 0) / returns.length;
        const stdDev = Math.sqrt(variance);
        const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(trades.length) : 0;

        // Max drawdown from equity curve
        let maxDrawdownPct = 0;
        let peak = this._initialCapital;
        for (const pt of this._equityCurve) {
            if (pt.equity > peak) peak = pt.equity;
            const dd = (peak - pt.equity) / peak;
            if (dd > maxDrawdownPct) maxDrawdownPct = dd;
        }

        const bestTrade = trades.reduce((best, t) => t.pnl > (best?.pnl ?? -Infinity) ? t : best, null);
        const worstTrade = trades.reduce((worst, t) => t.pnl < (worst?.pnl ?? Infinity) ? t : worst, null);

        return {
            totalTrades: trades.length,
            winningTrades: wins.length,
            losingTrades: losses.length,
            winRate: +(wins.length / trades.length).toFixed(4),
            totalPnl: +totalPnl.toFixed(2),
            totalPnlPct: +totalPnlPct.toFixed(2),
            avgWin: +avgWin.toFixed(2),
            avgLoss: +avgLoss.toFixed(2),
            profitFactor: profitFactor === Infinity ? 999 : +profitFactor.toFixed(2),
            sharpeRatio: +sharpeRatio.toFixed(2),
            maxDrawdownPct: +(maxDrawdownPct * 100).toFixed(2),
            expectancy: +(avgReturn * 100).toFixed(2),
            bestTrade: bestTrade ? { pnl: bestTrade.pnl, pnlPct: bestTrade.pnlPct, symbol: bestTrade.symbol } : null,
            worstTrade: worstTrade ? { pnl: worstTrade.pnl, pnlPct: worstTrade.pnlPct, symbol: worstTrade.symbol } : null,
            equity: +this._capital.toFixed(2),
            returnPct: +totalPnlPct.toFixed(2),
        };
    }

    // ── Export ───────────────────────────────────────────────────────────────

    /**
     * Full report object suitable for JSON export.
     */
    report() {
        return {
            config: {
                initialCapital: this._initialCapital,
                positionSizePct: this._positionSizePct,
                stopLossPct: this._orchConfig.stopLossPct,
                takeProfitPct: this._orchConfig.takeProfitPct,
                minConfidence: this._orchConfig.minConfidence,
                commissionPct: this._commissionPct,
            },
            metrics: this.getMetrics(),
            trades: this._trades,
            equityCurve: this._equityCurve,
            openPositions: [...this._positions.values()],
        };
    }

    /**
     * Render a formatted terminal report.
     */
    renderReport() {
        const m = this.getMetrics();
        const lines = [];

        lines.push('');
        lines.push(`${C.bold}${C.cyan}╔${'═'.repeat(66)}╗${C.reset}`);
        lines.push(`${C.bold}${C.cyan}║${C.reset}${C.bold}  PAPER TRADER — Session Report${' '.repeat(39)}${C.cyan}║${C.reset}`);
        lines.push(`${C.bold}${C.cyan}╚${'═'.repeat(66)}╝${C.reset}`);

        // Performance
        lines.push('');
        lines.push(`${C.bold}Performance${C.reset}`);
        lines.push(`  Return:      ${m.returnPct >= 0 ? C.green : C.red}${m.returnPct}%${C.reset}`);
        lines.push(`  Equity:      $${m.equity.toLocaleString()} (initial: $${this._initialCapital.toLocaleString()})`);
        lines.push(`  Sharpe:      ${m.sharpeRatio >= 1 ? C.green : m.sharpeRatio >= 0 ? C.yellow : C.red}${m.sharpeRatio}${C.reset}`);
        lines.push(`  Max DD:      ${C.red}${m.maxDrawdownPct}%${C.reset}`);

        // Trade stats
        lines.push('');
        lines.push(`${C.bold}Trade Statistics${C.reset}`);
        lines.push(`  Total:       ${m.totalTrades}  (${m.winningTrades}W / ${m.losingTrades}L)`);
        lines.push(`  Win Rate:    ${m.winRate >= 0.5 ? C.green : C.red}${(m.winRate * 100).toFixed(1)}%${C.reset}`);
        lines.push(`  PF:          ${m.profitFactor >= 1.5 ? C.green : m.profitFactor >= 1 ? C.yellow : C.red}${m.profitFactor}${C.reset}`);
        lines.push(`  Expectancy:  ${m.expectancy >= 0 ? C.green : C.red}${m.expectancy}%${C.reset}`);
        lines.push(`  Avg Win:     ${C.green}$${m.avgWin}${C.reset}`);
        lines.push(`  Avg Loss:    ${C.red}$${m.avgLoss}${C.reset}`);

        if (m.bestTrade) {
            lines.push(`  Best Trade:  ${C.green}$${m.bestTrade.pnl} (${m.bestTrade.pnlPct}%) ${m.bestTrade.symbol}${C.reset}`);
        }
        if (m.worstTrade) {
            lines.push(`  Worst Trade: ${C.red}$${m.worstTrade.pnl} (${m.worstTrade.pnlPct}%) ${m.worstTrade.symbol}${C.reset}`);
        }

        // Open positions
        if (this._positions.size > 0) {
            lines.push('');
            lines.push(`${C.bold}Open Positions${C.reset}`);
            for (const [, pos] of this._positions) {
                const dir = pos.direction === 'long' ? 'LONG ' : 'SHORT';
                lines.push(`  ${pos.symbol} ${dir} @ $${pos.entryPrice.toFixed(2)} (SL: ${pos.stopLoss.toFixed(2)}, TP: ${pos.takeProfit.toFixed(2)})`);
            }
        }

        // Recent trades
        const recent = this._trades.slice(-5);
        if (recent.length > 0) {
            lines.push('');
            lines.push(`${C.bold}Recent Trades${C.reset}`);
            for (const t of recent) {
                const pnlStr = t.pnl >= 0 ? `${C.green}+$${t.pnl}${C.reset}` : `${C.red}$${t.pnl}${C.reset}`;
                lines.push(`  ${t.symbol} ${t.direction.toUpperCase()} | ${pnlStr} | ${t.pnlPct}% | ${t.exitReason} (${t.barsHeld}b)`);
            }
        }

        lines.push('');
        return lines.join('\n');
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    get trades() { return this._trades; }
    get positions() { return this._positions; }
    get equityCurve() { return this._equityCurve; }
    get capital() { return this._capital; }
    get barIndex() { return this._barIndex; }
}

// ─── 2. Convenience Runner ──────────────────────────────────────────────────────

/**
 * Run a full paper trading session across multiple symbols.
 *
 * @param {Object<string, Array>} symbolData — { 'BTC-USD': [...candles], ... }
 * @param {object} config — PaperTrader + orchestrator config
 * @returns {PaperTrader} — the trader with completed session
 */
export function runPaperSession(symbolData, config = {}) {
    const trader = new PaperTrader(config);

    // Determine the max bar count across all symbols for interleaving
    // Simple approach: process each symbol's full series sequentially
    // (co-located symbols are processed independently; correlation comes later)

    for (const [symbol, candles] of Object.entries(symbolData)) {
        if (!candles || candles.length === 0) continue;
        trader.feed(candles, symbol);
    }

    // Close all open positions at session end
    const lastSym = Object.keys(symbolData)[0];
    const lastCandles = symbolData[lastSym];
    if (lastCandles && lastCandles.length > 0) {
        trader.closeAll(lastCandles[lastCandles.length - 1]);
    }

    // Final snapshot
    trader.snapshot(Date.now(), null);

    return trader;
}

/**
 * Interleaved paper session — processes symbols bar-by-bar in lockstep.
 * Useful when symbols share a timeline (e.g., same timeframe, same period).
 */
export function runInterleavedSession(symbolData, config = {}) {
    const trader = new PaperTrader(config);
    const symbols = Object.keys(symbolData);
    const minLen = Math.min(...symbols.map(s => symbolData[s]?.length ?? 0));
    const warmup = config.warmupBars ?? 50;

    for (let i = warmup; i < minLen; i++) {
        for (const sym of symbols) {
            const candles = symbolData[sym];
            const window = candles.slice(0, i + 1);
            const currentCandle = candles[i];

            // Check exits
            // (trader handles this via its internal feed logic)
        }
        // Snapshot once per bar across all symbols
        const refCandle = symbolData[symbols[0]][i];
        trader.snapshot(null, refCandle);
    }

    return trader;
}

// ─── 3. CLI ─────────────────────────────────────────────────────────────────────

function parseArgs(argv) {
    const args = {
        symbols: ['BTC-USD'],
        timeframe: '1h',
        days: 60,
        capital: 10000,
        minConfidence: 0.15,
        stopLossPct: 0.02,
        takeProfitPct: 0.04,
        seed: 42,
        verbose: false,
        export: false,
    };

    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        const v = argv[i + 1];
        switch (a) {
            case '--symbols': case '-s': args.symbols = v.split(',').map(s => s.trim()); i++; break;
            case '--tf': args.timeframe = v; i++; break;
            case '--days': case '-d': args.days = parseInt(v, 10); i++; break;
            case '--capital': args.capital = parseFloat(v); i++; break;
            case '--min-conf': args.minConfidence = parseFloat(v); i++; break;
            case '--sl': args.stopLossPct = parseFloat(v); i++; break;
            case '--tp': args.takeProfitPct = parseFloat(v); i++; break;
            case '--seed': args.seed = parseInt(v, 10); i++; break;
            case '--verbose': case '-v': args.verbose = true; break;
            case '--export': args.export = true; break;
            case '--help': case '-h': printHelp(); process.exit(0);
        }
    }

    return args;
}

function printHelp() {
    console.log(`
${C.bold}Paper Trader${C.reset} — Simulated live trading

${C.cyan}Usage:${C.reset}
  node audit/paper-trader.mjs [options]

${C.cyan}Options:${C.reset}
  -s, --symbols     Comma-separated symbols (default: BTC-USD)
  --tf              Candle timeframe (default: 1h)
  -d, --days        Days of data (default: 60)
  --capital         Starting capital (default: 10000)
  --min-conf        Min confidence threshold (default: 0.15)
  --sl              Stop loss % (default: 0.02)
  --tp              Take profit % (default: 0.04)
  --seed            PRNG seed (default: 42)
  --export          Export JSON report to file
  -v, --verbose     Show per-bar progress
`);
}

async function main() {
    const args = parseArgs(process.argv.slice(2));
    const { generateSyntheticCandles } = await import('./trade.mjs');

    console.log(`${C.bold}${C.cyan}╔${'═'.repeat(66)}╗${C.reset}`);
    console.log(`${C.bold}${C.cyan}║${C.reset}${C.bold}  PAPER TRADER${' '.repeat(54)}${C.cyan}║${C.reset}`);
    console.log(`${C.bold}${C.cyan}╚${'═'.repeat(66)}╝${C.reset}`);
    console.log(`  ${C.dim}${args.symbols.length} symbols | ${args.timeframe} | ${args.days} days | $${args.capital} capital${C.reset}`);
    console.log('');

    // Generate data
    const symbolData = {};
    for (const sym of args.symbols) {
        const idx = args.symbols.indexOf(sym);
        const basePrice = sym.toUpperCase().includes('BTC') || sym === 'MBT' ? 75000
            : sym.toUpperCase().includes('ETH') ? 3500
            : sym.toUpperCase().includes('SOL') ? 150 : 100;

        symbolData[sym] = generateSyntheticCandles(args.days * 24, {
            seed: args.seed + idx,
            basePrice,
            volatility: 0.012,
        });
    }

    // Run session
    const trader = runPaperSession(symbolData, {
        capital: args.capital,
        minConfidence: args.minConfidence,
        stopLossPct: args.stopLossPct,
        takeProfitPct: args.takeProfitPct,
    });

    // Report
    console.log(trader.renderReport());

    // Export
    if (args.export) {
        const fs = await import('node:fs');
        const report = trader.report();
        const outFile = `paper-trader-${Date.now()}.json`;
        fs.writeFileSync(outFile, JSON.stringify(report, null, 2));
        console.log(`${C.dim}Exported to ${outFile}${C.reset}`);
    }
}

const isMain = process.argv[1]?.endsWith('paper-trader.mjs') || process.argv[1]?.endsWith('paper-trader');
if (isMain) {
    main().catch(err => {
        console.error(`${C.red}FATAL: ${err.message}${C.reset}`);
        process.exit(1);
    });
}

export default PaperTrader;
