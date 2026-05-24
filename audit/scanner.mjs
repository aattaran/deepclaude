/**
 * Multi-Symbol Scanner — runs optimizer across symbols simultaneously and ranks by trading viability.
 *
 * For each symbol:
 *   1. Generate synthetic candles (generateSyntheticCandles from trade.mjs)
 *   2. Run optimize() from optimizer.mjs with a small coarse grid
 *   3. Collect bestScore and bestStats
 *
 * Results are sorted by score descending.
 *
 * Usage:
 *   import { scanSymbols, scanResultsToTable, recommendSymbols } from './scanner.mjs';
 *   const results = await scanSymbols(['BTC-USD', 'ETH-USD', 'SOL-USD']);
 *   console.log(scanResultsToTable(results));
 *   const picks = recommendSymbols(results);
 *
 * ES module. Zero npm dependencies.
 */

import { generateSyntheticCandles } from './trade.mjs';
import { optimize } from './optimizer.mjs';

// ─── Constants ──────────────────────────────────────────────────────────────────

const BARS_PER_TF = {
  '1m': 1440,
  '5m': 288,
  '15m': 96,
  '1h': 24,
  '4h': 6,
  '1d': 1,
};

const DEFAULT_BASE_PRICES = {
  'BTC-USD': 50000,
  'ETH-USD': 3000,
  'SOL-USD': 150,
  'XRP-USD': 0.5,
  'ADA-USD': 0.6,
  'DOGE-USD': 0.15,
  'DOT-USD': 7,
  'AVAX-USD': 35,
  'LINK-USD': 14,
  'MATIC-USD': 0.7,
  'UNI-USD': 10,
  'ATOM-USD': 12,
  'BCH-USD': 400,
  'LTC-USD': 90,
  'NEAR-USD': 5,
  'TRX-USD': 0.08,
  'FIL-USD': 6,
  'APT-USD': 9,
  'ARB-USD': 1.2,
  'OP-USD': 2.5,
};

/**
 * Efficient coarse grid for the scanner — keeps runtime reasonable.
 * 2 SL × 2 TP × 1 cooldown × 1 warmup × 2 confidence × 1 fusion = 8 combinations.
 */
const DEFAULT_COARSE_SPACE = {
  stopLossPct: [0.02, 0.04],
  takeProfitPct: [0.04, 0.08],
  cooldownBars: [10],
  warmupBars: [50],
  minConfidence: [0.15, 0.25],
  fusionMethod: ['weighted'],
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Map common symbols to realistic base prices for synthetic data generation.
 * Unknown symbols default to 100.
 */
function symbolBasePrice(symbol) {
  return DEFAULT_BASE_PRICES[symbol] ?? 100;
}

/**
 * Deterministic hash from a string — used to derive reproducible seeds per symbol.
 */
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash |= 0;
  }
  return Math.abs(hash) || 1;
}

// ─── Main Scanner ─────────────────────────────────────────────────────────────

/**
 * Run the optimizer across multiple symbols simultaneously and rank by score.
 *
 * @param {string[]} symbols  Array of trading symbols (e.g. ['BTC-USD', 'ETH-USD'])
 * @param {object}  [options]
 * @param {number}  [options.days=30]          Days of synthetic data per symbol
 * @param {string}  [options.tf='1h']           Timeframe key ('1m','5m','15m','1h','4h','1d')
 * @param {number}  [options.volatility=0.012]  Volatility for synthetic candle generation
 * @param {object|number} [options.basePrice]   Per-symbol base price map or single default number
 * @param {string}  [options.objective]         Objective function name for optimizer
 * @param {number}  [options.maxCombinations]   Grid search max combinations limit
 * @param {number}  [options.maxIterations=3]   Hill climb max iterations
 * @param {boolean} [options.verbose=false]     Log progress to stderr
 * @returns {Promise<Array<{
 *   symbol: string,
 *   bestScore: number,
 *   bestParams: object,
 *   bestStats: object,
 *   objective: string,
 *   rank: number
 * }>>}
 */
export async function scanSymbols(symbols, options = {}) {
  const {
    days = 30,
    tf = '1h',
    volatility = 0.012,
    basePrice: basePriceOption,
    objective,
    maxCombinations,
    maxIterations = 3,
    verbose = false,
  } = options;

  if (!symbols || symbols.length === 0) return [];

  const barsPerDay = BARS_PER_TF[tf] ?? 24;
  const totalBars = days * barsPerDay;

  if (verbose) {
    console.error(`Scanner: ${symbols.length} symbols, ${totalBars} ${tf} bars each (${days} days)`);
  }

  const results = [];

  for (let i = 0; i < symbols.length; i++) {
    const symbol = symbols[i];

    if (verbose) {
      console.error(`  [${i + 1}/${symbols.length}] ${symbol} — generating candles...`);
    }

    // Resolve base price: per-symbol map, single number, or default
    const basePrice =
      (basePriceOption && typeof basePriceOption === 'object'
        ? basePriceOption[symbol]
        : undefined) ??
      (typeof basePriceOption === 'number' ? basePriceOption : null) ??
      symbolBasePrice(symbol);

    // Deterministic seed so re-scans with same params produce identical results
    const seed = hashString(`${symbol}:${days}:${tf}:${volatility}`);

    const candles = generateSyntheticCandles(totalBars, {
      seed,
      basePrice,
      volatility,
    });

    if (verbose) {
      console.error(`  [${i + 1}/${symbols.length}] ${symbol} — running optimizer...`);
    }

    const optResult = optimize(candles, {
      objective,
      maxCombinations,
      coarseSpace: DEFAULT_COARSE_SPACE,
      maxIterations,
      verbose: false,
      symbol,
    });

    results.push({
      symbol,
      bestScore: optResult.bestScore,
      bestParams: { ...optResult.bestParams },
      bestStats: { ...optResult.bestStats },
      objective: optResult.objective,
    });
  }

  // Sort by score descending, then assign ranks
  results.sort((a, b) => b.bestScore - a.bestScore);
  results.forEach((r, i) => { r.rank = i + 1; });

  return results;
}

// ─── Table Formatter ──────────────────────────────────────────────────────────

/**
 * Format scan results as a plain-text table (no ANSI codes).
 *
 * Columns: RANK, SYMBOL, SCORE, TRADES, WIN%, PF, PNL%, BEST_SL, BEST_TP
 *
 * @param {Array} results  Output from scanSymbols
 * @returns {string}  Formatted table string
 */
export function scanResultsToTable(results) {
  if (!results || results.length === 0) {
    return '(no results)';
  }

  const COLUMNS = [
    { label: 'RANK', width: 5, align: 'right' },
    { label: 'SYMBOL', width: 10, align: 'left' },
    { label: 'SCORE', width: 8, align: 'right' },
    { label: 'TRADES', width: 7, align: 'right' },
    { label: 'WIN%', width: 7, align: 'right' },
    { label: 'PF', width: 8, align: 'right' },
    { label: 'PNL%', width: 8, align: 'right' },
    { label: 'BEST_SL', width: 8, align: 'right' },
    { label: 'BEST_TP', width: 8, align: 'right' },
  ];

  function pad(val, col) {
    const s = String(val);
    return col.align === 'right' ? s.padStart(col.width) : s.padEnd(col.width);
  }

  const lines = [];

  // Header row
  const header = COLUMNS.map(c => c.label.padEnd(c.width)).join(' ');
  const sep = COLUMNS.map(c => '─'.repeat(c.width)).join(' ');
  lines.push(header);
  lines.push(sep);

  // Data rows
  for (const r of results) {
    const s = r.bestStats ?? {};
    const trades = s.totalTrades ?? 0;
    const winPct = s.winRate !== undefined ? (s.winRate * 100).toFixed(1) : 'N/A';
    const pf =
      s.profitFactor === Infinity
        ? 'Inf'
        : s.profitFactor !== undefined && s.profitFactor !== null
          ? s.profitFactor.toFixed(2)
          : 'N/A';
    const pnlPct = s.totalPnl !== undefined
      ? (s.totalPnl * 100).toFixed(2)
      : 'N/A';
    const sl = r.bestParams?.stopLossPct !== undefined
      ? (r.bestParams.stopLossPct * 100).toFixed(1) + '%'
      : 'N/A';
    const tp = r.bestParams?.takeProfitPct !== undefined
      ? (r.bestParams.takeProfitPct * 100).toFixed(1) + '%'
      : 'N/A';

    const row = [
      pad(r.rank ?? '-', COLUMNS[0]),
      pad(r.symbol ?? '?', COLUMNS[1]),
      pad(r.bestScore?.toFixed(4) ?? 'N/A', COLUMNS[2]),
      pad(trades, COLUMNS[3]),
      pad(winPct, COLUMNS[4]),
      pad(pf, COLUMNS[5]),
      pad(pnlPct, COLUMNS[6]),
      pad(sl, COLUMNS[7]),
      pad(tp, COLUMNS[8]),
    ];

    lines.push(row.join(' '));
  }

  return lines.join('\n');
}

// ─── Recommender ──────────────────────────────────────────────────────────────

/**
 * Filter results by minimum score and provide human-readable reasons.
 *
 * @param {Array}  results    Output from scanSymbols
 * @param {number} [minScore=1.0]  Minimum score threshold
 * @returns {Array<{ symbol: string, score: number, reason: string }>}
 */
export function recommendSymbols(results, minScore = 1.0) {
  if (!results || results.length === 0) return [];

  return results
    .filter(r => r.bestScore >= minScore)
    .map(r => {
      const reasons = [];
      const s = r.bestStats;

      if (s) {
        if (s.profitFactor >= 2) reasons.push('Strong PF');
        else if (s.profitFactor >= 1.5) reasons.push('Good PF');
        else if (s.profitFactor !== undefined && s.profitFactor !== null) reasons.push(`PF=${s.profitFactor.toFixed(2)}`);

        if (s.winRate >= 0.6) reasons.push('High Win Rate');
        else if (s.winRate >= 0.5) reasons.push('Pos Win Rate');
        else if (s.winRate !== undefined) reasons.push(`Win=${(s.winRate * 100).toFixed(0)}%`);

        if (s.totalPnl > 0.05) reasons.push('Strong PnL');
        else if (s.totalPnl > 0) reasons.push('Pos PnL');
        else if (s.totalPnl !== undefined) reasons.push(`PnL=${(s.totalPnl * 100).toFixed(1)}%`);

        if (s.totalTrades >= 10) reasons.push('Enough Trades');
        else if (s.totalTrades >= 5) reasons.push('Some Trades');
      }

      if (reasons.length === 0) {
        reasons.push(`Score=${r.bestScore.toFixed(2)}`);
      }

      return {
        symbol: r.symbol,
        score: r.bestScore,
        reason: reasons.join(' + '),
      };
    });
}
