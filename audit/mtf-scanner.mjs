/**
 * Multi-Timeframe Confluence Scanner
 *
 * Runs the SAME symbol across multiple timeframes, gets an orchestrator decision
 * for each, and computes a conviction score based on directional agreement.
 *
 * Higher timeframes are weighted more heavily in the conviction calculation,
 * so a SELL on the daily chart carries more weight than a SELL on the 15m chart.
 *
 * Usage:
 *   import { scanMTF, renderMTFReport, computeConviction } from './mtf-scanner.mjs';
 *   const result = await scanMTF('BTC-USD', ['15m', '1h', '4h', '1d']);
 *   console.log(renderMTFReport(result));
 *
 * ES module. Zero npm dependencies.
 */

import { loadOrFetch } from './datafeed.mjs';
import { createOrchestrator } from './orchestrator.mjs';
import { generateSyntheticCandles } from './trade.mjs';

// ─── Constants ─────────────────────────────────────────────────────────────────────

const DEFAULT_TIMEFRAMES = ['15m', '1h', '4h', '1d'];

/** Default days of data per timeframe — higher TFs get more days. */
const DEFAULT_DAYS_PER_TF = {
    '1m': 1, '5m': 2, '15m': 3, '30m': 5,
    '1h': 7, '4h': 14, '1d': 30, '1w': 90,
};

const BARS_PER_DAY = {
    '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
    '1h': 24, '4h': 6, '1d': 1, '1w': 1 / 7,
};

const VERDICT = {
    BULLISH_CONFLUENCE: 'BULLISH_CONFLUENCE',
    BEARISH_CONFLUENCE: 'BEARISH_CONFLUENCE',
    MIXED: 'MIXED',
};

// ─── Helpers ────────────────────────────────────────────────────────────────────────

function signalFromAction(action) {
    if (action === 'BUY') return 1;
    if (action === 'SELL') return -1;
    return 0; // HOLD or unknown
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

/**
 * Assign a weight to a timeframe based on its position in the timeframes array.
 * Higher timeframes (later in the array) get proportionally higher weight.
 *
 * @param {string} tf — timeframe label (e.g., '15m', '1h')
 * @param {string[]} timeframes — ordered list of all timeframes
 * @returns {number} weight value (>= 1)
 */
function weightForTimeframe(tf, timeframes) {
    const idx = timeframes.indexOf(tf);
    if (idx === -1) return 1;
    // 1-based weighting: first TF gets 1, last gets timeframes.length
    return idx + 1;
}

/**
 * Resolve the number of calendar days for a specific timeframe.
 */
function resolveDays(tf, daysOverride) {
    if (daysOverride !== undefined && daysOverride !== null && typeof daysOverride === 'object') {
        return daysOverride[tf] ?? DEFAULT_DAYS_PER_TF[tf] ?? 7;
    }
    if (typeof daysOverride === 'number') {
        return daysOverride;
    }
    return DEFAULT_DAYS_PER_TF[tf] ?? 7;
}

/**
 * Fetch or generate candle data for a single timeframe.
 *
 * When `live` is true, attempts real Binance data via loadOrFetch with synthetic
 * fallback. When `live` is false, generates synthetic data directly (fast path
 * for testing and offline use).
 */
async function fetchDataForTF(symbol, tf, days, opts, tfIndex) {
    const { live = false, cacheDir, seed = 42 } = opts;

    if (live) {
        return loadOrFetch(symbol, tf, days, {
            cacheDir,
            allowSynthetic: true,
            seed: seed + tfIndex,
        });
    }

    // Fast synthetic path — no network calls
    const barsPerDay = BARS_PER_DAY[tf] ?? 24;
    const barCount = Math.max(Math.floor(days * barsPerDay), 50);
    const basePrice = estimateBasePrice(symbol);
    return generateSyntheticCandles(barCount, {
        seed: seed + tfIndex,
        basePrice,
        volatility: 0.012,
    });
}

// ─── Core Computation ───────────────────────────────────────────────────────────────

/**
 * Compute a weighted conviction score from multi-TF decisions.
 *
 * Higher timeframes are weighted more heavily, so a SELL on the daily chart
 * carries more weight than a SELL on the 15m chart when both are present.
 *
 * @param {Array<{ timeframe: string, action: string }>} decisions
 * @param {string[]} timeframes — ordered timeframe labels used for weighting
 * @returns {{ conviction: number, buys: number, sells: number, holds: number, verdict: string }}
 */
export function computeConviction(decisions, timeframes) {
    if (!decisions || decisions.length === 0) {
        return { conviction: 0, buys: 0, sells: 0, holds: 0, verdict: VERDICT.MIXED };
    }

    const tfs = timeframes && timeframes.length > 0
        ? timeframes
        : decisions.map(d => d.timeframe);

    let weightedSum = 0;
    let totalWeight = 0;
    let buys = 0;
    let sells = 0;
    let holds = 0;

    for (const dec of decisions) {
        const signal = signalFromAction(dec.action);
        const weight = weightForTimeframe(dec.timeframe, tfs);

        weightedSum += signal * weight;
        totalWeight += weight;

        if (signal === 1) buys++;
        else if (signal === -1) sells++;
        else holds++;
    }

    const conviction = totalWeight > 0 ? weightedSum / totalWeight : 0;

    let verdict;
    if (conviction >= 0.5) {
        verdict = VERDICT.BULLISH_CONFLUENCE;
    } else if (conviction <= -0.5) {
        verdict = VERDICT.BEARISH_CONFLUENCE;
    } else {
        verdict = VERDICT.MIXED;
    }

    return { conviction, buys, sells, holds, verdict };
}

// ─── Main Scanner ───────────────────────────────────────────────────────────────────

/**
 * Scan a single symbol across multiple timeframes.
 *
 * For each timeframe, fetches or generates candle data, runs the orchestrator
 * to produce a trading decision, then computes a multi-TF conviction score
 * that weights higher timeframes more heavily.
 *
 * @param {string} symbol — e.g. 'BTC-USD', 'SOL-USD'
 * @param {string[]} [timeframes] — timeframe labels (default: ['15m', '1h', '4h', '1d'])
 * @param {number|object} [days] — days per timeframe. A number applies equally to all
 *   TFs; an object maps TF -> days (e.g. { '15m': 2, '1h': 7 }). Falls back to
 *   DEFAULT_DAYS_PER_TF when omitted.
 * @param {object} [opts]
 * @param {number} [opts.minConfidence=0.15] — minimum confidence threshold for
 *   the orchestrator
 * @param {boolean} [opts.live=false] — attempt real Binance data; falls back to
 *   synthetic if unavailable
 * @param {number} [opts.seed=42] — PRNG seed for deterministic synthetic data
 * @param {string} [opts.cacheDir] — custom cache directory for datafeed
 * @returns {Promise<{
 *   symbol: string,
 *   decisions: Array<{
 *     timeframe: string,
 *     action: string,
 *     confidence: number,
 *     compositeScore: number,
 *     reasoning: string,
 *     price: number|null,
 *     regime: string,
 *     candles: number
 *   }>,
 *   conviction: number,
 *   buys: number,
 *   sells: number,
 *   holds: number,
 *   verdict: string,
 *   error?: string
 * }>}
 */
export async function scanMTF(symbol, timeframes = DEFAULT_TIMEFRAMES, days, opts = {}) {
    // Empty timeframes → early return
    if (!timeframes || timeframes.length === 0) {
        return {
            symbol,
            decisions: [],
            conviction: 0,
            buys: 0,
            sells: 0,
            holds: 0,
            verdict: VERDICT.MIXED,
            error: 'no timeframes provided',
        };
    }

    const { minConfidence = 0.15 } = opts;
    const decisions = [];
    const orchestrator = createOrchestrator({ minConfidence });

    for (let i = 0; i < timeframes.length; i++) {
        const tf = timeframes[i];
        const tfDays = resolveDays(tf, days);

        // Fetch or generate candles for this timeframe
        let candles;
        try {
            candles = await fetchDataForTF(symbol, tf, tfDays, opts, i);
        } catch (err) {
            // Last-resort emergency fallback
            const barsPerDay = BARS_PER_DAY[tf] ?? 24;
            const barCount = Math.max(Math.floor(tfDays * barsPerDay), 20);
            const basePrice = estimateBasePrice(symbol);
            candles = generateSyntheticCandles(barCount, {
                seed: opts.seed ?? 42 + i,
                basePrice,
                volatility: 0.012,
            });
        }

        // Insufficient data → HOLD
        if (!candles || candles.length < 10) {
            decisions.push({
                timeframe: tf,
                action: 'HOLD',
                confidence: 0,
                compositeScore: 0,
                reasoning: 'insufficient candle data',
                price: null,
                regime: 'unknown',
                candles: candles?.length ?? 0,
            });
            continue;
        }

        // Run orchestrator
        const decision = orchestrator.run(candles, { symbol });

        decisions.push({
            timeframe: tf,
            action: decision.action,
            confidence: decision.confidence,
            compositeScore: decision.compositeScore,
            reasoning: decision.reasoning,
            price: decision.price,
            regime: decision.regime,
            candles: candles.length,
        });
    }

    // Compute weighted conviction
    const { conviction, buys, sells, holds, verdict } = computeConviction(decisions, timeframes);

    return {
        symbol,
        decisions,
        conviction,
        buys,
        sells,
        holds,
        verdict,
    };
}

// ─── Report Renderer ────────────────────────────────────────────────────────────────

// ANSI escape codes
const C = {
    reset: '\x1b[0m',
    bold: '\x1b[1m',
    dim: '\x1b[2m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
};

function ansiAction(action) {
    switch (action) {
        case 'BUY':  return C.green + C.bold + 'BUY ' + C.reset;
        case 'SELL': return C.red + C.bold + 'SELL' + C.reset;
        case 'HOLD': return C.yellow + 'HOLD' + C.reset;
        default:     return action;
    }
}

function ansiVerdictColor(verdict) {
    if (verdict === 'BULLISH_CONFLUENCE') return C.green;
    if (verdict === 'BEARISH_CONFLUENCE') return C.red;
    return C.yellow;
}

/**
 * Render an MTF scan result as a colorized terminal string.
 *
 * Displays a table with one row per timeframe showing:
 *   TIMEFRAME | ACTION | SCORE | CONFIDENCE | REGIME | CANDLES
 *
 * Followed by a summary line with overall conviction and verdict.
 *
 * @param {object} result — output from scanMTF()
 * @returns {string} — formatted string with ANSI color codes
 */
export function renderMTFReport(result) {
    if (!result) return '(no data)';

    const { symbol, decisions, conviction, buys, sells, holds, verdict } = result;
    const lines = [];

    // ── Header ──
    lines.push('');
    lines.push(`${C.cyan}${C.bold}╔${'═'.repeat(60)}╗${C.reset}`);
    lines.push(`${C.cyan}${C.bold}║${C.reset}${C.bold}  MTF CONFLUENCE SCANNER  |  ${(symbol ?? 'UNKNOWN').padEnd(27)}${C.cyan}${C.bold}║${C.reset}`);
    lines.push(`${C.cyan}${C.bold}╚${'═'.repeat(60)}╝${C.reset}`);
    lines.push('');

    // ── Column Headers ──
    const colPad = (s, w, align = 'left') =>
        align === 'right' ? s.padStart(w) : s.padEnd(w);

    lines.push(
        C.bold +
        colPad('TIMEFRAME', 12) + '  ' +
        colPad('ACTION', 8) + '  ' +
        colPad('SCORE', 8, 'right') + '  ' +
        colPad('CONFIDENCE', 10, 'right') + '  ' +
        colPad('REGIME', 22) + '  ' +
        colPad('CANDLES', 7, 'right') +
        C.reset,
    );
    lines.push(C.dim + '─'.repeat(76) + C.reset);

    // ── Decision Rows ──
    for (const d of decisions) {
        const tfStr = colPad(d.timeframe, 12);

        const actionStr = ansiAction(d.action);

        const scoreVal = d.compositeScore ?? 0;
        const scoreColor = scoreVal > 0.1 ? C.green : scoreVal < -0.1 ? C.red : C.dim;
        const scoreStr = scoreColor + colPad(scoreVal.toFixed(3), 8, 'right') + C.reset;

        const confPct = (d.confidence * 100).toFixed(0) + '%';
        const confColor = d.confidence >= 0.6 ? C.green : d.confidence >= 0.3 ? C.cyan : C.dim;
        const confStr = confColor + colPad(confPct, 10, 'right') + C.reset;

        const regimeStr = C.dim + colPad(d.regime ?? 'unknown', 22) + C.reset;
        const candlesStr = colPad(String(d.candles ?? '?'), 7, 'right');

        lines.push(`  ${tfStr}  ${actionStr}  ${scoreStr}  ${confStr}  ${regimeStr}  ${candlesStr}`);
    }

    // ── Separator ──
    lines.push(C.dim + '─'.repeat(76) + C.reset);

    // ── Summary ──
    const vColor = ansiVerdictColor(verdict);
    const convStr = conviction.toFixed(3);
    const convColor = conviction >= 0.5 ? C.green + C.bold
        : conviction <= -0.5 ? C.red + C.bold
        : C.yellow + C.bold;

    lines.push('');
    lines.push(
        `  ${C.bold}Conviction:${C.reset} ${convColor}${convStr}${C.reset}  ` +
        `|  ${C.bold}Verdict:${C.reset} ${vColor}${C.bold}${verdict}${C.reset}`,
    );
    lines.push(
        `  ${C.dim}BUY: ${buys ?? 0}  ` +
        `SELL: ${sells ?? 0}  ` +
        `HOLD: ${holds ?? 0}  ` +
        `|  Timeframes: ${decisions.length}${C.reset}`,
    );
    lines.push('');

    return lines.join('\n');
}
