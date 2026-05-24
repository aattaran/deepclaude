/**
 * DeepClaude Trading Dashboard -- terminal live dashboard
 *
 * Renders zones, regime, signals, and decisions in a colorized CLI interface.
 * Zero npm dependencies.  ESM module.  Uses only Node.js built-ins (readline,
 * process.stdout/stderr).
 *
 * Usage:
 *   import { renderDashboard, colorize, createLiveDashboard }
 *     from './audit/dashboard.mjs';
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_WIDTH = 61;
const MIN_WIDTH = 40;
const TITLE = 'DEEPCLAUDE';

// ANSI escape codes used by colorize()
const A = {
    reset: '\x1b[0m',
    green: '\x1b[32m',
    red: '\x1b[31m',
    yellow: '\x1b[33m',
    cyan: '\x1b[36m',
    bold: '\x1b[1m',
    dim: '\x1b[2m',
    white: '\x1b[37m',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(n, lo, hi) { return Math.min(Math.max(n, lo), hi); }

/**
 * Truncate string with a trailing ellipsis (U+2026) if it exceeds maxLen.
 */
function truncate(str, maxLen) {
    if (maxLen <= 0) return '';
    const s = String(str ?? '');
    if (s.length <= maxLen) return s;
    return s.slice(0, Math.max(maxLen - 1, 0)) + '…';
}

/**
 * Right-pad (or truncate) a value to exactly `len` characters.
 */
function padRight(str, len) {
    const s = truncate(str, len);
    return s + ' '.repeat(Math.max(0, len - s.length));
}

/**
 * Left-pad (or truncate) a value to exactly `len` characters.
 */
function padLeft(str, len) {
    const s = truncate(str, len);
    return ' '.repeat(Math.max(0, len - s.length)) + s;
}

/**
 * Repeat a character N times.
 */
function repeat(ch, n) {
    return ch.repeat(Math.max(0, n));
}

// ---------------------------------------------------------------------------
// Row builders
// ---------------------------------------------------------------------------

/**
 * Build a horizontal border / separator line.
 *
 * @param {number} width  Total line width (including corners).
 * @param {string} left   Left character (┌ ├ └).
 * @param {string} fill   Fill character (─).
 * @param {string} right  Right character (┐ ┤ ┘).
 * @returns {string}
 */
function borderLine(width, left, fill, right) {
    return left + repeat(fill, width - 2) + right;
}

/**
 * Build a data row with N equally-spaced columns.
 *
 * Layout: │  col0  │  col1  │ ... │
 * Overhead = (5 * N + 1) characters  (N+1 pipe chars + 2N space-pairs)
 *
 * @param {number} width    Total line width.
 * @param {string[]} columns  Column display strings (before padding).
 * @returns {string}
 */
function dataRow(width, columns) {
    const n = columns.length;
    if (n === 0) return '│' + repeat(' ', width - 2) + '│';

    const overhead = 5 * n + 1;
    const available = width - overhead;

    if (available <= 0) {
        // Extremely narrow fallback — use 2-space padding
        const min = columns.map(c => truncate(c, 1));
        return '│  ' + min.join('  │  ') + '  │';
    }

    const baseW = Math.floor(available / n);
    const remainder = available - baseW * n;

    const parts = columns.map((col, i) => {
        const w = baseW + (i < remainder ? 1 : 0);
        return '  ' + padRight(col, w) + '  ';
    });

    return '│' + parts.join('│') + '│';
}

// ---------------------------------------------------------------------------
// Value formatters (used by renderDashboard)
// ---------------------------------------------------------------------------

/**
 * Format a numeric score with sign.
 *  +0.65  /  -0.30  /   0.00
 */
function formatScore(value) {
    if (value == null || Number.isNaN(Number(value))) return ' 0.00';
    const v = Number(value);
    const sign = v >= 0 ? '+' : '';
    return sign + v.toFixed(2);
}

/**
 * Format a zone row item:  $price  ██  strength
 */
function formatZoneItem(price, strength, colWidth) {
    const priceStr = '$' + Number(price).toFixed(2);
    const strengthStr = Number(strength).toFixed(1);
    // Two spaces: one before bar, one before strength
    const barMax = Math.max(colWidth - priceStr.length - strengthStr.length - 2, 0);
    const blocks = Math.round(Number(strength) * barMax);
    const bar = '█'.repeat(Math.min(blocks, barMax));
    return priceStr + ' ' + bar + ' ' + strengthStr;
}

/**
 * Format an active-zone entry:  TYPE  $price
 */
function formatActiveZone(zone, colWidth) {
    const type = truncate((zone.type || '').toUpperCase(), Math.max(colWidth - 8, 2));
    const priceStr = zone.price != null ? '$' + Number(zone.price).toFixed(2) : '';
    return (type + ' ' + priceStr).trim();
}

/**
 * Format a param key/value pair for the params column.
 *
 * Known keys map to human-readable labels; unknown keys are shown as-is.
 */
function formatParam(key, value) {
    const labelPad = 8;
    let label;
    switch (key) {
        case 'stopMultiplier':       label = 'Stop:';      break;
        case 'tpAggressiveness':     label = 'TP:';        break;
        case 'positionSizeFactor':   label = 'Pos Size:';  break;
        case 'trailingStopPct':      label = 'Trail:';     break;
        case 'maxHoldingBars':       label = 'Max Bars:';  break;
        default:                     label = key + ':';     break;
    }
    const fv = typeof value === 'number' ? value.toFixed(1) : String(value ?? '');
    if (key === 'trailingStopPct' || key === 'trailingStop') {
        return padRight(label, labelPad) + ' ' + fv + '%';
    }
    if (key === 'stopMultiplier') {
        return padRight(label, labelPad) + ' ' + fv + 'x ATR';
    }
    if (key === 'tpAggressiveness') {
        return padRight(label, labelPad) + ' ' + fv + ' agg';
    }
    if (key === 'positionSizeFactor') {
        return padRight(label, labelPad) + ' ' + fv + 'x';
    }
    return padRight(label, labelPad) + ' ' + fv;
}

/**
 * Build the regime description line.
 */
function formatRegimeLine(regime, confidence) {
    const c = (confidence != null ? Number(confidence) : 0).toFixed(2);
    const desc = regime.replace(/_/g, ' ');
    return 'REGIME: ' + regime + ' (' + c + ') — ' + desc;
}

// ---------------------------------------------------------------------------
// renderDashboard  (plain text, no ANSI codes)
// ---------------------------------------------------------------------------

/**
 * Render a complete trading dashboard as a plain-text string.
 *
 * @param {object}  data              Trading decision (TradingDecision shape).
 * @param {string}  [options.width]   Total dashboard width in chars.
 * @returns {string}                  Multi-line plain-text dashboard.
 */
export function renderDashboard(data, options) {
    if (!data || typeof data !== 'object') {
        return 'No data';
    }

    const opts = options || {};
    const width = clamp(
        opts.width != null ? opts.width : DEFAULT_WIDTH,
        MIN_WIDTH,
        200,
    );

    const {
        symbol = 'UNKNOWN',
        price,
        action = 'HOLD',
        confidence = 0,
        compositeScore = 0,
        regime = 'unknown',
        regimeConfidence = 0,
        support,
        resistance,
        activeZones = [],
        signals = [],
        params = {},
        reasoning = '',
    } = data;

    const lines = [];

    // --- Price string ---
    const priceStr = price != null ? '$' + Number(price).toFixed(2) : 'N/A';

    // --- Top border ---
    lines.push(borderLine(width, '┌', '─', '┐'));

    // --- Header: TITLE | SYMBOL | PRICE | REGIME ---
    lines.push(dataRow(width, [TITLE, symbol, priceStr, regime.toUpperCase()]));

    // --- Separator ---
    lines.push(borderLine(width, '├', '─', '┤'));

    // --- Composite / Confidence ---
    const arrow = compositeScore > 0
        ? '▲'
        : compositeScore < 0
            ? '▼'
            : '◆';
    const compText = 'Composite: ' + arrow + ' '
        + Math.abs(Number(compositeScore)).toFixed(2)
        + ' (' + action + ')';
    const confText = 'Confidence: ' + (Number(confidence) * 100).toFixed(0) + '%';
    lines.push(dataRow(width, [compText, confText]));

    // --- Separator ---
    lines.push(borderLine(width, '├', '─', '┤'));

    // --- Zone section header ---
    lines.push(dataRow(width, ['SUPPORTS', 'RESISTANCES', 'ACTIVE ZONES']));

    // --- Zone data rows ---
    const zcw = Math.floor((width - 16) / 3); // 3 columns
    const maxZoneRows = Math.max(
        support && support.price != null ? 1 : 0,
        resistance && resistance.price != null ? 1 : 0,
        activeZones.length,
        1,
    );

    for (let i = 0; i < maxZoneRows; i++) {
        let suppText = '';
        if (i === 0 && support && support.price != null) {
            suppText = formatZoneItem(support.price, support.strength || 0, zcw);
        }

        let resText = '';
        if (i === 0 && resistance && resistance.price != null) {
            resText = formatZoneItem(resistance.price, resistance.strength || 0, zcw);
        }

        let zoneText = '';
        if (i < activeZones.length) {
            zoneText = formatActiveZone(activeZones[i], zcw);
        }

        lines.push(dataRow(width, [suppText, resText, zoneText]));
    }

    // --- Separator ---
    lines.push(borderLine(width, '├', '─', '┤'));

    // --- Signals / Params section header ---
    lines.push(dataRow(width, ['SIGNALS', 'PARAMS']));

    // --- Signal rows ---
    const sigColWidth = Math.floor((width - 11) / 2); // 2 columns
    const sigNameWidth = Math.max(sigColWidth - 11, 2);
    const maxSigRows = Math.max(signals.length, Object.keys(params).length, 1);

    for (let i = 0; i < maxSigRows; i++) {
        let sigText = '';
        if (i < signals.length) {
            const s = signals[i];
            sigText = padRight(s.name || '', sigNameWidth)
                + ' ' + formatScore(s.value)
                + ' ' + (Number(s.confidence) || 0).toFixed(2);
        }

        let paramText = '';
        const pKeys = Object.keys(params);
        if (i < pKeys.length) {
            paramText = formatParam(pKeys[i], params[pKeys[i]]);
        }

        lines.push(dataRow(width, [sigText, paramText]));
    }

    // --- Separator ---
    lines.push(borderLine(width, '├', '─', '┤'));

    // --- Regime description ---
    lines.push(dataRow(width, [formatRegimeLine(regime, regimeConfidence)]));

    // --- Decision line ---
    const decBase = 'DECISION: ' + action + ' — '
        + (reasoning || 'composite=' + Number(compositeScore).toFixed(2));
    lines.push(dataRow(width, [decBase]));

    // --- Bottom border ---
    lines.push(borderLine(width, '└', '─', '┘'));

    return lines.join('\n');
}

// ---------------------------------------------------------------------------
// colorize  (ANSI escape code wrapper)
// ---------------------------------------------------------------------------

/**
 * Wrap plain dashboard text with ANSI escape codes for terminal coloring.
 *
 * @param {string} text   Plain text from renderDashboard().
 * @returns {string}      Text with ANSI color codes applied.
 */
export function colorize(text) {
    const RST = A.reset;
    const GRN = A.green;
    const RED = A.red;
    const YEL = A.yellow;
    const CYN = A.cyan;
    const BLD = A.bold;
    const DIM = A.dim;
    const WHT = A.white;
    const BWHT = BLD + A.white;
    const BGRN = BLD + A.green;
    const BRED = BLD + A.red;

    const lines = (text || '').split('\n');

    return lines.map(line => {
        // Border / ruler lines -- leave as-is
        if (/[┌┐└┘├┤─]/.test(line)) {
            return line;
        }

        let c = line;

        // Colour header labels (all-caps section headings)
        c = c.replace(
            /(SUPPORTS|RESISTANCES|ACTIVE ZONES|SIGNALS|PARAMS)/g,
            BLD + CYN + '$1' + RST,
        );

        // Colour action words
        c = c.replace(/\bBUY\b/g, BGRN + 'BUY' + RST);
        c = c.replace(/\bSELL\b/g, BRED + 'SELL' + RST);
        c = c.replace(/\bHOLD\b/g, BLD + YEL + 'HOLD' + RST);

        // Colour prices ($X.XX)
        c = c.replace(/\$\d+\.\d{2}/g, BWHT + '$&' + RST);

        // Colour confidence based on value
        c = c.replace(/Confidence: (\d+)%/g, (_m, pct) => {
            const n = parseInt(pct, 10);
            if (n > 70) return 'Confidence: ' + BGRN + pct + '%' + RST;
            if (n < 30) return 'Confidence: ' + DIM + WHT + pct + '%' + RST;
            return 'Confidence: ' + pct + '%';
        });

        // Colour DECISION and REGIME labels
        c = c.replace(/\b(DECISION|REGIME):/g, BLD + CYN + '$1' + RST + ':');

        // Colour action arrows
        c = c.replace(/▲/g, GRN + BLD + '$&' + RST);
        c = c.replace(/▼/g, RED + BLD + '$&' + RST);
        c = c.replace(/◆/g, YEL + BLD + '$&' + RST);

        return c;
    }).join('\n');
}

// ---------------------------------------------------------------------------
// createLiveDashboard
// ---------------------------------------------------------------------------

/**
 * Create a live terminal dashboard controller.
 *
 * @param {object}  [options]
 * @param {string}  [options.symbol='UNKNOWN']  Trading pair symbol.
 * @param {number}  [options.refreshMs=1000]     Refresh interval in ms.
 * @param {number}  [options.width]               Dashboard width.
 * @returns {{ update: Function, start: Function, stop: Function }}
 */
export function createLiveDashboard(options) {
    const opts = options || {};
    const refreshMs = opts.refreshMs || 1000;
    const width = opts.width != null ? opts.width : DEFAULT_WIDTH;

    let currentData = null;
    let intervalId = null;

    /**
     * Update the dashboard with new data and render it immediately.
     */
    function update(data) {
        currentData = data;
        const plain = renderDashboard(data, { width });
        const colored = colorize(plain);
        console.log(colored);
    }

    /**
     * Start real-time refresh loop.  Clears screen and begins periodic
     * re-renders.  Safe to call multiple times (no-op if already running).
     */
    function start() {
        if (intervalId) return;
        // Clear screen and home cursor
        process.stdout.write('\x1b[2J\x1b[3J\x1b[H');
        if (currentData) {
            const plain = renderDashboard(currentData, { width });
            const colored = colorize(plain);
            console.log(colored);
        }
        intervalId = setInterval(() => {
            if (currentData) {
                process.stdout.write('\x1b[H\x1b[J');
                const plain = renderDashboard(currentData, { width });
                const colored = colorize(plain);
                console.log(colored);
            }
        }, refreshMs);
        // Allow process to exit even if interval is active
        if (intervalId && typeof intervalId.unref === 'function') {
            intervalId.unref();
        }
    }

    /**
     * Stop the real-time refresh loop.
     */
    function stop() {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
    }

    return { update, start, stop };
}
