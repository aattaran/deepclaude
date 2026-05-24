/**
 * Portfolio Correlation Matrix Module
 *
 * Pure computation module that takes OHLCV candle arrays for multiple symbols
 * and computes correlation matrices, portfolio variance, diversification
 * scores, and formatted terminal tables.
 *
 * ES module. Zero npm dependencies.
 *
 * Exports:
 *   alignSeries, computeCorrelationMatrix, computePortfolioVariance,
 *   diversificationScore, renderCorrelationTable
 */

// ─── Statistics Helpers ────────────────────────────────────────────────────────

/**
 * Pearson correlation coefficient between two arrays of equal length.
 * Uses a mean-centered (computational) formula for better numerical
 * stability. Returns 0 when either array has zero variance or lengths
 * differ. Clamped to [-1, 1] to correct floating-point drift.
 */
function pearsonCorrelation(x, y) {
    const n = x.length;
    if (n !== y.length || n < 2) return 0;

    // Mean-center for numerical stability (avoids catastrophic cancellation
    // in the n*sumX2 - sumX^2 path).
    let sumX = 0, sumY = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
    }
    const meanX = sumX / n;
    const meanY = sumY / n;

    let cov = 0, varX = 0, varY = 0;
    for (let i = 0; i < n; i++) {
        const dx = x[i] - meanX;
        const dy = y[i] - meanY;
        cov  += dx * dy;
        varX += dx * dx;
        varY += dy * dy;
    }

    const den = Math.sqrt(varX * varY);
    if (den === 0) return 0;
    return Math.max(-1, Math.min(1, cov / den));
}

/**
 * Compute daily log returns from an array of close prices.
 * Returns an array of length closes.length - 1.
 */
function computeLogReturns(closes) {
    const returns = [];
    for (let i = 1; i < closes.length; i++) {
        const prev = closes[i - 1];
        returns.push(prev === 0 ? 0 : Math.log(closes[i] / prev));
    }
    return returns;
}

/**
 * Extract the `close` field from each candle.
 */
function extractCloses(candles) {
    return candles.map(c => c.close);
}

// ─── Series Alignment ──────────────────────────────────────────────────────────

/**
 * Trim all candle series to the same length (the shortest input array).
 *
 * @param {Object<string, Array>} symbolData - { 'BTC-USD': [...candles], ... }
 * @returns {Object<string, Array>} Aligned symbol → candles
 */
export function alignSeries(symbolData) {
    const symbols = Object.keys(symbolData);
    if (symbols.length === 0) return {};

    const minLen = Math.min(...symbols.map(s => symbolData[s].length));
    if (minLen === 0) return {};

    const result = {};
    for (const sym of symbols) {
        const arr = symbolData[sym];
        result[sym] = arr.slice(arr.length - minLen);
    }
    return result;
}

// ─── Cluster Detection ─────────────────────────────────────────────────────────

/**
 * Build connected-component clusters from a correlation matrix.
 * Two symbols are connected if their correlation exceeds the threshold.
 * Only returns clusters with 2+ members.
 */
function buildClusters(symbols, matrix, threshold) {
    const adj = {};
    for (const sym of symbols) adj[sym] = [];

    for (let i = 0; i < symbols.length; i++) {
        for (let j = i + 1; j < symbols.length; j++) {
            const a = symbols[i];
            const b = symbols[j];
            if ((matrix[a]?.[b] ?? 0) > threshold) {
                adj[a].push(b);
                adj[b].push(a);
            }
        }
    }

    const visited = new Set();
    const clusters = [];

    for (const sym of symbols) {
        if (visited.has(sym)) continue;

        const component = [];
        const queue = [sym];
        visited.add(sym);

        while (queue.length > 0) {
            const node = queue.shift();
            component.push(node);
            for (const neighbor of adj[node]) {
                if (!visited.has(neighbor)) {
                    visited.add(neighbor);
                    queue.push(neighbor);
                }
            }
        }

        if (component.length > 1) {
            clusters.push([...component].sort());
        }
    }

    return clusters;
}

// ─── Core: Correlation Matrix ──────────────────────────────────────────────────

/**
 * Compute the Pearson correlation matrix for a set of symbol → candle arrays.
 *
 * @param {Object<string, Array>} symbolData - { 'BTC-USD': [...candles], ... }
 * @param {Object} [options]
 * @param {number} [options.clusterThreshold=0.7]  Threshold for cluster grouping
 * @returns {{ matrix: Object, symbols: string[], clusters: string[][], warnings: string[] }}
 */
export function computeCorrelationMatrix(symbolData, options = {}) {
    const clusterThreshold = options.clusterThreshold ?? 0.7;

    const symbols = Object.keys(symbolData);
    if (symbols.length === 0) {
        return { matrix: {}, symbols: [], clusters: [], warnings: [] };
    }

    // Align all series to the same length
    const aligned = alignSeries(symbolData);

    // Compute log returns for each symbol
    const returnMap = {};
    for (const sym of symbols) {
        returnMap[sym] = computeLogReturns(extractCloses(aligned[sym]));
    }

    const n = symbols.length;
    const matrix = {};
    const warnings = [];

    for (let i = 0; i < n; i++) {
        const symI = symbols[i];
        matrix[symI] = {};

        for (let j = 0; j < n; j++) {
            const symJ = symbols[j];

            if (i === j) {
                matrix[symI][symJ] = 1.0;
            } else if (j < i) {
                // Mirror the already-computed half
                matrix[symI][symJ] = matrix[symJ][symI];
            } else {
                const corr = pearsonCorrelation(returnMap[symI], returnMap[symJ]);
                matrix[symI][symJ] = corr;

                if (corr > 0.85) {
                    warnings.push(
                        `High correlation: ${symI} ↔ ${symJ} = ${corr.toFixed(4)}`
                    );
                }
            }
        }
    }

    const clusters = buildClusters(symbols, matrix, clusterThreshold);

    return {
        matrix,
        symbols: [...symbols].sort(),
        clusters,
        warnings,
    };
}

// ─── Portfolio Variance ────────────────────────────────────────────────────────

/**
 * Compute portfolio variance using the Markowitz formula:
 *   sigma^2 = w' * (diag(sigma) * corr * diag(sigma)) * w
 *
 * @param {Object<string, number>} weights     - { BTC-USD: 0.4, ... }
 * @param {Object<string, Object<string, number>>} corrMatrix - correlation matrix
 * @param {Object<string, number>} vols        - per-asset volatility (std dev)
 * @returns {number}
 */
export function computePortfolioVariance(weights, corrMatrix, vols) {
    const symbols = Object.keys(weights);
    if (symbols.length === 0) return 0;

    let variance = 0;
    for (let i = 0; i < symbols.length; i++) {
        const symI = symbols[i];
        const wI = weights[symI] ?? 0;
        const sigI = vols[symI] ?? 0;

        for (let j = 0; j < symbols.length; j++) {
            const symJ = symbols[j];
            const wJ = weights[symJ] ?? 0;
            const sigJ = vols[symJ] ?? 0;
            const rhoIJ = corrMatrix[symI]?.[symJ] ?? 0;

            variance += wI * wJ * sigI * sigJ * rhoIJ;
        }
    }
    return variance;
}

// ─── Diversification Score ─────────────────────────────────────────────────────

/**
 * Compute a 0–1 diversification score.
 *   1 = perfectly uncorrelated (all off-diagonals are 0 or negative)
 *   0 = all assets move in lockstep (all off-diagonals near 1)
 *
 * Formula: 1 - average(absolute value of off-diagonal correlations)
 *
 * @param {Object<string, Object<string, number>>} corrMatrix
 * @returns {number}
 */
export function diversificationScore(corrMatrix) {
    const symbols = Object.keys(corrMatrix);
    if (symbols.length <= 1) return 1.0;

    let sum = 0;
    let count = 0;

    for (let i = 0; i < symbols.length; i++) {
        const symI = symbols[i];
        for (let j = i + 1; j < symbols.length; j++) {
            const symJ = symbols[j];
            const val = corrMatrix[symI]?.[symJ];
            if (val !== undefined) {
                sum += Math.abs(val);
                count++;
            }
        }
    }

    if (count === 0) return 1.0;
    return 1 - sum / count;
}

// ─── Terminal Table ────────────────────────────────────────────────────────────

const ANSI = {
    reset:  '\x1b[0m',
    green:  '\x1b[32m',
    yellow: '\x1b[33m',
    red:    '\x1b[31m',
};

/**
 * Format a single correlation value with ANSI color:
 *   green  — |value| < 0.3
 *   yellow — 0.3 <= |value| < 0.7
 *   red    — 0.7 <= |value|
 */
function colorCell(val, width) {
    const formatted = val.toFixed(4).padStart(width);
    const absVal = Math.abs(val);
    if (absVal < 0.3) return `${ANSI.green}${formatted}${ANSI.reset}`;
    if (absVal < 0.7) return `${ANSI.yellow}${formatted}${ANSI.reset}`;
    return `${ANSI.red}${formatted}${ANSI.reset}`;
}

/**
 * Render the correlation matrix as a compact lower-triangular table
 * suitable for terminal display.
 *
 * @param {Object<string, Object<string, number>>} matrix   - correlation matrix
 * @param {string[]} symbols   - symbol list (order determines row/col order)
 * @returns {string}  Formatted terminal string
 */
export function renderCorrelationTable(matrix, symbols) {
    if (symbols.length === 0) return '(empty)';

    const sorted = [...symbols].sort();
    const colWidth = Math.max(8, ...sorted.map(s => s.length));

    let out = '';

    // Header row
    out += ''.padEnd(colWidth) + '  ';
    for (const sym of sorted) {
        out += sym.padEnd(colWidth + 2);
    }
    out += '\n';

    // Separator
    out += '─'.repeat(colWidth) + '──';
    for (let i = 0; i < sorted.length; i++) {
        out += '─'.repeat(colWidth + 2);
    }
    out += '\n';

    // Lower-triangular data rows
    for (let i = 0; i < sorted.length; i++) {
        const rowSym = sorted[i];
        out += rowSym.padEnd(colWidth + 2);

        for (let j = 0; j < sorted.length; j++) {
            if (j <= i) {
                const val = matrix[rowSym]?.[sorted[j]];
                if (val !== undefined) {
                    out += colorCell(val, colWidth) + '  ';
                } else {
                    out += 'N/A'.padEnd(colWidth) + '  ';
                }
            } else {
                out += ''.padEnd(colWidth + 2);
            }
        }
        out += '\n';
    }

    return out;
}
