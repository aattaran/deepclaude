/**
 * Strategy Parameter Optimizer — grid search + hill climbing for optimal backtest params.
 *
 * Finds the best stop-loss, take-profit, cooldown, warmup, minConfidence, and
 * fusion method for a given symbol and data regime. Supports per-regime optimization.
 *
 * Usage:
 *   import { optimize, gridSearch, hillClimb } from './optimizer.mjs';
 *   const result = optimize(candles, { objective: 'sharpe' });
 *   // => { bestParams, bestScore, allResults, topN }
 *
 * ES module. Zero npm dependencies.
 */

import { runBacktest } from './orchestrator.mjs';

// ─── Constants ──────────────────────────────────────────────────────────────────

const DEFAULT_OBJECTIVE = 'profitFactor';

const OBJECTIVES = {
    profitFactor: (stats) => {
        if (stats.totalTrades === 0) return 0;
        return stats.profitFactor;
    },
    sharpe: (stats) => {
        if (stats.totalTrades === 0) return 0;
        const winRate = stats.winRate;
        const avgWin = Math.abs(stats.avgWin);
        const avgLoss = Math.abs(stats.avgLoss);
        if (avgLoss === 0) return stats.totalPnl > 0 ? 10 : 0;
        const expectancy = winRate * avgWin - (1 - winRate) * avgLoss;
        const stdDev = Math.sqrt(
            winRate * (avgWin - expectancy) ** 2 + (1 - winRate) * (avgLoss + expectancy) ** 2,
        );
        if (stdDev === 0) return expectancy > 0 ? 10 : 0;
        return expectancy / stdDev;
    },
    totalPnl: (stats) => stats.totalPnl,
    winRate: (stats) => {
        if (stats.totalTrades === 0) return 0;
        // Blend win rate with trade count to prefer high-win-rate strategies with volume
        return stats.winRate + stats.totalTrades * 0.001;
    },
    calmar: (stats) => {
        if (stats.totalTrades === 0) return 0;
        // Approximate Calmar: total return / max observed drawdown from trade sequence
        return stats.profitFactor * stats.winRate;
    },
    expectancy: (stats) => {
        if (stats.totalTrades === 0) return 0;
        const avgWin = Math.abs(stats.avgWin);
        const avgLoss = Math.abs(stats.avgLoss);
        if (avgLoss === 0) return stats.totalPnl > 0 ? 10 : 0;
        return stats.winRate * avgWin - (1 - stats.winRate) * avgLoss;
    },
};

// ─── Parameter Space ────────────────────────────────────────────────────────────

/**
 * Default parameter ranges for grid search.
 * Each range is [min, max, step] or a discrete set of values.
 */
const DEFAULT_PARAM_SPACE = {
    stopLossPct: [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05],
    takeProfitPct: [0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
    cooldownBars: [3, 5, 10, 15, 20, 30],
    warmupBars: [30, 50, 70],
    minConfidence: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    fusionMethod: ['weighted', 'bayesian', 'voting'],
};

// ─── 1. Grid Search ─────────────────────────────────────────────────────────────

/**
 * Exhaustive grid search over parameter space.
 *
 * @param {Array} candles — historical OHLCV data
 * @param {object} [opts]
 * @param {string} [opts.objective='profitFactor'] — objective function name
 * @param {object} [opts.paramSpace] — custom parameter ranges (default: DEFAULT_PARAM_SPACE)
 * @param {string} [opts.symbol='BTC-USD']
 * @param {number} [opts.maxCombinations=500] — bail if grid exceeds this
 * @param {boolean} [opts.verbose=false] — log progress
 * @returns {{ bestParams: object, bestScore: number, bestStats: object, allResults: Array, topN: Array }}
 */
export function gridSearch(candles, opts = {}) {
    const objectiveName = opts.objective ?? DEFAULT_OBJECTIVE;
    const objectiveFn = OBJECTIVES[objectiveName] ?? OBJECTIVES.profitFactor;
    const paramSpace = opts.paramSpace ?? DEFAULT_PARAM_SPACE;
    const symbol = opts.symbol ?? 'BTC-USD';
    const maxCombinations = opts.maxCombinations ?? 500;
    const verbose = opts.verbose ?? false;

    // Generate all combinations
    const keys = Object.keys(paramSpace);
    const combinations = cartesianProduct(
        keys.map(k => Array.isArray(paramSpace[k]) ? paramSpace[k] : [paramSpace[k]]),
    );

    if (combinations.length > maxCombinations) {
        throw new Error(
            `Grid size ${combinations.length} exceeds maxCombinations ${maxCombinations}. ` +
            `Reduce paramSpace or increase maxCombinations.`,
        );
    }

    const results = [];
    let bestScore = -Infinity;
    let bestParams = null;
    let bestStats = null;

    for (let i = 0; i < combinations.length; i++) {
        const values = combinations[i];
        const params = {};
        for (let j = 0; j < keys.length; j++) {
            params[keys[j]] = values[j];
        }

        const backtestResult = runBacktest(candles, { ...params, symbol });
        const score = objectiveFn(backtestResult.stats);

        results.push({ params: { ...params }, score, stats: backtestResult.stats });

        if (score > bestScore) {
            bestScore = score;
            bestParams = { ...params };
            bestStats = backtestResult.stats;
        }

        if (verbose && (i + 1) % 50 === 0) {
            console.error(`  grid: ${i + 1}/${combinations.length} — best ${objectiveName}=${bestScore.toFixed(4)}`);
        }
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    return {
        bestParams,
        bestScore,
        bestStats,
        allResults: results,
        topN: results.slice(0, 10),
        objective: objectiveName,
        totalEvaluations: combinations.length,
    };
}

// ─── 2. Hill Climbing ────────────────────────────────────────────────────────────

/**
 * Hill climbing optimizer — starts from an initial guess and iteratively
 * explores neighboring parameter values, moving to improvements.
 *
 * @param {Array} candles
 * @param {object} [opts]
 * @param {string} [opts.objective='profitFactor']
 * @param {object} [opts.initialParams] — starting parameter values
 * @param {object} [opts.paramSpace] — parameter ranges for neighbor generation
 * @param {number} [opts.maxIterations=50] — max hill climb steps
 * @param {number} [opts.neighborsPerIteration=8] — neighbors to evaluate per step
 * @param {string} [opts.symbol='BTC-USD']
 * @param {boolean} [opts.verbose=false]
 * @returns {{ bestParams, bestScore, bestStats, path: Array, iterations: number }}
 */
export function hillClimb(candles, opts = {}) {
    const objectiveName = opts.objective ?? DEFAULT_OBJECTIVE;
    const objectiveFn = OBJECTIVES[objectiveName] ?? OBJECTIVES.profitFactor;
    const paramSpace = opts.paramSpace ?? DEFAULT_PARAM_SPACE;
    const symbol = opts.symbol ?? 'BTC-USD';
    const maxIterations = opts.maxIterations ?? 50;
    const neighborsPerIteration = opts.neighborsPerIteration ?? 8;
    const verbose = opts.verbose ?? false;

    // Initialize from provided params or midpoints of paramSpace
    let current = opts.initialParams ?? midpoints(paramSpace);
    let currentResult = runBacktest(candles, { ...current, symbol });
    let currentScore = objectiveFn(currentResult.stats);

    const path = [{ params: { ...current }, score: currentScore }];
    let iterations = 0;
    let improved = true;

    while (improved && iterations < maxIterations) {
        improved = false;
        iterations++;

        // Generate neighbors
        const neighbors = generateNeighbors(current, paramSpace, neighborsPerIteration);

        let bestNeighborScore = -Infinity;
        let bestNeighborParams = null;
        let bestNeighborStats = null;

        for (const neighbor of neighbors) {
            const result = runBacktest(candles, { ...neighbor, symbol });
            const score = objectiveFn(result.stats);

            if (score > bestNeighborScore) {
                bestNeighborScore = score;
                bestNeighborParams = neighbor;
                bestNeighborStats = result.stats;
            }
        }

        if (bestNeighborScore > currentScore) {
            current = bestNeighborParams;
            currentScore = bestNeighborScore;
            currentResult = { stats: bestNeighborStats };
            path.push({ params: { ...current }, score: currentScore });
            improved = true;

            if (verbose) {
                console.error(`  hill: iter ${iterations} — score ${currentScore.toFixed(4)}`);
            }
        }
    }

    return {
        bestParams: current,
        bestScore: currentScore,
        bestStats: currentResult.stats,
        path,
        iterations,
        objective: objectiveName,
    };
}

// ─── 3. Combined Optimize — grid + hill climb ────────────────────────────────────

/**
 * Run a coarse grid search followed by hill climbing from the best grid result.
 *
 * @param {Array} candles
 * @param {object} [opts]
 * @returns {{ bestParams, bestScore, bestStats, gridResult, hillResult, objective }}
 */
export function optimize(candles, opts = {}) {
    const verbose = opts.verbose ?? false;

    // Phase 1: Coarse grid
    const coarseSpace = opts.coarseSpace ?? {
        stopLossPct: [0.01, 0.02, 0.03, 0.05],
        takeProfitPct: [0.02, 0.04, 0.06, 0.10],
        cooldownBars: [5, 10, 20],
        warmupBars: [50],
        minConfidence: [0.15, 0.25],
        fusionMethod: ['weighted'],
    };

    if (verbose) console.error('Phase 1: Coarse grid search...');
    const gridResult = gridSearch(candles, {
        ...opts,
        paramSpace: coarseSpace,
        verbose,
    });

    if (verbose) {
        console.error(`  Grid best: ${gridResult.objective}=${gridResult.bestScore.toFixed(4)}`);
        console.error('Phase 2: Hill climbing from grid best...');
    }

    // Phase 2: Hill climb from grid best
    const hillResult = hillClimb(candles, {
        ...opts,
        initialParams: gridResult.bestParams,
        verbose,
    });

    if (verbose) {
        console.error(`  Hill best: ${hillResult.objective}=${hillResult.bestScore.toFixed(4)}`);
        console.error(`  Improvement: ${((hillResult.bestScore - gridResult.bestScore) / Math.max(Math.abs(gridResult.bestScore), 0.001) * 100).toFixed(1)}%`);
    }

    return {
        bestParams: hillResult.bestParams,
        bestScore: hillResult.bestScore,
        bestStats: hillResult.bestStats,
        gridResult,
        hillResult,
        objective: gridResult.objective,
    };
}

// ─── 4. Per-Regime Optimization ─────────────────────────────────────────────────

/**
 * Optimize parameters separately for each market regime.
 *
 * @param {Array} candles — must have _regime tags (from generateSyntheticCandles)
 * @param {object} [opts]
 * @returns {object} — { regimeName: { bestParams, bestScore, bestStats } }
 */
export function optimizePerRegime(candles, opts = {}) {
    const verbose = opts.verbose ?? false;

    // Partition candles by their regime tag
    const byRegime = {};
    for (const c of candles) {
        const r = c._regime ?? 'unknown';
        if (!byRegime[r]) byRegime[r] = [];
        byRegime[r].push(c);
    }

    const results = {};
    for (const [regime, regimeCandles] of Object.entries(byRegime)) {
        if (regimeCandles.length < 60) {
            if (verbose) console.error(`  Skipping ${regime}: only ${regimeCandles.length} candles`);
            continue;
        }

        if (verbose) console.error(`Optimizing ${regime} (${regimeCandles.length} candles)...`);
        results[regime] = optimize(regimeCandles, { ...opts, verbose: false });

        if (verbose) {
            console.error(`  ${regime}: best score=${results[regime].bestScore.toFixed(4)}`);
        }
    }

    return results;
}

// ─── 5. Sensitivity Analysis ────────────────────────────────────────────────────

/**
 * Vary one parameter while holding others fixed to measure sensitivity.
 *
 * @param {Array} candles
 * @param {object} baseParams — baseline parameter values
 * @param {string} varyKey — which parameter to vary
 * @param {Array} varyValues — values to test for the varied parameter
 * @param {object} [opts]
 * @returns {Array<{ params, score, stats }>}
 */
export function sensitivity(candles, baseParams, varyKey, varyValues, opts = {}) {
    const objectiveName = opts.objective ?? DEFAULT_OBJECTIVE;
    const objectiveFn = OBJECTIVES[objectiveName] ?? OBJECTIVES.profitFactor;
    const symbol = opts.symbol ?? 'BTC-USD';

    const results = [];
    for (const value of varyValues) {
        const params = { ...baseParams, [varyKey]: value };
        const btResult = runBacktest(candles, { ...params, symbol });
        const score = objectiveFn(btResult.stats);
        results.push({ params: { ...params }, score, stats: btResult.stats });
    }
    return results;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────────

function cartesianProduct(arrays) {
    if (arrays.length === 0) return [[]];
    const rest = cartesianProduct(arrays.slice(1));
    const result = [];
    for (const val of arrays[0]) {
        for (const combo of rest) {
            result.push([val, ...combo]);
        }
    }
    return result;
}

function midpoints(paramSpace) {
    const params = {};
    for (const [key, values] of Object.entries(paramSpace)) {
        if (!Array.isArray(values) || values.length === 0) {
            params[key] = values;
        } else if (typeof values[0] === 'number') {
            const sorted = [...values].sort((a, b) => a - b);
            params[key] = sorted[Math.floor(sorted.length / 2)];
        } else {
            params[key] = values[0];
        }
    }
    return params;
}

function generateNeighbors(current, paramSpace, count) {
    const neighbors = [];
    const numericKeys = Object.keys(paramSpace).filter(
        k => Array.isArray(paramSpace[k]) && typeof paramSpace[k][0] === 'number',
    );
    const discreteKeys = Object.keys(paramSpace).filter(
        k => Array.isArray(paramSpace[k]) && typeof paramSpace[k][0] !== 'number',
    );

    for (let i = 0; i < count; i++) {
        const neighbor = { ...current };

        // Perturb one numeric parameter
        if (numericKeys.length > 0) {
            const key = numericKeys[Math.floor(Math.random() * numericKeys.length)];
            const values = paramSpace[key].slice().sort((a, b) => a - b);
            const currentIdx = values.indexOf(current[key]);

            if (currentIdx >= 0) {
                // Step to adjacent value in the sorted list
                const dir = Math.random() < 0.5 ? -1 : 1;
                const newIdx = Math.max(0, Math.min(values.length - 1, currentIdx + dir));
                neighbor[key] = values[newIdx];
            } else {
                // Current value not in the discrete list — pick a random value
                neighbor[key] = values[Math.floor(Math.random() * values.length)];
            }
        }

        // Occasionally flip a discrete parameter
        if (discreteKeys.length > 0 && Math.random() < 0.3) {
            const key = discreteKeys[Math.floor(Math.random() * discreteKeys.length)];
            const values = paramSpace[key];
            const currentIdx = values.indexOf(current[key]);
            const newIdx = (currentIdx + 1 + Math.floor(Math.random() * (values.length - 1))) % values.length;
            neighbor[key] = values[newIdx];
        }

        neighbors.push(neighbor);
    }

    return neighbors;
}

// ─── Type Definitions ────────────────────────────────────────────────────────────

/**
 * @typedef {object} OptimizeResult
 * @property {object} bestParams
 * @property {number} bestScore
 * @property {object} bestStats
 * @property {object} gridResult
 * @property {object} hillResult
 * @property {string} objective
 */

/**
 * @typedef {object} GridResult
 * @property {object} bestParams
 * @property {number} bestScore
 * @property {object} bestStats
 * @property {Array} allResults
 * @property {Array} topN
 * @property {string} objective
 * @property {number} totalEvaluations
 */
