/**
 * Signal Fusion Engine — cross-domain signal fusion for trading decisions.
 *
 * Fuses signals from alpha-parser, microstructure analysis, liquidity analysis,
 * backtest evaluations, and sentiment into a composite confidence score.
 * Supports weighted average, naive Bayes, and voting fusion methods.
 *
 * Usage:
 *   import { SignalFusionEngine, SignalQualityAnalyzer } from './signal-fusion.mjs';
 *
 *   const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
 *   engine.addSignal('alpha', 'momentum', 0.8, 0.9);
 *   engine.addSignal('microstructure', 'orderflow', -0.3, 0.7);
 *   const decision = engine.getDecision();
 *   // => { action: 'BUY', confidence: 0.65, reasoning: '...' }
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Valid signal source identifiers. */
export const VALID_SOURCES = ['alpha', 'microstructure', 'liquidity', 'backtest', 'sentiment'];

/** Valid fusion method names. */
export const VALID_METHODS = ['weighted', 'bayesian', 'voting'];

/** Default per-source weights (sum to 1). */
export const DEFAULT_WEIGHTS = {
    alpha: 0.30,
    microstructure: 0.25,
    liquidity: 0.20,
    backtest: 0.15,
    sentiment: 0.10,
};

const DEFAULT_MIN_CONFIDENCE = 0.6;
const DEFAULT_FUSION_METHOD = 'weighted';

// ---------------------------------------------------------------------------
// Pure utility functions
// ---------------------------------------------------------------------------

/**
 * Compute the Pearson correlation coefficient between two arrays.
 *
 * Returns a value in [-1, 1] where 1 = perfect positive correlation,
 * -1 = perfect negative correlation, and 0 = no linear correlation.
 * Returns 0 when either array has fewer than 2 elements or zero variance.
 *
 * @param {number[]} xs - first array of numbers
 * @param {number[]} ys - second array of numbers (same length as xs)
 * @returns {number} Pearson correlation coefficient
 */
export function pearsonCorrelation(xs, ys) {
    if (!Array.isArray(xs) || !Array.isArray(ys)) return 0;
    if (xs.length !== ys.length || xs.length < 2) return 0;

    const n = xs.length;
    const sumX = xs.reduce((s, v) => s + v, 0);
    const sumY = ys.reduce((s, v) => s + v, 0);
    const sumXY = xs.reduce((s, v, i) => s + v * ys[i], 0);
    const sumX2 = xs.reduce((s, v) => s + v * v, 0);
    const sumY2 = ys.reduce((s, v) => s + v * v, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt(
        (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY),
    );

    if (denominator === 0) return 0;
    return Math.max(-1, Math.min(1, numerator / denominator));
}

/**
 * Compute the weighted average of an array of values.
 *
 * When `weights` is omitted, returns the simple arithmetic mean.
 * When provided, each value is multiplied by its corresponding weight.
 *
 * @param {number[]} values - array of values to average
 * @param {number[]} [weights] - parallel array of weights (same length)
 * @returns {number} weighted (or simple) average
 */
export function weightedAverage(values, weights) {
    if (!Array.isArray(values) || values.length === 0) return 0;

    if (weights === undefined) {
        return values.reduce((s, v) => s + v, 0) / values.length;
    }

    if (values.length !== weights.length) return 0;

    const totalWeight = weights.reduce((s, w) => s + w, 0);
    if (totalWeight === 0) return 0;

    const weightedSum = values.reduce((s, v, i) => s + v * weights[i], 0);
    return weightedSum / totalWeight;
}

/**
 * Normalize signal values to the -1..1 range.
 *
 * Scales by the maximum absolute value across all signals, so relative
 * relationships between signals are preserved.  If all values are zero
 * (or the array is empty), the original array is returned unchanged.
 *
 * @param {{source: string, name: string, value: number, confidence: number}[]} signals
 * @returns {{source: string, name: string, value: number, confidence: number}[]}
 */
export function normalizeScores(signals) {
    if (!Array.isArray(signals) || signals.length === 0) return [];

    const maxAbs = Math.max(...signals.map(s => Math.abs(s.value)), 0);
    if (maxAbs === 0) return signals.map(s => ({ ...s }));

    return signals.map(s => ({
        ...s,
        value: Math.max(-1, Math.min(1, s.value / maxAbs)),
    }));
}

/**
 * Fuse signals using naive Bayes under conditional independence.
 *
 * Each signal's value is treated as evidence for a bullish or bearish
 * hypothesis.  The value is mapped through a sigmoid-derived likelihood
 * ratio and combined with prior odds.
 *
 * @param {{source: string, name: string, value: number, confidence: number}[]} signals
 * @param {Object} [priors]
 * @param {number} [priors.bullish=0.5] - prior probability of bullish outcome
 * @param {number} [priors.bearish=0.5] - prior probability of bearish outcome
 * @returns {{ score: number, confidence: number }}
 *   score: fused score in [-1, 1] (negative = bearish, positive = bullish)
 *   confidence: posterior probability of the dominant hypothesis (0-1)
 */
export function fuseBayesian(signals, priors = {}) {
    if (!Array.isArray(signals) || signals.length === 0) {
        return { score: 0, confidence: 0 };
    }

    const priorBullish = priors.bullish ?? 0.5;
    const priorBearish = priors.bearish ?? 0.5;

    // Work in log-space for numerical stability
    let logPostBullish = Math.log(priorBullish);
    let logPostBearish = Math.log(priorBearish);

    for (const signal of signals) {
        // Strength: product of direction and confidence
        // Positive value -> bullish evidence, negative -> bearish evidence
        const strength = signal.value * signal.confidence;

        // Likelihood ratio for each hypothesis given this signal
        // strength > 0  => lrBullish > 1 > lrBearish
        // strength < 0  => lrBearish > 1 > lrBullish
        logPostBullish += strength;   // ln(exp(strength))
        logPostBearish -= strength;   // ln(exp(-strength))
    }

    // Normalize via log-sum-exp trick
    const maxLog = Math.max(logPostBullish, logPostBearish);
    const expBullish = Math.exp(logPostBullish - maxLog);
    const expBearish = Math.exp(logPostBearish - maxLog);
    const sumExp = expBullish + expBearish;

    const posteriorBullish = sumExp > 0 ? expBullish / sumExp : 0.5;

    // Map [0, 1] posterior to [-1, 1] score
    const score = 2 * posteriorBullish - 1;
    const confidence = Math.max(posteriorBullish, 1 - posteriorBullish);

    return { score, confidence };
}

// ---------------------------------------------------------------------------
// SignalFusionEngine
// ---------------------------------------------------------------------------

/**
 * Cross-domain signal fusion engine.
 *
 * Accepts signals from multiple domains (alpha, microstructure, liquidity,
 * backtest, sentiment) and fuses them into a single composite score and
 * trading decision using a configurable fusion strategy.
 */
export class SignalFusionEngine {
    /**
     * @param {Object} [config]
     * @param {Object} [config.weights] - per-source weights
     *   (default: { alpha: 0.30, microstructure: 0.25, liquidity: 0.20, backtest: 0.15, sentiment: 0.10 })
     * @param {number} [config.minConfidence=0.6] - minimum |composite| to trigger BUY/SELL
     * @param {'weighted'|'bayesian'|'voting'} [config.fusionMethod='weighted'] - fusion strategy
     */
    constructor(config = {}) {
        this._weights = { ...DEFAULT_WEIGHTS, ...config.weights };
        this._minConfidence = config.minConfidence ?? DEFAULT_MIN_CONFIDENCE;
        this._fusionMethod = VALID_METHODS.includes(config.fusionMethod)
            ? config.fusionMethod
            : DEFAULT_FUSION_METHOD;

        /** @type {{source: string, name: string, value: number, confidence: number, timestamp: number}[]} */
        this._signals = [];

        /**
         * Historical snapshots of per-source aggregates, recorded on each
         * call to getCompositeScore() / getDecision().
         * @type {{timestamp: number, sources: Object<string, {value: number, confidence: number}>}[]}
         */
        this._history = [];
    }

    /**
     * Register a single signal.
     *
     * @param {'alpha'|'microstructure'|'liquidity'|'backtest'|'sentiment'} source
     * @param {string} name - descriptive name for the signal
     * @param {number} value - direction/strength in [-1, 1] (negative = bearish)
     * @param {number} confidence - confidence in [0, 1]
     * @returns {SignalFusionEngine} this (for chaining)
     */
    addSignal(source, name, value, confidence) {
        if (!VALID_SOURCES.includes(source)) {
            return this;
        }
        this._signals.push({
            source,
            name,
            value: Math.max(-1, Math.min(1, value)),
            confidence: Math.max(0, Math.min(1, confidence)),
            timestamp: Date.now(),
        });
        return this;
    }

    /**
     * Register multiple signals in batch.
     *
     * @param {{source: string, name: string, value: number, confidence: number}[]} signals
     * @returns {SignalFusionEngine} this (for chaining)
     */
    addSignals(signals) {
        if (!Array.isArray(signals)) return this;
        for (const s of signals) {
            this.addSignal(s.source, s.name, s.value, s.confidence);
        }
        return this;
    }

    /**
     * Fuse all registered signals into a single composite score.
     *
     * The fusion method is determined by the `fusionMethod` config option.
     * This call also records a history snapshot for correlation analysis.
     *
     * @returns {number} composite score in [-1, 1]
     *   negative = bearish, positive = bullish, 0 = neutral
     */
    getCompositeScore() {
        if (this._signals.length === 0) return 0;

        this._recordRound();

        switch (this._fusionMethod) {
            case 'weighted': return this._fuseWeighted();
            case 'bayesian': return this._fuseBayesian();
            case 'voting':   return this._fuseVoting();
            default:         return this._fuseWeighted();
        }
    }

    // -- Private fusion implementations --

    /**
     * Weighted average fusion.
     * Each source's aggregate value is weighted by (source weight * average confidence).
     * @returns {number}
     */
    _fuseWeighted() {
        const breakdown = this._aggregateBySource();
        let numerator = 0;
        let denominator = 0;

        for (const [source, data] of Object.entries(breakdown)) {
            const weight = this._weights[source] ?? DEFAULT_WEIGHTS[source] ?? 0;
            const effective = weight * data.confidence;
            numerator += data.value * effective;
            denominator += effective;
        }

        return denominator > 0 ? numerator / denominator : 0;
    }

    /**
     * Naive Bayes fusion.
     * Delegates to the pure `fuseBayesian` function with per-source aggregates.
     * @returns {number}
     */
    _fuseBayesian() {
        const breakdown = this._aggregateBySource();
        const signals = Object.entries(breakdown).map(([source, data]) => ({
            source,
            name: source,
            value: data.value,
            confidence: data.confidence,
        }));
        return fuseBayesian(signals).score;
    }

    /**
     * Majority-vote fusion.
     * Each source casts a confidence-weighted vote; the net score reflects
     * the margin of victory scaled by total vote weight.
     * @returns {number}
     */
    _fuseVoting() {
        const breakdown = this._aggregateBySource();
        let bullishWeight = 0;
        let bearishWeight = 0;

        for (const [source, data] of Object.entries(breakdown)) {
            const weight = this._weights[source] ?? DEFAULT_WEIGHTS[source] ?? 0;
            const votePower = weight * data.confidence;

            if (data.value > 0) {
                bullishWeight += votePower * data.value;
            } else if (data.value < 0) {
                bearishWeight += votePower * Math.abs(data.value);
            }
        }

        const total = bullishWeight + bearishWeight;
        if (total === 0) return 0;

        return (bullishWeight - bearishWeight) / total;
    }

    /**
     * Group current signals by source and compute per-source aggregates.
     *
     * For each source:
     *   - `value` is the confidence-weighted average of its signal values
     *   - `confidence` is the arithmetic mean of individual confidences
     *   - `signalCount` is the number of signals from that source
     *
     * @returns {Object<string, {value: number, confidence: number, signalCount: number}>}
     */
    _aggregateBySource() {
        /** @type {Object<string, {totalValue: number, totalConfidence: number, totalWeight: number, count: number}>} */
        const groups = {};

        for (const s of this._signals) {
            if (!s.source || !VALID_SOURCES.includes(s.source)) continue;

            if (!groups[s.source]) {
                groups[s.source] = {
                    totalValue: 0,
                    totalConfidence: 0,
                    totalWeight: 0,
                    count: 0,
                };
            }
            const g = groups[s.source];
            g.totalValue += s.value * s.confidence;
            g.totalConfidence += s.confidence;
            g.totalWeight += s.confidence;
            g.count++;
        }

        const result = {};
        for (const [source, g] of Object.entries(groups)) {
            result[source] = {
                value: g.totalWeight > 0 ? g.totalValue / g.totalWeight : 0,
                confidence: g.count > 0 ? g.totalConfidence / g.count : 0,
                signalCount: g.count,
            };
        }
        return result;
    }

    /**
     * Snapshot current per-source aggregates into the history log.
     */
    _recordRound() {
        const breakdown = this._aggregateBySource();
        const sources = {};
        for (const [source, data] of Object.entries(breakdown)) {
            sources[source] = { value: data.value, confidence: data.confidence };
        }
        this._history.push({
            timestamp: Date.now(),
            sources,
        });
    }

    // -- Public analysis methods --

    /**
     * Get a per-source breakdown of aggregated signal data.
     *
     * @returns {Object<string, {score: number, weight: number, confidence: number, signalCount: number}>}
     */
    getSourceBreakdown() {
        const breakdown = this._aggregateBySource();
        const result = {};
        for (const [source, data] of Object.entries(breakdown)) {
            result[source] = {
                score: data.value,
                weight: this._weights[source] ?? DEFAULT_WEIGHTS[source] ?? 0,
                confidence: data.confidence,
                signalCount: data.signalCount,
            };
        }
        return result;
    }

    /**
     * Get a trading decision based on the composite score.
     *
     * Returns BUY when composite > minConfidence, SELL when
     * composite < -minConfidence, and HOLD otherwise.
     *
     * @returns {{action: 'BUY'|'SELL'|'HOLD', confidence: number, reasoning: string}}
     */
    getDecision() {
        const score = this.getCompositeScore();
        const absScore = Math.abs(score);

        let action = 'HOLD';
        if (absScore >= this._minConfidence) {
            action = score > 0 ? 'BUY' : 'SELL';
        }

        const breakdown = this.getSourceBreakdown();
        const summaries = Object.entries(breakdown)
            .map(([s, d]) => `${s}: ${d.score.toFixed(2)} (${d.signalCount} sig, ${(d.confidence * 100).toFixed(0)}% conf)`)
            .join('; ');

        const reasoning = `${this._fusionMethod} fusion score: ${score.toFixed(4)}. `
            + `Sources: [${summaries}]`;

        return { action, confidence: absScore, reasoning };
    }

    /**
     * Compute the pairwise Pearson correlation matrix between signal sources
     * over all recorded history rounds.
     *
     * Diagonal entries are always 1.0.  Off-diagonal entries are computed
     * from rounds where both sources had data (minimum 2 overlapping rounds
     * required; otherwise 0).
     *
     * @returns {Object<string, Object<string, number>>}
     *   e.g. { alpha: { microstructure: 0.42, liquidity: -0.15, ... }, ... }
     */
    getCorrelationMatrix() {
        /** @type {Object<string, number[]>} */
        const sourceSeries = {};

        for (const round of this._history) {
            for (const [source, data] of Object.entries(round.sources)) {
                if (!sourceSeries[source]) sourceSeries[source] = [];
                sourceSeries[source].push(data.value);
            }
        }

        const names = Object.keys(sourceSeries);
        const matrix = {};

        for (const a of names) {
            matrix[a] = {};
            for (const b of names) {
                if (a === b) {
                    matrix[a][b] = 1;
                    continue;
                }

                // Align paired values across overlapping rounds
                const xs = [];
                const ys = [];
                for (const round of this._history) {
                    if (round.sources[a] !== undefined && round.sources[b] !== undefined) {
                        xs.push(round.sources[a].value);
                        ys.push(round.sources[b].value);
                    }
                }

                matrix[a][b] = xs.length >= 2 ? pearsonCorrelation(xs, ys) : 0;
            }
        }

        return matrix;
    }

    /**
     * Clear all signals for a new evaluation.
     *
     * History of past rounds is preserved for correlation analysis
     * across evaluations.
     *
     * @returns {SignalFusionEngine} this (for chaining)
     */
    reset() {
        this._signals = [];
        return this;
    }
}

// ---------------------------------------------------------------------------
// SignalQualityAnalyzer
// ---------------------------------------------------------------------------

/**
 * Tracks signal accuracy over time and calibrates source weights.
 *
 * Records whether past signals were correct (addOutcome), reports per-source
 * accuracy, and produces calibrated weights using exponential recency
 * weighting so that more recent performance is weighted more heavily.
 */
export class SignalQualityAnalyzer {
    /**
     * @param {Object} [config]
     * @param {number} [config.decayRate=0.01] - exponential decay rate for
     *   recency weighting (higher = faster decay of old observations)
     */
    constructor(config = {}) {
        /** @type {{source: string, name: string, value: number, confidence: number, wasCorrect: boolean, timestamp: number}[]} */
        this._outcomes = [];
        this._lambda = config.decayRate ?? 0.01;
    }

    /**
     * Record whether a past signal was correct.
     *
     * @param {{source: string, name: string, value: number, confidence: number}} signalContext
     * @param {boolean} wasCorrect - true if the signal's prediction was accurate
     */
    addOutcome(signalContext, wasCorrect) {
        this._outcomes.push({
            source: signalContext.source,
            name: signalContext.name,
            value: signalContext.value,
            confidence: signalContext.confidence,
            wasCorrect,
            timestamp: Date.now(),
        });
    }

    /**
     * Get the raw accuracy rate for a specific source.
     *
     * @param {string} source - one of 'alpha', 'microstructure', 'liquidity',
     *   'backtest', 'sentiment'
     * @returns {number} accuracy in [0, 1] (0 if no outcomes for source)
     */
    getSourceAccuracy(source) {
        const relevant = this._outcomes.filter(o => o.source === source);
        if (relevant.length === 0) return 0;
        return relevant.filter(o => o.wasCorrect).length / relevant.length;
    }

    /**
     * Calibrate source weights based on historical accuracy using exponential
     * recency weighting.
     *
     * Each source's base weight is scaled by its recency-weighted accuracy.
     * More recent outcomes have greater influence.  The result is normalized
     * to sum to 1.
     *
     * @param {Object} [baseWeights] - base weights to calibrate (default:
     *   built-in DEFAULT_WEIGHTS)
     * @returns {Object} calibrated weights object with the same keys
     */
    calibrateWeights(baseWeights = DEFAULT_WEIGHTS) {
        const now = Date.now();
        const outcomes = this._outcomes;

        // Compute the span of the observation window for normalizing time deltas
        const oldest = outcomes.length > 0
            ? Math.min(...outcomes.map(o => o.timestamp))
            : now;
        const maxAge = Math.max(now - oldest, 1);

        // Group outcomes by source
        /** @type {Object<string, Array>} */
        const bySource = {};
        for (const o of outcomes) {
            if (!bySource[o.source]) bySource[o.source] = [];
            bySource[o.source].push(o);
        }

        const calibrated = {};
        for (const source of VALID_SOURCES) {
            const sourceOutcomes = bySource[source] || [];
            const baseWeight = baseWeights[source] ?? 0;

            if (sourceOutcomes.length === 0) {
                calibrated[source] = baseWeight;
                continue;
            }

            // Compute recency-weighted accuracy
            let totalDecay = 0;
            let correctDecay = 0;

            for (const o of sourceOutcomes) {
                const age = now - o.timestamp;
                const decay = Math.exp(-this._lambda * (age / maxAge));
                totalDecay += decay;
                if (o.wasCorrect) correctDecay += decay;
            }

            const recencyAccuracy = totalDecay > 0 ? correctDecay / totalDecay : 0.5;

            // Scale base weight by accuracy:
            //   accuracy=1.0 -> 1.5x base
            //   accuracy=0.5 -> 1.0x base (no adjustment)
            //   accuracy=0.0 -> 0.5x base
            const adjustment = 0.5 + recencyAccuracy;
            calibrated[source] = baseWeight * adjustment;
        }

        // Normalize to sum to 1
        const total = Object.values(calibrated).reduce((s, w) => s + w, 0);
        if (total > 0) {
            for (const source of Object.keys(calibrated)) {
                calibrated[source] /= total;
            }
        }

        return calibrated;
    }

    /**
     * Get a detailed reliability report per source.
     *
     * @returns {Object<string, {accuracy: number, recencyWeightedAccuracy: number, signalCount: number, avgConfidence: number, calibrationError: number}>}
     */
    getReliabilityReport() {
        const now = Date.now();
        const outcomes = this._outcomes;

        const oldest = outcomes.length > 0
            ? Math.min(...outcomes.map(o => o.timestamp))
            : now;
        const maxAge = Math.max(now - oldest, 1);

        /** @type {Object<string, Array>} */
        const bySource = {};
        for (const o of outcomes) {
            if (!bySource[o.source]) bySource[o.source] = [];
            bySource[o.source].push(o);
        }

        const report = {};
        for (const source of VALID_SOURCES) {
            const sourceOutcomes = bySource[source] || [];

            if (sourceOutcomes.length === 0) {
                report[source] = {
                    accuracy: 0,
                    recencyWeightedAccuracy: 0,
                    signalCount: 0,
                    avgConfidence: 0,
                    calibrationError: 0,
                };
                continue;
            }

            // Raw accuracy
            const correct = sourceOutcomes.filter(o => o.wasCorrect).length;
            const accuracy = sourceOutcomes.length > 0
                ? correct / sourceOutcomes.length
                : 0;

            // Recency-weighted accuracy
            let totalDecay = 0;
            let correctDecay = 0;
            let totalConf = 0;

            for (const o of sourceOutcomes) {
                const age = now - o.timestamp;
                const decay = Math.exp(-this._lambda * (age / maxAge));
                totalDecay += decay;
                if (o.wasCorrect) correctDecay += decay;
                totalConf += o.confidence;
            }

            const recencyWeightedAccuracy = totalDecay > 0
                ? correctDecay / totalDecay
                : 0;
            const avgConfidence = sourceOutcomes.length > 0
                ? totalConf / sourceOutcomes.length
                : 0;

            // Calibration error: absolute difference between accuracy and
            // average confidence.  Low values mean the source is well-calibrated
            // (its confidence matches its actual accuracy).
            const calibrationError = Math.abs(accuracy - avgConfidence);

            report[source] = {
                accuracy,
                recencyWeightedAccuracy,
                signalCount: sourceOutcomes.length,
                avgConfidence,
                calibrationError,
            };
        }

        return report;
    }
}
