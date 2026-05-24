/**
 * Decision Graph — knowledge graph for trading decisions.
 *
 * Records every trading decision with its full reasoning chain, signal sources,
 * outcomes, and inter-decision relationships. Supports queries, pattern analysis,
 * and trace following.
 *
 * Usage:
 *   import { DecisionGraph, GraphQuery, generateDecisionId, decisionToText } from './decision-graph.mjs';
 *
 *   const graph = new DecisionGraph();
 *   const id = graph.addDecision({ symbol: 'AAPL', action: 'BUY', ... });
 *   graph.recordOutcome(id, { pnl: 250, exitReason: 'take_profit', wasCorrect: true });
 *   const patterns = graph.getPatterns();
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VALID_ACTIONS = ['BUY', 'SELL', 'HOLD'];
const VALID_RELATIONSHIPS = ['refines', 'contradicts', 'confirms', 'leads_to', 'exit_of'];

// ---------------------------------------------------------------------------
// Internal ID counter (module-level, shared across all instances)
// ---------------------------------------------------------------------------

let _idCounter = 0;

// ---------------------------------------------------------------------------
// Pure utility functions
// ---------------------------------------------------------------------------

/**
 * Generate a short unique decision ID based on timestamp and a monotonic counter.
 *
 * The ID is formatted as `D-<base36-timestamp>-<padded-counter>` and is
 * suitable for human-readable references (not cryptographically secure).
 *
 * @param {number} [timestamp] - Unix timestamp in ms (defaults to Date.now())
 * @returns {string} unique decision ID
 */
export function generateDecisionId(timestamp) {
    const ts = typeof timestamp === 'number' ? timestamp : Date.now();
    const suffix = String(_idCounter++).padStart(4, '0');
    return `D-${ts.toString(36).toUpperCase()}-${suffix}`;
}

/**
 * Format a decision as a human-readable summary string.
 *
 * Includes the decision ID, action, symbol, timestamp, signals breakdown,
 * reasoning, and outcome (if recorded).
 *
 * @param {Object} decision - a decision node as stored in the graph
 * @returns {string} multi-line human-readable summary
 */
export function decisionToText(decision) {
    if (!decision) return '(no decision)';

    const id = decision.id || '(no id)';
    const ts = new Date(decision.timestamp).toISOString();
    const lines = [
        `[${id}] ${decision.action} ${decision.symbol} @ ${ts}`,
    ];

    if (Array.isArray(decision.signals) && decision.signals.length > 0) {
        lines.push('  Signals:');
        for (const sig of decision.signals) {
            const sign = sig.value >= 0 ? '+' : '';
            lines.push(
                `    ${sig.source}:${sig.name} (${sign}${sig.value.toFixed(2)} @ ${(sig.confidence * 100).toFixed(0)}%)`,
            );
        }
    }

    if (decision.reasoning) {
        lines.push(`  Reasoning: ${decision.reasoning}`);
    }

    if (decision.outcome) {
        const o = decision.outcome;
        const pnlStr = o.pnl !== undefined
            ? `${o.pnl >= 0 ? '+' : ''}${o.pnl.toFixed(2)}`
            : '';
        const exitStr = o.exitReason || '';
        const correctMark = o.wasCorrect ? '✓' : '✗';
        lines.push(`  Outcome: ${pnlStr}${exitStr ? ' (' + exitStr + ')' : ''} ${correctMark}`);
    }

    if (decision.metadata) {
        const metaStr = Object.entries(decision.metadata)
            .map(([k, v]) => `${k}=${v}`)
            .join(', ');
        if (metaStr) lines.push(`  Metadata: ${metaStr}`);
    }

    return lines.join('\n');
}

/**
 * Compute a similarity score between two decisions based on overlapping signals.
 *
 * Uses Jaccard similarity on the set of (source, name) pairs, combined with
 * value agreement for overlapping signals. Returns a value in [0, 1] where
 * 1 = identical signal profiles and 0 = no overlap.
 *
 * @param {Object} a - first decision node
 * @param {Object} b - second decision node
 * @returns {number} similarity score in [0, 1]
 */
export function compareDecisions(a, b) {
    if (!a || !b) return 0;
    if (!Array.isArray(a.signals) || !Array.isArray(b.signals)) return 0;
    if (a.signals.length === 0 && b.signals.length === 0) return 1;
    if (a.signals.length === 0 || b.signals.length === 0) return 0;

    // Build signal maps keyed by source:name
    const mapA = new Map();
    for (const sig of a.signals) {
        mapA.set(`${sig.source}:${sig.name}`, sig.value);
    }
    const mapB = new Map();
    for (const sig of b.signals) {
        mapB.set(`${sig.source}:${sig.name}`, sig.value);
    }

    // Jaccard similarity on signal keys
    const keysA = new Set(mapA.keys());
    const keysB = new Set(mapB.keys());
    const intersection = new Set([...keysA].filter(k => keysB.has(k)));
    const union = new Set([...keysA, ...keysB]);

    if (union.size === 0) return 1;

    const jaccard = intersection.size / union.size;

    // Value agreement: for overlapping signals, how close are the values?
    let valueAgreement = 0;
    if (intersection.size > 0) {
        let totalDiff = 0;
        for (const key of intersection) {
            totalDiff += Math.abs(mapA.get(key) - mapB.get(key));
        }
        valueAgreement = 1 - (totalDiff / intersection.size / 2);
    }

    // 60% jaccard + 40% value agreement
    // When signals are disjoint (no overlap), score is 0
    if (intersection.size === 0) return 0;
    return +(0.6 * jaccard + 0.4 * Math.max(0, valueAgreement)).toFixed(4);
}

/**
 * Filter an array of decisions to only those with average signal confidence
 * above (or equal to) a given threshold.
 *
 * Decisions with no signals return false (confidence 0 < any positive threshold).
 *
 * @param {Object[]} decisions - array of decision nodes
 * @param {number} minConfidence - minimum average confidence threshold [0, 1]
 * @returns {Object[]} filtered decisions
 */
export function filterByConfidence(decisions, minConfidence) {
    if (!Array.isArray(decisions)) return [];
    return decisions.filter(d => {
        if (!Array.isArray(d.signals) || d.signals.length === 0) return false;
        const avg = d.signals.reduce((s, sig) => s + sig.confidence, 0) / d.signals.length;
        return avg >= minConfidence;
    });
}

// ---------------------------------------------------------------------------
// DecisionGraph
// ---------------------------------------------------------------------------

/**
 * Knowledge graph for recording, linking, and analyzing trading decisions.
 *
 * Maintains a directed graph where nodes are trading decisions and edges
 * represent relationships (refines, contradicts, confirms, leads_to, exit_of).
 * Supports pattern detection, trace following, and structured queries.
 */
export class DecisionGraph {
    constructor() {
        /** @type {Map<string, Object>} */
        this._decisions = new Map();
        /** @type {Map<string, Array<{childId: string, relationship: string}>>} */
        this._edges = new Map();
        /** @type {Map<string, Array<{parentId: string, relationship: string}>>} */
        this._reverseEdges = new Map();
    }

    /**
     * Record a new trading decision.
     *
     * If the entry does not include an `id`, one is auto-generated via
     * `generateDecisionId(timestamp)`.
     *
     * @param {Object} entry - decision entry
     * @param {string} [entry.id] - unique identifier (auto-generated if omitted)
     * @param {number} entry.timestamp - Unix timestamp in ms
     * @param {'BUY'|'SELL'|'HOLD'} entry.action - trading action taken
     * @param {string} entry.symbol - trading symbol (e.g. 'AAPL')
     * @param {Array<{source: string, name: string, value: number, confidence: number}>} entry.signals - signals that informed the decision
     * @param {string} entry.reasoning - free-text reasoning for the decision
     * @param {Object} [entry.metadata] - optional additional data
     * @returns {string} the decision ID
     * @throws {Error} if entry is invalid or action is not BUY/SELL/HOLD
     */
    addDecision(entry) {
        if (!entry || typeof entry !== 'object') {
            throw new Error('addDecision: entry must be an object');
        }
        if (typeof entry.timestamp !== 'number') {
            throw new Error('addDecision: entry.timestamp is required and must be a number');
        }
        if (!VALID_ACTIONS.includes(entry.action)) {
            throw new Error(`addDecision: action must be one of ${VALID_ACTIONS.join(', ')}`);
        }
        if (!entry.symbol || typeof entry.symbol !== 'string') {
            throw new Error('addDecision: entry.symbol is required');
        }
        if (!entry.reasoning || typeof entry.reasoning !== 'string') {
            throw new Error('addDecision: entry.reasoning is required');
        }

        const id = entry.id || generateDecisionId(entry.timestamp);

        if (this._decisions.has(id)) {
            throw new Error(`addDecision: decision with id "${id}" already exists`);
        }

        const signals = Array.isArray(entry.signals) ? entry.signals.map(s => ({
            source: String(s.source || ''),
            name: String(s.name || ''),
            value: Math.max(-1, Math.min(1, typeof s.value === 'number' ? s.value : 0)),
            confidence: Math.max(0, Math.min(1, typeof s.confidence === 'number' ? s.confidence : 0)),
        })) : [];

        this._decisions.set(id, {
            id,
            timestamp: entry.timestamp,
            action: entry.action,
            symbol: entry.symbol,
            signals,
            reasoning: entry.reasoning,
            metadata: entry.metadata ? { ...entry.metadata } : undefined,
            outcome: undefined,
        });

        return id;
    }

    /**
     * Create a directed edge between two decisions.
     *
     * Both parent and child decisions must already exist in the graph.
     *
     * @param {string} parentId - the upstream decision ID
     * @param {string} childId - the downstream decision ID
     * @param {'refines'|'contradicts'|'confirms'|'leads_to'|'exit_of'} relationship - type of relationship
     * @returns {boolean} true if the edge was created
     * @throws {Error} if either ID is not found or relationship is invalid
     */
    linkDecisions(parentId, childId, relationship) {
        if (!this._decisions.has(parentId)) {
            throw new Error(`linkDecisions: parent decision "${parentId}" not found`);
        }
        if (!this._decisions.has(childId)) {
            throw new Error(`linkDecisions: child decision "${childId}" not found`);
        }
        if (!VALID_RELATIONSHIPS.includes(relationship)) {
            throw new Error(`linkDecisions: relationship must be one of ${VALID_RELATIONSHIPS.join(', ')}`);
        }

        // Forward edge
        if (!this._edges.has(parentId)) {
            this._edges.set(parentId, []);
        }
        this._edges.get(parentId).push({ childId, relationship });

        // Reverse edge
        if (!this._reverseEdges.has(childId)) {
            this._reverseEdges.set(childId, []);
        }
        this._reverseEdges.get(childId).push({ parentId, relationship });

        return true;
    }

    /**
     * Attach an outcome to a previously recorded decision.
     *
     * @param {string} decisionId - ID of the decision to update
     * @param {Object} outcome - outcome data
     * @param {number} [outcome.pnl] - profit/loss in USD
     * @param {string} [outcome.exitReason] - reason for exiting (e.g. 'take_profit', 'stop_loss')
     * @param {number} [outcome.holdDuration] - duration of the hold in ms
     * @param {boolean} outcome.wasCorrect - whether the decision was correct
     * @returns {boolean} true if the outcome was recorded
     * @throws {Error} if decisionId is not found
     */
    recordOutcome(decisionId, outcome) {
        if (!this._decisions.has(decisionId)) {
            throw new Error(`recordOutcome: decision "${decisionId}" not found`);
        }

        this._decisions.get(decisionId).outcome = {
            pnl: outcome.pnl !== undefined ? outcome.pnl : undefined,
            exitReason: outcome.exitReason || undefined,
            holdDuration: outcome.holdDuration !== undefined ? outcome.holdDuration : undefined,
            wasCorrect: outcome.wasCorrect === true,
        };

        return true;
    }

    /**
     * Retrieve a single decision node with its inbound and outbound edges.
     *
     * @param {string} id - decision ID
     * @returns {Object|null} the decision node with attached edges, or null if not found
     */
    getDecision(id) {
        const decision = this._decisions.get(id);
        if (!decision) return null;

        return {
            ...decision,
            edges: {
                outgoing: this._edges.get(id) || [],
                incoming: this._reverseEdges.get(id) || [],
            },
        };
    }

    /**
     * Query decisions by filter criteria.
     *
     * All filter properties are optional and combined with AND logic.
     * Results are sorted by timestamp ascending.
     *
     * @param {Object} [filter] - filter criteria
     * @param {string} [filter.symbol] - match symbol exactly
     * @param {'BUY'|'SELL'|'HOLD'} [filter.action] - match action exactly
     * @param {string} [filter.source] - match if any signal has this source
     * @param {string} [filter.signalName] - match if any signal has this name
     * @param {boolean} [filter.wasCorrect] - match outcome correctness
     * @param {{start: number, end: number}} [filter.timeRange] - timestamp range {start, end} inclusive
     * @returns {Object[]} matching decisions sorted by timestamp
     */
    getDecisions(filter = {}) {
        const results = [];

        for (const decision of this._decisions.values()) {
            if (filter.symbol !== undefined && decision.symbol !== filter.symbol) continue;
            if (filter.action !== undefined && decision.action !== filter.action) continue;

            if (filter.source !== undefined) {
                const hasSource = decision.signals.some(s => s.source === filter.source);
                if (!hasSource) continue;
            }

            if (filter.signalName !== undefined) {
                const hasName = decision.signals.some(s => s.name === filter.signalName);
                if (!hasName) continue;
            }

            if (filter.wasCorrect !== undefined) {
                if (!decision.outcome) continue;
                if (decision.outcome.wasCorrect !== filter.wasCorrect) continue;
            }

            if (filter.timeRange !== undefined) {
                const { start, end } = filter.timeRange;
                if (start !== undefined && decision.timestamp < start) continue;
                if (end !== undefined && decision.timestamp > end) continue;
            }

            results.push({ ...decision });
        }

        results.sort((a, b) => a.timestamp - b.timestamp);
        return results;
    }

    /**
     * Follow the chain of linked decisions forward and backward from a starting point.
     *
     * Performs a bidirectional traversal to collect all decisions connected
     * through edges (in any direction, regardless of relationship type).
     * Returns an ordered array of decision nodes (without edge annotations).
     *
     * @param {string} decisionId - starting decision ID
     * @returns {Object[]} ordered array of connected decision nodes (including the start)
     */
    getTrace(decisionId) {
        if (!this._decisions.has(decisionId)) return [];

        // BFS both directions
        const visited = new Set();
        const queue = [decisionId];
        visited.add(decisionId);

        while (queue.length > 0) {
            const current = queue.shift();

            // Follow outgoing edges
            const outgoing = this._edges.get(current) || [];
            for (const edge of outgoing) {
                if (!visited.has(edge.childId)) {
                    visited.add(edge.childId);
                    queue.push(edge.childId);
                }
            }

            // Follow incoming edges
            const incoming = this._reverseEdges.get(current) || [];
            for (const edge of incoming) {
                if (!visited.has(edge.parentId)) {
                    visited.add(edge.parentId);
                    queue.push(edge.parentId);
                }
            }
        }

        // Sort by timestamp
        const trace = [...visited]
            .map(id => ({ ...this._decisions.get(id) }))
            .sort((a, b) => a.timestamp - b.timestamp);

        return trace;
    }

    /**
     * Analyze the graph for common patterns across all decisions with outcomes.
     *
     * Computes:
     * - Signal combinations appearing in winning decisions
     * - Signal combinations appearing in losing decisions
     * - Source accuracy breakdown
     * - Average confidence of winning vs losing decisions
     * - Most common exit reasons
     *
     * @returns {Object} pattern analysis results
     * @property {Array<{signal: string, winCount: number, totalCount: number, winRate: number}>} winningCombos - per-signal win rates
     * @property {Array<{signal: string, lossCount: number, totalCount: number, lossRate: number}>} losingCombos - per-signal loss rates
     * @property {Array<{source: string, totalDecisions: number, winningDecisions: number, accuracy: number}>} bestSources - sources ranked by accuracy
     * @property {{winning: number, losing: number}} avgConfidence - average signal confidence split by outcome
     * @property {Array<{reason: string, count: number}>} commonExitReasons - exit reasons sorted by frequency
     */
    getPatterns() {
        const decisions = [...this._decisions.values()];
        const withOutcomes = decisions.filter(d => d.outcome !== undefined);
        const winning = withOutcomes.filter(d => d.outcome.wasCorrect === true);
        const losing = withOutcomes.filter(d => d.outcome.wasCorrect === false);

        return {
            winningCombos: this._analyzeSignalCombos(winning, true),
            losingCombos: this._analyzeSignalCombos(losing, false),
            bestSources: this._computeSourceAccuracy(withOutcomes),
            avgConfidence: {
                winning: this._avgSignalConfidence(winning),
                losing: this._avgSignalConfidence(losing),
            },
            commonExitReasons: this._exitReasonBreakdown(withOutcomes),
        };
    }

    /**
     * Total number of decisions in the graph.
     * @returns {number}
     */
    get size() {
        return this._decisions.size;
    }

    /**
     * Total number of edges in the graph.
     * @returns {number}
     */
    get edgeCount() {
        let count = 0;
        for (const edges of this._edges.values()) {
            count += edges.length;
        }
        return count;
    }

    /**
     * Get all decisions (without edge annotations) as a shallow-copied array.
     * Useful for passing to GraphQuery or other utilities.
     * @returns {Object[]}
     */
    all() {
        return [...this._decisions.values()].map(d => ({ ...d }));
    }

    // -- Private helpers --

    /**
     * Analyze signal frequency and win/loss rates across a set of decisions.
     * @param {Object[]} decisions - filtered decisions (winning or losing)
     * @param {boolean} isWinning - true if analyzing winning decisions
     * @returns {Array<{signal: string, winCount: number, totalCount: number, winRate: number}>}
     */
    _analyzeSignalCombos(decisions, isWinning) {
        const signalCounts = {};
        const totalDecisions = decisions.length;

        for (const d of decisions) {
            const seen = new Set();
            for (const sig of d.signals) {
                const key = `${sig.source}:${sig.name}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    signalCounts[key] = (signalCounts[key] || 0) + 1;
                }
            }
        }

        // Total decisions with each signal across full graph (for context)
        const totalWithSignal = {};
        for (const d of this._decisions.values()) {
            const seen = new Set();
            for (const sig of d.signals) {
                const key = `${sig.source}:${sig.name}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    totalWithSignal[key] = (totalWithSignal[key] || 0) + 1;
                }
            }
        }

        return Object.entries(signalCounts)
            .map(([signal, count]) => ({
                signal,
                [isWinning ? 'winCount' : 'lossCount']: count,
                totalCount: totalWithSignal[signal] || 0,
                [isWinning ? 'winRate' : 'lossRate']: totalDecisions > 0
                    ? +(count / totalDecisions).toFixed(4)
                    : 0,
            }))
            .sort((a, b) => {
                const countKey = isWinning ? 'winCount' : 'lossCount';
                return b[countKey] - a[countKey];
            });
    }

    /**
     * Compute per-source accuracy from decisions with outcomes.
     * @param {Object[]} withOutcomes
     * @returns {Array<{source: string, totalDecisions: number, winningDecisions: number, accuracy: number}>}
     */
    _computeSourceAccuracy(withOutcomes) {
        const sourceStats = {};

        for (const d of withOutcomes) {
            const usedSources = new Set(d.signals.map(s => s.source));
            for (const source of usedSources) {
                if (!sourceStats[source]) {
                    sourceStats[source] = { total: 0, winning: 0 };
                }
                sourceStats[source].total++;
                if (d.outcome.wasCorrect) {
                    sourceStats[source].winning++;
                }
            }
        }

        return Object.entries(sourceStats)
            .map(([source, stats]) => ({
                source,
                totalDecisions: stats.total,
                winningDecisions: stats.winning,
                accuracy: stats.total > 0 ? +(stats.winning / stats.total).toFixed(4) : 0,
            }))
            .sort((a, b) => b.accuracy - a.accuracy);
    }

    /**
     * Compute average signal confidence across a set of decisions.
     * @param {Object[]} decisions
     * @returns {number}
     */
    _avgSignalConfidence(decisions) {
        if (decisions.length === 0) return 0;
        let totalConf = 0;
        let count = 0;
        for (const d of decisions) {
            for (const sig of d.signals) {
                totalConf += sig.confidence;
                count++;
            }
        }
        return count > 0 ? +(totalConf / count).toFixed(4) : 0;
    }

    /**
     * Break down exit reasons by frequency.
     * @param {Object[]} withOutcomes
     * @returns {Array<{reason: string, count: number}>}
     */
    _exitReasonBreakdown(withOutcomes) {
        const reasons = {};
        for (const d of withOutcomes) {
            if (d.outcome.exitReason) {
                reasons[d.outcome.exitReason] = (reasons[d.outcome.exitReason] || 0) + 1;
            }
        }
        return Object.entries(reasons)
            .map(([reason, count]) => ({ reason, count }))
            .sort((a, b) => b.count - a.count);
    }
}

// ---------------------------------------------------------------------------
// GraphQuery — Fluent query API
// ---------------------------------------------------------------------------

/**
 * Fluent query builder for filtering and analyzing arrays of decisions.
 *
 * All filter methods return `this` for chaining. Terminal methods
 * (execute, count, avgConfidence, topSignal) produce the result.
 *
 * Usage:
 *   const query = new GraphQuery(decisions)
 *     .symbol('AAPL')
 *     .action('BUY')
 *     .profitable();
 *   const count = query.count();
 *   const avg = query.avgConfidence();
 *   const results = query.execute();
 */
export class GraphQuery {
    /**
     * @param {Object[]} decisions - array of decision nodes to query
     */
    constructor(decisions) {
        /** @type {Object[]} */
        this._working = Array.isArray(decisions) ? [...decisions] : [];
    }

    /**
     * Filter by symbol (exact match).
     * @param {string} sym - symbol to match
     * @returns {GraphQuery} this
     */
    symbol(sym) {
        this._working = this._working.filter(d => d.symbol === sym);
        return this;
    }

    /**
     * Filter by action (BUY, SELL, or HOLD).
     * @param {string} action - action to match
     * @returns {GraphQuery} this
     */
    action(action) {
        this._working = this._working.filter(d => d.action === action);
        return this;
    }

    /**
     * Filter to decisions that include at least one signal from the given source.
     * @param {string} src - source to match (e.g. 'alpha', 'microstructure')
     * @returns {GraphQuery} this
     */
    source(src) {
        this._working = this._working.filter(d =>
            Array.isArray(d.signals) && d.signals.some(s => s.source === src),
        );
        return this;
    }

    /**
     * Filter to decisions that were profitable (outcome.wasCorrect === true).
     * @returns {GraphQuery} this
     */
    profitable() {
        this._working = this._working.filter(d => d.outcome && d.outcome.wasCorrect === true);
        return this;
    }

    /**
     * Filter to decisions that were unprofitable (outcome.wasCorrect === false).
     * @returns {GraphQuery} this
     */
    unprofitable() {
        this._working = this._working.filter(d => d.outcome && d.outcome.wasCorrect === false);
        return this;
    }

    /**
     * Filter to decisions whose average signal confidence is at least n.
     * @param {number} n - minimum confidence threshold [0, 1]
     * @returns {GraphQuery} this
     */
    confidenceAbove(n) {
        this._working = this._working.filter(d => {
            if (!Array.isArray(d.signals) || d.signals.length === 0) return false;
            const avg = d.signals.reduce((s, sig) => s + sig.confidence, 0) / d.signals.length;
            return avg >= n;
        });
        return this;
    }

    /**
     * Execute the query and return the filtered results sorted by timestamp.
     * @returns {Object[]} filtered decisions
     */
    execute() {
        return [...this._working].sort((a, b) => a.timestamp - b.timestamp);
    }

    /**
     * Return the number of matching decisions.
     * @returns {number}
     */
    count() {
        return this._working.length;
    }

    /**
     * Compute the average signal confidence across all matching decisions.
     * Decisions with no signals contribute 0 to the average.
     * @returns {number}
     */
    avgConfidence() {
        if (this._working.length === 0) return 0;
        let total = 0;
        let count = 0;
        for (const d of this._working) {
            if (Array.isArray(d.signals)) {
                for (const sig of d.signals) {
                    total += sig.confidence;
                    count++;
                }
            }
        }
        return count > 0 ? +(total / count).toFixed(4) : 0;
    }

    /**
     * Find the most frequently occurring signal key across matching decisions.
     * Returns a string in the format `source:name` or null if no signals exist.
     * @returns {string|null}
     */
    topSignal() {
        const counts = {};
        for (const d of this._working) {
            if (Array.isArray(d.signals)) {
                for (const sig of d.signals) {
                    const key = `${sig.source}:${sig.name}`;
                    counts[key] = (counts[key] || 0) + 1;
                }
            }
        }
        let best = null;
        let bestCount = 0;
        for (const [key, count] of Object.entries(counts)) {
            if (count > bestCount) {
                bestCount = count;
                best = key;
            }
        }
        return best;
    }
}
