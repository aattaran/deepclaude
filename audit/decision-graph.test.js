/**
 * Decision Graph — unit tests
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/decision-graph.test.js
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    DecisionGraph,
    GraphQuery,
    generateDecisionId,
    decisionToText,
    compareDecisions,
    filterByConfidence,
} from './decision-graph.mjs';

// ===========================================================================
// 1. generateDecisionId
// ===========================================================================
describe('generateDecisionId', () => {

    it('returns a string starting with D-', () => {
        const id = generateDecisionId(1000000);
        assert.ok(id.startsWith('D-'));
    });

    it('produces unique IDs on successive calls', () => {
        const a = generateDecisionId(2000000);
        const b = generateDecisionId(2000000);
        assert.notEqual(a, b);
    });

    it('uses the provided timestamp in base36', () => {
        const ts = 1716000000000;
        const id = generateDecisionId(ts);
        const expectedBase = ts.toString(36).toUpperCase();
        assert.ok(id.includes(expectedBase));
    });

    it('defaults to Date.now() when timestamp is omitted', () => {
        const before = Date.now();
        const id = generateDecisionId();
        const after = Date.now();
        // Extract the base36 portion between D- and the last -
        const match = id.match(/^D-([A-Z0-9]+)-/);
        assert.ok(match !== null, 'id format mismatch');
        const decoded = parseInt(match[1], 36);
        assert.ok(decoded >= before && decoded <= after,
            `expected decoded ${decoded} to be between ${before} and ${after}`);
    });

    it('pads the counter to at least 4 digits', () => {
        const id = generateDecisionId(3000000);
        const parts = id.split('-');
        assert.ok(parts.length === 3);
        assert.ok(parts[2].length >= 4);
    });
});

// ===========================================================================
// 2. decisionToText
// ===========================================================================
describe('decisionToText', () => {

    it('returns a string for a complete decision with outcome', () => {
        const text = decisionToText({
            id: 'D-TEST-0001',
            timestamp: 1716000000000,
            action: 'BUY',
            symbol: 'AAPL',
            signals: [{ source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 }],
            reasoning: 'Strong momentum',
            outcome: { pnl: 250, exitReason: 'take_profit', wasCorrect: true },
        });
        assert.ok(text.includes('D-TEST-0001'));
        assert.ok(text.includes('BUY'));
        assert.ok(text.includes('AAPL'));
        assert.ok(text.includes('+250.00'));
        assert.ok(text.includes('take_profit'));
        assert.ok(text.includes('✓'));
    });

    it('returns (no decision) for null input', () => {
        assert.equal(decisionToText(null), '(no decision)');
    });

    it('returns (no decision) for undefined input', () => {
        assert.equal(decisionToText(undefined), '(no decision)');
    });

    it('includes signals in the output', () => {
        const text = decisionToText({
            id: 'D-TEST-0002',
            timestamp: 1716000000000,
            action: 'SELL',
            symbol: 'TSLA',
            signals: [
                { source: 'microstructure', name: 'orderflow', value: -0.6, confidence: 0.85 },
            ],
            reasoning: 'Weak order flow',
        });
        assert.ok(text.includes('microstructure:orderflow'));
        assert.ok(text.includes('-0.60'));
        assert.ok(text.includes('85%'));
    });

    it('includes metadata when present', () => {
        const text = decisionToText({
            id: 'D-TEST-0003',
            timestamp: 1716000000000,
            action: 'HOLD',
            symbol: 'GOOGL',
            signals: [],
            reasoning: 'No clear signal',
            metadata: { riskScore: '3', timeframe: '1d' },
        });
        assert.ok(text.includes('riskScore=3'));
        assert.ok(text.includes('timeframe=1d'));
    });

    it('shows losing outcome with ✗', () => {
        const text = decisionToText({
            id: 'D-TEST-0004',
            timestamp: 1716000000000,
            action: 'BUY',
            symbol: 'MSFT',
            signals: [],
            reasoning: 'Gut feeling',
            outcome: { pnl: -150, exitReason: 'stop_loss', wasCorrect: false },
        });
        assert.ok(text.includes('-150.00'));
        assert.ok(text.includes('✗'));
    });
});

// ===========================================================================
// 3. compareDecisions
// ===========================================================================
describe('compareDecisions', () => {

    it('returns 1 for decisions with identical signals', () => {
        const a = {
            signals: [
                { source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 },
                { source: 'microstructure', name: 'orderflow', value: 0.5, confidence: 0.7 },
            ],
        };
        const b = {
            signals: [
                { source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 },
                { source: 'microstructure', name: 'orderflow', value: 0.5, confidence: 0.7 },
            ],
        };
        assert.equal(compareDecisions(a, b), 1);
    });

    it('returns 0 for decisions with no overlapping signals', () => {
        const a = { signals: [{ source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 }] };
        const b = { signals: [{ source: 'liquidity', name: 'spread', value: -0.3, confidence: 0.6 }] };
        assert.equal(compareDecisions(a, b), 0);
    });

    it('returns 1 when both decisions have empty signal arrays', () => {
        assert.equal(compareDecisions({ signals: [] }, { signals: [] }), 1);
    });

    it('returns 0 when one decision has no signals and the other does', () => {
        assert.equal(compareDecisions(
            { signals: [] },
            { signals: [{ source: 'a', name: 'b', value: 1, confidence: 1 }] },
        ), 0);
    });

    it('returns 0 for null or undefined inputs', () => {
        assert.equal(compareDecisions(null, { signals: [] }), 0);
        assert.equal(compareDecisions(undefined, { signals: [] }), 0);
        assert.equal(compareDecisions(null, null), 0);
    });

    it('computes partial similarity for partially overlapping signals', () => {
        const a = {
            signals: [
                { source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 },
                { source: 'alpha', name: 'trend', value: 0.6, confidence: 0.8 },
            ],
        };
        const b = {
            signals: [
                { source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 },
                { source: 'microstructure', name: 'orderflow', value: 0.5, confidence: 0.7 },
            ],
        };
        const score = compareDecisions(a, b);
        assert.ok(score > 0 && score < 1,
            `expected partial similarity, got ${score}`);
    });

    it('penalizes overlapping signals with different values', () => {
        const a = { signals: [{ source: 'a', name: 'x', value: 0.9, confidence: 1 }] };
        const b = { signals: [{ source: 'a', name: 'x', value: -0.9, confidence: 1 }] };
        // Same key but opposite values => lower value agreement
        const score = compareDecisions(a, b);
        assert.ok(score > 0 && score < 0.7,
            `expected reduced similarity for opposite values, got ${score}`);
    });
});

// ===========================================================================
// 4. filterByConfidence
// ===========================================================================
describe('filterByConfidence', () => {

    it('returns decisions with average confidence >= threshold', () => {
        const decisions = [
            { signals: [{ source: 'a', name: 'x', confidence: 0.9 }] },
            { signals: [{ source: 'a', name: 'y', confidence: 0.5 }] },
            { signals: [{ source: 'a', name: 'z', confidence: 0.7 }] },
        ];
        const filtered = filterByConfidence(decisions, 0.7);
        assert.equal(filtered.length, 2);
    });

    it('returns empty array when no decisions meet threshold', () => {
        const decisions = [
            { signals: [{ source: 'a', name: 'x', confidence: 0.3 }] },
            { signals: [{ source: 'a', name: 'y', confidence: 0.4 }] },
        ];
        const filtered = filterByConfidence(decisions, 0.5);
        assert.equal(filtered.length, 0);
    });

    it('returns empty array for decisions with no signals', () => {
        const decisions = [
            { signals: [] },
            { signals: [] },
        ];
        const filtered = filterByConfidence(decisions, 0);
        assert.equal(filtered.length, 0);
    });

    it('returns empty array for empty input', () => {
        assert.deepEqual(filterByConfidence([], 0.5), []);
    });

    it('returns empty array for non-array input', () => {
        assert.deepEqual(filterByConfidence(null, 0.5), []);
        assert.deepEqual(filterByConfidence(undefined, 0.5), []);
    });

    it('handles threshold at exactly 0', () => {
        const decisions = [
            { signals: [{ source: 'a', name: 'x', confidence: 0 }] },
            { signals: [{ source: 'a', name: 'y', confidence: 0.5 }] },
        ];
        // With minConfidence=0, [0.0, 0.5] average is 0.25 >= 0
        const filtered = filterByConfidence(decisions, 0);
        assert.equal(filtered.length, 2);
    });
});

// ===========================================================================
// 5. DecisionGraph — addDecision and getDecision
// ===========================================================================
describe('DecisionGraph — addDecision / getDecision', () => {

    it('stores and retrieves a decision by auto-generated ID', () => {
        const graph = new DecisionGraph();
        const id = graph.addDecision({
            timestamp: 1716000000000,
            action: 'BUY',
            symbol: 'AAPL',
            signals: [{ source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 }],
            reasoning: 'Strong momentum',
        });
        const retrieved = graph.getDecision(id);
        assert.ok(retrieved !== null);
        assert.equal(retrieved.symbol, 'AAPL');
        assert.equal(retrieved.action, 'BUY');
        assert.equal(retrieved.signals.length, 1);
        assert.equal(retrieved.signals[0].name, 'momentum');
    });

    it('stores and retrieves a decision with a custom ID', () => {
        const graph = new DecisionGraph();
        graph.addDecision({
            id: 'MY-DEC-001',
            timestamp: 1716000000000,
            action: 'SELL',
            symbol: 'TSLA',
            signals: [],
            reasoning: 'Technical breakdown',
        });
        const retrieved = graph.getDecision('MY-DEC-001');
        assert.ok(retrieved !== null);
        assert.equal(retrieved.id, 'MY-DEC-001');
    });

    it('returns null for non-existent decision', () => {
        const graph = new DecisionGraph();
        assert.equal(graph.getDecision('NONEXISTENT'), null);
    });

    it('throws for invalid entry (missing fields)', () => {
        const graph = new DecisionGraph();
        assert.throws(() => graph.addDecision({}), /timestamp/);
        assert.throws(() => graph.addDecision({ timestamp: 1 }), /action/);
        assert.throws(() => graph.addDecision({ timestamp: 1, action: 'BUY' }), /symbol/);
        assert.throws(() => graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'AAPL' }), /reasoning/);
    });

    it('throws for invalid action', () => {
        const graph = new DecisionGraph();
        assert.throws(() => graph.addDecision({
            timestamp: 1716000000000,
            action: 'INVALID',
            symbol: 'AAPL',
            signals: [],
            reasoning: 'bad action',
        }), /action/);
    });

    it('throws for duplicate custom ID', () => {
        const graph = new DecisionGraph();
        graph.addDecision({
            id: 'DUP-001',
            timestamp: 1716000000000,
            action: 'HOLD',
            symbol: 'AAPL',
            signals: [],
            reasoning: 'First',
        });
        assert.throws(() => graph.addDecision({
            id: 'DUP-001',
            timestamp: 1716000000001,
            action: 'BUY',
            symbol: 'TSLA',
            signals: [],
            reasoning: 'Second',
        }), /already exists/);
    });

    it('exposes decision edges on getDecision', () => {
        const graph = new DecisionGraph();
        const parentId = graph.addDecision({
            timestamp: 1000, action: 'BUY', symbol: 'AAPL',
            signals: [], reasoning: 'first',
        });
        const childId = graph.addDecision({
            timestamp: 2000, action: 'SELL', symbol: 'AAPL',
            signals: [], reasoning: 'second',
        });
        graph.linkDecisions(parentId, childId, 'leads_to');
        const parent = graph.getDecision(parentId);
        assert.equal(parent.edges.outgoing.length, 1);
        assert.equal(parent.edges.outgoing[0].childId, childId);
        assert.equal(parent.edges.outgoing[0].relationship, 'leads_to');
        assert.equal(parent.edges.incoming.length, 0);
        const child = graph.getDecision(childId);
        assert.equal(child.edges.incoming.length, 1);
        assert.equal(child.edges.incoming[0].parentId, parentId);
    });

    it('track graph size', () => {
        const graph = new DecisionGraph();
        assert.equal(graph.size, 0);
        graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        assert.equal(graph.size, 1);
        graph.addDecision({ timestamp: 2, action: 'SELL', symbol: 'B', signals: [], reasoning: 'r' });
        assert.equal(graph.size, 2);
    });
});

// ===========================================================================
// 6. DecisionGraph — linkDecisions
// ===========================================================================
describe('DecisionGraph — linkDecisions', () => {

    it('creates a directed edge between two decisions', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'SELL', symbol: 'A', signals: [], reasoning: 'r' });
        assert.ok(graph.linkDecisions(a, b, 'leads_to'));
        assert.equal(graph.edgeCount, 1);
    });

    it('throws for non-existent parent', () => {
        const graph = new DecisionGraph();
        const b = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        assert.throws(() => graph.linkDecisions('NOPE', b, 'confirms'), /not found/);
    });

    it('throws for non-existent child', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        assert.throws(() => graph.linkDecisions(a, 'NOPE', 'confirms'), /not found/);
    });

    it('throws for invalid relationship type', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'SELL', symbol: 'A', signals: [], reasoning: 'r' });
        assert.throws(() => graph.linkDecisions(a, b, 'invalid_rel'), /relationship/);
    });

    it('supports all five relationship types', () => {
        const graph = new DecisionGraph();
        const nodes = [];
        for (let i = 0; i < 6; i++) {
            nodes.push(graph.addDecision({
                timestamp: i, action: 'BUY', symbol: 'X', signals: [], reasoning: String(i),
            }));
        }
        const rels = ['refines', 'contradicts', 'confirms', 'leads_to', 'exit_of'];
        for (let i = 0; i < 5; i++) {
            assert.ok(graph.linkDecisions(nodes[i], nodes[i + 1], rels[i]));
        }
        assert.equal(graph.edgeCount, 5);
    });

    it('allows multiple edges from the same parent', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'BUY', symbol: 'B', signals: [], reasoning: 'r' });
        const c = graph.addDecision({ timestamp: 3, action: 'BUY', symbol: 'C', signals: [], reasoning: 'r' });
        graph.linkDecisions(a, b, 'leads_to');
        graph.linkDecisions(a, c, 'leads_to');
        assert.equal(graph.edgeCount, 2);
        const retrieved = graph.getDecision(a);
        assert.equal(retrieved.edges.outgoing.length, 2);
    });
});

// ===========================================================================
// 7. DecisionGraph — recordOutcome
// ===========================================================================
describe('DecisionGraph — recordOutcome', () => {

    it('records outcome on a decision', () => {
        const graph = new DecisionGraph();
        const id = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        graph.recordOutcome(id, { pnl: 250, exitReason: 'take_profit', holdDuration: 7200000, wasCorrect: true });
        const decision = graph.getDecision(id);
        assert.equal(decision.outcome.pnl, 250);
        assert.equal(decision.outcome.exitReason, 'take_profit');
        assert.equal(decision.outcome.holdDuration, 7200000);
        assert.equal(decision.outcome.wasCorrect, true);
    });

    it('throws for non-existent decision', () => {
        const graph = new DecisionGraph();
        assert.throws(() => graph.recordOutcome('FAKE', { wasCorrect: true }), /not found/);
    });

    it('records a losing outcome', () => {
        const graph = new DecisionGraph();
        const id = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        graph.recordOutcome(id, { pnl: -150, exitReason: 'stop_loss', wasCorrect: false });
        const decision = graph.getDecision(id);
        assert.equal(decision.outcome.wasCorrect, false);
        assert.equal(decision.outcome.pnl, -150);
    });

    it('defaults wasCorrect to false when not boolean true', () => {
        const graph = new DecisionGraph();
        const id = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        graph.recordOutcome(id, { wasCorrect: 0 });
        assert.equal(graph.getDecision(id).outcome.wasCorrect, false);
    });

    it('records minimal outcome (only wasCorrect)', () => {
        const graph = new DecisionGraph();
        const id = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        graph.recordOutcome(id, { wasCorrect: true });
        const decision = graph.getDecision(id);
        assert.equal(decision.outcome.wasCorrect, true);
        assert.equal(decision.outcome.pnl, undefined);
        assert.equal(decision.outcome.exitReason, undefined);
    });
});

// ===========================================================================
// 8. DecisionGraph — getDecisions (filtering)
// ===========================================================================
describe('DecisionGraph — getDecisions (filtering)', () => {

    function setupGraph() {
        const g = new DecisionGraph();
        const d1 = g.addDecision({ timestamp: 1000, action: 'BUY', symbol: 'AAPL', signals: [{ source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 }], reasoning: 'r1' });
        const d2 = g.addDecision({ timestamp: 2000, action: 'SELL', symbol: 'TSLA', signals: [{ source: 'microstructure', name: 'orderflow', value: -0.6, confidence: 0.85 }], reasoning: 'r2' });
        const d3 = g.addDecision({ timestamp: 3000, action: 'BUY', symbol: 'AAPL', signals: [{ source: 'alpha', name: 'momentum', value: 0.5, confidence: 0.7 }], reasoning: 'r3' });
        g.recordOutcome(d1, { pnl: 200, wasCorrect: true });
        g.recordOutcome(d2, { pnl: -100, wasCorrect: false });
        g.recordOutcome(d3, { pnl: 50, wasCorrect: true });
        return { graph: g, ids: { d1, d2, d3 } };
    }

    it('returns all decisions when filter is empty', () => {
        const { graph } = setupGraph();
        const all = graph.getDecisions();
        assert.equal(all.length, 3);
    });

    it('filters by symbol', () => {
        const { graph } = setupGraph();
        const aapl = graph.getDecisions({ symbol: 'AAPL' });
        assert.equal(aapl.length, 2);
        assert.ok(aapl.every(d => d.symbol === 'AAPL'));
    });

    it('filters by action', () => {
        const { graph } = setupGraph();
        const buys = graph.getDecisions({ action: 'BUY' });
        assert.equal(buys.length, 2);
        assert.ok(buys.every(d => d.action === 'BUY'));
    });

    it('filters by source', () => {
        const { graph } = setupGraph();
        const alpha = graph.getDecisions({ source: 'alpha' });
        assert.equal(alpha.length, 2);
    });

    it('filters by signal name', () => {
        const { graph } = setupGraph();
        const orderflow = graph.getDecisions({ signalName: 'orderflow' });
        assert.equal(orderflow.length, 1);
        assert.equal(orderflow[0].action, 'SELL');
    });

    it('filters by wasCorrect', () => {
        const { graph } = setupGraph();
        const winners = graph.getDecisions({ wasCorrect: true });
        assert.equal(winners.length, 2);
        const losers = graph.getDecisions({ wasCorrect: false });
        assert.equal(losers.length, 1);
    });

    it('filters by combined criteria (AND logic)', () => {
        const { graph } = setupGraph();
        const filtered = graph.getDecisions({ symbol: 'AAPL', action: 'BUY', wasCorrect: true });
        assert.equal(filtered.length, 2);
    });

    it('filters by timeRange', () => {
        const { graph } = setupGraph();
        const mid = graph.getDecisions({ timeRange: { start: 1500, end: 2500 } });
        assert.equal(mid.length, 1);
        assert.equal(mid[0].timestamp, 2000);
    });

    it('returns empty array when no matches', () => {
        const { graph } = setupGraph();
        const empty = graph.getDecisions({ symbol: 'NONEXISTENT' });
        assert.deepEqual(empty, []);
    });

    it('returns decisions sorted by timestamp', () => {
        const { graph } = setupGraph();
        const all = graph.getDecisions();
        for (let i = 1; i < all.length; i++) {
            assert.ok(all[i - 1].timestamp <= all[i].timestamp);
        }
    });
});

// ===========================================================================
// 9. DecisionGraph — getTrace
// ===========================================================================
describe('DecisionGraph — getTrace', () => {

    it('returns an empty array for non-existent decision', () => {
        const graph = new DecisionGraph();
        assert.deepEqual(graph.getTrace('NOPE'), []);
    });

    it('returns a single decision when no edges exist', () => {
        const graph = new DecisionGraph();
        const id = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const trace = graph.getTrace(id);
        assert.equal(trace.length, 1);
        assert.equal(trace[0].id, id);
    });

    it('follows a linear chain', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'SELL', symbol: 'A', signals: [], reasoning: 'r' });
        const c = graph.addDecision({ timestamp: 3, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        graph.linkDecisions(a, b, 'leads_to');
        graph.linkDecisions(b, c, 'leads_to');

        const traceA = graph.getTrace(a);
        assert.equal(traceA.length, 3);
        assert.equal(traceA[0].id, a);
        assert.equal(traceA[1].id, b);
        assert.equal(traceA[2].id, c);

        const traceB = graph.getTrace(b);
        assert.equal(traceB.length, 3);
    });

    it('follows incoming edges backward', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'SELL', symbol: 'A', signals: [], reasoning: 'r' });
        graph.linkDecisions(a, b, 'leads_to');

        const traceFromB = graph.getTrace(b);
        assert.equal(traceFromB.length, 2);
        assert.equal(traceFromB[0].id, a);
        assert.equal(traceFromB[1].id, b);
    });

    it('follows branching chains', () => {
        const graph = new DecisionGraph();
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'BUY', symbol: 'B', signals: [], reasoning: 'r' });
        const c = graph.addDecision({ timestamp: 3, action: 'BUY', symbol: 'C', signals: [], reasoning: 'r' });
        graph.linkDecisions(a, b, 'leads_to');
        graph.linkDecisions(a, c, 'leads_to');

        const trace = graph.getTrace(a);
        assert.equal(trace.length, 3);
    });

    it('returns decisions sorted by timestamp', () => {
        const graph = new DecisionGraph();
        const c = graph.addDecision({ timestamp: 3, action: 'BUY', symbol: 'C', signals: [], reasoning: 'r' });
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'BUY', symbol: 'B', signals: [], reasoning: 'r' });
        graph.linkDecisions(a, b, 'leads_to');
        graph.linkDecisions(b, c, 'leads_to');

        const trace = graph.getTrace(a);
        assert.equal(trace[0].id, a);
        assert.equal(trace[1].id, b);
        assert.equal(trace[2].id, c);
    });
});

// ===========================================================================
// 10. DecisionGraph — getPatterns
// ===========================================================================
describe('DecisionGraph — getPatterns', () => {

    it('returns empty patterns when there are no outcomes', () => {
        const graph = new DecisionGraph();
        graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const patterns = graph.getPatterns();
        assert.deepEqual(patterns.winningCombos, []);
        assert.deepEqual(patterns.losingCombos, []);
        assert.deepEqual(patterns.bestSources, []);
        assert.equal(patterns.avgConfidence.winning, 0);
        assert.equal(patterns.avgConfidence.losing, 0);
        assert.deepEqual(patterns.commonExitReasons, []);
    });

    it('detects winning signal combos', () => {
        const graph = new DecisionGraph();
        const d1 = graph.addDecision({
            timestamp: 1, action: 'BUY', symbol: 'A',
            signals: [{ source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 }],
            reasoning: 'r',
        });
        const d2 = graph.addDecision({
            timestamp: 2, action: 'BUY', symbol: 'B',
            signals: [{ source: 'alpha', name: 'momentum', value: 0.7, confidence: 0.85 }],
            reasoning: 'r',
        });
        graph.recordOutcome(d1, { pnl: 100, wasCorrect: true });
        graph.recordOutcome(d2, { pnl: 50, wasCorrect: true });
        const patterns = graph.getPatterns();
        assert.ok(patterns.winningCombos.length >= 1);
        const momentum = patterns.winningCombos.find(c => c.signal === 'alpha:momentum');
        assert.ok(momentum !== undefined);
        assert.equal(momentum.winCount, 2);
    });

    it('computes source accuracy correctly', () => {
        const graph = new DecisionGraph();
        const d1 = graph.addDecision({
            timestamp: 1, action: 'BUY', symbol: 'A',
            signals: [{ source: 'alpha', name: 'mom', value: 0.8, confidence: 0.9 }],
            reasoning: 'r',
        });
        const d2 = graph.addDecision({
            timestamp: 2, action: 'SELL', symbol: 'B',
            signals: [{ source: 'alpha', name: 'mom', value: -0.5, confidence: 0.7 }],
            reasoning: 'r',
        });
        const d3 = graph.addDecision({
            timestamp: 3, action: 'BUY', symbol: 'C',
            signals: [{ source: 'microstructure', name: 'flow', value: 0.6, confidence: 0.8 }],
            reasoning: 'r',
        });
        graph.recordOutcome(d1, { pnl: 100, wasCorrect: true });
        graph.recordOutcome(d2, { pnl: -50, wasCorrect: false });
        graph.recordOutcome(d3, { pnl: 75, wasCorrect: true });
        const patterns = graph.getPatterns();
        const alpha = patterns.bestSources.find(s => s.source === 'alpha');
        assert.ok(alpha !== undefined);
        assert.equal(alpha.totalDecisions, 2);
        assert.equal(alpha.winningDecisions, 1);
        assert.equal(alpha.accuracy, 0.5);
        const micro = patterns.bestSources.find(s => s.source === 'microstructure');
        assert.equal(micro.accuracy, 1);
    });

    it('computes average confidence for winning vs losing', () => {
        const graph = new DecisionGraph();
        const d1 = graph.addDecision({
            timestamp: 1, action: 'BUY', symbol: 'A',
            signals: [{ source: 'a', name: 'x', value: 1, confidence: 0.9 }],
            reasoning: 'r',
        });
        const d2 = graph.addDecision({
            timestamp: 2, action: 'SELL', symbol: 'B',
            signals: [{ source: 'a', name: 'x', value: -1, confidence: 0.5 }],
            reasoning: 'r',
        });
        graph.recordOutcome(d1, { pnl: 100, wasCorrect: true });
        graph.recordOutcome(d2, { pnl: -50, wasCorrect: false });
        const patterns = graph.getPatterns();
        assert.equal(patterns.avgConfidence.winning, 0.9);
        assert.equal(patterns.avgConfidence.losing, 0.5);
    });

    it('returns common exit reasons sorted by frequency', () => {
        const graph = new DecisionGraph();
        const d1 = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const d2 = graph.addDecision({ timestamp: 2, action: 'BUY', symbol: 'B', signals: [], reasoning: 'r' });
        const d3 = graph.addDecision({ timestamp: 3, action: 'BUY', symbol: 'C', signals: [], reasoning: 'r' });
        graph.recordOutcome(d1, { pnl: 100, exitReason: 'take_profit', wasCorrect: true });
        graph.recordOutcome(d2, { pnl: -50, exitReason: 'stop_loss', wasCorrect: false });
        graph.recordOutcome(d3, { pnl: 25, exitReason: 'take_profit', wasCorrect: true });
        const patterns = graph.getPatterns();
        assert.equal(patterns.commonExitReasons.length, 2);
        assert.equal(patterns.commonExitReasons[0].reason, 'take_profit');
        assert.equal(patterns.commonExitReasons[0].count, 2);
        assert.equal(patterns.commonExitReasons[1].reason, 'stop_loss');
        assert.equal(patterns.commonExitReasons[1].count, 1);
    });

    it('handles empty graph gracefully', () => {
        const graph = new DecisionGraph();
        const patterns = graph.getPatterns();
        assert.equal(patterns.avgConfidence.winning, 0);
        assert.equal(patterns.avgConfidence.losing, 0);
    });
});

// ===========================================================================
// 11. GraphQuery
// ===========================================================================
describe('GraphQuery', () => {

    function sampleDecisions() {
        return [
            {
                id: 'd1',
                timestamp: 1000,
                action: 'BUY',
                symbol: 'AAPL',
                signals: [{ source: 'alpha', name: 'momentum', value: 0.8, confidence: 0.9 }],
                reasoning: 'Strong',
                outcome: { pnl: 200, wasCorrect: true },
            },
            {
                id: 'd2',
                timestamp: 2000,
                action: 'SELL',
                symbol: 'TSLA',
                signals: [
                    { source: 'microstructure', name: 'orderflow', value: -0.6, confidence: 0.85 },
                    { source: 'alpha', name: 'trend', value: -0.4, confidence: 0.7 },
                ],
                reasoning: 'Weak flow',
                outcome: { pnl: -100, wasCorrect: false },
            },
            {
                id: 'd3',
                timestamp: 3000,
                action: 'BUY',
                symbol: 'AAPL',
                signals: [{ source: 'alpha', name: 'momentum', value: 0.5, confidence: 0.7 }],
                reasoning: 'Moderate',
                outcome: { pnl: 50, wasCorrect: true },
            },
            {
                id: 'd4',
                timestamp: 4000,
                action: 'HOLD',
                symbol: 'GOOGL',
                signals: [],
                reasoning: 'No signal',
                outcome: { pnl: 0, wasCorrect: true },
            },
        ];
    }

    it('execute() returns all decisions when no filters applied', () => {
        const q = new GraphQuery(sampleDecisions());
        assert.equal(q.execute().length, 4);
    });

    it('symbol() filters by symbol', () => {
        const q = new GraphQuery(sampleDecisions()).symbol('AAPL');
        assert.equal(q.count(), 2);
        assert.ok(q.execute().every(d => d.symbol === 'AAPL'));
    });

    it('action() filters by action', () => {
        const q = new GraphQuery(sampleDecisions()).action('BUY');
        assert.equal(q.count(), 2);
    });

    it('source() filters by signal source', () => {
        const q = new GraphQuery(sampleDecisions()).source('microstructure');
        assert.equal(q.count(), 1);
    });

    it('profitable() filters to winning decisions only', () => {
        const q = new GraphQuery(sampleDecisions()).profitable();
        assert.equal(q.count(), 3); // d1, d3, d4 (d4 has wasCorrect: true)
    });

    it('unprofitable() filters to losing decisions only', () => {
        const q = new GraphQuery(sampleDecisions()).unprofitable();
        assert.equal(q.count(), 1);
        assert.equal(q.execute()[0].id, 'd2');
    });

    it('confidenceAbove() filters by min average confidence', () => {
        const q = new GraphQuery(sampleDecisions()).confidenceAbove(0.8);
        // d1: 0.9, d2: avg(0.85,0.7)=0.775, d3: 0.7, d4: no signals
        assert.equal(q.count(), 1);
    });

    it('chained filters combine with AND logic', () => {
        const q = new GraphQuery(sampleDecisions())
            .symbol('AAPL')
            .action('BUY')
            .profitable();
        assert.equal(q.count(), 2);
    });

    it('count() returns correct count after filtering', () => {
        const q = new GraphQuery(sampleDecisions()).symbol('AAPL');
        assert.equal(q.count(), 2);
    });

    it('avgConfidence() computes correct average', () => {
        const q = new GraphQuery(sampleDecisions()).symbol('AAPL');
        // d1: 0.9, d3: 0.7 => avg = 0.8
        assert.equal(q.avgConfidence(), 0.8);
    });

    it('avgConfidence() returns 0 for empty result set', () => {
        const q = new GraphQuery(sampleDecisions()).symbol('NONEXISTENT');
        assert.equal(q.avgConfidence(), 0);
    });

    it('avgConfidence() for decisions with no signals', () => {
        const q = new GraphQuery(sampleDecisions()).symbol('GOOGL');
        assert.equal(q.avgConfidence(), 0);
    });

    it('topSignal() returns the most common signal key', () => {
        const q = new GraphQuery(sampleDecisions());
        const top = q.topSignal();
        assert.equal(top, 'alpha:momentum'); // appears in d1 and d3
    });

    it('topSignal() returns null when no signals', () => {
        const q = new GraphQuery([]);
        assert.equal(q.topSignal(), null);
    });

    it('execute() returns decisions sorted by timestamp', () => {
        const q = new GraphQuery(sampleDecisions());
        const results = q.execute();
        for (let i = 1; i < results.length; i++) {
            assert.ok(results[i - 1].timestamp <= results[i].timestamp);
        }
    });

    it('handles empty initial array', () => {
        const q = new GraphQuery([]);
        assert.equal(q.count(), 0);
        assert.deepEqual(q.execute(), []);
        assert.equal(q.avgConfidence(), 0);
        assert.equal(q.topSignal(), null);
    });

    it('handles non-array initial input', () => {
        const q = new GraphQuery(null);
        assert.equal(q.count(), 0);
        assert.deepEqual(q.execute(), []);
    });
});

// ===========================================================================
// 12. DecisionGraph — all() helper and edge count
// ===========================================================================
describe('DecisionGraph — all() and edgeCount', () => {

    it('all() returns a shallow copy of all decisions', () => {
        const graph = new DecisionGraph();
        graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        graph.addDecision({ timestamp: 2, action: 'SELL', symbol: 'B', signals: [], reasoning: 'r' });
        const all = graph.all();
        assert.equal(all.length, 2);
        // Mutation safety: modifying returned array doesn't affect graph
        all.length = 0;
        assert.equal(graph.size, 2);
    });

    it('edgeCount reflects total edges', () => {
        const graph = new DecisionGraph();
        assert.equal(graph.edgeCount, 0);
        const a = graph.addDecision({ timestamp: 1, action: 'BUY', symbol: 'A', signals: [], reasoning: 'r' });
        const b = graph.addDecision({ timestamp: 2, action: 'BUY', symbol: 'B', signals: [], reasoning: 'r' });
        const c = graph.addDecision({ timestamp: 3, action: 'BUY', symbol: 'C', signals: [], reasoning: 'r' });
        graph.linkDecisions(a, b, 'confirms');
        graph.linkDecisions(a, c, 'confirms');
        assert.equal(graph.edgeCount, 2);
    });
});
