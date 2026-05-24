/**
 * Cost Tracker — unit tests (node:test runner)
 * Run: node --test proxy/cost-tracker.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { computeCost, createCostTracker } from './cost-tracker.js';

describe('computeCost', () => {
    it('returns 0 for zero tokens', () => {
        assert.equal(computeCost(0, 0), 0);
    });

    it('computes cost for deepseek with 75% discount', () => {
        // DeepSeek discounted: input 0.44*0.25=0.11, output 0.87*0.25=0.2175 per 1M
        const cost = computeCost(1_000_000, 1_000_000, 0, 0, 'deepseek');
        // 1M * 0.11 + 1M * 0.2175 = 0.3275
        assert.ok(cost > 0.30 && cost < 0.35, `expected ~0.3275, got ${cost}`);
    });

    it('computes anthropic cost at full price', () => {
        const cost = computeCost(1_000_000, 1_000_000, 0, 0, 'anthropic');
        // 1M * 3.00 + 1M * 15.00 = 18.00
        assert.ok(cost > 17 && cost < 19, `expected ~18, got ${cost}`);
    });

    it('handles fractional tokens', () => {
        const cost = computeCost(500, 2000, 0, 0, 'anthropic');
        assert.ok(cost > 0.03 && cost < 0.04, `expected ~0.0315, got ${cost}`);
    });

    it('gemini is most expensive for output-heavy workloads', () => {
        const geminiCost = computeCost(100, 10000, 0, 0, 'gemini');
        const deepseekCost = computeCost(100, 10000, 0, 0, 'deepseek');
        assert.ok(geminiCost > deepseekCost);
    });

    it('includes cache read tokens in cost', () => {
        const withCache = computeCost(0, 0, 1_000_000, 0, 'anthropic');
        // 1M cache reads at $0.30/M
        assert.ok(withCache > 0.25 && withCache < 0.35, `got ${withCache}`);
    });

    it('includes cache write tokens in cost', () => {
        const withCache = computeCost(0, 0, 0, 1_000_000, 'anthropic');
        // 1M cache writes at $3.00/M
        assert.ok(withCache > 2.50 && withCache < 3.50, `got ${withCache}`);
    });

    it('defaults to deepseek full-price for unknown backend (no discount applied)', () => {
        const cost = computeCost(1_000_000, 1_000_000, 0, 0, 'unknown');
        // Falls back to PRICING.deepseek at full price: $0.44 + $0.87 = $1.31
        assert.ok(cost > 1.25 && cost < 1.40, `expected ~1.31, got ${cost}`);
    });
});

describe('createCostTracker', () => {
    it('starts with zero cost', () => {
        const ct = createCostTracker();
        const s = ct.getSummary();
        assert.equal(s.requestCount, 0);
        assert.equal(s.totals.cost, 0);
        assert.equal(s.totals.inputTokens, 0);
    });

    it('tracks a single request', () => {
        const ct = createCostTracker();
        ct.record({ model: 'claude-opus-4-7', backend: 'deepseek', inputTokens: 5000, outputTokens: 2000 });
        const s = ct.getSummary();
        assert.equal(s.requestCount, 1);
        assert.equal(s.totals.inputTokens, 5000);
        assert.equal(s.totals.outputTokens, 2000);
        assert.ok(s.totals.cost > 0);
    });

    it('tracks multiple requests with running totals', () => {
        const ct = createCostTracker();
        ct.record({ model: 'm1', inputTokens: 100, outputTokens: 200 });
        ct.record({ model: 'm2', inputTokens: 300, outputTokens: 400 });
        const s = ct.getSummary();
        assert.equal(s.requestCount, 2);
        assert.equal(s.totals.inputTokens, 400);
        assert.equal(s.totals.outputTokens, 600);
    });

    it('tracks per-backend breakdown', () => {
        const ct = createCostTracker();
        ct.record({ model: 'm1', backend: 'deepseek', inputTokens: 1000, outputTokens: 500 });
        ct.record({ model: 'm2', backend: 'gemini', inputTokens: 1000, outputTokens: 500 });
        const s = ct.getSummary();
        assert.ok(s.byBackend.deepseek);
        assert.ok(s.byBackend.gemini);
        assert.equal(s.byBackend.deepseek.count, 1);
        assert.equal(s.byBackend.gemini.count, 1);
    });

    it('tracks per-model breakdown', () => {
        const ct = createCostTracker();
        ct.record({ model: 'claude-opus-4-7', inputTokens: 1000, outputTokens: 500 });
        ct.record({ model: 'claude-haiku-4-5', inputTokens: 100, outputTokens: 50 });
        ct.record({ model: 'claude-opus-4-7', inputTokens: 2000, outputTokens: 1000 });
        const s = ct.getSummary();
        assert.equal(s.byModel['claude-opus-4-7'].count, 2);
        assert.equal(s.byModel['claude-haiku-4-5'].count, 1);
    });

    it('defaults backend when not specified', () => {
        const ct = createCostTracker({ defaultBackend: 'gemini' });
        ct.record({ model: 'test', inputTokens: 1000, outputTokens: 1000 });
        const s = ct.getSummary();
        assert.ok(s.byBackend.gemini);
    });

    it('uses custom sessionId', () => {
        const ct = createCostTracker({ sessionId: 'my-session-123' });
        assert.equal(ct.sessionId, 'my-session-123');
        assert.equal(ct.getSummary().sessionId, 'my-session-123');
    });

    it('getReport returns formatted text', () => {
        const ct = createCostTracker({ sessionId: 'test' });
        ct.record({ model: 'm1', backend: 'deepseek', inputTokens: 10000, outputTokens: 5000 });
        const report = ct.getReport();
        assert.ok(report.includes('test'));
        assert.ok(report.includes('$'));
        assert.ok(report.includes('deepseek'));
    });

    it('compareWith shows savings vs anthropic', () => {
        const ct = createCostTracker();
        ct.record({ model: 'm1', backend: 'deepseek', inputTokens: 100000, outputTokens: 50000 });
        ct.record({ model: 'm2', backend: 'deepseek', inputTokens: 50000, outputTokens: 25000 });
        const cmp = ct.compareWith('anthropic');
        assert.ok(cmp.actual > 0);
        assert.ok(cmp.compared > cmp.actual, `compared ${cmp.compared} should be > actual ${cmp.actual}`);
        assert.ok(cmp.savings > 0);
        assert.ok(cmp.savingsPct > 0);
    });

    it('compareWith returns zeros for empty tracker', () => {
        const ct = createCostTracker();
        const cmp = ct.compareWith('anthropic');
        assert.equal(cmp.actual, 0);
        assert.equal(cmp.compared, 0);
        assert.equal(cmp.savings, 0);
    });

    it('topRequests returns most expensive requests sorted', () => {
        const ct = createCostTracker();
        ct.record({ model: 'cheap', backend: 'deepseek', inputTokens: 100, outputTokens: 100 });
        ct.record({ model: 'expensive', backend: 'deepseek', inputTokens: 100000, outputTokens: 50000 });
        ct.record({ model: 'medium', backend: 'deepseek', inputTokens: 1000, outputTokens: 500 });
        const s = ct.getSummary();
        assert.equal(s.topRequests.length, 3);
        // Most expensive first
        assert.ok(s.topRequests[0].cost >= s.topRequests[1].cost);
        assert.ok(s.topRequests[1].cost >= s.topRequests[2].cost);
    });

    it('reset clears all data', () => {
        const ct = createCostTracker();
        ct.record({ model: 'm1', inputTokens: 1000, outputTokens: 500 });
        ct.reset();
        const s = ct.getSummary();
        assert.equal(s.requestCount, 0);
        assert.equal(s.totals.cost, 0);
        assert.deepEqual(s.byBackend, {});
        assert.deepEqual(s.byModel, {});
    });

    it('averages are computed correctly', () => {
        const ct = createCostTracker();
        ct.record({ model: 'm1', inputTokens: 1000, outputTokens: 200 });
        ct.record({ model: 'm2', inputTokens: 2000, outputTokens: 400 });
        const s = ct.getSummary();
        assert.equal(s.averages.inputPerRequest, 1500);
        assert.equal(s.averages.outputPerRequest, 300);
    });

    it('handles cache read/write tokens', () => {
        const ct = createCostTracker();
        ct.record({
            model: 'm1',
            backend: 'deepseek',
            inputTokens: 5000,
            outputTokens: 2000,
            cacheReadTokens: 10000,
            cacheWriteTokens: 5000,
        });
        const s = ct.getSummary();
        assert.equal(s.totals.cacheReadTokens, 10000);
        assert.equal(s.totals.cacheWriteTokens, 5000);
    });

    it('handles latencyMs tracking', () => {
        const ct = createCostTracker();
        ct.record({ model: 'm1', inputTokens: 100, outputTokens: 100, latencyMs: 250 });
        const s = ct.getSummary();
        assert.equal(s.topRequests[0].latencyMs, 250);
    });

    it('handles missing optional fields gracefully', () => {
        const ct = createCostTracker();
        // Minimal record — only model and input tokens
        ct.record({ model: 'minimal', inputTokens: 100, outputTokens: 0 });
        const s = ct.getSummary();
        assert.equal(s.requestCount, 1);
    });
});
