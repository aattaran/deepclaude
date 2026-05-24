/**
 * Cost Tracker — per-session cost tracking for the model proxy
 *
 * Tracks token usage and estimates cost across backends, models, and sessions.
 * All monetary values in USD.
 *
 * ES module. Zero dependencies.
 */

// Pricing per 1M tokens (May 2026). Keep in sync with model-proxy.js PRICING_PER_M.
const PRICING = {
    deepseek:   { input: 0.44,  output: 0.87,  cacheRead: 0.004, cacheWrite: 0.044 },
    openrouter: { input: 0.44,  output: 0.87,  cacheRead: 0.004, cacheWrite: 0.044 },
    fireworks:  { input: 1.74,  output: 3.48,  cacheRead: 0.017, cacheWrite: 0.174 },
    gemini:     { input: 1.50,  output: 9.00,  cacheRead: 0.15,  cacheWrite: 1.50 },
    anthropic:  { input: 3.00,  output: 15.00, cacheRead: 0.30,  cacheWrite: 3.00 },
};

// DeepSeek 75% discount until May 31, 2026
const DEEPSEEK_DISCOUNT = 0.25; // pay 25% of list price
const DISCOUNT_END = new Date('2026-06-01T00:00:00Z');

/**
 * Get effective pricing for a backend, applying any active discounts.
 * @param {string} backend
 * @returns {{ input: number, output: number, cacheRead: number, cacheWrite: number }}
 */
function effectivePricing(backend) {
    const base = PRICING[backend] || PRICING.deepseek;
    const now = new Date();
    if ((backend === 'deepseek' || backend === 'openrouter') && now < DISCOUNT_END) {
        return {
            input:     base.input  * DEEPSEEK_DISCOUNT,
            output:    base.output * DEEPSEEK_DISCOUNT,
            cacheRead: base.cacheRead * DEEPSEEK_DISCOUNT,
            cacheWrite: base.cacheWrite * DEEPSEEK_DISCOUNT,
        };
    }
    return { ...base };
}

/**
 * Compute cost from token counts.
 * @param {number} inputTokens
 * @param {number} outputTokens
 * @param {number} [cacheReadTokens=0]
 * @param {number} [cacheWriteTokens=0]
 * @param {string} [backend='deepseek']
 * @returns {number} cost in USD
 */
export function computeCost(inputTokens, outputTokens, cacheReadTokens = 0, cacheWriteTokens = 0, backend = 'deepseek') {
    const p = effectivePricing(backend);
    return (
        (inputTokens     / 1_000_000) * p.input +
        (outputTokens    / 1_000_000) * p.output +
        (cacheReadTokens / 1_000_000) * p.cacheRead +
        (cacheWriteTokens / 1_000_000) * p.cacheWrite
    );
}

/**
 * Create a session cost tracker.
 *
 * @param {object} [options]
 * @param {string} [options.sessionId] — identifier for this session
 * @param {string} [options.defaultBackend='deepseek']
 * @returns {CostTracker}
 */
export function createCostTracker(options = {}) {
    const sessionId = options.sessionId ?? `session-${Date.now()}`;
    const defaultBackend = options.defaultBackend ?? 'deepseek';

    let totalCost = 0;
    let totalInput = 0;
    let totalOutput = 0;
    let totalCacheRead = 0;
    let totalCacheWrite = 0;
    let requestCount = 0;

    /** @type {Array<{ ts: string, model: string, backend: string, input: number, output: number, cacheRead: number, cacheWrite: number, cost: number, latencyMs: number }>} */
    const requests = [];

    /** @type {Record<string, { count: number, input: number, output: number, cacheRead: number, cacheWrite: number, cost: number }>} */
    const byBackend = {};

    /** @type {Record<string, { count: number, input: number, output: number, cacheRead: number, cacheWrite: number, cost: number }>} */
    const byModel = {};

    /**
     * Record a completed request.
     *
     * @param {object} entry
     * @param {string} entry.model       — model name used
     * @param {string} [entry.backend]   — backend that served the request
     * @param {number} entry.inputTokens
     * @param {number} entry.outputTokens
     * @param {number} [entry.cacheReadTokens=0]
     * @param {number} [entry.cacheWriteTokens=0]
     * @param {number} [entry.latencyMs=0]
     */
    function record(entry) {
        const backend = entry.backend ?? defaultBackend;
        const input = entry.inputTokens ?? 0;
        const output = entry.outputTokens ?? 0;
        const cacheRead = entry.cacheReadTokens ?? 0;
        const cacheWrite = entry.cacheWriteTokens ?? 0;
        const latency = entry.latencyMs ?? 0;
        const model = entry.model ?? 'unknown';

        const cost = computeCost(input, output, cacheRead, cacheWrite, backend);

        totalCost += cost;
        totalInput += input;
        totalOutput += output;
        totalCacheRead += cacheRead;
        totalCacheWrite += cacheWrite;
        requestCount++;

        requests.push({
            ts: new Date().toISOString(),
            model,
            backend,
            input,
            output,
            cacheRead,
            cacheWrite,
            cost,
            latencyMs: latency,
        });

        // Per-backend
        if (!byBackend[backend]) {
            byBackend[backend] = { count: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 };
        }
        byBackend[backend].count++;
        byBackend[backend].input += input;
        byBackend[backend].output += output;
        byBackend[backend].cacheRead += cacheRead;
        byBackend[backend].cacheWrite += cacheWrite;
        byBackend[backend].cost += cost;

        // Per-model
        if (!byModel[model]) {
            byModel[model] = { count: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0 };
        }
        byModel[model].count++;
        byModel[model].input += input;
        byModel[model].output += output;
        byModel[model].cacheRead += cacheRead;
        byModel[model].cacheWrite += cacheWrite;
        byModel[model].cost += cost;
    }

    /**
     * Get a summary of all costs in this session.
     * @returns {CostSummary}
     */
    function getSummary() {
        const avgCostPerReq = requestCount > 0 ? totalCost / requestCount : 0;
        const avgInputPerReq = requestCount > 0 ? totalInput / requestCount : 0;
        const avgOutputPerReq = requestCount > 0 ? totalOutput / requestCount : 0;

        const top3ByCost = [...requests]
            .sort((a, b) => b.cost - a.cost)
            .slice(0, 3);

        return {
            sessionId,
            requestCount,
            totals: {
                cost: +totalCost.toFixed(6),
                inputTokens: totalInput,
                outputTokens: totalOutput,
                cacheReadTokens: totalCacheRead,
                cacheWriteTokens: totalCacheWrite,
            },
            averages: {
                costPerRequest: +avgCostPerReq.toFixed(6),
                inputPerRequest: +avgInputPerReq.toFixed(1),
                outputPerRequest: +avgOutputPerReq.toFixed(1),
            },
            byBackend: Object.fromEntries(
                Object.entries(byBackend).map(([k, v]) => [k, { ...v, cost: +v.cost.toFixed(6) }]),
            ),
            byModel: Object.fromEntries(
                Object.entries(byModel).map(([k, v]) => [k, { ...v, cost: +v.cost.toFixed(6) }]),
            ),
            topRequests: top3ByCost,
        };
    }

    /**
     * Compare actual cost against what it would have cost on another backend.
     *
     * @param {string} compareBackend — backend to compare against
     * @returns {{ actual: number, compared: number, savings: number, savingsPct: number }}
     */
    function compareWith(compareBackend = 'anthropic') {
        if (requestCount === 0) return { actual: 0, compared: 0, savings: 0, savingsPct: 0 };

        // Recompute all requests with compareBackend pricing
        let comparedCost = 0;
        for (const req of requests) {
            comparedCost += computeCost(req.input, req.output, req.cacheRead, req.cacheWrite, compareBackend);
        }

        const actual = totalCost;
        const savings = comparedCost - actual;
        const savingsPct = comparedCost > 0 ? (savings / comparedCost) * 100 : 0;

        return {
            actual: +actual.toFixed(6),
            compared: +comparedCost.toFixed(6),
            savings: +savings.toFixed(6),
            savingsPct: +savingsPct.toFixed(1),
        };
    }

    /**
     * Get a formatted text report.
     * @returns {string}
     */
    function getReport() {
        const s = getSummary();
        const cmp = compareWith('anthropic');
        const lines = [];

        lines.push('═'.repeat(60));
        lines.push('  Cost Report');
        lines.push('═'.repeat(60));
        lines.push(`  Session:     ${s.sessionId}`);
        lines.push(`  Requests:    ${s.requestCount}`);
        lines.push(`  Total cost:  $${s.totals.cost.toFixed(4)}`);
        lines.push('');
        lines.push(`  Input:       ${s.totals.inputTokens.toLocaleString()} tokens`);
        lines.push(`  Output:      ${s.totals.outputTokens.toLocaleString()} tokens`);
        lines.push(`  Avg/req:     $${s.averages.costPerRequest.toFixed(4)}`);
        lines.push('');
        lines.push('  vs Anthropic:');
        lines.push(`    Saved:     $${cmp.savings.toFixed(4)} (${cmp.savingsPct.toFixed(0)}%)`);

        if (Object.keys(s.byBackend).length) {
            lines.push('');
            lines.push('  By Backend:');
            for (const [name, stats] of Object.entries(s.byBackend)) {
                lines.push(`    ${name}: ${stats.count} reqs, $${stats.cost.toFixed(4)}`);
            }
        }

        if (Object.keys(s.byModel).length > 1) {
            lines.push('');
            lines.push('  By Model:');
            for (const [name, stats] of Object.entries(s.byModel)) {
                lines.push(`    ${name}: ${stats.count} reqs, $${stats.cost.toFixed(4)}`);
            }
        }

        lines.push('═'.repeat(60));
        return lines.join('\n');
    }

    /**
     * Reset all counters for a new session.
     */
    function reset() {
        totalCost = 0;
        totalInput = 0;
        totalOutput = 0;
        totalCacheRead = 0;
        totalCacheWrite = 0;
        requestCount = 0;
        requests.length = 0;
        for (const k of Object.keys(byBackend)) delete byBackend[k];
        for (const k of Object.keys(byModel)) delete byModel[k];
    }

    return {
        record,
        getSummary,
        compareWith,
        getReport,
        reset,
        get sessionId() { return sessionId; },
        get requestCount() { return requestCount; },
        get totalCost() { return totalCost; },
    };
}

/**
 * @typedef {object} CostSummary
 * @property {string} sessionId
 * @property {number} requestCount
 * @property {{ cost: number, inputTokens: number, outputTokens: number, cacheReadTokens: number, cacheWriteTokens: number }} totals
 * @property {{ costPerRequest: number, inputPerRequest: number, outputPerRequest: number }} averages
 * @property {Record<string, { count: number, input: number, output: number, cacheRead: number, cacheWrite: number, cost: number }>} byBackend
 * @property {Record<string, { count: number, input: number, output: number, cacheRead: number, cacheWrite: number, cost: number }>} byModel
 * @property {Array<{ ts: string, model: string, backend: string, input: number, output: number, cacheRead: number, cacheWrite: number, cost: number, latencyMs: number }>} topRequests
 */
