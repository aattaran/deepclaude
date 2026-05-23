// ---------------------------------------------------------------------------
// observability.js — structured logging, metrics, tracing for the model proxy
// ---------------------------------------------------------------------------
// All output goes to stderr as JSON lines, keeping stdout clean for proxy data.
// Every export is a factory or pure function; no global state is maintained.
// ---------------------------------------------------------------------------

import { randomBytes } from 'crypto';

// ─── Constants ─────────────────────────────────────────────────────────────

const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };

const HISTOGRAM_BUCKETS_MS = [
    50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, Infinity,
];

// ─── 1. Structured Logger ──────────────────────────────────────────────────

/**
 * Create a structured JSON-lines logger that writes to stderr.
 *
 * @param {string} [defaultLevel='info']  Initial log-level threshold.
 * @returns {{
 *   debug:  (msg: string, ctx?: object) => void,
 *   info:   (msg: string, ctx?: object) => void,
 *   warn:   (msg: string, ctx?: object) => void,
 *   error:  (msg: string, ctx?: object) => void,
 *   setLevel: (level: string) => void,
 *   child:   (defaultCtx: object) => Logger,
 * }}
 */
export function createLogger(defaultLevel = 'info') {
    let currentLevel = LOG_LEVELS[defaultLevel];
    if (currentLevel === undefined) currentLevel = 1; // default to info

    /**
     * Core log function.
     * @param {string} level   One of 'debug', 'info', 'warn', 'error'.
     * @param {string} msg     Log message.
     * @param {object} [ctx]   Additional structured context fields.
     */
    function log(level, msg, ctx = {}) {
        const sev = LOG_LEVELS[level];
        if (sev === undefined || sev < currentLevel) return;

        const entry = {
            ts: new Date().toISOString(),
            level,
            msg,
            ...ctx,
        };

        // Write to stderr as a single JSON line
        process.stderr.write(JSON.stringify(entry) + '\n');
    }

    return {
        debug(msg, ctx) { log('debug', msg, ctx); },
        info(msg, ctx)  { log('info',  msg, ctx); },
        warn(msg, ctx)  { log('warn',  msg, ctx); },
        error(msg, ctx) { log('error', msg, ctx); },

        /**
         * Dynamically change the log level at runtime.
         * @param {string} level  One of 'debug', 'info', 'warn', 'error'.
         */
        setLevel(level) {
            const sev = LOG_LEVELS[level];
            if (sev !== undefined) currentLevel = sev;
        },

        /**
         * Create a child logger that includes `defaultCtx` in every log call.
         * Useful for pre-populating correlationId, service, component, etc.
         * @param {object} defaultCtx  Fields merged into every log entry.
         * @returns {Logger}
         */
        child(defaultCtx) {
            const parent = this;
            const childFn = (level, msg, ctx) =>
                log(level, msg, { ...defaultCtx, ...ctx });
            return {
                debug(msg, ctx) { childFn('debug', msg, ctx); },
                info(msg, ctx)  { childFn('info',  msg, ctx); },
                warn(msg, ctx)  { childFn('warn',  msg, ctx); },
                error(msg, ctx) { childFn('error', msg, ctx); },
                setLevel(l) { parent.setLevel(l); },
                child(extraCtx) { return parent.child({ ...defaultCtx, ...extraCtx }); },
            };
        },
    };
}

// ─── 2. Correlation ID Generator ───────────────────────────────────────────

/**
 * Generate a short, sortable, unique correlation ID.
 *
 * Format: `<8-hex-chars>-<4-hex-chars>`  (e.g. `"a1b2c3d4-e5f6"`)
 * The leading bytes are from crypto.randomBytes, giving ~2^48 unique values.
 *
 * @returns {string}
 */
export function correlationId() {
    const buf = randomBytes(6);
    const a = buf.readUInt32BE(0);          // first 4 bytes → 8 hex chars
    const b = buf.readUInt16BE(4);          // next  2 bytes → 4 hex chars
    return a.toString(16).padStart(8, '0') + '-' + b.toString(16).padStart(4, '0');
}

// ─── 3. Latency Histogram ──────────────────────────────────────────────────

/**
 * A fixed-bucket latency histogram for recording request durations.
 *
 * Buckets (ms): 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, +Inf
 *
 * Usage:
 *   const h = new LatencyHistogram();
 *   h.record(47);
 *   h.record(3200);
 *   h.getSnapshot();  // { count, sum, buckets, p50, p95, p99 }
 */
export class LatencyHistogram {
    constructor() {
        /** @type {number[]} Upper bounds for each bucket (monotonically increasing). */
        this._bounds = [...HISTOGRAM_BUCKETS_MS];
        /** @type {number[]} Per-bucket counters, parallel to _bounds. */
        this._buckets = new Array(this._bounds.length).fill(0);
        /** @type {number} Total observations. */
        this._count = 0;
        /** @type {number} Sum of all observed values (ms). */
        this._sum = 0;
        /** @type {number[]} Raw samples for percentile computation (kept small). */
        this._samples = [];
    }

    /**
     * Record a single latency observation.
     * @param {number} ms  Latency in milliseconds.
     */
    record(ms) {
        this._count++;
        this._sum += ms;

        // Find the first bucket whose bound is >= ms
        for (let i = 0; i < this._bounds.length; i++) {
            if (ms <= this._bounds[i]) {
                this._buckets[i]++;
                break;
            }
        }

        // Keep a bounded sorted sample for percentile calculation.
        // We cap at 5000 samples to avoid unbounded memory growth.
        if (this._samples.length < 5000) {
            // Insert-sort into the sample array (it stays small enough).
            const idx = lowerBound(this._samples, ms);
            this._samples.splice(idx, 0, ms);
        }
    }

    /**
     * Return a snapshot of the histogram.
     *
     * @returns {{
     *   count:   number,
     *   sum:     number,
     *   avg:     number,
     *   buckets: Record<string, number>,
     *   p50:     number,
     *   p95:     number,
     *   p99:     number,
     * }}
     */
    getSnapshot() {
        const buckets = {};
        for (let i = 0; i < this._bounds.length; i++) {
            const label = this._bounds[i] === Infinity
                ? '+Inf'
                : String(this._bounds[i]);
            buckets[label] = this._buckets[i];
        }

        return {
            count: this._count,
            sum:   this._sum,
            avg:   this._count > 0 ? this._sum / this._count : 0,
            buckets,
            p50:   percentile(this._samples, 0.50),
            p95:   percentile(this._samples, 0.95),
            p99:   percentile(this._samples, 0.99),
        };
    }

    /**
     * Reset all counters and samples (useful for periodic reporting).
     */
    reset() {
        this._buckets.fill(0);
        this._count = 0;
        this._sum = 0;
        this._samples = [];
    }
}

// ─── 4. Request Metrics Collector ──────────────────────────────────────────

/**
 * Create a metrics collector that tracks request counts, errors, and latency
 * across backends and models.
 *
 * @returns {{
 *   recordRequest: (backend: string, model: string, latencyMs: number, status: number, errorType?: string) => void,
 *   getSnapshot:   () => object,
 *   getPrometheus: () => string,
 * }}
 */
export function createMetrics() {
    let totalRequests = 0;
    let activeRequests = 0;
    let totalErrors = 0;
    let totalLatency = 0; // sum for averaging

    /** @type {Record<string, { count: number, errors: number, latencySum: number }>} */
    const byBackend = {};

    /** @type {Record<string, { count: number, errors: number, latencySum: number }>} */
    const byModel = {};

    /** @type {LatencyHistogram} */
    const histogram = new LatencyHistogram();

    /**
     * Record the outcome of a single proxied request.
     *
     * @param {string}  backend     Backend name (e.g. 'deepseek', 'gemini', 'anthropic').
     * @param {string}  model       Model name that was called (the remapped name, if any).
     * @param {number}  latencyMs   Total request latency in milliseconds.
     * @param {number}  status      HTTP status code returned by the upstream.
     * @param {string}  [errorType] Optional error category (e.g. 'timeout', 'connection', 'upstream_5xx').
     */
    function recordRequest(backend, model, latencyMs, status, errorType) {
        totalRequests++;
        activeRequests++;
        totalLatency += latencyMs;

        const isError = status >= 400 || !!errorType;
        if (isError) totalErrors++;

        histogram.record(latencyMs);

        // Per-backend breakdown
        if (!byBackend[backend]) {
            byBackend[backend] = { count: 0, errors: 0, latencySum: 0 };
        }
        byBackend[backend].count++;
        byBackend[backend].latencySum += latencyMs;
        if (isError) byBackend[backend].errors++;

        // Per-model breakdown
        const modelKey = model || 'unknown';
        if (!byModel[modelKey]) {
            byModel[modelKey] = { count: 0, errors: 0, latencySum: 0 };
        }
        byModel[modelKey].count++;
        byModel[modelKey].latencySum += latencyMs;
        if (isError) byModel[modelKey].errors++;
    }

    /**
     * Decrement the active-request counter when a request completes or fails.
     */
    function finishRequest() {
        if (activeRequests > 0) activeRequests--;
    }

    /**
     * Return a plain-object snapshot of all current metrics.
     *
     * @returns {{
     *   totalRequests:  number,
     *   activeRequests: number,
     *   totalErrors:    number,
     *   totalLatency:   number,
     *   avgLatency:     number,
     *   errorRate:      number,
     *   histogram:      ReturnType<LatencyHistogram['getSnapshot']>,
     *   byBackend:      Record<string, { count: number, errors: number, avgLatency: number }>,
     *   byModel:        Record<string, { count: number, errors: number, avgLatency: number }>,
     * }}
     */
    function getSnapshot() {
        const avgLatency = totalRequests > 0 ? totalLatency / totalRequests : 0;
        const errorRate  = totalRequests > 0 ? totalErrors / totalRequests : 0;

        const backends = {};
        for (const [name, stats] of Object.entries(byBackend)) {
            backends[name] = {
                count:      stats.count,
                errors:     stats.errors,
                avgLatency: stats.count > 0 ? stats.latencySum / stats.count : 0,
            };
        }

        const models = {};
        for (const [name, stats] of Object.entries(byModel)) {
            models[name] = {
                count:      stats.count,
                errors:     stats.errors,
                avgLatency: stats.count > 0 ? stats.latencySum / stats.count : 0,
            };
        }

        return {
            totalRequests,
            activeRequests,
            totalErrors,
            totalLatency,
            avgLatency: +avgLatency.toFixed(2),
            errorRate:  +errorRate.toFixed(6),
            histogram:  histogram.getSnapshot(),
            byBackend:  backends,
            byModel:    models,
        };
    }

    /**
     * Render metrics in Prometheus exposition format (text/plain; version=0.0.4).
     *
     * @returns {string}
     */
    function getPrometheus() {
        const snap = getSnapshot();
        const nl = '\n';
        let out = '';

        // # HELP and # TYPE comments
        out += `# HELP proxy_requests_total Total proxied requests${nl}`;
        out += `# TYPE proxy_requests_total counter${nl}`;
        out += `proxy_requests_total ${snap.totalRequests}${nl}${nl}`;

        out += `# HELP proxy_requests_active Currently in-flight requests${nl}`;
        out += `# TYPE proxy_requests_active gauge${nl}`;
        out += `proxy_requests_active ${snap.activeRequests}${nl}${nl}`;

        out += `# HELP proxy_errors_total Total request errors${nl}`;
        out += `# TYPE proxy_errors_total counter${nl}`;
        out += `proxy_errors_total ${snap.totalErrors}${nl}${nl}`;

        out += `# HELP proxy_latency_seconds Latency in seconds${nl}`;
        out += `# TYPE proxy_latency_seconds summary${nl}`;
        out += `proxy_latency_seconds{quantile="0.5"} ${snap.histogram.p50 / 1000}${nl}`;
        out += `proxy_latency_seconds{quantile="0.95"} ${snap.histogram.p95 / 1000}${nl}`;
        out += `proxy_latency_seconds{quantile="0.99"} ${snap.histogram.p99 / 1000}${nl}`;
        out += `proxy_latency_seconds_sum ${snap.totalLatency / 1000}${nl}`;
        out += `proxy_latency_seconds_count ${snap.totalRequests}${nl}${nl}`;

        // Per-backend breakdown
        out += `# HELP proxy_backend_requests_total Requests per backend${nl}`;
        out += `# TYPE proxy_backend_requests_total counter${nl}`;
        for (const [backend, stats] of Object.entries(snap.byBackend)) {
            out += `proxy_backend_requests_total{backend="${backend}"} ${stats.count}${nl}`;
        }
        out += nl;

        out += `# HELP proxy_backend_errors_total Errors per backend${nl}`;
        out += `# TYPE proxy_backend_errors_total counter${nl}`;
        for (const [backend, stats] of Object.entries(snap.byBackend)) {
            out += `proxy_backend_errors_total{backend="${backend}"} ${stats.errors}${nl}`;
        }
        out += nl;

        out += `# HELP proxy_backend_latency_seconds Avg latency per backend${nl}`;
        out += `# TYPE proxy_backend_latency_seconds gauge${nl}`;
        for (const [backend, stats] of Object.entries(snap.byBackend)) {
            out += `proxy_backend_latency_seconds{backend="${backend}"} ${(stats.avgLatency / 1000).toFixed(4)}${nl}`;
        }
        out += nl;

        // Per-model breakdown
        out += `# HELP proxy_model_requests_total Requests per model${nl}`;
        out += `# TYPE proxy_model_requests_total counter${nl}`;
        for (const [model, stats] of Object.entries(snap.byModel)) {
            out += `proxy_model_requests_total{model="${model}"} ${stats.count}${nl}`;
        }

        return out;
    }

    return {
        recordRequest,
        finishRequest,
        getSnapshot,
        getPrometheus,
    };
}

// ─── 5. Health Check Reporter ──────────────────────────────────────────────

/**
 * Build a health-report object from metrics and backend status information.
 *
 * @param {object}   metricsSnapshot  Result of `metrics.getSnapshot()`.
 * @param {number}   uptimeSeconds    Server uptime in seconds.
 * @param {Array<{name: string, status: 'ok'|'degraded'|'down', errors?: number, avgLatency?: number}>} backends
 *   List of backends with their current health status. `errors` and `avgLatency`
 *   are optional; if omitted the function will use the metrics breakdown.
 * @returns {{
 *   status:     'ok'|'degraded'|'down',
 *   uptime:     number,
 *   backends:   Record<string, { status: string, errors: number, avgLatency: number }>,
 *   aggregate:  { totalRequests: number, errorRate: number, avgLatency: number },
 * }}
 */
export function getHealthReport(metricsSnapshot, uptimeSeconds, backends) {
    const { totalRequests, errorRate, avgLatency, byBackend } = metricsSnapshot;

    const backendReport = {};
    let worstStatus = 'ok';

    for (const b of backends) {
        const meta = byBackend[b.name] || { count: 0, errors: 0, avgLatency: 0 };
        const errors = b.errors ?? meta.errors;
        const latency = b.avgLatency ?? meta.avgLatency;

        backendReport[b.name] = {
            status:     b.status,
            errors,
            avgLatency: +latency.toFixed(2),
        };

        if (b.status === 'down') worstStatus = 'down';
        else if (b.status === 'degraded' && worstStatus !== 'down') worstStatus = 'degraded';
    }

    return {
        status: worstStatus,
        uptime: uptimeSeconds,
        backends: backendReport,
        aggregate: {
            totalRequests,
            errorRate: +errorRate.toFixed(6),
            avgLatency: +avgLatency.toFixed(2),
        },
    };
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/**
 * Binary-search lower bound: return the leftmost index where `arr[i] >= val`.
 * Assumes `arr` is sorted ascending.
 * @param {number[]} arr
 * @param {number}   val
 * @returns {number}
 */
function lowerBound(arr, val) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/**
 * Compute the p-th percentile from a sorted sample array using linear
 * interpolation (same method as Prometheus / HDR Histogram).
 * @param {number[]} sorted  Sorted ascending observations.
 * @param {number}   p       Quantile in [0, 1].
 * @returns {number}
 */
function percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0];

    const rank = p * (sorted.length - 1);
    const lo = Math.floor(rank);
    const hi = Math.ceil(rank);
    if (lo === hi) return sorted[lo];

    const frac = rank - lo;
    return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}
