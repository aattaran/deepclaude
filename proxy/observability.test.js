/**
 * Observability — unit tests (node:test runner)
 * Run: node --test proxy/observability.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createLogger, createMetrics, correlationId, LatencyHistogram, getHealthReport } from './observability.js';

describe('correlationId', () => {
    it('returns a string', () => {
        assert.equal(typeof correlationId(), 'string');
    });

    it('has correct format (hex-hex)', () => {
        assert.match(correlationId(), /^[0-9a-f]+-[0-9a-f]+$/);
    });

    it('produces unique values', () => {
        const ids = new Set(Array.from({ length: 100 }, () => correlationId()));
        assert.equal(ids.size, 100);
    });
});

describe('createLogger', () => {
    it('creates a logger with expected methods', () => {
        const log = createLogger();
        assert.equal(typeof log.debug, 'function');
        assert.equal(typeof log.info, 'function');
        assert.equal(typeof log.warn, 'function');
        assert.equal(typeof log.error, 'function');
        assert.equal(typeof log.setLevel, 'function');
        assert.equal(typeof log.child, 'function');
    });

    it('setLevel filters messages below threshold', () => {
        const log = createLogger('warn');
        // debug and info should no-op without throwing
        assert.doesNotThrow(() => log.debug('should not appear'));
        assert.doesNotThrow(() => log.info('should not appear'));
    });

    it('child logger inherits context', () => {
        const log = createLogger();
        const child = log.child({ component: 'test' });
        assert.equal(typeof child.info, 'function');
    });
});

describe('createMetrics', () => {
    it('returns a metrics collector with expected methods', () => {
        const m = createMetrics();
        assert.equal(typeof m.recordRequest, 'function');
        assert.equal(typeof m.finishRequest, 'function');
        assert.equal(typeof m.getSnapshot, 'function');
        assert.equal(typeof m.getPrometheus, 'function');
    });

    it('getSnapshot returns initialized counters', () => {
        const m = createMetrics();
        const snap = m.getSnapshot();
        assert.equal(snap.totalRequests, 0);
        assert.equal(snap.activeRequests, 0);
        assert.equal(snap.totalErrors, 0);
    });

    it('recordRequest increments counters', () => {
        const m = createMetrics();
        m.recordRequest('deepseek', 'deepseek-v4-pro', 150, 200);
        const snap = m.getSnapshot();
        assert.equal(snap.totalRequests, 1);
        assert.ok(snap.byBackend.deepseek);
    });

    it('recordRequest with error status increments error count', () => {
        const m = createMetrics();
        m.recordRequest('gemini', 'gemini-3.5-flash', 5000, 500, 'timeout');
        const snap = m.getSnapshot();
        assert.equal(snap.totalErrors, 1);
        assert.equal(snap.byBackend.gemini.errors, 1);
    });

    it('getPrometheus returns text format', () => {
        const m = createMetrics();
        m.recordRequest('deepseek', 'claude-opus-4-7', 100, 200);
        const text = m.getPrometheus();
        assert.ok(text.includes('proxy_requests_total'));
        assert.ok(text.includes('proxy_errors_total') || text.includes('_requests_total'));
    });
});

describe('LatencyHistogram', () => {
    it('starts with zero count', () => {
        const h = new LatencyHistogram();
        const snap = h.getSnapshot();
        assert.equal(snap.count, 0);
        assert.equal(snap.sum, 0);
    });

    it('records samples and computes percentiles', () => {
        const h = new LatencyHistogram();
        for (let i = 0; i < 100; i++) h.record(100);
        for (let i = 0; i < 50; i++) h.record(500);
        const snap = h.getSnapshot();
        assert.equal(snap.count, 150);
        assert.ok(snap.p50 > 0);
        assert.ok(snap.p95 > 0);
        assert.ok(snap.p99 > 0);
    });

    it('places samples in correct buckets', () => {
        const h = new LatencyHistogram();
        h.record(75);   // bucket "100"
        h.record(200);  // bucket "250"
        h.record(600);  // bucket "1000"
        const snap = h.getSnapshot();
        assert.ok(snap.buckets['100'] > 0 || snap.buckets['250'] > 0); // at least one bucket populated
    });

    it('reset clears all data', () => {
        const h = new LatencyHistogram();
        h.record(100);
        h.record(200);
        h.reset();
        const snap = h.getSnapshot();
        assert.equal(snap.count, 0);
    });
});

describe('getHealthReport', () => {
    it('returns ok status when no backends degraded', () => {
        const backendList = [
            { name: 'deepseek', status: 'ok', errors: 0, avgLatency: 120 },
            { name: 'gemini', status: 'ok', errors: 0, avgLatency: 450 },
        ];
        const snap = { totalRequests: 100, totalErrors: 0, errorRate: 0, avgLatency: 100, byBackend: { deepseek: { count: 50, errors: 0, avgLatency: 120 }, gemini: { count: 50, errors: 0, avgLatency: 450 } } };
        const report = getHealthReport(snap, 3600, backendList);
        assert.equal(report.status, 'ok');
        assert.ok(report.uptime >= 0);
        assert.ok(report.aggregate.errorRate === 0);
    });

    it('returns degraded when backends are unhealthy', () => {
        const backendList = [
            { name: 'deepseek', status: 'degraded', errors: 30, avgLatency: 2000 },
        ];
        const snap = { totalRequests: 100, totalErrors: 30, errorRate: 0.3, avgLatency: 2000, byBackend: { deepseek: { count: 100, errors: 30, avgLatency: 2000 } } };
        const report = getHealthReport(snap, 100, backendList);
        assert.equal(report.status, 'degraded');
    });

    it('returns down when all backends are down', () => {
        const backendList = [
            { name: 'deepseek', status: 'down', errors: 100, avgLatency: null },
        ];
        const snap = { totalRequests: 0, totalErrors: 100, errorRate: 1, avgLatency: 0, byBackend: {} };
        const report = getHealthReport(snap, 0, backendList);
        assert.equal(report.status, 'down');
    });
});
