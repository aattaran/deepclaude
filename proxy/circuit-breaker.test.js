/**
 * Circuit Breaker & Resilience — unit tests (node:test runner)
 * Run: node --test proxy/circuit-breaker.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    CircuitOpenError,
    CircuitBreaker,
    isRetryableError,
    withRetry,
    TokenBucket,
    BackendHealthTracker,
    selectFallback,
    CIRCUIT_STATES,
} from './circuit-breaker.js';

describe('CircuitOpenError', () => {
    it('is an instance of Error', () => {
        const err = new CircuitOpenError('deepseek', { failures: 5, openedAt: Date.now() });
        assert.ok(err instanceof Error);
    });

    it('has name CircuitOpenError', () => {
        const err = new CircuitOpenError('gemini', { failures: 3, openedAt: Date.now() });
        assert.equal(err.name, 'CircuitOpenError');
    });

    it('stores backend and state', () => {
        const state = { failures: 7, openedAt: Date.now() };
        const err = new CircuitOpenError('anthropic', state);
        assert.equal(err.backend, 'anthropic');
        assert.equal(err.state, state);
    });

    it('includes meaningful message', () => {
        const now = Date.now();
        const err = new CircuitOpenError('deepseek', { failures: 4, openedAt: now });
        assert.ok(err.message.includes('deepseek'));
        assert.ok(err.message.includes('4'));
    });
});

describe('CircuitBreaker', () => {
    it('starts in CLOSED state', () => {
        const cb = new CircuitBreaker();
        const state = cb.getState();
        assert.equal(state.state, CIRCUIT_STATES.CLOSED);
        assert.equal(state.failures, 0);
    });

    it('executes successfully in CLOSED state', async () => {
        const cb = new CircuitBreaker();
        const result = await cb.execute(async () => 'ok');
        assert.equal(result, 'ok');
    });

    it('increments failure count on error', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 5 });
        for (let i = 0; i < 3; i++) {
            try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        }
        const state = cb.getState();
        assert.equal(state.failures, 3);
        assert.equal(state.state, CIRCUIT_STATES.CLOSED);
    });

    it('opens circuit after threshold failures', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 3 });
        for (let i = 0; i < 3; i++) {
            try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        }
        const state = cb.getState();
        assert.equal(state.state, CIRCUIT_STATES.OPEN);
        assert.equal(state.failures, 3);
    });

    it('throws CircuitOpenError when OPEN', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 1 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}

        try {
            await cb.execute(async () => 'should not run', { backend: 'test' });
            assert.fail('should have thrown');
        } catch (err) {
            assert.ok(err instanceof CircuitOpenError);
            assert.equal(err.backend, 'test');
        }
    });

    it('resets failures on success', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 5 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        assert.equal(cb.getState().failures, 2);

        await cb.execute(async () => 'ok');
        assert.equal(cb.getState().failures, 0);
    });

    it('transitions to HALF_OPEN after resetTimeout', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeout: 30 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        assert.equal(cb.getState().state, CIRCUIT_STATES.OPEN);

        // Wait for reset timeout
        await new Promise(r => setTimeout(r, 40));
        const state = cb.getState();
        assert.equal(state.state, CIRCUIT_STATES.HALF_OPEN);
    });

    it('closes circuit on success in HALF_OPEN', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeout: 20 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        await new Promise(r => setTimeout(r, 30)); // wait for half-open

        const result = await cb.execute(async () => 'recovered');
        assert.equal(result, 'recovered');
        assert.equal(cb.getState().state, CIRCUIT_STATES.CLOSED);
    });

    it('re-opens on failure in HALF_OPEN', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeout: 20 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        await new Promise(r => setTimeout(r, 30));

        try {
            await cb.execute(async () => { throw new Error('still broken'); });
        } catch (_) {}

        assert.equal(cb.getState().state, CIRCUIT_STATES.OPEN);
    });

    it('limits probes in HALF_OPEN to halfOpenMaxRequests', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 1, resetTimeout: 20, halfOpenMaxRequests: 1 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        await new Promise(r => setTimeout(r, 30));

        // Start first probe that blocks (increments halfOpenRequests but doesn't resolve yet)
        const probe1 = cb.execute(async () => {
            await new Promise(r => setTimeout(r, 80));
            return 'ok';
        });

        // Give the first probe time to consume the only half-open slot
        await new Promise(r => setTimeout(r, 5));

        // Second probe should fail because halfOpenMaxRequests=1 limit already consumed
        try {
            await cb.execute(async () => 'should not run', { backend: 'test' });
            assert.fail('should have thrown');
        } catch (err) {
            assert.ok(err instanceof CircuitOpenError);
        }

        // Let first probe finish and close the circuit
        await probe1;
        assert.equal(cb.getState().state, CIRCUIT_STATES.CLOSED);
    });

    it('forceClose resets circuit to CLOSED', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 1 });
        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        assert.equal(cb.getState().state, CIRCUIT_STATES.OPEN);

        cb.forceClose();
        const state = cb.getState();
        assert.equal(state.state, CIRCUIT_STATES.CLOSED);
        assert.equal(state.failures, 0);
        assert.equal(state.openedAt, null);
    });

    it('forceOpen opens circuit immediately', () => {
        const cb = new CircuitBreaker();
        cb.forceOpen();
        assert.equal(cb.getState().state, CIRCUIT_STATES.OPEN);
    });

    it('custom failureThreshold works', async () => {
        const cb = new CircuitBreaker({ failureThreshold: 7 });
        for (let i = 0; i < 6; i++) {
            try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        }
        assert.equal(cb.getState().state, CIRCUIT_STATES.CLOSED);

        try { await cb.execute(async () => { throw new Error('fail'); }); } catch (_) {}
        assert.equal(cb.getState().state, CIRCUIT_STATES.OPEN);
    });
});

describe('isRetryableError', () => {
    it('returns false for null/undefined', () => {
        assert.equal(isRetryableError(null), false);
        assert.equal(isRetryableError(undefined), false);
    });

    it('returns true for network errors (no statusCode)', () => {
        assert.ok(isRetryableError(new Error('ECONNREFUSED')));
        assert.ok(isRetryableError(new Error('socket hang up')));
        assert.ok(isRetryableError(new Error('ETIMEDOUT')));
    });

    it('returns false for error with empty message', () => {
        assert.equal(isRetryableError(new Error('')), false);
    });

    it('returns true for 5xx server errors', () => {
        const err = new Error('Server Error');
        err.statusCode = 500;
        assert.ok(isRetryableError(err));

        err.statusCode = 503;
        assert.ok(isRetryableError(err));

        err.statusCode = 502;
        assert.ok(isRetryableError(err));
    });

    it('returns true for 429 rate limit', () => {
        const err = new Error('Rate limited');
        err.statusCode = 429;
        assert.ok(isRetryableError(err));
    });

    it('returns false for 4xx client errors', () => {
        const err = new Error('Bad Request');
        err.statusCode = 400;
        assert.equal(isRetryableError(err), false);

        err.statusCode = 403;
        assert.equal(isRetryableError(err), false);

        err.statusCode = 404;
        assert.equal(isRetryableError(err), false);
    });
});

describe('withRetry', () => {
    it('succeeds on first attempt', async () => {
        const { result, attempts } = await withRetry(async () => 'ok');
        assert.equal(result, 'ok');
        assert.equal(attempts, 1);
    });

    it('retries on transient error and succeeds', async () => {
        let calls = 0;
        const { result, attempts } = await withRetry(
            async () => {
                calls++;
                if (calls < 2) throw Object.assign(new Error('fail'), { statusCode: 503 });
                return 'recovered';
            },
            { baseDelay: 10 },
        );
        assert.equal(result, 'recovered');
        assert.equal(attempts, 2);
        assert.equal(calls, 2);
    });

    it('gives up after maxRetries and throws last error', async () => {
        let calls = 0;
        try {
            await withRetry(
                async () => {
                    calls++;
                    throw Object.assign(new Error('persistent'), { statusCode: 503 });
                },
                { maxRetries: 2, baseDelay: 10 },
            );
            assert.fail('should have thrown');
        } catch (err) {
            assert.equal(err.message, 'persistent');
            assert.equal(calls, 3); // original + 2 retries
        }
    });

    it('does not retry on 4xx errors (except 429)', async () => {
        let calls = 0;
        try {
            await withRetry(
                async () => {
                    calls++;
                    throw Object.assign(new Error('bad request'), { statusCode: 400 });
                },
                { baseDelay: 10 },
            );
            assert.fail('should have thrown');
        } catch (err) {
            assert.equal(err.statusCode, 400);
            assert.equal(calls, 1); // no retries
        }
    });

    it('returns totalTime as a positive number', async () => {
        const { totalTime } = await withRetry(async () => 'fast', { baseDelay: 10 });
        assert.ok(totalTime >= 0);
    });

    it('uses custom isRetryable predicate', async () => {
        let calls = 0;
        try {
            await withRetry(
                async () => {
                    calls++;
                    throw new Error('custom error');
                },
                {
                    maxRetries: 2,
                    baseDelay: 10,
                    isRetryable: () => false, // never retry
                },
            );
            assert.fail('should have thrown');
        } catch (_) {
            assert.equal(calls, 1);
        }
    });

    it('respects exponential backoff', async () => {
        const t0 = Date.now();
        let calls = 0;
        try {
            await withRetry(
                async () => {
                    calls++;
                    throw Object.assign(new Error('retry'), { statusCode: 503 });
                },
                { maxRetries: 2, baseDelay: 50, backoffMultiplier: 4 },
            );
        } catch (_) {}
        const elapsed = Date.now() - t0;
        // First retry: ~50ms (±25%), Second: ~200ms (±25%)
        assert.ok(elapsed >= 30, `expected >= 30ms, got ${elapsed}ms`);
        assert.equal(calls, 3);
    });
});

describe('TokenBucket', () => {
    it('starts full by default', () => {
        const tb = new TokenBucket({ capacity: 10, fillRate: 5 });
        const state = tb.getState();
        assert.equal(state.tokens, 10);
        assert.equal(state.capacity, 10);
    });

    it('tryConsume returns true when tokens available', () => {
        const tb = new TokenBucket({ capacity: 10, fillRate: 5 });
        assert.ok(tb.tryConsume());
        const state = tb.getState();
        assert.equal(state.tokens, 9);
    });

    it('tryConsume returns false when empty', () => {
        const tb = new TokenBucket({ capacity: 1, fillRate: 0 });
        assert.ok(tb.tryConsume());
        assert.equal(tb.tryConsume(), false);
    });

    it('consumes multiple tokens at once', () => {
        const tb = new TokenBucket({ capacity: 10, fillRate: 5 });
        assert.ok(tb.tryConsume(3));
        assert.equal(tb.getState().tokens, 7);
    });

    it('refills tokens over time', async () => {
        const tb = new TokenBucket({ capacity: 10, fillRate: 100 }); // 100 tokens/sec
        // Drain it
        while (tb.tryConsume()) {}
        assert.equal(tb.getState().tokens, 0);

        // Wait for refill (100 tokens/sec => ~0.05s for 5 tokens)
        await new Promise(r => setTimeout(r, 60));
        const state = tb.getState();
        assert.ok(state.tokens >= 4, `expected >= 4 tokens, got ${state.tokens}`);
    });

    it('never exceeds capacity', async () => {
        const tb = new TokenBucket({ capacity: 3, fillRate: 100 });
        await new Promise(r => setTimeout(r, 100));
        const state = tb.getState();
        assert.ok(state.tokens <= 3);
    });

    it('waitForToken resolves immediately when tokens available', async () => {
        const tb = new TokenBucket({ capacity: 10, fillRate: 5 });
        const result = await tb.waitForToken(1000);
        assert.ok(result);
    });

    it('waitForToken returns false on timeout', async () => {
        const tb = new TokenBucket({ capacity: 0, fillRate: 0, initialTokens: 0 });
        const t0 = Date.now();
        const result = await tb.waitForToken(50);
        assert.equal(result, false);
        assert.ok(Date.now() - t0 >= 50);
    });

    it('custom initialTokens works', () => {
        const tb = new TokenBucket({ capacity: 10, fillRate: 5, initialTokens: 3 });
        assert.equal(tb.getState().tokens, 3);
    });
});

describe('BackendHealthTracker', () => {
    it('starts with no backends', () => {
        const ht = new BackendHealthTracker();
        assert.deepEqual(ht.getAllScores(), []);
        assert.deepEqual(ht.getDegradedBackends(), []);
    });

    it('returns score 100 for unknown backend', () => {
        const ht = new BackendHealthTracker();
        assert.equal(ht.getScore('unknown'), 100);
    });

    it('isHealthy returns true for score > 50', () => {
        const ht = new BackendHealthTracker();
        assert.ok(ht.isHealthy('new-backend')); // 100 by default
    });

    it('records successes and computes score', () => {
        const ht = new BackendHealthTracker({ rollingWindow: 100 });
        ht.recordSuccess('deepseek');
        ht.recordSuccess('deepseek');
        ht.recordSuccess('deepseek');
        ht.recordFailure('deepseek');
        assert.equal(ht.getScore('deepseek'), 75); // 3/4 = 75%
    });

    it('detects degraded backends (score < 80)', () => {
        const ht = new BackendHealthTracker({ rollingWindow: 10 });
        ht.recordSuccess('backend-a');
        ht.recordSuccess('backend-a');
        ht.recordFailure('backend-a'); // 66%
        ht.recordSuccess('backend-b');
        ht.recordSuccess('backend-b');
        ht.recordSuccess('backend-b');
        ht.recordSuccess('backend-b'); // 100%
        ht.recordSuccess('backend-b');

        const degraded = ht.getDegradedBackends();
        assert.ok(degraded.some(d => d.backend === 'backend-a'));
        assert.equal(degraded.some(d => d.backend === 'backend-b'), false);
    });

    it('getAllScores returns all tracked backends', () => {
        const ht = new BackendHealthTracker();
        ht.recordSuccess('a');
        ht.recordSuccess('b');
        const scores = ht.getAllScores();
        assert.equal(scores.length, 2);
        assert.ok(scores.every(s => typeof s.healthy === 'boolean'));
    });

    it('returns 100 for backend with no requests recorded', () => {
        const ht = new BackendHealthTracker();
        // record a different backend so tracker has entries
        ht.recordSuccess('other');
        assert.equal(ht.getScore('untracked'), 100);
    });
});

describe('selectFallback', () => {
    it('returns current backend when healthy', () => {
        const ht = new BackendHealthTracker();
        ht.recordSuccess('deepseek');
        ht.recordSuccess('deepseek');
        ht.recordSuccess('deepseek');
        const result = selectFallback('deepseek', ht);
        assert.equal(result.backend, 'deepseek');
        assert.equal(result.reason, 'healthy');
    });

    it('selects healthier alternative when current is degraded', () => {
        const ht = new BackendHealthTracker();
        // deepseek: 50% (degraded)
        ht.recordSuccess('deepseek');
        ht.recordFailure('deepseek');
        // gemini: 100% (healthy)
        ht.recordSuccess('gemini');
        ht.recordSuccess('gemini');
        ht.recordSuccess('gemini');
        const result = selectFallback('deepseek', ht, ['deepseek', 'gemini']);
        assert.equal(result.backend, 'gemini');
        assert.ok(result.reason.includes('fallback'));
    });

    it('keeps current when no better alternative', () => {
        const ht = new BackendHealthTracker();
        // deepseek: 50%
        ht.recordSuccess('deepseek');
        ht.recordFailure('deepseek');
        // gemini: 0% (worse)
        ht.recordFailure('gemini');
        ht.recordFailure('gemini');
        const result = selectFallback('deepseek', ht, ['deepseek', 'gemini']);
        assert.equal(result.backend, 'deepseek');
    });

    it('handles unknown backend with default healthy score', () => {
        const ht = new BackendHealthTracker();
        const result = selectFallback('not-yet-seen', ht, ['not-yet-seen', 'deepseek']);
        assert.equal(result.reason, 'healthy');
    });

    it('falls back when current is critically low even if alternative is not great', () => {
        const ht = new BackendHealthTracker();
        // deepseek: 0%
        ht.recordFailure('deepseek');
        ht.recordFailure('deepseek');
        // gemini: 30% (also bad, but not 0)
        ht.recordFailure('gemini');
        ht.recordFailure('gemini');
        ht.recordSuccess('gemini');
        const result = selectFallback('deepseek', ht, ['deepseek', 'gemini']);
        assert.equal(result.backend, 'gemini');
    });

    it('handles single backend — no alternatives', () => {
        const ht = new BackendHealthTracker();
        ht.recordFailure('deepseek');
        ht.recordFailure('deepseek');
        const result = selectFallback('deepseek', ht, ['deepseek']);
        assert.equal(result.backend, 'deepseek');
        assert.ok(result.reason.includes('no fallback'));
    });
});
