/**
 * Circuit Breaker & Resilience Module
 *
 * Production-grade fault tolerance patterns for the model proxy:
 *   - CircuitBreaker  — fail-fast when a backend is degraded
 *   - withRetry       — exponential backoff with jitter
 *   - TokenBucket     — rate limiting via token bucket algorithm
 *   - BackendHealthTracker — rolling-window health scoring per backend
 *   - selectFallback  — graceful degradation to healthiest alternative
 *
 * ES module. All exports at the bottom.
 */

// ---------------------------------------------------------------------------
// Custom error — thrown when the circuit is OPEN and rejects a call
// ---------------------------------------------------------------------------
export class CircuitOpenError extends Error {
    constructor(backend, state) {
        super(`Circuit breaker is OPEN for "${backend}" (failures: ${state.failures}, opened: ${new Date(state.openedAt).toISOString()})`);
        this.name = 'CircuitOpenError';
        this.backend = backend;
        this.state = state;
    }
}

// ---------------------------------------------------------------------------
// 1. CircuitBreaker
// ---------------------------------------------------------------------------
const CIRCUIT_STATES = Object.freeze({
    CLOSED: 'CLOSED',
    OPEN: 'OPEN',
    HALF_OPEN: 'HALF_OPEN',
});

export class CircuitBreaker {
    /**
     * @param {object} options
     * @param {number}  [options.failureThreshold=5]    — consecutive failures before opening
     * @param {number}  [options.resetTimeout=30000]     — ms before transitioning to HALF_OPEN
     * @param {number}  [options.halfOpenMaxRequests=3]  — max requests allowed in HALF_OPEN
     */
    constructor(options = {}) {
        this._failureThreshold = options.failureThreshold ?? 5;
        this._resetTimeout = options.resetTimeout ?? 30000;
        this._halfOpenMaxRequests = options.halfOpenMaxRequests ?? 3;

        this._state = CIRCUIT_STATES.CLOSED;
        this._failures = 0;
        this._lastFailure = null;
        this._openedAt = null;
        this._halfOpenSuccesses = 0;
        this._halfOpenRequests = 0;
    }

    /**
     * Execute a function through the circuit breaker.
     * - CLOSED:   call passes through; failure increments counter
     * - OPEN:     throws CircuitOpenError immediately (fail-fast)
     * - HALF_OPEN: allows limited probes; success closes, failure re-opens
     *
     * @template T
     * @param {() => Promise<T>} fn  — the async function to protect
     * @param {object}          [options]
     * @param {string}          [options.backend]  — backend name for error context
     * @returns {Promise<T>}
     */
    async execute(fn, options = {}) {
        this._maybeTransitionToHalfOpen();

        if (this._state === CIRCUIT_STATES.OPEN) {
            throw new CircuitOpenError(options.backend ?? 'unknown', this.getState());
        }

        if (this._state === CIRCUIT_STATES.HALF_OPEN) {
            if (this._halfOpenRequests >= this._halfOpenMaxRequests) {
                throw new CircuitOpenError(options.backend ?? 'unknown', this.getState());
            }
            this._halfOpenRequests++;
        }

        try {
            const result = await fn();
            this._onSuccess();
            return result;
        } catch (err) {
            this._onFailure(err);
            throw err;
        }
    }

    /** Record a success — resets failure count, closes circuit if half-open */
    _onSuccess() {
        this._failures = 0;
        this._lastFailure = null;
        if (this._state === CIRCUIT_STATES.HALF_OPEN) {
            this._halfOpenSuccesses++;
            if (this._halfOpenSuccesses >= 1) {
                // One success in half-open is enough to close
                this._close();
            }
        }
    }

    /** Record a failure */
    _onFailure(err) {
        this._failures++;
        this._lastFailure = Date.now();

        if (this._state === CIRCUIT_STATES.CLOSED && this._failures >= this._failureThreshold) {
            this._open();
        } else if (this._state === CIRCUIT_STATES.HALF_OPEN) {
            this._open();
        }
    }

    _open() {
        this._state = CIRCUIT_STATES.OPEN;
        this._openedAt = Date.now();
        this._halfOpenRequests = 0;
        this._halfOpenSuccesses = 0;
    }

    _close() {
        this._state = CIRCUIT_STATES.CLOSED;
        this._failures = 0;
        this._lastFailure = null;
        this._openedAt = null;
        this._halfOpenRequests = 0;
        this._halfOpenSuccesses = 0;
    }

    /** Check if resetTimeout has elapsed and transition to HALF_OPEN */
    _maybeTransitionToHalfOpen() {
        if (this._state !== CIRCUIT_STATES.OPEN) return;
        if (this._openedAt && (Date.now() - this._openedAt) >= this._resetTimeout) {
            this._state = CIRCUIT_STATES.HALF_OPEN;
            this._halfOpenRequests = 0;
            this._halfOpenSuccesses = 0;
        }
    }

    /** @returns {{ state: string, failures: number, lastFailure: number|null, openedAt: number|null }} */
    getState() {
        this._maybeTransitionToHalfOpen();
        return {
            state: this._state,
            failures: this._failures,
            lastFailure: this._lastFailure,
            openedAt: this._openedAt,
        };
    }

    /** Force the circuit closed (e.g. after manual intervention) */
    forceClose() {
        this._close();
    }

    /** Force the circuit open */
    forceOpen() {
        this._open();
    }
}

// ---------------------------------------------------------------------------
// 2. RetryPolicy — withRetry
// ---------------------------------------------------------------------------

/**
 * Parse the Retry-After header value (seconds or HTTP-date).
 * @param {string|null} headerValue
 * @returns {number} seconds to wait (0 if unparsable)
 */
function parseRetryAfter(headerValue) {
    if (!headerValue) return 0;
    const asSeconds = parseInt(headerValue, 10);
    if (!isNaN(asSeconds) && asSeconds >= 0) return asSeconds;
    // Could be an HTTP-date — fall back to a safe default
    const parsed = Date.parse(headerValue);
    if (!isNaN(parsed)) {
        return Math.max(0, Math.ceil((parsed - Date.now()) / 1000));
    }
    return 0;
}

/**
 * Generate a jittered delay. Adds ±25% uniform jitter to the base.
 * @param {number} baseDelay — delay in ms
 * @returns {number} jittered delay in ms
 */
function jitter(baseDelay) {
    const jitterRange = baseDelay * 0.25;
    return baseDelay + (Math.random() * jitterRange * 2 - jitterRange);
}

/**
 * Default function to determine whether an error is retryable.
 * Retries on network errors (no statusCode) and 5xx (server errors).
 * Does NOT retry on 4xx except 429 (rate limited).
 *
 * @param {Error & { statusCode?: number }} err
 * @returns {boolean}
 */
export function isRetryableError(err) {
    if (!err) return false;
    // Network / connection errors have no status code
    if (err.statusCode === undefined && err.message) return true;
    // 5xx — server errors
    if (err.statusCode >= 500 && err.statusCode < 600) return true;
    // 429 — rate limited (retry with Retry-After)
    if (err.statusCode === 429) return true;
    return false;
}

/**
 * Execute an async function with exponential backoff and jitter.
 * Does NOT retry on 4xx errors (except 429).
 *
 * @template T
 * @param {() => Promise<T>} fn  — the async function to retry
 * @param {object}          [options]
 * @param {number}          [options.maxRetries=3]        — max number of retry attempts
 * @param {number}          [options.baseDelay=1000]       — initial delay in ms
 * @param {number}          [options.maxDelay=30000]       — max delay in ms
 * @param {number}          [options.backoffMultiplier=2]  — exponential factor
 * @param {(err: Error) => boolean} [options.isRetryable] — custom retry predicate
 * @returns {Promise<{ result: T, attempts: number, totalTime: number }>}
 */
export async function withRetry(fn, options = {}) {
    const maxRetries = options.maxRetries ?? 3;
    const baseDelay = options.baseDelay ?? 1000;
    const maxDelay = options.maxDelay ?? 30000;
    const multiplier = options.backoffMultiplier ?? 2;
    const shouldRetry = options.isRetryable ?? isRetryableError;

    const t0 = Date.now();
    let lastError = null;

    for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
        try {
            const result = await fn();
            return {
                result,
                attempts: attempt,
                totalTime: Date.now() - t0,
            };
        } catch (err) {
            lastError = err;

            // If this was the last attempt, stop
            if (attempt > maxRetries) break;

            // Check if the error is retryable
            if (!shouldRetry(err)) break;

            // Compute delay with exponential backoff
            let delay = baseDelay * Math.pow(multiplier, attempt - 1);
            delay = Math.min(delay, maxDelay);

            // Respect Retry-After header if present
            const retryAfter = parseRetryAfter(err.retryAfter);
            if (retryAfter > 0) {
                delay = Math.max(delay, retryAfter * 1000);
            }

            // Apply jitter
            const sleepMs = Math.round(jitter(delay));

            await new Promise(resolve => setTimeout(resolve, sleepMs));
        }
    }

    // All attempts exhausted
    throw lastError;
}

// ---------------------------------------------------------------------------
// 3. RateLimiter — TokenBucket
// ---------------------------------------------------------------------------

export class TokenBucket {
    /**
     * @param {object} options
     * @param {number} [options.capacity=60]         — max tokens the bucket can hold
     * @param {number} [options.fillRate=10]          — tokens added per second
     * @param {number} [options.initialTokens]        — defaults to full capacity
     */
    constructor(options = {}) {
        this._capacity = options.capacity ?? 60;
        this._fillRate = options.fillRate ?? 10; // per second
        this._tokens = options.initialTokens ?? this._capacity;
        this._lastRefill = Date.now();
    }

    /** Refill tokens based on elapsed time since last operation */
    _refill() {
        const now = Date.now();
        const elapsed = (now - this._lastRefill) / 1000; // seconds
        if (elapsed <= 0) return;

        const added = elapsed * this._fillRate;
        this._tokens = Math.min(this._capacity, this._tokens + added);
        this._lastRefill = now;
    }

    /**
     * Try to consume `count` tokens.
     * @param {number} [count=1]
     * @returns {boolean} true if tokens were consumed, false if rate limited
     */
    tryConsume(count = 1) {
        this._refill();
        if (this._tokens >= count) {
            this._tokens -= count;
            return true;
        }
        return false;
    }

    /**
     * Wait until a token is available, up to `timeout` ms.
     * @param {number} timeout  — max wait time in ms (0 = no wait, Infinity = forever)
     * @returns {Promise<boolean>} true if token acquired, false on timeout
     */
    async waitForToken(timeout = 0) {
        if (this.tryConsume(1)) return true;
        if (timeout === 0) return false;

        const deadline = Date.now() + timeout;

        while (Date.now() < deadline) {
            // Wait for the next refill increment (at most a reasonable polling interval)
            const remaining = deadline - Date.now();
            if (remaining <= 0) break;

            // Poll at a reasonable interval — we want to wake up when a token
            // would be available. Worst-case: 1 token at current fillRate.
            const msPerToken = 1000 / this._fillRate;
            const sleepMs = Math.min(remaining, Math.ceil(msPerToken));

            await new Promise(resolve => setTimeout(resolve, sleepMs));

            if (this.tryConsume(1)) return true;
        }

        return false;
    }

    /**
     * @returns {{ tokens: number, capacity: number, fillRate: number }}
     */
    getState() {
        this._refill();
        return {
            tokens: this._tokens,
            capacity: this._capacity,
            fillRate: this._fillRate,
        };
    }
}

// ---------------------------------------------------------------------------
// 4. BackendHealthTracker
// ---------------------------------------------------------------------------

const DEFAULT_ROLLING_WINDOW = 100;

export class BackendHealthTracker {
    /**
     * @param {object}  [options]
     * @param {number}  [options.rollingWindow=100]  — number of recent requests to track
     */
    constructor(options = {}) {
        this._windowSize = options.rollingWindow ?? DEFAULT_ROLLING_WINDOW;
        /** @type {Map<string, { successes: number[], failures: number[] }>} */
        this._backends = new Map();
    }

    /** Ensure a backend entry exists */
    _ensure(backend) {
        if (!this._backends.has(backend)) {
            this._backends.set(backend, { successes: [], failures: [] });
        }
        return this._backends.get(backend);
    }

    /** Prune entries outside the rolling window */
    _prune(entries) {
        const threshold = Date.now();
        // We use timestamps; keep only the last windowSize entries
        while (entries.successes.length > this._windowSize) {
            entries.successes.shift();
        }
        while (entries.failures.length > this._windowSize) {
            entries.failures.shift();
        }
    }

    /**
     * Record a successful request for a backend.
     * @param {string} backend
     */
    recordSuccess(backend) {
        const entries = this._ensure(backend);
        entries.successes.push(Date.now());
        this._prune(entries);
    }

    /**
     * Record a failed request for a backend.
     * @param {string} backend
     * @param {Error}  [error]
     */
    recordFailure(backend, error) {
        const entries = this._ensure(backend);
        entries.failures.push(Date.now());
        this._prune(entries);
    }

    /**
     * Check if a backend is healthy (error rate <= 50% in the rolling window).
     * @param {string} backend
     * @returns {boolean}
     */
    isHealthy(backend) {
        return this.getScore(backend) > 50;
    }

    /**
     * Compute a 0-100 health score for a backend.
     * Score = (successes / total) * 100, where total = successes + failures
     * Returns 100 if no requests recorded.
     * @param {string} backend
     * @returns {number}
     */
    getScore(backend) {
        const entries = this._backends.get(backend);
        if (!entries) return 100;

        const successes = entries.successes.length;
        const failures = entries.failures.length;
        const total = successes + failures;

        if (total === 0) return 100;

        return Math.round((successes / total) * 100);
    }

    /**
     * Get list of backends with health score below 80.
     * @returns {Array<{ backend: string, score: number }>}
     */
    getDegradedBackends() {
        const degraded = [];
        for (const [backend] of this._backends) {
            const score = this.getScore(backend);
            if (score < 80) {
                degraded.push({ backend, score });
            }
        }
        return degraded;
    }

    /**
     * Get all tracked backends with their scores.
     * @returns {Array<{ backend: string, score: number, healthy: boolean }>}
     */
    getAllScores() {
        const scores = [];
        for (const [backend] of this._backends) {
            const score = this.getScore(backend);
            scores.push({ backend, score, healthy: score > 50 });
        }
        return scores;
    }
}

// ---------------------------------------------------------------------------
// 5. GracefulDegradation — selectFallback
// ---------------------------------------------------------------------------

/**
 * Given a desired backend and a BackendHealthTracker, select the healthiest
 * fallback if the desired backend is unhealthy.
 *
 * @param {string}                  currentBackend  — the backend the caller wants to use
 * @param {BackendHealthTracker}    healthTracker   — health tracker instance
 * @param {string[]}                [availableBackends] — list of all candidate backends.
 *                                                        Defaults to all tracked backends.
 * @returns {{ backend: string, reason: string }}
 *   - If currentBackend is healthy: { backend: currentBackend, reason: 'healthy' }
 *   - If currentBackend is unhealthy and alternatives exist:
 *     { backend: '<healthiest>', reason: 'fallback: <current> score <N> below threshold' }
 *   - If no alternatives exist: { backend: currentBackend, reason: 'no fallback available' }
 */
export function selectFallback(currentBackend, healthTracker, availableBackends) {
    const candidates = availableBackends ?? Array.from(
        healthTracker._backends.keys(),
        // Fall back to tracking the current backend if it's not yet known
        (k) => k,
    );

    // If current backend is not even in the tracker yet, it's healthy by default
    const currentScore = healthTracker.getScore(currentBackend);
    if (currentScore >= 80 && candidates.includes(currentBackend)) {
        return { backend: currentBackend, reason: 'healthy' };
    }

    // Find the healthiest alternative that is not the current backend
    const alternatives = candidates.filter(b => b !== currentBackend);

    if (alternatives.length === 0) {
        return {
            backend: currentBackend,
            reason: `no fallback available (current score: ${currentScore})`,
        };
    }

    // Sort by score descending (highest first)
    const scored = alternatives.map(b => ({ backend: b, score: healthTracker.getScore(b) }));
    scored.sort((a, b) => b.score - a.score);
    const best = scored[0];

    if (best.score >= currentScore) {
        return {
            backend: best.backend,
            reason: `fallback: ${currentBackend} score ${currentScore} below threshold, using ${best.backend} (score ${best.score})`,
        };
    }

    // Fallback is worse than current — still return it if current is really bad
    if (currentScore < 50 && best.score >= 0) {
        return {
            backend: best.backend,
            reason: `fallback: ${currentBackend} score ${currentScore} critically low, using ${best.backend} (score ${best.score})`,
        };
    }

    // Current backend is degraded but alternatives are no better — keep using it
    return {
        backend: currentBackend,
        reason: `degraded but no better alternative (current: ${currentScore}, best alternative: ${best.backend} ${best.score})`,
    };
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
export {
    CIRCUIT_STATES,
    CircuitBreaker as default,
};
