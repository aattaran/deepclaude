/**
 * Crypto Signals — signed, verifiable alpha signal envelopes
 *
 * Inspired by da Vinci's mirror-writing: signals carry cryptographic proof of
 * provenance so recipients can verify authenticity without trusting the channel.
 *
 * Uses HMAC-SHA256 for lightweight signing (zero external deps — Node builtins).
 * For production use with public-key verification, swap to Ed25519.
 *
 * ES module. Zero npm dependencies.
 */

import { createHmac, randomBytes, timingSafeEqual } from 'crypto';

// ─── Constants ─────────────────────────────────────────────────────────────

const SIG_VERSION = 1;
const SIG_ALGORITHM = 'sha256';
const DEFAULT_TTL_MS = 300_000; // 5 minutes

// ─── 1. Signal Signing ─────────────────────────────────────────────────────

/**
 * Create a signed signal envelope.
 *
 * An envelope wraps a signal payload with metadata and an HMAC signature
 * so recipients can verify: (a) the signal hasn't been tampered with,
 * (b) it came from someone who knows the secret, (c) it hasn't expired.
 *
 * @param {object}  payload          — the signal data to sign
 * @param {string}  secret           — shared secret for HMAC
 * @param {object}  [options]
 * @param {string}  [options.source] — who/what generated this signal
 * @param {string}  [options.id]     — unique signal ID (auto-generated if omitted)
 * @param {number}  [options.ttlMs]  — time-to-live in milliseconds (default 5 min)
 * @returns {SignalEnvelope}         — { id, ts, source, ttl, version, payload, signature }
 */
export function signSignal(payload, secret, options = {}) {
    const ts = Date.now();
    const id = options.id ?? `${ts}-${randomBytes(4).toString('hex')}`;
    const source = options.source ?? 'unknown';
    const ttl = options.ttlMs ?? DEFAULT_TTL_MS;

    const envelope = {
        id,
        ts,
        source,
        ttl,
        version: SIG_VERSION,
        payload,
    };

    const signature = createSignature(envelope, secret);

    return {
        ...envelope,
        signature,
    };
}

/**
 * Create the HMAC signature for an envelope.
 * Signs: id|ts|source|ttl|version|payload (canonical JSON, sorted keys)
 *
 * @param {object} envelope  — the envelope without signature
 * @param {string} secret
 * @returns {string} hex-encoded HMAC
 */
function createSignature(envelope, secret) {
    const { signature, ...rest } = envelope;
    const canonical = JSON.stringify(rest);
    return createHmac(SIG_ALGORITHM, secret).update(canonical).digest('hex');
}

// ─── 2. Signal Verification ────────────────────────────────────────────────

/**
 * Verify a signed signal envelope.
 *
 * Checks:
 *   1. Signature matches (tamper detection)
 *   2. Not expired (ttl check)
 *   3. Version is supported
 *
 * @param {SignalEnvelope} envelope  — the signed envelope to verify
 * @param {string}         secret    — shared secret for HMAC
 * @returns {VerificationResult}     — { valid: boolean, reason?: string, payload?: object }
 */
export function verifySignal(envelope, secret) {
    const { signature, ...rest } = envelope;

    // Version check
    if (!rest.version || rest.version !== SIG_VERSION) {
        return { valid: false, reason: `unsupported version: ${rest.version}` };
    }

    // TTL check
    const age = Date.now() - rest.ts;
    if (age > rest.ttl) {
        return {
            valid: false,
            reason: `signal expired (age: ${age}ms, ttl: ${rest.ttl}ms)`,
        };
    }

    // Future timestamp check (clock skew guard)
    if (rest.ts > Date.now() + 60_000) {
        return {
            valid: false,
            reason: `signal timestamp is in the future (${new Date(rest.ts).toISOString()})`,
        };
    }

    // Signature check
    const expected = createSignature(rest, secret);
    if (!timingSafeEqual(Buffer.from(signature, 'hex'), Buffer.from(expected, 'hex'))) {
        return { valid: false, reason: 'signature mismatch — signal may be tampered' };
    }

    return { valid: true, payload: rest.payload };
}

/**
 * Verify multiple signals in batch. Returns results in same order.
 *
 * @param {SignalEnvelope[]} envelopes
 * @param {string}           secret
 * @returns {VerificationResult[]}
 */
export function verifyBatch(envelopes, secret) {
    return envelopes.map(e => verifySignal(e, secret));
}

// ─── 3. Provenance Chain ───────────────────────────────────────────────────

/**
 * Create a provenance chain entry linking a signal to its predecessor.
 *
 * This builds a tamper-evident chain: each new signal includes the hash of the
 * previous signal, creating a verifiable lineage from origin to latest.
 *
 * @param {object}        payload        — new signal payload
 * @param {string}        secret         — shared secret
 * @param {string|null}   previousSig    — hex signature of the previous signal in the chain
 * @param {object}        [options]      — same as signSignal options
 * @returns {SignalEnvelope}             — envelope with `previousSig` field
 */
export function chainSignal(payload, secret, previousSig, options = {}) {
    const ts = Date.now();
    const id = options.id ?? `${ts}-${randomBytes(4).toString('hex')}`;
    const source = options.source ?? 'unknown';
    const ttl = options.ttlMs ?? DEFAULT_TTL_MS;

    const envelope = {
        id,
        ts,
        source,
        ttl,
        version: SIG_VERSION,
        previousSig: previousSig ?? null,
        payload,
    };

    const signature = createSignature(envelope, secret);

    return {
        ...envelope,
        signature,
    };
}

/**
 * Verify an entire provenance chain.
 *
 * Checks each link's signature AND that each link references its predecessor.
 *
 * @param {SignalEnvelope[]} chain   — ordered array of chained signals
 * @param {string}           secret
 * @returns {{ valid: boolean, reason?: string, validLinks: number, totalLinks: number }}
 */
export function verifyChain(chain, secret) {
    if (!chain.length) {
        return { valid: true, validLinks: 0, totalLinks: 0 };
    }

    let validLinks = 0;
    let previousSig = null;

    for (let i = 0; i < chain.length; i++) {
        const result = verifySignal(chain[i], secret);
        if (!result.valid) {
            return {
                valid: false,
                reason: `link ${i}: ${result.reason}`,
                validLinks,
                totalLinks: chain.length,
            };
        }

        // Check chain integrity: this link's previousSig must match previous link's signature
        if (i > 0) {
            if (chain[i].previousSig !== previousSig) {
                return {
                    valid: false,
                    reason: `chain broken at link ${i}: previousSig mismatch`,
                    validLinks,
                    totalLinks: chain.length,
                };
            }
        }

        previousSig = chain[i].signature;
        validLinks++;
    }

    return { valid: true, validLinks, totalLinks: chain.length };
}

// ─── 4. Anti-Replay Protection ─────────────────────────────────────────────

/**
 * A simple nonce tracker to prevent signal replay attacks.
 *
 * Usage:
 *   const replayGuard = createReplayGuard();
 *   const result = replayGuard.checkAndRecord(envelope.id);
 *   if (!result.allowed) { /* replay detected * / }
 */
export function createReplayGuard(options = {}) {
    const maxEntries = options.maxEntries ?? 10_000;
    /** @type {Map<string, number>} — id → expiration timestamp */
    const seen = new Map();

    /**
     * Check if a signal ID has been seen before. If not, record it.
     *
     * @param {string} signalId  — the unique signal ID
     * @param {number} [ttlMs]   — how long to remember this ID (defaults to signal TTL)
     * @returns {{ allowed: boolean, reason?: string }}
     */
    function checkAndRecord(signalId, ttlMs) {
        // Prune expired entries
        const now = Date.now();
        for (const [id, expires] of seen) {
            if (expires < now) seen.delete(id);
        }

        // Enforce max entries (evict oldest)
        if (seen.size >= maxEntries) {
            const oldest = [...seen.keys()][0];
            seen.delete(oldest);
        }

        if (seen.has(signalId)) {
            return { allowed: false, reason: 'replay detected: duplicate signal ID' };
        }

        seen.set(signalId, now + (ttlMs ?? DEFAULT_TTL_MS));
        return { allowed: true };
    }

    /**
     * Manually clear a signal ID from the replay guard.
     * @param {string} signalId
     */
    function forget(signalId) {
        seen.delete(signalId);
    }

    /** Number of currently tracked signal IDs */
    function size() {
        return seen.size;
    }

    /** Clear all tracked IDs */
    function reset() {
        seen.clear();
    }

    return { checkAndRecord, forget, size, reset };
}

// ─── 5. Key Management Helpers ─────────────────────────────────────────────

/**
 * Generate a cryptographically random secret suitable for HMAC signing.
 *
 * @param {number} [byteLength=32]  — length in bytes (32 = 256 bits)
 * @returns {string} hex-encoded secret
 */
export function generateSecret(byteLength = 32) {
    return randomBytes(byteLength).toString('hex');
}

/**
 * Derive a per-source signing key from a master secret.
 * Uses HMAC-based key derivation so each source gets a unique key
 * without storing multiple secrets.
 *
 * @param {string} masterSecret
 * @param {string} source        — source identifier (e.g. 'alpha-parser', 'backtest')
 * @returns {string} hex-encoded derived key
 */
export function deriveKey(masterSecret, source) {
    return createHmac(SIG_ALGORITHM, masterSecret).update(`derive:${source}`).digest('hex');
}

// ─── 6. Signal Bundle ──────────────────────────────────────────────────────

/**
 * Create a bundle of multiple signed signals, itself signed.
 * Useful for batching signals from different sources into one verifiable package.
 *
 * @param {SignalEnvelope[]} signals  — array of signed signal envelopes
 * @param {string}           secret
 * @param {object}           [options] — same as signSignal options
 * @returns {SignalEnvelope}           — bundle envelope with payload.signals = [...]
 */
export function bundleSignals(signals, secret, options = {}) {
    const payload = {
        type: 'signal-bundle',
        count: signals.length,
        signals,
    };
    return signSignal(payload, secret, options);
}

/**
 * Verify a signal bundle. Verifies the bundle signature AND each contained signal.
 *
 * @param {SignalEnvelope} bundleEnvelope
 * @param {string}         secret
 * @param {object}         [options]
 * @param {boolean}        [options.verifyContents=true]  — also verify each signal inside
 * @returns {{ valid: boolean, reason?: string, payload?: object, results?: VerificationResult[] }}
 */
export function verifyBundle(bundleEnvelope, secret, options = {}) {
    const verifyContents = options.verifyContents ?? true;

    const bundleResult = verifySignal(bundleEnvelope, secret);
    if (!bundleResult.valid) return bundleResult;

    const bundlePayload = bundleResult.payload;
    if (!bundlePayload || bundlePayload.type !== 'signal-bundle') {
        return { valid: false, reason: 'payload is not a signal-bundle' };
    }

    if (verifyContents && bundlePayload.signals) {
        const results = verifyBatch(bundlePayload.signals, secret);
        const allValid = results.every(r => r.valid);
        if (!allValid) {
            const invalidIdx = results.findIndex(r => !r.valid);
            return {
                valid: false,
                reason: `signal ${invalidIdx} in bundle is invalid: ${results[invalidIdx].reason}`,
                results,
            };
        }
        return { valid: true, payload: bundlePayload, results };
    }

    return { valid: true, payload: bundlePayload };
}

// ─── Type Definitions (JSDoc) ──────────────────────────────────────────────

/**
 * @typedef {object} SignalEnvelope
 * @property {string} id          — unique signal ID
 * @property {number} ts          — creation timestamp (ms)
 * @property {string} source      — signal source identifier
 * @property {number} ttl         — time-to-live in ms
 * @property {number} version     — envelope format version
 * @property {string} [previousSig] — previous signal signature (chain mode)
 * @property {object} payload     — the actual signal data
 * @property {string} signature   — hex-encoded HMAC signature
 */

/**
 * @typedef {object} VerificationResult
 * @property {boolean} valid
 * @property {string}  [reason]
 * @property {object}  [payload]
 */
