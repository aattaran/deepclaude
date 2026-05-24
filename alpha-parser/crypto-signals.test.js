/**
 * Crypto Signals — unit tests (node:test runner)
 * Run: node --test alpha-parser/crypto-signals.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    signSignal,
    verifySignal,
    verifyBatch,
    chainSignal,
    verifyChain,
    createReplayGuard,
    generateSecret,
    deriveKey,
    bundleSignals,
    verifyBundle,
} from './crypto-signals.mjs';

const SECRET = 'test-secret-key-for-unit-tests-32b';

describe('signSignal + verifySignal', () => {
    it('produces a valid signed envelope', () => {
        const env = signSignal({ action: 'BUY', token: 'SOL' }, SECRET, { source: 'test' });
        assert.ok(env.id);
        assert.ok(env.signature);
        assert.equal(env.source, 'test');
        assert.equal(env.version, 1);
        assert.deepEqual(env.payload, { action: 'BUY', token: 'SOL' });
    });

    it('round-trips: sign then verify succeeds', () => {
        const env = signSignal({ action: 'SELL', confidence: 0.8 }, SECRET);
        const result = verifySignal(env, SECRET);
        assert.ok(result.valid);
        assert.deepEqual(result.payload, { action: 'SELL', confidence: 0.8 });
    });

    it('detects tampered payload', () => {
        const env = signSignal({ action: 'BUY' }, SECRET);
        // Create a forgery: same envelope but with different payload and old signature
        const forged = { ...env, payload: { action: 'SELL' } };
        const result = verifySignal(forged, SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('signature'));
    });

    it('detects tampered source', () => {
        const env = signSignal({ x: 1 }, SECRET, { source: 'alpha' });
        env.source = 'hacker';
        const result = verifySignal(env, SECRET);
        assert.equal(result.valid, false);
    });

    it('detects wrong secret', () => {
        const env = signSignal({ x: 1 }, SECRET);
        const result = verifySignal(env, 'wrong-secret');
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('signature'));
    });

    it('detects expired signal', async () => {
        const env = signSignal({ x: 1 }, SECRET, { ttlMs: 10 });
        await new Promise(r => setTimeout(r, 20));
        const result = verifySignal(env, SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('expired'));
    });

    it('accepts non-expired signal', () => {
        const env = signSignal({ x: 1 }, SECRET, { ttlMs: 60_000 });
        const result = verifySignal(env, SECRET);
        assert.ok(result.valid);
    });

    it('rejects future timestamps > 60s', () => {
        const env = signSignal({ x: 1 }, SECRET);
        env.ts = Date.now() + 120_000; // 2 minutes in the future
        const result = verifySignal(env, SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('future'));
    });

    it('generates unique IDs by default', () => {
        const e1 = signSignal({ x: 1 }, SECRET);
        const e2 = signSignal({ x: 1 }, SECRET);
        assert.notEqual(e1.id, e2.id);
    });

    it('accepts custom ID', () => {
        const env = signSignal({ x: 1 }, SECRET, { id: 'custom-123' });
        assert.equal(env.id, 'custom-123');
    });

    it('uses hex-encoded sha256 signature', () => {
        const env = signSignal({ x: 1 }, SECRET);
        assert.equal(env.signature.length, 64); // sha256 hex = 64 chars
        assert.match(env.signature, /^[0-9a-f]{64}$/);
    });
});

describe('verifyBatch', () => {
    it('verifies multiple envelopes', () => {
        const envs = [
            signSignal({ a: 1 }, SECRET),
            signSignal({ b: 2 }, SECRET),
            signSignal({ c: 3 }, SECRET),
        ];
        const results = verifyBatch(envs, SECRET);
        assert.equal(results.length, 3);
        assert.ok(results.every(r => r.valid));
    });

    it('detects invalid in batch', () => {
        const envs = [
            signSignal({ a: 1 }, SECRET),
            signSignal({ b: 2 }, 'wrong'),
        ];
        // Tamper the second one
        envs[1].payload.b = 'tampered';

        const results = verifyBatch(envs, SECRET);
        assert.ok(results[0].valid);
        assert.equal(results[1].valid, false);
    });

    it('handles empty batch', () => {
        assert.deepEqual(verifyBatch([], SECRET), []);
    });
});

describe('chainSignal + verifyChain', () => {
    it('creates chained signals with previousSig references', () => {
        const s1 = chainSignal({ step: 1 }, SECRET, null);
        const s2 = chainSignal({ step: 2 }, SECRET, s1.signature);
        const s3 = chainSignal({ step: 3 }, SECRET, s2.signature);

        assert.equal(s1.previousSig, null);
        assert.equal(s2.previousSig, s1.signature);
        assert.equal(s3.previousSig, s2.signature);
    });

    it('verifyChain succeeds for valid chain', () => {
        const s1 = chainSignal({ step: 1 }, SECRET, null);
        const s2 = chainSignal({ step: 2 }, SECRET, s1.signature);
        const s3 = chainSignal({ step: 3 }, SECRET, s2.signature);

        const result = verifyChain([s1, s2, s3], SECRET);
        assert.ok(result.valid);
        assert.equal(result.validLinks, 3);
        assert.equal(result.totalLinks, 3);
    });

    it('verifyChain detects broken chain (wrong previousSig)', () => {
        const s1 = chainSignal({ step: 1 }, SECRET, null);
        // Create a link that's validly self-signed but points to wrong previous
        const s2 = chainSignal({ step: 2 }, SECRET, '0'.repeat(64));

        const result = verifyChain([s1, s2], SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('chain broken'));
        assert.equal(result.validLinks, 1);
    });

    it('verifyChain detects expired link mid-chain', async () => {
        const s1 = chainSignal({ step: 1 }, SECRET, null, { ttlMs: 10 });
        const s2 = chainSignal({ step: 2 }, SECRET, s1.signature);

        await new Promise(r => setTimeout(r, 20));

        const result = verifyChain([s1, s2], SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('link 0'));
    });

    it('verifyChain handles empty chain', () => {
        const result = verifyChain([], SECRET);
        assert.ok(result.valid);
        assert.equal(result.validLinks, 0);
    });
});

describe('createReplayGuard', () => {
    it('allows first occurrence', () => {
        const guard = createReplayGuard();
        const result = guard.checkAndRecord('sig-1');
        assert.ok(result.allowed);
    });

    it('blocks duplicate (replay)', () => {
        const guard = createReplayGuard();
        guard.checkAndRecord('sig-1');
        const result = guard.checkAndRecord('sig-1');
        assert.equal(result.allowed, false);
        assert.ok(result.reason.includes('replay'));
    });

    it('allows different IDs', () => {
        const guard = createReplayGuard();
        assert.ok(guard.checkAndRecord('sig-1').allowed);
        assert.ok(guard.checkAndRecord('sig-2').allowed);
        assert.ok(guard.checkAndRecord('sig-3').allowed);
    });

    it('forget removes a tracked ID', () => {
        const guard = createReplayGuard();
        guard.checkAndRecord('sig-1');
        guard.forget('sig-1');
        assert.ok(guard.checkAndRecord('sig-1').allowed);
    });

    it('reset clears all', () => {
        const guard = createReplayGuard();
        guard.checkAndRecord('a');
        guard.checkAndRecord('b');
        guard.checkAndRecord('c');
        guard.reset();
        assert.equal(guard.size(), 0);
        assert.ok(guard.checkAndRecord('a').allowed);
    });

    it('size reflects tracked count', () => {
        const guard = createReplayGuard();
        assert.equal(guard.size(), 0);
        guard.checkAndRecord('a');
        guard.checkAndRecord('b');
        assert.equal(guard.size(), 2);
    });

    it('expires entries after TTL', async () => {
        const guard = createReplayGuard();
        guard.checkAndRecord('sig-1', 10); // 10ms TTL
        guard.checkAndRecord('sig-2', 60_000); // long TTL

        await new Promise(r => setTimeout(r, 20));

        // sig-1 should be expired on next checkAndRecord
        guard.checkAndRecord('sig-3'); // triggers prune
        assert.ok(guard.checkAndRecord('sig-1').allowed); // can now re-use
        assert.equal(guard.checkAndRecord('sig-2').allowed, false); // still tracked
    });
});

describe('generateSecret', () => {
    it('returns 64-char hex string by default (32 bytes)', () => {
        const s = generateSecret();
        assert.equal(s.length, 64);
        assert.match(s, /^[0-9a-f]{64}$/);
    });

    it('produces unique values', () => {
        const a = generateSecret();
        const b = generateSecret();
        assert.notEqual(a, b);
    });

    it('respects byteLength parameter', () => {
        assert.equal(generateSecret(16).length, 32);
        assert.equal(generateSecret(64).length, 128);
    });
});

describe('deriveKey', () => {
    it('derives different keys for different sources', () => {
        const k1 = deriveKey(SECRET, 'alpha-parser');
        const k2 = deriveKey(SECRET, 'backtest');
        assert.notEqual(k1, k2);
    });

    it('is deterministic — same input = same key', () => {
        const k1 = deriveKey(SECRET, 'alpha');
        const k2 = deriveKey(SECRET, 'alpha');
        assert.equal(k1, k2);
    });

    it('different master = different keys', () => {
        const k1 = deriveKey('secret-a', 'test');
        const k2 = deriveKey('secret-b', 'test');
        assert.notEqual(k1, k2);
    });

    it('produces 64-char hex output', () => {
        const key = deriveKey(SECRET, 'test');
        assert.equal(key.length, 64);
        assert.match(key, /^[0-9a-f]{64}$/);
    });
});

describe('bundleSignals + verifyBundle', () => {
    it('bundles multiple signals into a signed envelope', () => {
        const s1 = signSignal({ a: 1 }, SECRET);
        const s2 = signSignal({ b: 2 }, SECRET);
        const bundle = bundleSignals([s1, s2], SECRET);

        assert.equal(bundle.payload.type, 'signal-bundle');
        assert.equal(bundle.payload.count, 2);
        assert.equal(bundle.payload.signals.length, 2);
    });

    it('verifyBundle succeeds with valid contents', () => {
        const s1 = signSignal({ a: 1 }, SECRET);
        const s2 = signSignal({ b: 2 }, SECRET);
        const bundle = bundleSignals([s1, s2], SECRET);

        const result = verifyBundle(bundle, SECRET);
        assert.ok(result.valid);
        assert.equal(result.payload.count, 2);
    });

    it('verifyBundle detects tampered inner signal', () => {
        // Create a signal, then tamper its payload AFTER it was signed
        const s1 = signSignal({ a: 1 }, SECRET);
        s1.payload.a = 999; // s1's own signature is now invalid
        const s2 = signSignal({ b: 2 }, SECRET);

        // Bundle signed with tampered s1 — bundle sig is valid, but inner sig is broken
        const bundle = bundleSignals([s1, s2], SECRET);

        const result = verifyBundle(bundle, SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('signal 0'));
    });

    it('verifyBundle detects invalid bundle signature', () => {
        const s1 = signSignal({ a: 1 }, SECRET);
        const bundle = bundleSignals([s1], SECRET);
        bundle.signature = '0'.repeat(64);

        const result = verifyBundle(bundle, SECRET);
        assert.equal(result.valid, false);
    });

    it('verifyBundle skips content verification when option set', () => {
        // Tamper the inner signal BEFORE bundling so bundle signature covers it
        const s1 = signSignal({ a: 1 }, SECRET);
        s1.payload.a = 999; // breaks s1's own signature but bundle will re-sign it
        const bundle = bundleSignals([s1], SECRET);

        // With verifyContents=false, inner signatures are skipped — bundle is valid
        const result = verifyBundle(bundle, SECRET, { verifyContents: false });
        assert.ok(result.valid);
        // With verifyContents=true (default), it should fail
        const result2 = verifyBundle(bundle, SECRET);
        assert.equal(result2.valid, false);
    });

    it('verifyBundle rejects non-bundle payload', () => {
        const env = signSignal({ not: 'a-bundle' }, SECRET);
        const result = verifyBundle(env, SECRET);
        assert.equal(result.valid, false);
        assert.ok(result.reason.includes('not a signal-bundle'));
    });
});
