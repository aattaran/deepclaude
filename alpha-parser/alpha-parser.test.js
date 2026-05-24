/**
 * Alpha Parser — unit tests (node:test runner)
 * Run: node --test alpha-parser/alpha-parser.test.js
 */
import { describe, it, before } from 'node:test';
import assert from 'node:assert/strict';
import { extractLinks, auditMessage, scanMessage, RED_FLAGS, SAFETY } from './engine.mjs';
import { createMockFetch, ALL_FIXTURES, CLEAN_CARD, CLEAN_EXPECT,
         HYPE_CARD, HYPE_EXPECT, MULTI_LINK, MULTI_EXPECT,
         EMOJI_VARIANTS, EMOJI_EXPECT, NO_AUDIT_CARD, NO_AUDIT_EXPECT,
         NUMERIC_EDGES, NUMERIC_EXPECT, NO_LINK, NO_LINK_EXPECT } from './test-fixtures.mjs';

const mockFetch = createMockFetch();

// ---------------------------------------------------------------------------
describe('extractLinks', () => {
    it('extracts a single tokenscan address', () => {
        const links = extractLinks(HYPE_CARD);
        assert.equal(links.length, 1);
        assert.match(links[0], /^[1-9A-HJ-NP-Za-km-z]{32,44}$/);
    });

    it('extracts two addresses from multi-link message', () => {
        const links = extractLinks(MULTI_LINK);
        assert.equal(links.length, 2);
    });

    it('returns empty array when no tokenscan link present', () => {
        const links = extractLinks(NO_LINK);
        assert.equal(links.length, 0);
    });

    it('returns empty array for empty string', () => {
        assert.equal(extractLinks('').length, 0);
    });

    it('returns empty array for null/undefined input', () => {
        assert.equal(extractLinks(null).length, 0);
        assert.equal(extractLinks(undefined).length, 0);
    });
});

// ---------------------------------------------------------------------------
describe('scanMessage', () => {
    it('returns null when no link is present', async () => {
        const result = await scanMessage(NO_LINK, { fetch: mockFetch });
        assert.equal(result, null);
    });

    it('returns results for one valid link', async () => {
        const result = await scanMessage(HYPE_CARD, { fetch: mockFetch });
        assert.ok(Array.isArray(result));
        assert.equal(result.length, 1);
        assert.ok(result[0].address);
        assert.ok(typeof result[0].score === 'number');
        assert.ok(['PASS', 'WARN', 'REJECT'].includes(result[0].verdict));
    });

    it('returns results for multi-link message', async () => {
        const result = await scanMessage(MULTI_LINK, { fetch: mockFetch });
        assert.equal(result.length, 2);
    });
});

// ---------------------------------------------------------------------------
describe('auditMessage — fixtures', () => {
    it('HYPE — known garbage (5 flags) → REJECT', async () => {
        const result = await auditMessage(HYPE_CARD, { fetch: mockFetch });
        assert.equal(result.found, true);
        const a = result.audits[0];
        assert.equal(a.verdict, HYPE_EXPECT.verdict);
        for (const flag of HYPE_EXPECT.flags) {
            assert.ok(a.flags_triggered.some(f => f.key === flag),
                `expected flag "${flag}" not found in ${JSON.stringify(a.flags_triggered)}`);
        }
    });

    it('Clean token — all green → PASS', async () => {
        const result = await auditMessage(CLEAN_CARD, { fetch: mockFetch });
        const a = result.audits[0];
        assert.equal(a.verdict, CLEAN_EXPECT.verdict);
        assert.ok(a.safety_score >= CLEAN_EXPECT.minScore);
    });

    it('Multi-link — two tokens, first WARN second PASS', async () => {
        const result = await auditMessage(MULTI_LINK, { fetch: mockFetch });
        assert.equal(result.audits.length, MULTI_EXPECT.linkCount);
        assert.equal(result.audits[0].verdict, MULTI_EXPECT.firstVerdict);
        assert.equal(result.audits[1].verdict, MULTI_EXPECT.secondVerdict);
    });

    it('Emoji variants — handles both ⚠️ and ⚠', async () => {
        const result = await auditMessage(EMOJI_VARIANTS, { fetch: mockFetch });
        const a = result.audits[0];
        assert.equal(a.verdict, EMOJI_EXPECT.verdict);
        for (const flag of EMOJI_EXPECT.flags) {
            assert.ok(a.flags_triggered.some(f => f.key === flag),
                `expected flag "${flag}" not found`);
        }
    });

    it('No audit section — DEX-only scoring', async () => {
        const result = await auditMessage(NO_AUDIT_CARD, { fetch: mockFetch });
        // card_parsed is an object with empty flags/warnings when no audit lines exist
        assert.ok(typeof result.audits[0].card_parsed === 'object');
        assert.equal(result.audits[0].card_parsed.flags.length, 0);
        // Should still produce a verdict from DEX data alone
        assert.ok(['PASS', 'WARN', 'REJECT'].includes(result.audits[0].verdict));
    });

    it('Numeric edges — K suffix and commas parsed correctly', async () => {
        const result = await auditMessage(NUMERIC_EDGES, { fetch: mockFetch });
        const a = result.audits[0];
        assert.equal(a.verdict, NUMERIC_EXPECT.verdict);
        for (const flag of NUMERIC_EXPECT.flags) {
            assert.ok(a.flags_triggered.some(f => f.key === flag),
                `expected flag "${flag}" not found`);
        }
    });

    it('False positive — no link → found: false', async () => {
        const result = await auditMessage(NO_LINK);
        assert.equal(result.found, NO_LINK_EXPECT.found);
    });
});

// ---------------------------------------------------------------------------
describe('auditMessage — structure and edge cases', () => {
    it('returns correct result shape for a valid audit', async () => {
        const result = await auditMessage(HYPE_CARD, { fetch: mockFetch });
        assert.equal(result.found, true);
        assert.ok(Array.isArray(result.audits));
        const a = result.audits[0];
        assert.ok(a.address);
        assert.ok(typeof a.safety_score === 'number');
        assert.ok(a.safety_score >= 0 && a.safety_score <= 100);
        assert.ok(Array.isArray(a.flags_triggered));
        assert.ok(typeof a.verdict === 'string');
        assert.ok(typeof a.message === 'string' || !('message' in a));
    });

    it('unknown URL returns no DEX data but does not crash', async () => {
        // Using a random valid base58 address not in our mock
        const result = await auditMessage(
            'https://t.me/tokenscan?start=scan-HZRC7PqJLj4R1XgcLrj6p3MbZnNR2mJqTkVkAvpHWJ8x',
            { fetch: mockFetch }
        );
        assert.equal(result.found, true);
        assert.equal(result.audits[0].dex_summary, null);
    });
});

// ---------------------------------------------------------------------------
describe('SAFETY thresholds', () => {
    it('defines all expected safety constants', () => {
        assert.ok(typeof SAFETY.MIN_LP_RATIO === 'number');
        assert.ok(typeof SAFETY.MAX_TOP10_HOLDERS === 'number');
        assert.ok(typeof SAFETY.MIN_LIQUIDITY_USD === 'number');
        assert.ok(typeof SAFETY.MIN_HOLDER_COUNT === 'number');
    });
});

// ---------------------------------------------------------------------------
describe('RED_FLAGS', () => {
    it('has exactly the expected flag keys', () => {
        const keys = RED_FLAGS.map(f => f.key);
        assert.deepEqual(keys, [
            'mintable', 'mutable', 'top10_heavy', 'low_lp',
            'low_holders', 'dex_unpaid', 'no_data', 'new_token',
        ]);
    });

    it('every flag has a non-empty label', () => {
        for (const f of RED_FLAGS) {
            assert.ok(f.label.length > 0, `flag ${f.key} has empty label`);
        }
    });

    it('every flag has a positive weight', () => {
        for (const f of RED_FLAGS) {
            assert.ok(f.weight > 0, `flag ${f.key} has non-positive weight`);
        }
    });
});
