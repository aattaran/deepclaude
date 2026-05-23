#!/usr/bin/env node
/**
 * Alpha Parser Test Runner — validates parser accuracy against fixtures.
 *
 *   node test.mjs                       Run all tests (real API calls)
 *   node test.mjs --offline             Run all tests with mock fetch (no network)
 *   node test.mjs --verbose             Show full DEX + card data per test
 *   OFFLINE=true node --test test.mjs   Run with mock fetch via node --test
 */

import { auditMessage, scanMessage, parseTokenScanCards, extractLinks } from './engine.mjs';
import { ALL_FIXTURES, createMockFetch } from './test-fixtures.mjs';

const verbose = process.argv.includes('--verbose');
const offline = process.argv.includes('--offline') || process.env.OFFLINE === 'true' || process.env.OFFLINE === '1';
let passed = 0;
let failed = 0;
const failures = [];

const mockFetch = offline ? createMockFetch() : null;

console.log('\n' + '═'.repeat(60));
console.log('  ALPHA PARSER — Test Suite');
if (offline) console.log('  [OFFLINE MODE — using mock fetch]');
console.log('═'.repeat(60) + '\n');

for (const fixture of ALL_FIXTURES) {
    const { name, input, expect } = fixture;
    console.log(`  📋 ${name}`);

    try {
        // Test link extraction separately for NO_LINK case
        if (expect.found === false) {
            const links = extractLinks(input);
            if (links.length === 0) {
                console.log('     ✅ Correctly found no links\n');
                passed++;
            } else {
                console.log(`     ❌ Expected no links, found ${links.length}\n`);
                failed++;
                failures.push(name);
            }
            continue;
        }

        // Full audit (with mock fetch in offline mode)
        const result = await auditMessage(input, offline ? { fetch: mockFetch } : {});

        if (!result.found) {
            console.log('     ❌ Expected to find links, found none\n');
            failed++;
            failures.push(name);
            continue;
        }

        let testPassed = true;

        // Check link count for multi-link tests
        if (expect.linkCount !== undefined) {
            if (result.links.length === expect.linkCount) {
                console.log(`     ✅ Link count: ${result.links.length}`);
            } else {
                console.log(`     ❌ Link count: expected ${expect.linkCount}, got ${result.links.length}`);
                testPassed = false;
            }
        }

        // Check per-token verdicts for multi-link
        if (expect.firstVerdict && result.audits[0]) {
            if (result.audits[0].verdict === expect.firstVerdict) {
                console.log(`     ✅ Token 1 verdict: ${result.audits[0].verdict} (${result.audits[0].safety_score}/100)`);
            } else {
                console.log(`     ❌ Token 1 verdict: expected ${expect.firstVerdict}, got ${result.audits[0].verdict}`);
                testPassed = false;
            }
        }
        if (expect.secondVerdict && result.audits[1]) {
            if (result.audits[1].verdict === expect.secondVerdict) {
                console.log(`     ✅ Token 2 verdict: ${result.audits[1].verdict} (${result.audits[1].safety_score}/100)`);
            } else {
                console.log(`     ❌ Token 2 verdict: expected ${expect.secondVerdict}, got ${result.audits[1].verdict}`);
                testPassed = false;
            }
        }

        // For single-token tests
        const audit = result.audits[0];

        // Verdict check
        if (expect.verdict && audit) {
            if (audit.verdict === expect.verdict) {
                console.log(`     ✅ Verdict: ${audit.verdict} (${audit.safety_score}/100)`);
            } else {
                console.log(`     ❌ Verdict: expected ${expect.verdict}, got ${audit.verdict} (${audit.safety_score}/100)`);
                testPassed = false;
            }
        }

        // Score range check
        if (expect.minScore !== undefined && audit && audit.safety_score < expect.minScore) {
            console.log(`     ❌ Score too low: ${audit.safety_score} < ${expect.minScore}`);
            testPassed = false;
        }
        if (expect.maxScore !== undefined && audit && audit.safety_score > expect.maxScore) {
            console.log(`     ❌ Score too high: ${audit.safety_score} > ${expect.maxScore}`);
            testPassed = false;
        }

        // Flag check
        if (expect.flags && audit) {
            const gotFlags = audit.flags_triggered.map(f => f.key || f);
            const missingFlags = expect.flags.filter(f => !gotFlags.includes(f));
            const extraFlags = gotFlags.filter(f => !expect.flags.includes(f));

            if (missingFlags.length === 0 && extraFlags.length === 0) {
                console.log(`     ✅ Flags: ${gotFlags.join(', ') || '(none)'}`);
            } else {
                if (missingFlags.length > 0) console.log(`     ❌ Missing flags: ${missingFlags.join(', ')}`);
                if (extraFlags.length > 0) console.log(`     ⚠️  Extra flags: ${extraFlags.join(', ')}`);
                if (missingFlags.length > 0) testPassed = false;
            }
        }

        // Card parsed check
        if (expect.cardParsed === false && audit) {
            const hasCardData = audit.card_parsed && Object.keys(audit.card_parsed.raw).length > 0;
            if (!hasCardData) {
                console.log('     ✅ Card correctly not parsed (no audit lines)');
            } else {
                console.log('     ❌ Card should not have parsed data');
                testPassed = false;
            }
        }

        // Card clean check — all audit flags show safe values
        if (expect.cardClean && audit?.card_parsed) {
            const critFlags = audit.card_parsed.flags || [];
            if (critFlags.length === 0) {
                console.log('     ✅ Card clean: no critical flags from audit');
            } else {
                console.log(`     ❌ Card expected clean but flagged: ${critFlags.join(', ')}`);
                testPassed = false;
            }
        }

        // Card MC/Liq checks
        if (expect.cardMc && audit?.card_parsed?.mc) {
            if (audit.card_parsed.mc === expect.cardMc) {
                console.log(`     ✅ Card MC: $${(audit.card_parsed.mc / 1e6).toFixed(2)}M`);
            } else {
                console.log(`     ⚠️  Card MC: expected $${(expect.cardMc / 1e6).toFixed(2)}M, got $${(audit.card_parsed.mc / 1e6).toFixed(2)}M`);
                // Not a hard failure — DEX may differ
            }
        }
        if (expect.cardLiq && audit?.card_parsed?.liquidity) {
            if (audit.card_parsed.liquidity === expect.cardLiq) {
                console.log(`     ✅ Card Liq: $${(audit.card_parsed.liquidity / 1e6).toFixed(2)}M`);
            } else {
                console.log(`     ⚠️  Card Liq: expected $${(expect.cardLiq / 1e6).toFixed(2)}M, got $${(audit.card_parsed.liquidity / 1e6).toFixed(2)}M`);
            }
        }

        // Sources check
        if (audit) {
            const sources = [];
            if (audit.sources?.card) sources.push('card');
            if (audit.sources?.dex) sources.push('dex');
            if (audit.sources?.rugcheck) sources.push('rugcheck');
            console.log(`     📡 Sources: ${sources.join(' + ')}`);
        }

        // Verbose: full dump
        if (verbose && audit) {
            console.log('');
            if (audit.card_parsed?.raw && Object.keys(audit.card_parsed.raw).length > 0) {
                console.log(`     [verbose] Card raw: ${JSON.stringify(audit.card_parsed.raw)}`);
            }
            if (audit.dex_summary) {
                console.log(`     [verbose] DEX: price=${audit.dex_summary.price} mc=${audit.dex_summary.mc} liq=${audit.dex_summary.liquidity} lp=${audit.dex_summary.lpRatio}`);
            }
            for (const f of (audit.flags_triggered || [])) {
                console.log(`     [verbose] Flag: [${f.source}] ${f.label}`);
            }
        }

        if (testPassed) {
            passed++;
        } else {
            failed++;
            failures.push(name);
        }
    } catch (e) {
        console.log(`     💥 ERROR: ${e.message}`);
        if (verbose) console.log(e.stack);
        failed++;
        failures.push(name);
    }

    console.log('');
}

// Summary
console.log('═'.repeat(60));
const total = passed + failed;
console.log(`  Results: ${passed}/${total} passed`);
if (failed > 0) {
    console.log(`  Failed:  ${failures.join(', ')}`);
}
console.log('═'.repeat(60) + '\n');

process.exit(failed > 0 ? 1 : 0);
