#!/usr/bin/env node
/**
 * Alpha Parser CLI — multi-source token audit (TokenScan card + DEX + RugCheck).
 *
 *   node cli.mjs verify "message"    Full breakdown from all sources
 *   node cli.mjs scan "message"      Only PASS/WARN tokens
 *   node cli.mjs test-hype           Test against $HYPE with full card text
 *   node cli.mjs test "full card"    Paste a complete TokenScan card
 *
 * Nothing sends messages. Everything is read-only.
 */

import { auditMessage, scanMessage } from './engine.mjs';

const mode = process.argv[2];
const input = process.argv.slice(3).join(' ');

const W = '═'.repeat(62);
const w = '─'.repeat(62);

console.log(`\n  ${W}`);
console.log(`  🔍 ALPHA PARSER — Multi-Source Audit`);
console.log(`  ${W}`);

switch (mode) {
    case 'verify':
    case 'audit': {
        if (!input) {
            console.log('\n  Paste a Telegram message containing TokenScan links.\n');
            console.log('  Usage:  node cli.mjs verify "message text"\n');
            console.log('  Shows:  TokenScan card parse → DEX Screener → RugCheck → Verdict');
            process.exit(0);
        }

        console.log(`\n  📋 AUDITING:\n     "${input.slice(0, 180)}${input.length > 180 ? '...' : ''}"\n`);

        const result = await auditMessage(input);
        if (!result.found) {
            console.log(`  ${result.message}\n`);
            process.exit(0);
        }

        console.log(`  Found ${result.links.length} TokenScan link(s):\n`);

        for (let i = 0; i < result.audits.length; i++) {
            const a = result.audits[i];
            const vEmoji = a.verdict === 'PASS' ? '🟢' : a.verdict === 'WARN' ? '🟡' : '🔴';

            console.log(`  ${w}`);
            console.log(`  Token #${i + 1}  ${vEmoji} ${a.verdict}  (${a.safety_score}/100)`);
            console.log(`  ${w}`);
            console.log(`  Address:  ${a.address}`);

            // Data sources used
            const srcs = [];
            if (a.sources.card) srcs.push('TokenScan card');
            if (a.sources.dex) srcs.push('DEX Screener');
            if (a.sources.rugcheck) srcs.push('RugCheck');
            console.log(`  Sources:  ${srcs.join(' + ')}`);

            // TokenScan card data
            if (a.card_parsed) {
                const cp = a.card_parsed;
                console.log(`\n  ── TokenScan Card ──`);
                if (cp.mc) console.log(`  MC:        $${(cp.mc / 1e6).toFixed(2)}M`);
                if (cp.liquidity) console.log(`  Liquidity: $${(cp.liquidity / 1e6).toFixed(2)}M`);
                if (cp.volume24h) console.log(`  Volume 24h: $${(cp.volume24h / 1e6).toFixed(2)}M`);
                if (cp.ageDays) console.log(`  Age:       ${cp.ageDays}d`);
                if (cp.holders) console.log(`  Holders:   ${cp.holders.toLocaleString()}`);
                if (cp.priceUsd) console.log(`  Price:     $${cp.priceUsd}`);
                if (Object.keys(cp.raw).length > 0) {
                    console.log(`  Raw flags: ${JSON.stringify(cp.raw)}`);
                }
            }

            // DEX data
            if (a.dex_summary) {
                const ds = a.dex_summary;
                console.log(`\n  ── DEX Screener ──`);
                console.log(`  Price:      ${ds.price}`);
                console.log(`  MC:         ${ds.mc}`);
                console.log(`  Liquidity:  ${ds.liquidity}`);
                console.log(`  LP/MC:      ${ds.lpRatio}`);
                console.log(`  Volume 24h: ${ds.volume24h}`);
                console.log(`  Change 24h: ${ds.priceChange24h}`);
                console.log(`  Age:        ${ds.age}`);
                console.log(`  Txns 24h:   ${ds.buys24h}B / ${ds.sells24h}S`);
                console.log(`  DEX:        ${ds.dexId}`);
            }

            // RugCheck data
            if (a.rugcheck_available && !a.card_parsed) {
                // Only show if we actually used it (no card available)
                console.log(`\n  ── RugCheck (fallback) ──`);
                console.log(`  Used because no TokenScan card text found in message.`);
            }

            // Flags
            if (a.flags_triggered.length > 0) {
                console.log(`\n  🚩 FLAGS (${a.flags_triggered.length}):`);
                for (const f of a.flags_triggered) {
                    const srcTag = `[${f.source}]`.padEnd(14);
                    console.log(`     ❌ ${srcTag} ${f.label}`);
                }
            } else {
                console.log(`\n  ✅ No flags triggered`);
            }

            console.log(`\n  📊 VERDICT: ${a.verdict} (${a.safety_score}/100)`);
            if (a.verdict === 'REJECT') {
                console.log(`     → SILENTLY FILTERED in live mode`);
            } else if (a.verdict === 'WARN') {
                console.log(`     → Shown with warnings in live mode`);
            } else {
                console.log(`     → SURFACED as quality signal`);
            }
            console.log('');
        }

        console.log(`  ${result.message}\n`);
        break;
    }

    case 'scan': {
        if (!input) {
            console.log('\n  Usage: node cli.mjs scan "message"\n');
            process.exit(0);
        }

        const results = await scanMessage(input);
        if (!results) {
            console.log('\n  No TokenScan links found.\n');
            process.exit(0);
        }

        const passing = results.filter(r => r.verdict !== 'REJECT');
        if (passing.length === 0) {
            console.log(`\n  🔴 All ${results.length} token(s) REJECTED. Nothing to look at.\n`);
            process.exit(0);
        }

        console.log(`\n  🟢 ${passing.length}/${results.length} token(s) passed:\n`);
        for (const r of passing) {
            console.log(`  ${r.summary}`);
        }
        console.log('');
        break;
    }

    case 'test-hype':
    case 'test': {
        // Full $HYPE card text from earlier
        const hypeCard = input || `HYPE (https://t.me/tokenscan?start=scan-98sMhvDwXj1RQi5c5Mndm3vPe9cBqPrbLaufMXFNMh5g) ($HYPE)
➰🟣 🌱225d 👀92

📊 Token Stats
➰ MC:   $54.96M
➰ ATH:  $58.52M (-6.09% / 3h)
➰ USD:  58.83 (13.7%)
➰ LIQ:  $3.44M
➰ VOL:  $8.19M (24h)
➰ 1H:   B 442 / S 410 (7.8%)
➰ HLD:  13,047

❌ Audit 🟥🟥
❌ Mintable [Yes]
❌ Token Data Mutable [Yes]
⚠️ LP Ratio [6.26%]
❌ DEX [NOT PAID]
❌ Top 10 Holders [52.64%]`;

        console.log('\n  🧪 Testing against $HYPE with full card text...\n');

        const result = await auditMessage(hypeCard);
        if (result.found) {
            for (const a of result.audits) {
                console.log(`  Score:    ${a.safety_score}/100`);
                console.log(`  Verdict:  ${a.verdict}`);
                console.log(`  Sources:  ${[a.sources.card ? 'card' : '', a.sources.dex ? 'dex' : '', a.sources.rugcheck ? 'rugcheck' : ''].filter(Boolean).join(', ')}`);
                console.log(`  Card raw: ${a.card_parsed ? JSON.stringify(a.card_parsed.raw) : 'NOT PARSED'} `);
                console.log(`  Card MC:  ${a.card_parsed?.mc ? '$' + (a.card_parsed.mc / 1e6).toFixed(2) + 'M' : 'N/A'}`);
                console.log(`  Card Liq: ${a.card_parsed?.liquidity ? '$' + (a.card_parsed.liquidity / 1e6).toFixed(2) + 'M' : 'N/A'}`);
                console.log(`  Flags:`);
                for (const f of a.flags_triggered) {
                    console.log(`    ❌ [${f.source}] ${f.label}`);
                }
            }
        } else {
            console.log(`  ${result.message}`);
        }
        console.log('');
        break;
    }

    default: {
        console.log('\n  Commands:');
        console.log('    verify "msg"   Full multi-source audit');
        console.log('    scan "msg"     Quick PASS/WARN only');
        console.log('    test-hype       Test against $HYPE (known garbage)');
        console.log('    test "card"     Test against custom card text');
        console.log('');
        break;
    }
}

console.log(`  ${W}\n`);
