/**
 * Alpha Chat Parser Engine — stealth token extraction + multi-source safety scoring.
 *
 * Data sources (in priority order):
 *   1. TokenScan card text — parsed inline from TG message (free, no API call)
 *   2. DEX Screener API — price, liquidity, volume, pair age
 *   3. RugCheck API — mint authority, mutability, holder concentration (fallback)
 *
 * Usage:
 *   import { scanMessage, auditMessage } from './engine.mjs';
 *   const result = await scanMessage(tgMessageText);
 */

const DEXSCREENER_BASE = 'https://api.dexscreener.com/latest/dex/tokens';
const RUGCHECK_BASE = 'https://api.rugcheck.xyz/v1/tokens';
const TOKENSCAN_URL_PATTERN = /https?:\/\/t\.me\/tokenscan\?start=scan-([1-9A-HJ-NP-Za-km-z]{32,44})/g;

// ── TokenScan card text parsers ──
// Matches lines like: ❌ Mintable [Yes] or ⚠️ LP Ratio [6.26%]
// Handles both plain ⚠ and emoji ⚠️ (with variation selector U+FE0F)
const CARD_PARSERS = [
    { key: 'mintable',     pattern: /Mintable\s*\[([^\]]+)\]/i,          crit: v => v.toLowerCase() === 'yes' },
    { key: 'mutable',      pattern: /Token\s*Data\s*Mutable\s*\[([^\]]+)\]/i, crit: v => v.toLowerCase() === 'yes' },
    { key: 'top10_heavy',  pattern: /Top\s*10\s*Holders?\s*\[([^\]]+)\]/i,
        crit: v => { const pct = parseFloat(v); return pct > 30; },
        warn: v => { const pct = parseFloat(v); return pct > 20 && pct <= 30; } },
    { key: 'low_lp',       pattern: /LP\s*Ratio\s*\[([^\]]+)\]/i,
        crit: v => { const pct = parseFloat(v); return pct < 5; },
        warn: v => { const pct = parseFloat(v); return pct >= 5 && pct < 10; } },
    { key: 'dex_unpaid',   pattern: /DEX\s*\[([^\]]+)\]/i,               crit: v => v.toUpperCase().includes('NOT PAID') },
];

// ── Safety thresholds ──
const SAFETY = {
    MIN_LP_RATIO: 0.08,         // 8% LP/MC minimum
    MAX_TOP10_HOLDERS: 0.30,    // 30% max concentration
    MIN_LIQUIDITY_USD: 5000,    // $5k minimum liquidity
    MIN_HOLDER_COUNT: 50,       // at least 50 holders
};

const RED_FLAGS = [
    { key: 'mintable',     label: 'Mintable — dev can dilute supply at will',                    weight: 40 },
    { key: 'mutable',      label: 'Metadata mutable — name/symbol can change anytime',           weight: 25 },
    { key: 'top10_heavy',  label: 'Top 10 holders concentrated — coordinated dump risk',         weight: 30 },
    { key: 'low_lp',       label: 'LP ratio too low — severe slippage on exit',                  weight: 20 },
    { key: 'low_holders',  label: 'Too few holders — likely bundled or farmed',                  weight: 15 },
    { key: 'dex_unpaid',   label: 'DEX not paid — low legitimacy signal',                        weight: 10 },
    { key: 'no_data',      label: 'No DEX data — token may not exist or has no pairs',           weight: 30 },
    { key: 'new_token',    label: 'Token too new — insufficient history to assess',              weight: 15 },
];

// ── Public API ──

/**
 * @typedef {Object} TokenVerdict
 * @property {string} address
 * @property {'PASS'|'WARN'|'REJECT'} verdict
 * @property {Object[]} flags — {key, label, weight, source} for each triggered flag
 * @property {number} score — 0-100
 * @property {Object} dex — DEX Screener summary
 * @property {Object} card — parsed TokenScan card data
 * @property {Object} [rugcheck] — RugCheck API data
 * @property {string} summary
 */

export async function scanMessage(text, { fetch: _fetch = fetch } = {}) {
    const links = extractLinks(text);
    if (links.length === 0) return null;

    // Parse TokenScan cards from the full message text (once, shared across links)
    const cards = parseTokenScanCards(text);

    const results = [];
    for (const [i, address] of links.entries()) {
        const card = cards[i] || null;
        const verdict = await evaluateToken(address, card, { fetch: _fetch });
        results.push(verdict);
    }
    return results;
}

export async function auditMessage(text, { fetch: _fetch = fetch } = {}) {
    const links = extractLinks(text);
    if (links.length === 0) return { found: false, links: [], message: 'No TokenScan links found.' };

    const cards = parseTokenScanCards(text);

    const audits = [];
    for (const [i, address] of links.entries()) {
        const card = cards[i] || null;
        const dex = await fetchDexScreener(address, { fetch: _fetch });
        const hasAuditData = card && (card.flags.length > 0 || card.warnings.length > 0);
        const rugcheck = !hasAuditData ? await fetchRugCheck(address, { fetch: _fetch }) : null;

        const dexFlags = scoreDex(dex);
        const cardFlags = card ? scoreCard(card) : [];
        const rugFlags = rugcheck ? scoreRugCheck(rugcheck) : [];

        // Merge flags: card data takes precedence, RugCheck fills gaps
        const allFlags = mergeFlags(cardFlags, dexFlags, rugFlags);
        const score = calculateScore(allFlags);

        audits.push({
            address,
            dex_summary: dex ? summarizeDex(dex) : null,
            card_parsed: card,
            rugcheck_available: !!rugcheck,
            flags_triggered: allFlags.map(f => ({ key: f.key, label: RED_FLAGS.find(rf => rf.key === f.key)?.label || f.key, source: f.source })),
            safety_score: score,
            verdict: score >= 70 ? 'PASS' : score >= 40 ? 'WARN' : 'REJECT',
            sources: {
                card: !!card,
                dex: !!dex,
                rugcheck: !!rugcheck,
            },
        });
    }

    return {
        found: true,
        links,
        audits,
        message: `${links.length} token(s). ${audits.filter(a => a.verdict === 'PASS').length} PASS / ${audits.filter(a => a.verdict === 'WARN').length} WARN / ${audits.filter(a => a.verdict === 'REJECT').length} REJECT`,
    };
}

export async function scanBatch(messages, { fetch: _fetch = fetch } = {}) {
    const allResults = [];
    const CONCURRENCY = 5;

    for (let i = 0; i < messages.length; i += CONCURRENCY) {
        const chunk = messages.slice(i, i + CONCURRENCY);
        const settled = await Promise.allSettled(
            chunk.map(msg =>
                scanMessage(typeof msg === 'string' ? msg : msg.text || '', { fetch: _fetch })
            )
        );
        for (const s of settled) {
            if (s.status === 'fulfilled' && s.value) {
                allResults.push(...s.value);
            }
        }
    }

    return allResults;
}

// ── TokenScan card text parser ──

/**
 * Parse TokenScan audit cards from raw Telegram message text.
 * Each card starts with the URL and is followed by audit lines.
 * Returns an array of parsed card objects, one per TokenScan link found.
 */
function parseTokenScanCards(text) {
    // Split text by TokenScan URL boundaries
    const segments = splitByTokenScanUrls(text);
    const cards = [];

    for (const seg of segments) {
        const card = { flags: [], warnings: [], raw: {} };
        let hasAudit = false;

        for (const { key, pattern, crit, warn } of CARD_PARSERS) {
            const m = seg.match(pattern);
            if (!m) continue;
            hasAudit = true;
            const value = m[1].trim();
            card.raw[key] = value;

            const isCrit = crit ? crit(value) : false;
            const isWarn = warn ? warn(value) : false;

            if (isCrit) {
                card.flags.push(key);
            } else if (isWarn) {
                card.warnings.push(key);
            }
        }

        // Also extract numeric fields for display
        const mcMatch = seg.match(/MC:\s*\$?([\d.]+)([KMB]?)/i);
        if (mcMatch) card.mc = parseAmount(mcMatch[1], mcMatch[2]);

        const liqMatch = seg.match(/LIQ:\s*\$?([\d.]+)([KMB]?)/i);
        if (liqMatch) card.liquidity = parseAmount(liqMatch[1], liqMatch[2]);

        const volMatch = seg.match(/VOL:\s*\$?([\d.]+)([KMB]?)/i);
        if (volMatch) card.volume24h = parseAmount(volMatch[1], volMatch[2]);

        const ageMatch = seg.match(/🌱(\d+)d/);
        if (ageMatch) card.ageDays = parseInt(ageMatch[1]);

        const holdersMatch = seg.match(/HLD:\s*([\d,]+)/i);
        if (holdersMatch) card.holders = parseInt(holdersMatch[1].replace(/,/g, ''));

        const usdMatch = seg.match(/USD:\s*([\d.]+)/);
        if (usdMatch) card.priceUsd = parseFloat(usdMatch[1]);

        cards.push(hasAudit ? card : { flags: [], warnings: [], raw: {} });
    }

    return cards;
}

function splitByTokenScanUrls(text) {
    const segments = [];
    let lastIndex = 0;
    let m;

    TOKENSCAN_URL_PATTERN.lastIndex = 0;
    const urls = [];
    while ((m = TOKENSCAN_URL_PATTERN.exec(text)) !== null) {
        urls.push({ index: m.index, end: m.index + m[0].length });
    }
    TOKENSCAN_URL_PATTERN.lastIndex = 0;

    for (let i = 0; i < urls.length; i++) {
        const start = urls[i].end;
        const end = i + 1 < urls.length ? urls[i + 1].index : text.length;
        segments.push(text.substring(start, end));
    }

    if (segments.length === 0 && urls.length === 0) {
        // No URLs found but try parsing the text as-is (for manual audit paste)
        segments.push(text);
    }

    return segments;
}

function parseAmount(num, suffix) {
    const n = parseFloat(num);
    switch (suffix?.toUpperCase()) {
        case 'B': return n * 1e9;
        case 'M': return n * 1e6;
        case 'K': return n * 1e3;
        default: return n;
    }
}

// ── Scoring ──

function scoreDex(dex) {
    const flags = [];
    if (!dex) {
        flags.push({ key: 'no_data', source: 'dex' });
        return flags;
    }

    const pair = findBestPair(dex);
    if (!pair) {
        flags.push({ key: 'no_data', source: 'dex' });
        return flags;
    }

    const mc = pair.marketCap || pair.fdv || 0;
    const liq = pair.liquidity?.usd || 0;

    if (liq < SAFETY.MIN_LIQUIDITY_USD) {
        flags.push({ key: 'low_lp', source: 'dex' });
    }
    if (mc > 0 && liq / mc < SAFETY.MIN_LP_RATIO) {
        flags.push({ key: 'low_lp', source: 'dex' });
    }
    if (pair.pairCreatedAt) {
        const ageH = (Date.now() - pair.pairCreatedAt) / (1000 * 60 * 60);
        if (ageH < 2) flags.push({ key: 'new_token', source: 'dex' });
    }
    if (mc > 0 && mc < 10000 && pair.txns?.h24?.buys < 10) {
        flags.push({ key: 'low_holders', source: 'dex' });
    }

    return flags;
}

function scoreCard(card) {
    const flags = [];
    for (const f of card.flags) {
        flags.push({ key: f, source: 'card' });
    }
    // Warnings from card only count if not already flagged by DEX
    for (const w of card.warnings) {
        flags.push({ key: w, source: 'card', warn: true });
    }
    return flags;
}

function scoreRugCheck(rc) {
    const flags = [];
    if (!rc) return flags;

    if (rc.token?.mintAuthority) {
        flags.push({ key: 'mintable', source: 'rugcheck' });
    }
    if (rc.tokenMeta?.mutable) {
        flags.push({ key: 'mutable', source: 'rugcheck' });
    }
    if (rc.topHolders && rc.topHolders.length >= 10) {
        const top10Pct = rc.topHolders.slice(0, 10).reduce((s, h) => s + (h.pct || 0), 0);
        if (top10Pct > SAFETY.MAX_TOP10_HOLDERS * 100) {
            flags.push({ key: 'top10_heavy', source: 'rugcheck' });
        }
    }
    if (rc.risks && rc.risks.length > 0) {
        for (const risk of rc.risks) {
            if (risk.name?.toLowerCase().includes('lp')) {
                flags.push({ key: 'low_lp', source: 'rugcheck' });
            }
        }
    }
    const marketCount = rc.markets?.length || 0;
    if (marketCount === 0) {
        flags.push({ key: 'dex_unpaid', source: 'rugcheck' });
    }

    return flags;
}

function mergeFlags(cardFlags, dexFlags, rugFlags) {
    const seen = new Set();
    const merged = [];

    // Card takes priority (most trusted source)
    for (const f of [...cardFlags, ...dexFlags, ...rugFlags]) {
        if (seen.has(f.key)) continue;
        seen.add(f.key);
        merged.push(f);
    }

    return merged;
}

function calculateScore(flags) {
    let score = 100;
    for (const f of flags) {
        const def = RED_FLAGS.find(rf => rf.key === f.key);
        const weight = def?.weight || 15;
        // Warning-level flags deduct half
        score -= f.warn ? weight / 2 : weight;
    }
    return Math.max(0, score);
}

// ── API clients ──

async function fetchDexScreener(address, { fetch: _fetch = fetch } = {}) {
    try {
        const resp = await _fetch(`${DEXSCREENER_BASE}/${address}`, {
            signal: AbortSignal.timeout(8000),
        });
        if (!resp.ok) {
            console.warn(`[alpha-parser] DEX Screener returned ${resp.status} for ${address}`);
            return null;
        }
        return await resp.json();
    } catch (err) {
        console.warn(`[alpha-parser] DEX Screener fetch failed for ${address}: ${err?.message ?? err}`);
        return null;
    }
}

async function fetchRugCheck(address, { fetch: _fetch = fetch } = {}) {
    try {
        const resp = await _fetch(`${RUGCHECK_BASE}/${address}/report`, {
            signal: AbortSignal.timeout(8000),
        });
        if (!resp.ok) {
            console.warn(`[alpha-parser] RugCheck returned ${resp.status} for ${address}`);
            return null;
        }
        return await resp.json();
    } catch (err) {
        console.warn(`[alpha-parser] RugCheck fetch failed for ${address}: ${err?.message ?? err}`);
        return null;
    }
}

// ── Helpers ──

function extractLinks(text) {
    const addresses = new Set();
    let m;
    while ((m = TOKENSCAN_URL_PATTERN.exec(text)) !== null) {
        addresses.add(m[1]);
    }
    TOKENSCAN_URL_PATTERN.lastIndex = 0;
    return [...addresses];
}

async function evaluateToken(address, card, { fetch: _fetch = fetch } = {}) {
    const dex = await fetchDexScreener(address, { fetch: _fetch });
    const hasCardAudit = card && (card.flags.length > 0 || card.warnings.length > 0);
    const rugcheck = !hasCardAudit ? await fetchRugCheck(address, { fetch: _fetch }) : null;

    const dexFlags = scoreDex(dex);
    const cardFlags = card ? scoreCard(card) : [];
    const rugFlags = rugcheck ? scoreRugCheck(rugcheck) : [];

    const allFlags = mergeFlags(cardFlags, dexFlags, rugFlags);
    const score = calculateScore(allFlags);

    const verdict = score >= 70 ? 'PASS' : score >= 40 ? 'WARN' : 'REJECT';

    return {
        address,
        verdict,
        flags: allFlags.map(f => ({ key: f.key, label: RED_FLAGS.find(rf => rf.key === f.key)?.label || f.key, weight: RED_FLAGS.find(rf => rf.key === f.key)?.weight || 15, source: f.source })),
        score,
        dex: dex ? summarizeDex(dex) : null,
        card: card || null,
        rugcheck: rugcheck ? summarizeRugCheck(rugcheck) : null,
        summary: renderSummary(verdict, score, allFlags, dex),
    };
}

function formatUSD(value) {
    if (value == null || isNaN(value)) return 'N/A';
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
    if (value >= 1) return `$${value.toFixed(2)}`;
    return `$${value.toFixed(6)}`;
}

function findBestPair(dex) {
    if (!dex?.pairs?.length) return null;
    let best = dex.pairs[0];
    for (const p of dex.pairs) {
        if ((p.liquidity?.usd || 0) > (best.liquidity?.usd || 0)) best = p;
    }
    return best;
}

function summarizeDex(dex) {
    const pair = findBestPair(dex);
    if (!pair) return null;
    return {
        price: pair.priceUsd ? `$${Number(pair.priceUsd).toFixed(6)}` : 'N/A',
        mc: formatUSD(pair.fdv || pair.marketCap),
        liquidity: formatUSD(pair.liquidity?.usd),
        lpRatio: pair.fdv && pair.liquidity?.usd ? `${((pair.liquidity.usd / pair.fdv) * 100).toFixed(1)}%` : pair.marketCap && pair.liquidity?.usd ? `${((pair.liquidity.usd / pair.marketCap) * 100).toFixed(1)}%` : 'N/A',
        age: pair.pairCreatedAt ? `${((Date.now() - pair.pairCreatedAt) / (1000 * 60 * 60)).toFixed(0)}h` : 'N/A',
        volume24h: formatUSD(pair.volume?.h24),
        priceChange24h: pair.priceChange?.h24 ? `${pair.priceChange.h24.toFixed(1)}%` : 'N/A',
        buys24h: pair.txns?.h24?.buys || 0,
        sells24h: pair.txns?.h24?.sells || 0,
        pairAddress: pair.pairAddress,
        dexId: pair.dexId,
    };
}

function summarizeRugCheck(rc) {
    if (!rc) return null;
    const top10Pct = rc.topHolders?.slice(0, 10).reduce((s, h) => s + (h.pct || 0), 0) || 0;
    return {
        mintAuthority: rc.token?.mintAuthority || null,
        mutable: rc.tokenMeta?.mutable,
        top10Pct: `${top10Pct.toFixed(1)}%`,
        risks: rc.risks?.map(r => r.name) || [],
        markets: rc.markets?.length || 0,
    };
}

function renderSummary(verdict, score, flags, dex) {
    const emoji = verdict === 'PASS' ? '🟢' : verdict === 'WARN' ? '🟡' : '🔴';
    const summary = dex ? `${dex.price || '?'} | MC: ${dex.mc || '?'} | Liq: ${dex.liquidity || '?'}` : 'No DEX data';
    return `${emoji} ${verdict} (${score}/100) — ${summary} [${flags.length} flags]`;
}

export { extractLinks, parseTokenScanCards, RED_FLAGS, SAFETY };
