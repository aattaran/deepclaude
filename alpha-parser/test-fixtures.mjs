/**
 * Alpha Parser test fixtures — real TG message patterns and expected verdicts.
 * Feed these to the engine to validate parser accuracy.
 */

// Real Solana base58 addresses (match [1-9A-HJ-NP-Za-km-z]{32,44})
const ADDR1 = '98sMhvDwXj1RQi5c5Mndm3vPe9cBqPrbLaufMXFNMh5g'; // $HYPE
const ADDR2 = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'; // USDC
const ADDR3 = 'So11111111111111111111111111111111111111112'; // Wrapped SOL

// $HYPE — known garbage: mintable, mutable, concentrated, LP weak, DEX unpaid
export const HYPE_CARD = `HYPE (https://t.me/tokenscan?start=scan-${ADDR1}) ($HYPE)
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

export const HYPE_EXPECT = {
    verdict: 'REJECT',
    flags: ['mintable', 'mutable', 'top10_heavy', 'low_lp', 'dex_unpaid'],
    cardMc: 54_960_000,
    cardLiq: 3_440_000,
    minScore: 0,
    maxScore: 10,
};

// Clean token scenario — no audit flags, good DEX data
// Using a real base58-format address
export const CLEAN_CARD = `check this gem 💎 (https://t.me/tokenscan?start=scan-EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v)
📊 Token Stats
➰ MC:   $12.5M
➰ LIQ:  $2.1M
➰ VOL:  $1.8M (24h)
➰ HLD:  8,420

✅ Audit 🟢🟢
✅ Mintable [No]
✅ Token Data Mutable [No]
✅ LP Ratio [16.8%]
✅ DEX [PAID]
✅ Top 10 Holders [18.2%]`;

export const CLEAN_EXPECT = {
    verdict: 'PASS',  // Card shows clean audit — no card flags
    minScore: 60,     // DEX may flag LP ratio for high-cap tokens like USDC
    cardMc: 12_500_000,
    cardLiq: 2_100_000,
    cardClean: true,  // all card audit flags show "No"/safe values
};

// Multi-link: two tokens in one message
export const MULTI_LINK = `two plays:
First: https://t.me/tokenscan?start=scan-${ADDR1} ($HYPE)
❌ Audit 🟥🟥
❌ Mintable [Yes]
❌ DEX [NOT PAID]

Second: https://t.me/tokenscan?start=scan-${ADDR2}
✅ Audit 🟢🟢
✅ Mintable [No]
✅ DEX [PAID]`;

export const MULTI_EXPECT = {
    linkCount: 2,
    firstVerdict: 'WARN',   // 2 flags only (mintable + dex_unpaid) = 50/100 = WARN
    secondVerdict: 'PASS',
};

// Emoji variants — ⚠️ (U+26A0 + U+FE0F) vs ⚠ (U+26A0 plain)
export const EMOJI_VARIANTS = `token: https://t.me/tokenscan?start=scan-${ADDR2}
⚠️ Mintable [Yes]
⚠ Token Data Mutable [Yes]
⚠️ LP Ratio [7.5%]
⚠ DEX [NOT PAID]
⚠️ Top 10 Holders [35%]`;

export const EMOJI_EXPECT = {
    verdict: 'REJECT',
    flags: ['mintable', 'mutable', 'low_lp', 'dex_unpaid', 'top10_heavy'],
    minScore: 0,
    maxScore: 10,
};

// No audit section — only stats, no audit lines
export const NO_AUDIT_CARD = `https://t.me/tokenscan?start=scan-${ADDR3}
📊 Token Stats
➰ MC:   $5.2M
➰ LIQ:  $800K
➰ VOL:  $450K (24h)
➰ HLD:  3,200`;

export const NO_AUDIT_EXPECT = {
    verdict: 'PASS', // DEX data alone would PASS unless LP ratio is terrible
    cardParsed: false, // no audit lines to parse
};

// Numeric edge cases: commas in holders, K/M suffixes
export const NUMERIC_EDGES = `https://t.me/tokenscan?start=scan-${ADDR2}
📊 Token Stats
➰ MC:   $1.2K
➰ LIQ:  $300
➰ HLD:  1,234

❌ Audit
❌ Top 10 Holders [85%]
⚠️ LP Ratio [3.2%]`;

export const NUMERIC_EXPECT = {
    verdict: 'WARN',   // 2 flags (top10_heavy 30 + low_lp 20) = 50/100 = WARN
    flags: ['top10_heavy', 'low_lp'],
    cardMc: 1_200,
    cardLiq: 300,
};

// False positive test — message mentions tokenscan but no link
export const NO_LINK = `yo check tokenscan for this one, it's pumping rn fr fr
MC is huge, LP is locked, dev is based`;

export const NO_LINK_EXPECT = {
    found: false,
};

// ── Mock fetch for offline testing ──
// Returns deterministic DEX Screener and RugCheck responses matching
// the real API JSON structure, so no network calls are needed.
export function createMockFetch() {
    const dexResponses = {
        [ADDR1]: {
            pairs: [{
                chainId: 'solana',
                dexId: 'raydium',
                pairAddress: 'mockPair1',
                priceUsd: '0.05883',
                marketCap: 54960000,
                fdv: 54960000,
                liquidity: { usd: 55000000 },
                volume: { h24: 8190000 },
                priceChange: { h24: 7.8 },
                txns: { h24: { buys: 442, sells: 410 } },
                pairCreatedAt: Date.now() - (240 * 24 * 60 * 60 * 1000),
            }],
        },
        [ADDR2]: {
            pairs: [{
                chainId: 'solana',
                dexId: 'raydium',
                pairAddress: 'mockPair2',
                priceUsd: '1.00',
                marketCap: 50000000,
                fdv: 50000000,
                liquidity: { usd: 10000000 },
                volume: { h24: 50000000 },
                priceChange: { h24: 0.1 },
                txns: { h24: { buys: 5000, sells: 4800 } },
                pairCreatedAt: Date.now() - (730 * 24 * 60 * 60 * 1000),
            }],
        },
        [ADDR3]: {
            pairs: [{
                chainId: 'solana',
                dexId: 'jupiter',
                pairAddress: 'mockPair3',
                priceUsd: '150.00',
                marketCap: 10000000000,
                fdv: 10000000000,
                liquidity: { usd: 30000000 },
                volume: { h24: 100000000 },
                priceChange: { h24: -2.1 },
                txns: { h24: { buys: 15000, sells: 14500 } },
                pairCreatedAt: Date.now() - (1825 * 24 * 60 * 60 * 1000),
            }],
        },
    };

    const rugcheckResponses = {
        [ADDR3]: {
            token: { mintAuthority: null },
            tokenMeta: { mutable: false },
            topHolders: [
                { pct: 1.2 }, { pct: 0.8 }, { pct: 0.7 }, { pct: 0.6 }, { pct: 0.5 },
                { pct: 0.5 }, { pct: 0.4 }, { pct: 0.4 }, { pct: 0.3 }, { pct: 0.3 },
            ],
            risks: [],
            markets: [{ name: 'raydium' }, { name: 'orca' }],
        },
    };

    return async function mockFetch(url) {
        // DEX Screener API
        const dexMatch = typeof url === 'string' ? url.match(/dexscreener\.com\/latest\/dex\/tokens\/(\w+)/) : null;
        if (dexMatch) {
            const address = dexMatch[1];
            const data = dexResponses[address];
            return {
                ok: !!data,
                json: async () => data,
            };
        }

        // RugCheck API
        const rcMatch = typeof url === 'string' ? url.match(/rugcheck\.xyz\/v1\/tokens\/(\w+)\/report/) : null;
        if (rcMatch) {
            const address = rcMatch[1];
            const data = rugcheckResponses[address];
            return {
                ok: !!data,
                json: async () => data,
            };
        }

        // Unknown URL → 404
        return { ok: false, json: async () => null };
    };
}

// All fixtures in one array for easy test iteration
export const ALL_FIXTURES = [
    { name: 'HYPE — known garbage (5 flags)', input: HYPE_CARD, expect: HYPE_EXPECT },
    { name: 'Clean token — all green', input: CLEAN_CARD, expect: CLEAN_EXPECT },
    { name: 'Multi-link — two tokens', input: MULTI_LINK, expect: MULTI_EXPECT },
    { name: 'Emoji variants — ⚠️ vs ⚠', input: EMOJI_VARIANTS, expect: EMOJI_EXPECT },
    { name: 'No audit section — stats only', input: NO_AUDIT_CARD, expect: NO_AUDIT_EXPECT },
    { name: 'Numeric edges — K suffix, commas', input: NUMERIC_EDGES, expect: NUMERIC_EXPECT },
    { name: 'False positive — no link', input: NO_LINK, expect: NO_LINK_EXPECT },
];
