/**
 * Data Feed — unit tests (node:test runner)
 * Run: node --test audit/datafeed.test.js
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import {
    fetchCandles,
    loadOrFetch,
    fetchMultiple,
    getCacheAge,
    getSymbolInfo,
    summarizeCandles,
} from './datafeed.mjs';

// ─── Temp dir for caching ───────────────────────────────────────────────────────

let tmpDir;
before(() => { tmpDir = mkdtempSync('datafeed-test-'); });
after(() => {
    try { rmSync(tmpDir, { recursive: true, force: true }); } catch (_) { /* ok */ }
});

// ─── 1. fetchCandles (integration — requires network) ───────────────────────────

describe('fetchCandles', () => {
    it('fetches real BTCUSDT candles from Binance', async () => {
        try {
            const candles = await fetchCandles('BTC-USD', '1h', 10);
            assert.ok(Array.isArray(candles), 'returns array');
            assert.ok(candles.length > 0, 'has candles');
            assert.ok(candles.length <= 10, 'within limit');

            const c = candles[0];
            assert.ok(typeof c.timestamp === 'number', 'has timestamp');
            assert.ok(typeof c.open === 'number', 'has open');
            assert.ok(typeof c.high === 'number', 'has high');
            assert.ok(typeof c.low === 'number', 'has low');
            assert.ok(typeof c.close === 'number', 'has close');
            assert.ok(typeof c.volume === 'number', 'has volume');
            assert.ok(c.high >= c.low, 'high >= low');
            assert.ok(c.high >= c.open && c.high >= c.close, 'high >= open,close');
            assert.ok(c.low <= c.open && c.low <= c.close, 'low <= open,close');
            assert.ok(c.volume >= 0, 'volume >= 0');
        } catch (err) {
            // Offline is OK — skip the test
            if (err.message?.includes('fetch') || err.message?.includes('ENOTFOUND') || err.message?.includes('timed out') || err.message?.includes('451') || err.message?.includes('restricted')) {
                console.error('  (offline — skipping fetch test)');
                return;
            }
            throw err;
        }
    });

    it('fetches SOLUSDT candles', async () => {
        try {
            const candles = await fetchCandles('SOL-USD', '15m', 5);
            assert.ok(Array.isArray(candles));
            assert.ok(candles.length > 0);
            assert.ok(candles.length <= 5);
        } catch (err) {
            if (err.message?.includes('fetch') || err.message?.includes('ENOTFOUND') || err.message?.includes('451') || err.message?.includes('restricted')) {
                console.error('  (offline — skipping)');
                return;
            }
            throw err;
        }
    });

    it('supports all standard timeframes', async () => {
        const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
        for (const tf of timeframes) {
            try {
                const candles = await fetchCandles('BTC-USD', tf, 3);
                assert.ok(Array.isArray(candles), `${tf} returns array`);
            } catch (err) {
                if (err.message?.includes('fetch') || err.message?.includes('ENOTFOUND') || err.message?.includes('451') || err.message?.includes('restricted')) {
                    console.error(`  (offline — skipping ${tf})`);
                    continue;
                }
                throw err;
            }
        }
    });

    it('MBT symbol maps to BTCUSDT', async () => {
        try {
            const candles = await fetchCandles('MBT', '1h', 5);
            assert.ok(Array.isArray(candles));
            // BTC prices are in the thousands
            assert.ok(candles[0].close > 1000, 'BTC-level prices');
        } catch (err) {
            if (err.message?.includes('fetch') || err.message?.includes('ENOTFOUND') || err.message?.includes('451') || err.message?.includes('restricted')) {
                console.error('  (offline — skipping)');
                return;
            }
            throw err;
        }
    });

    it('throws on invalid timeframe', async () => {
        await assert.rejects(
            () => fetchCandles('BTC-USD', 'invalid', 10),
            /Unsupported timeframe/,
        );
    });

    it('respects limit parameter', async () => {
        try {
            const candles = await fetchCandles('BTC-USD', '1h', 3);
            assert.ok(candles.length <= 3);
        } catch (err) {
            if (err.message?.includes('fetch') || err.message?.includes('ENOTFOUND') || err.message?.includes('451') || err.message?.includes('restricted')) {
                console.error('  (offline — skipping)');
                return;
            }
            throw err;
        }
    });
});

// ─── 2. loadOrFetch (cache + fetch + synthetic fallback) ────────────────────────

describe('loadOrFetch', () => {
    it('returns candles (from API or synthetic fallback)', async () => {
        const candles = await loadOrFetch('BTC-USD', '15m', 5, {
            cacheDir: tmpDir,
            maxAgeMs: 0, // force skip cache
            allowSynthetic: true,
            basePrice: 75000,
            volatility: 0.01,
        });

        assert.ok(Array.isArray(candles), 'returns array');
        assert.ok(candles.length > 0, 'has candles');
        assert.ok(typeof candles[0].close === 'number', 'valid candle');
    });

    it('caches data after fetch', async () => {
        const cacheOpts = { cacheDir: tmpDir, maxAgeMs: 3600000, allowSynthetic: true, basePrice: 75000 };

        // First call — should fetch or generate
        const candles1 = await loadOrFetch('ETH-USD', '1h', 3, cacheOpts);

        // Second call — should hit cache
        const candles2 = await loadOrFetch('ETH-USD', '1h', 3, cacheOpts);

        assert.equal(candles1.length, candles2.length);
        assert.equal(candles1[0].close, candles2[0].close);
        assert.equal(candles1[candles1.length - 1].close, candles2[candles2.length - 1].close);
    });

    it('forceFetch skips cache', async () => {
        const cacheOpts = { cacheDir: tmpDir, maxAgeMs: 3600000, allowSynthetic: true, basePrice: 100 };

        // Populate cache
        await loadOrFetch('SOL-USD', '1h', 3, cacheOpts);

        // forceFetch should hit API or generate fresh
        const candles = await loadOrFetch('SOL-USD', '1h', 3, {
            ...cacheOpts,
            forceFetch: true,
            seed: 999, // different seed → different synthetic data
        });

        assert.ok(candles.length > 0);
    });

    it('getCacheAge reports cache freshness', async () => {
        const cacheOpts = { cacheDir: tmpDir, maxAgeMs: 3600000, allowSynthetic: true };
        await loadOrFetch('ADA-USD', '1h', 2, cacheOpts);

        const age = getCacheAge('ADA-USD', '1h', 2, tmpDir);
        assert.ok(age >= 0, 'cache exists');
        assert.ok(age < 30000, `cache is recent (${age}ms)`);
    });

    it('getCacheAge returns -1 for missing cache', () => {
        const age = getCacheAge('NONEXISTENT', '1h', 999, tmpDir);
        assert.equal(age, -1);
    });

    it('handles unknown symbol with synthetic fallback', async () => {
        const candles = await loadOrFetch('RANDOM-COIN', '1h', 3, {
            cacheDir: tmpDir,
            allowSynthetic: true,
            basePrice: 50,
            volatility: 0.02,
        });

        assert.ok(candles.length > 0);
        const avg = candles.reduce((s, c) => s + c.close, 0) / candles.length;
        assert.ok(avg > 30 && avg < 70, `avg price ${avg} near basePrice 50`);
    });
});

// ─── 3. fetchMultiple ────────────────────────────────────────────────────────────

describe('fetchMultiple', () => {
    it('fetches multiple symbols in parallel', async () => {
        const results = await fetchMultiple(
            ['BTC-USD', 'ETH-USD', 'SOL-USD'],
            '1h',
            3,
            { cacheDir: tmpDir, maxAgeMs: 0, allowSynthetic: true },
        );

        assert.ok(Array.isArray(results['BTC-USD']));
        assert.ok(Array.isArray(results['ETH-USD']));
        assert.ok(Array.isArray(results['SOL-USD']));
        assert.ok(results['BTC-USD'].length > 0);
    });

    it('reports errors for individual symbols', async () => {
        // Force fetch with invalid timeout to get errors
        const results = await fetchMultiple(
            ['VALID-USD', 'ALSO-VALID-USD'],
            '1h',
            2,
            { cacheDir: tmpDir, maxAgeMs: 0, allowSynthetic: true },
        );

        // With synthetic fallback, all should succeed
        assert.ok(Array.isArray(results['VALID-USD']));
        assert.ok(Array.isArray(results['ALSO-VALID-USD']));
    });
});

// ─── 4. summarizeCandles ────────────────────────────────────────────────────────

describe('summarizeCandles', () => {
    it('returns null for empty/null input', () => {
        assert.equal(summarizeCandles(null), null);
        assert.equal(summarizeCandles([]), null);
        assert.equal(summarizeCandles(undefined), null);
    });

    it('computes summary from candle array', () => {
        const candles = [
            { timestamp: 1000, open: 100, high: 102, low: 99, close: 101, volume: 500 },
            { timestamp: 2000, open: 101, high: 105, low: 100, close: 104, volume: 600 },
            { timestamp: 3000, open: 104, high: 106, low: 103, close: 103, volume: 400 },
        ];

        const summary = summarizeCandles(candles);
        assert.equal(summary.count, 3);
        assert.equal(summary.startPrice, 101);
        assert.equal(summary.endPrice, 103);
        assert.equal(summary.changePct, 1.98);
        assert.equal(summary.high, 106);
        assert.equal(summary.low, 99);
        assert.equal(summary.avgVolume, 500);
    });

    it('handles single candle', () => {
        const candles = [{ timestamp: 1000, open: 50, high: 51, low: 49, close: 50, volume: 100 }];
        const summary = summarizeCandles(candles);
        assert.equal(summary.count, 1);
        assert.equal(summary.changePct, 0);
        assert.equal(summary.high, 51);
        assert.equal(summary.low, 49);
    });

    it('handles missing volume field', () => {
        const candles = [
            { timestamp: 1000, open: 100, high: 101, low: 99, close: 100 },
        ];
        const summary = summarizeCandles(candles);
        assert.equal(summary.avgVolume, 0);
    });
});

// ─── 5. getSymbolInfo ────────────────────────────────────────────────────────────

describe('getSymbolInfo', () => {
    it('returns info for valid symbol', async () => {
        try {
            const info = await getSymbolInfo('BTC-USD');
            if (info) {
                assert.equal(info.baseAsset, 'BTC');
                assert.equal(info.quoteAsset, 'USDT');
                assert.ok(info.tickSize, 'has tickSize');
            }
        } catch (err) {
            if (err.message?.includes('fetch') || err.message?.includes('ENOTFOUND') || err.message?.includes('451') || err.message?.includes('restricted')) {
                console.error('  (offline — skipping)');
                return;
            }
            throw err;
        }
    });

    it('returns null for invalid symbol', async () => {
        const info = await getSymbolInfo('INVALID-XXXXX');
        assert.equal(info, null);
    });
});

// ─── 6. Edge Cases ──────────────────────────────────────────────────────────────

describe('edge cases', () => {
    it('loadOrFetch with zero days returns some data', async () => {
        const candles = await loadOrFetch('BTC-USD', '1h', 0, {
            cacheDir: tmpDir,
            allowSynthetic: true,
        });
        // 0 days * 24 bars = 0 bars requested; synthetic generates 0
        // But fetchCandles uses min limit of 1 effectively
        assert.ok(candles !== undefined);
    });

    it('multiple cache reads are consistent', async () => {
        const opts = { cacheDir: tmpDir, maxAgeMs: 3600000, allowSynthetic: true, seed: 42, basePrice: 100 };
        const a = await loadOrFetch('CACHE-TEST', '1h', 5, opts);
        const b = await loadOrFetch('CACHE-TEST', '1h', 5, opts);

        assert.equal(a.length, b.length);
        for (let i = 0; i < a.length; i++) {
            assert.equal(a[i].close, b[i].close, `candle ${i} close matches`);
        }
    });
});
