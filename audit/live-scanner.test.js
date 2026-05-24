/**
 * Live Scanner — unit tests (node:test runner)
 * Run: node --test audit/live-scanner.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    parseArgs,
    fetchSymbolData,
    scanSymbol,
    renderScanTable,
    renderVerdicts,
    estimateBasePrice,
} from './live-scanner.mjs';
import { generateSyntheticCandles } from './trade.mjs';

// ─── 1. parseArgs ──────────────────────────────────────────────────────────────

describe('parseArgs', () => {
    it('returns defaults with no args', () => {
        const args = parseArgs([]);
        assert.ok(Array.isArray(args.symbols));
        assert.ok(args.symbols.length >= 3, 'has default symbols');
        assert.equal(args.timeframe, '1h');
        assert.equal(args.days, 30);
        assert.equal(args.top, 3);
        assert.equal(args.minScore, 1.0);
        assert.equal(args.full, false);
        assert.equal(args.live, false);
        assert.equal(args.verbose, false);
        assert.equal(args.preset, true);
    });

    it('parses --symbols', () => {
        const args = parseArgs(['--symbols', 'BTC-USD,ETH-USD']);
        assert.deepEqual(args.symbols, ['BTC-USD', 'ETH-USD']);
    });

    it('parses -s short flag', () => {
        const args = parseArgs(['-s', 'SOL-USD,MBT']);
        assert.deepEqual(args.symbols, ['SOL-USD', 'MBT']);
    });

    it('parses --tf', () => {
        const args = parseArgs(['--tf', '15m']);
        assert.equal(args.timeframe, '15m');
    });

    it('parses --days', () => {
        const args = parseArgs(['--days', '60']);
        assert.equal(args.days, 60);
    });

    it('parses --top', () => {
        const args = parseArgs(['--top', '5']);
        assert.equal(args.top, 5);
    });

    it('parses --min-score', () => {
        const args = parseArgs(['--min-score', '1.5']);
        assert.equal(args.minScore, 1.5);
    });

    it('parses boolean flags', () => {
        const args = parseArgs(['--full', '--live', '--verbose']);
        assert.equal(args.full, true);
        assert.equal(args.live, true);
        assert.equal(args.verbose, true);
    });

    it('parses --no-preset', () => {
        const args = parseArgs(['--no-preset']);
        assert.equal(args.preset, false);
    });

    it('parses --seed', () => {
        const args = parseArgs(['--seed', '12345']);
        assert.equal(args.seed, 12345);
    });

    it('parses --cache-dir', () => {
        const args = parseArgs(['--cache-dir', '/tmp/test-cache']);
        assert.equal(args.cacheDir, '/tmp/test-cache');
    });
});

// ─── 2. estimateBasePrice ──────────────────────────────────────────────────────

describe('estimateBasePrice', () => {
    it('returns 75000 for BTC', () => {
        assert.equal(estimateBasePrice('BTC-USD'), 75000);
    });

    it('returns 75000 for MBT', () => {
        assert.equal(estimateBasePrice('MBT'), 75000);
    });

    it('returns 3500 for ETH', () => {
        assert.equal(estimateBasePrice('ETH-USD'), 3500);
    });

    it('returns 150 for SOL', () => {
        assert.equal(estimateBasePrice('SOL-USD'), 150);
    });

    it('returns 100 for unknown symbol', () => {
        assert.equal(estimateBasePrice('UNKNOWN-COIN'), 100);
    });

    it('is case-insensitive', () => {
        assert.equal(estimateBasePrice('btc-usd'), 75000);
        assert.equal(estimateBasePrice('eth-usd'), 3500);
    });
});

// ─── 3. fetchSymbolData ────────────────────────────────────────────────────────

describe('fetchSymbolData', () => {
    it('generates synthetic data for a symbol', async () => {
        const args = { timeframe: '1h', days: 5, live: false, seed: 42 };
        const candles = await fetchSymbolData('BTC-USD', args);
        assert.ok(Array.isArray(candles));
        assert.ok(candles.length > 0);
        assert.equal(typeof candles[0].close, 'number');
    });

    it('produces deterministic data with same seed', async () => {
        const args = { timeframe: '1h', days: 5, live: false, seed: 99 };
        const a = await fetchSymbolData('TEST-USD', args);
        const b = await fetchSymbolData('TEST-USD', args);
        assert.equal(a[0].close, b[0].close);
        assert.equal(a.length, b.length);
    });

    it('works for different timeframes', async () => {
        for (const tf of ['15m', '1h', '4h']) {
            const args = { timeframe: tf, days: 3, live: false, seed: 1 };
            const candles = await fetchSymbolData('BTC-USD', args);
            assert.ok(candles.length > 0, `${tf} produces candles`);
        }
    });

    it('handles unknown symbols', async () => {
        const args = { timeframe: '1h', days: 3, live: false, seed: 7 };
        const candles = await fetchSymbolData('RANDOM-COIN-999', args);
        assert.ok(Array.isArray(candles));
        assert.ok(candles.length > 0);
    });
});

// ─── 4. scanSymbol ─────────────────────────────────────────────────────────────

describe('scanSymbol', () => {
    it('returns result with expected shape', async () => {
        const candles = generateSyntheticCandles(300, { seed: 42, basePrice: 75000, volatility: 0.012 });
        const args = { timeframe: '1h', days: 10, preset: true };
        const result = await scanSymbol('BTC-USD', candles, args);

        assert.equal(result.symbol, 'BTC-USD');
        assert.ok(typeof result.bestScore === 'number', 'has bestScore');
        assert.ok(result.bestScore >= 0, 'score non-negative');
        assert.ok(result.bestParams, 'has bestParams');
        assert.ok(result.candleCount > 0, 'has candleCount');
    });

    it('produces a score > 0 with enough data and low minConfidence', async () => {
        const candles = generateSyntheticCandles(500, { seed: 42, basePrice: 75000, volatility: 0.012 });
        const args = { timeframe: '1h', days: 20, preset: false };
        const result = await scanSymbol('BTC-USD', candles, args);
        assert.ok(result.bestScore > 0, `expected positive score, got ${result.bestScore}`);
    });

    it('works with preset disabled', async () => {
        const candles = generateSyntheticCandles(300, { seed: 42, basePrice: 100, volatility: 0.012 });
        const args = { timeframe: '1h', days: 10, preset: false };
        const result = await scanSymbol('TEST-USD', candles, args);
        assert.equal(result.symbol, 'TEST-USD');
    });

    it('handles MBT symbol with high base price', async () => {
        const candles = generateSyntheticCandles(300, { seed: 42, basePrice: 75000, volatility: 0.012 });
        const args = { timeframe: '1h', days: 10, preset: true };
        const result = await scanSymbol('MBT', candles, args);
        assert.equal(result.symbol, 'MBT');
        assert.ok(typeof result.bestScore === 'number');
    });
});

// ─── 5. renderScanTable ────────────────────────────────────────────────────────

describe('renderScanTable', () => {
    const sampleResults = [
        {
            symbol: 'BTC-USD', bestScore: 15.5,
            bestStats: { totalTrades: 20, winRate: 0.75, profitFactor: 3.2, totalReturnPct: 12.5 },
            bestParams: { stopLossPct: 0.02, takeProfitPct: 0.04 },
        },
        {
            symbol: 'ETH-USD', bestScore: 5.2,
            bestStats: { totalTrades: 15, winRate: 0.6, profitFactor: 1.8, totalReturnPct: 5.0 },
            bestParams: { stopLossPct: 0.03, takeProfitPct: 0.06 },
        },
        {
            symbol: 'SOL-USD', bestScore: 0.8,
            bestStats: { totalTrades: 8, winRate: 0.25, profitFactor: 0.5, totalReturnPct: -2.0 },
            bestParams: { stopLossPct: 0.01, takeProfitPct: 0.02 },
        },
    ];

    it('renders a table with expected columns', () => {
        const table = renderScanTable(sampleResults, 3);
        assert.ok(table.includes('RANK'), 'has rank column');
        assert.ok(table.includes('SYMBOL'), 'has symbol column');
        assert.ok(table.includes('SCORE'), 'has score column');
        assert.ok(table.includes('TRADES'), 'has trades column');
        assert.ok(table.includes('WIN%'), 'has win% column');
        assert.ok(table.includes('PF'), 'has pf column');
        assert.ok(table.includes('BTC-USD'), 'includes first symbol');
        assert.ok(table.includes('ETH-USD'), 'includes second symbol');
        assert.ok(table.includes('SOL-USD'), 'includes third symbol');
    });

    it('handles empty results', () => {
        const table = renderScanTable([], 3);
        assert.ok(typeof table === 'string');
        assert.ok(table.length > 0);
    });

    it('handles null/undefined gracefully', () => {
        assert.doesNotThrow(() => renderScanTable(null, 3));
        assert.doesNotThrow(() => renderScanTable(undefined, 3));
    });

    it('shows only topN when fewer than total', () => {
        const table = renderScanTable(sampleResults, 2);
        assert.ok(table.includes('Showing 2 of 3'));
    });
});

// ─── 6. renderVerdicts ─────────────────────────────────────────────────────────

describe('renderVerdicts', () => {
    it('renders verdicts for multiple symbols', () => {
        const verdicts = [
            { symbol: 'BTC-USD', verdict: 'GRAIL', score: 15.5 },
            { symbol: 'ETH-USD', verdict: 'PROFITABLE', score: 5.2 },
            { symbol: 'DOGE-USD', verdict: 'NOT YIELDING', score: 0.3 },
        ];
        const output = renderVerdicts(verdicts);
        assert.ok(output.includes('GRAIL VERDICTS'), 'has header');
        assert.ok(output.includes('BTC-USD'), 'includes BTC');
        assert.ok(output.includes('GRAIL'), 'includes GRAIL');
        assert.ok(output.includes('PROFITABLE'), 'includes PROFITABLE');
    });

    it('returns empty string for empty array', () => {
        assert.equal(renderVerdicts([]), '');
    });

    it('returns empty string for null/undefined', () => {
        assert.equal(renderVerdicts(null), '');
        assert.equal(renderVerdicts(undefined), '');
    });
});

// ─── 7. Edge Cases ─────────────────────────────────────────────────────────────

describe('live-scanner edge cases', () => {
    it('scanSymbol handles very few candles', async () => {
        const candles = generateSyntheticCandles(60, { seed: 1, volatility: 0.01 });
        const args = { timeframe: '1h', days: 3, preset: false };
        const result = await scanSymbol('TINY-USD', candles, args);
        assert.ok(result.symbol === 'TINY-USD');
        // Even with few candles, it should not throw
    });

    it('parseArgs handles extra whitespace in symbols', () => {
        const args = parseArgs(['--symbols', ' BTC-USD , ETH-USD , SOL-USD ']);
        assert.deepEqual(args.symbols, ['BTC-USD', 'ETH-USD', 'SOL-USD']);
    });

    it('parseArgs ignores unknown flags', () => {
        assert.doesNotThrow(() => parseArgs(['--unknown-flag', 'value']));
    });
});
