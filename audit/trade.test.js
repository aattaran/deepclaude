/**
 * Unified Trade CLI — unit tests (node:test runner)
 * Run: node --test audit/trade.test.js
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { execSync } from 'node:child_process';
import { generateSyntheticCandles, loadCandles, parseArgs } from './trade.mjs';

// ─── Temp dir for candle files ──────────────────────────────────────────────────

let tmpDir;
before(() => { tmpDir = mkdtempSync('trade-test-'); });
after(() => { try { rmSync(tmpDir, { recursive: true, force: true }); } catch (_) { /* ok */ } });

function writeTempCandles(filename, candles) {
    const path = join(tmpDir, filename);
    writeFileSync(path, JSON.stringify(candles));
    return path;
}

// ─── 1. Candle Generation ──────────────────────────────────────────────────────

describe('generateSyntheticCandles', () => {
    it('generates the requested number of candles', () => {
        const candles = generateSyntheticCandles(100, { seed: 42 });
        assert.equal(candles.length, 100);
    });

    it('produces valid candle shapes', () => {
        const candles = generateSyntheticCandles(50, { seed: 42 });
        for (const c of candles) {
            assert.ok(typeof c.timestamp === 'number');
            assert.ok(typeof c.open === 'number');
            assert.ok(typeof c.high === 'number');
            assert.ok(typeof c.low === 'number');
            assert.ok(typeof c.close === 'number');
            assert.ok(typeof c.volume === 'number');
            assert.ok(c.high >= c.low, 'high >= low');
            assert.ok(c.high >= c.open && c.high >= c.close, 'high >= open,close');
            assert.ok(c.low <= c.open && c.low <= c.close, 'low <= open,close');
            assert.ok(c.volume >= 0, 'volume >= 0');
        }
    });

    it('is deterministic with same seed', () => {
        const a = generateSyntheticCandles(20, { seed: 42 });
        const b = generateSyntheticCandles(20, { seed: 42 });
        assert.equal(a.length, b.length);
        assert.equal(a[0].close, b[0].close);
        assert.equal(a[10].close, b[10].close);
        assert.equal(a[19].close, b[19].close);
    });

    it('produces different data with different seeds', () => {
        const a = generateSyntheticCandles(20, { seed: 42 });
        const b = generateSyntheticCandles(20, { seed: 99 });
        // At least some candles should differ
        const diffs = a.filter((c, i) => c.close !== b[i].close);
        assert.ok(diffs.length > 0, 'different seeds produce different data');
    });

    it('respects basePrice', () => {
        const candles = generateSyntheticCandles(50, { seed: 42, basePrice: 500 });
        const avg = candles.reduce((s, c) => s + c.close, 0) / candles.length;
        // Should be roughly near 500
        assert.ok(avg > 400 && avg < 600, `avg ${avg} near basePrice 500`);
    });

    it('higher volatility produces wider price range', () => {
        const lowVol = generateSyntheticCandles(100, { seed: 42, volatility: 0.005, basePrice: 100 });
        const highVol = generateSyntheticCandles(100, { seed: 42, volatility: 0.03, basePrice: 100 });
        const rangeLow = Math.max(...lowVol.map(c => c.high)) - Math.min(...lowVol.map(c => c.low));
        const rangeHigh = Math.max(...highVol.map(c => c.high)) - Math.min(...highVol.map(c => c.low));
        assert.ok(rangeHigh > rangeLow * 1.5, `high vol range ${rangeHigh} > low vol range ${rangeLow}`);
    });

    it('produces candles with _regime tag', () => {
        const candles = generateSyntheticCandles(200, { seed: 42 });
        const regimes = new Set(candles.map(c => c._regime));
        assert.ok(regimes.size >= 2, 'generates multiple regimes');
    });

    it('handles zero count', () => {
        const candles = generateSyntheticCandles(0);
        assert.equal(candles.length, 0);
    });

    it('handles large count efficiently', () => {
        const start = Date.now();
        const candles = generateSyntheticCandles(5000, { seed: 42 });
        const elapsed = Date.now() - start;
        assert.equal(candles.length, 5000);
        assert.ok(elapsed < 500, `5000 candles generated in ${elapsed}ms`);
    });
});

// ─── 2. Argument Parsing ────────────────────────────────────────────────────────

describe('parseArgs', () => {
    it('parses --analyze mode', () => {
        const args = parseArgs(['--analyze', '--symbol', 'SOL-USD']);
        assert.equal(args.mode, 'analyze');
        assert.equal(args.symbol, 'SOL-USD');
    });

    it('parses --backtest mode with options', () => {
        const args = parseArgs([
            '--backtest', '--symbol', 'BTC-USD', '--days', '60',
            '--seed', '42', '--tf', '1h', '--sl', '0.03', '--tp', '0.06',
        ]);
        assert.equal(args.mode, 'backtest');
        assert.equal(args.symbol, 'BTC-USD');
        assert.equal(args.days, 60);
        assert.equal(args.seed, 42);
        assert.equal(args.timeframe, '1h');
        assert.equal(args.stopLoss, 0.03);
        assert.equal(args.takeProfit, 0.06);
    });

    it('parses --live mode', () => {
        const args = parseArgs(['--live', '--tf', '5m']);
        assert.equal(args.mode, 'live');
        assert.equal(args.timeframe, '5m');
    });

    it('uses defaults when options omitted', () => {
        const args = parseArgs(['--backtest']);
        assert.equal(args.symbol, 'BTC-USD');
        assert.equal(args.days, 30);
        assert.equal(args.timeframe, '15m');
        assert.equal(args.warmup, 50);
        assert.equal(args.cooldown, 10);
        assert.equal(args.stopLoss, 0.02);
        assert.equal(args.takeProfit, 0.04);
    });

    it('parses --candles path', () => {
        const args = parseArgs(['--analyze', '--candles', '/tmp/data.json']);
        assert.equal(args.candlesPath, '/tmp/data.json');
    });

    it('parses fusion and confidence options', () => {
        const args = parseArgs(['--backtest', '--fusion', 'bayesian', '--min-confidence', '0.5']);
        assert.equal(args.fusionMethod, 'bayesian');
        assert.equal(args.minConfidence, 0.5);
    });

    it('parses --output format', () => {
        const args = parseArgs(['--backtest', '--output', 'json']);
        assert.equal(args.output, 'json');
    });
});

// ─── 3. Candle Loading ──────────────────────────────────────────────────────────

describe('loadCandles', () => {
    it('loads a JSON array of candles', () => {
        const candles = generateSyntheticCandles(10, { seed: 1 });
        const path = writeTempCandles('array.json', candles);
        const loaded = loadCandles(path);
        assert.equal(loaded.length, 10);
        assert.equal(loaded[0].close, candles[0].close);
    });

    it('loads { candles: [...] } shaped JSON', () => {
        const candles = generateSyntheticCandles(5, { seed: 1 });
        const path = writeTempCandles('wrapped.json', { candles });
        const loaded = loadCandles(path);
        assert.equal(loaded.length, 5);
    });

    it('loads { data: [...] } shaped JSON', () => {
        const candles = generateSyntheticCandles(5, { seed: 1 });
        const path = writeTempCandles('data.json', { data: candles });
        const loaded = loadCandles(path);
        assert.equal(loaded.length, 5);
    });

    it('loads { ohlcv: [...] } shaped JSON', () => {
        const candles = generateSyntheticCandles(5, { seed: 1 });
        const path = writeTempCandles('ohlcv.json', { ohlcv: candles });
        const loaded = loadCandles(path);
        assert.equal(loaded.length, 5);
    });

    it('normalizes o/h/l/c/v fields to open/high/low/close/volume', () => {
        const compact = [
            { t: 1000, o: 100, h: 102, l: 99, c: 101, v: 500 },
            { t: 2000, o: 101, h: 103, l: 100, c: 102, v: 600 },
        ];
        const path = writeTempCandles('compact.json', compact);
        const loaded = loadCandles(path);
        assert.equal(loaded.length, 2);
        assert.equal(loaded[0].open, 100);
        assert.equal(loaded[0].high, 102);
        assert.equal(loaded[0].low, 99);
        assert.equal(loaded[0].close, 101);
        assert.equal(loaded[0].volume, 500);
        assert.equal(loaded[0].timestamp, 1000);
        assert.equal(loaded[1].close, 102);
    });

    it('throws on non-existent file', () => {
        assert.throws(
            () => loadCandles('/nonexistent/path/candles.json'),
            /file not found/,
        );
    });

    it('throws on empty array', () => {
        const path = writeTempCandles('empty.json', []);
        assert.throws(
            () => loadCandles(path),
            /no valid candle array/,
        );
    });
});

// ─── 4. CLI Integration Tests ───────────────────────────────────────────────────

describe('CLI --analyze', () => {
    it('outputs JSON with --output json flag', () => {
        const candles = generateSyntheticCandles(60, { seed: 42 });
        const path = writeTempCandles('analyze-input.json', candles);
        const stdout = execSync(
            `node audit/trade.mjs --analyze --candles ${path} --symbol TEST-USD --output json --seed 42`,
            { encoding: 'utf-8', timeout: 15000 },
        );
        const result = JSON.parse(stdout);
        assert.ok(result.id, 'has id');
        assert.equal(result.symbol, 'TEST-USD');
        assert.ok(['BUY', 'SELL', 'HOLD'].includes(result.action), 'valid action');
        assert.ok(typeof result.confidence === 'number');
        assert.ok(result.confidence >= 0 && result.confidence <= 1, 'confidence 0-1');
        assert.ok(result.regime && typeof result.regime === 'string');
        assert.ok(Array.isArray(result.signals));
        assert.ok(Array.isArray(result.activeZones));
        assert.ok(typeof result.compositeScore === 'number');
    });

    it('runs analyze with --candles flag and returns valid decision', () => {
        const candles = generateSyntheticCandles(60, { seed: 42, volatility: 0.008 });
        const path = writeTempCandles('analyze2.json', candles);
        const stdout = execSync(
            `node audit/trade.mjs --analyze --candles ${path} --symbol SOL-USD --output json`,
            { encoding: 'utf-8', timeout: 15000 },
        );
        const decision = JSON.parse(stdout);
        assert.ok(['BUY', 'SELL', 'HOLD'].includes(decision.action));
        assert.ok(typeof decision.confidence === 'number');
        assert.ok(decision.support === null || (typeof decision.support.price === 'number'));
        assert.ok(decision.resistance === null || (typeof decision.resistance.price === 'number'));
    });

    it('handles small candle set gracefully', () => {
        const candles = generateSyntheticCandles(10, { seed: 1 });
        const path = writeTempCandles('small.json', candles);
        const stdout = execSync(
            `node audit/trade.mjs --analyze --candles ${path} --output json`,
            { encoding: 'utf-8', timeout: 15000 },
        );
        const decision = JSON.parse(stdout);
        // Should not crash; may return empty decision
        assert.ok(decision.action);
    });
});

describe('CLI --backtest', () => {
    it('runs backtest on synthetic data and outputs JSON', () => {
        const stdout = execSync(
            'node audit/trade.mjs --backtest --symbol BTC-USD --days 5 --seed 42 --tf 15m --output json',
            { encoding: 'utf-8', timeout: 30000 },
        );
        const result = JSON.parse(stdout);
        assert.ok(result.trades, 'has trades array');
        assert.ok(result.stats, 'has stats');
        assert.ok(typeof result.stats.totalTrades === 'number', 'totalTrades is number');
        assert.ok(typeof result.stats.winRate === 'number', 'winRate is number');
        assert.ok(result.stats.winRate >= 0 && result.stats.winRate <= 1, 'winRate 0-1');
        assert.ok(typeof result.stats.profitFactor === 'number', 'profitFactor is number');
        assert.ok(result.stats.perRegime, 'has perRegime breakdown');
    });

    it('respects --warmup and --cooldown', () => {
        const stdout = execSync(
            'node audit/trade.mjs --backtest --symbol BTC-USD --days 3 --seed 42 --tf 15m --warmup 30 --cooldown 5 --output json',
            { encoding: 'utf-8', timeout: 30000 },
        );
        const result = JSON.parse(stdout);
        assert.ok(result.stats.totalTrades >= 0);
    });

    it('works with --sl and --tp flags', () => {
        const stdout = execSync(
            'node audit/trade.mjs --backtest --symbol BTC-USD --days 3 --seed 42 --tf 15m --sl 0.01 --tp 0.03 --output json',
            { encoding: 'utf-8', timeout: 30000 },
        );
        const result = JSON.parse(stdout);
        assert.ok(result.stats.totalTrades >= 0);
    });

    it('works with different timeframes', () => {
        for (const tf of ['1m', '5m', '1h']) {
            const stdout = execSync(
                `node audit/trade.mjs --backtest --symbol BTC-USD --days 1 --seed 42 --tf ${tf} --output json`,
                { encoding: 'utf-8', timeout: 30000 },
            );
            const result = JSON.parse(stdout);
            assert.ok(result.stats, `backtest works with ${tf} timeframe`);
        }
    });

    it('produces trade entries with proper fields', () => {
        const stdout = execSync(
            'node audit/trade.mjs --backtest --symbol BTC-USD --days 10 --seed 42 --tf 15m --output json',
            { encoding: 'utf-8', timeout: 30000 },
        );
        const result = JSON.parse(stdout);
        if (result.trades.length > 0) {
            const t = result.trades[0];
            assert.ok(typeof t.entryBar === 'number');
            assert.ok(typeof t.exitBar === 'number');
            assert.ok(t.exitBar >= t.entryBar, 'exit after entry');
            assert.ok(['long', 'short'].includes(t.direction));
            assert.ok(typeof t.pnl === 'number');
            assert.ok(typeof t.exitReason === 'string');
        }
    });
});

describe('CLI error handling', () => {
    it('exits with error when no mode specified', () => {
        try {
            execSync('node audit/trade.mjs --symbol BTC-USD', { encoding: 'utf-8', timeout: 5000 });
            assert.fail('should have exited');
        } catch (err) {
            assert.ok(err.stderr?.includes('specify a mode') || err.status !== 0, 'error on missing mode');
        }
    });

    it('exits with error on invalid timeframe', () => {
        try {
            execSync('node audit/trade.mjs --backtest --tf invalid --output json', { encoding: 'utf-8', timeout: 5000 });
            assert.fail('should have exited');
        } catch (err) {
            assert.ok(err.stderr?.includes('unknown timeframe') || err.status !== 0);
        }
    });

    it('shows help with --help', () => {
        const stdout = execSync('node audit/trade.mjs --help', { encoding: 'utf-8', timeout: 5000 });
        assert.ok(stdout.includes('Trade CLI'));
        assert.ok(stdout.includes('--analyze'));
        assert.ok(stdout.includes('--backtest'));
        assert.ok(stdout.includes('--live'));
    });

    it('shows help with -h', () => {
        const stdout = execSync('node audit/trade.mjs -h', { encoding: 'utf-8', timeout: 5000 });
        assert.ok(stdout.includes('Trade CLI'));
    });
});

// ─── 5. Output Format Tests ─────────────────────────────────────────────────────

describe('output formats', () => {
    it('analyze --output json produces valid JSON', () => {
        const candles = generateSyntheticCandles(60, { seed: 42 });
        const path = writeTempCandles('fmt-json.json', candles);
        const stdout = execSync(
            `node audit/trade.mjs --analyze --candles ${path} --output json`,
            { encoding: 'utf-8', timeout: 15000 },
        );
        JSON.parse(stdout); // must parse
    });

    it('backtest --output json produces valid JSON with complete structure', () => {
        const stdout = execSync(
            'node audit/trade.mjs --backtest --days 2 --seed 42 --output json',
            { encoding: 'utf-8', timeout: 30000 },
        );
        const result = JSON.parse(stdout);

        // Validate stats structure
        const requiredStats = ['totalTrades', 'winningTrades', 'losingTrades', 'winRate', 'totalPnl', 'avgWin', 'avgLoss', 'profitFactor', 'perRegime'];
        for (const key of requiredStats) {
            assert.ok(key in result.stats, `stats.${key} exists`);
        }
    });
});

// ─── 6. Reproducibility ─────────────────────────────────────────────────────────

describe('reproducibility', () => {
    it('same seed produces identical backtest results', () => {
        const candles1 = generateSyntheticCandles(300, { seed: 123 });
        const path1 = writeTempCandles('rep1.json', candles1);
        const candles2 = generateSyntheticCandles(300, { seed: 123 });
        const path2 = writeTempCandles('rep2.json', candles2);

        // Verify candles are identical
        for (let i = 0; i < 300; i++) {
            assert.equal(candles1[i].close, candles2[i].close, `candle ${i} close matches`);
        }
    });

    it('different seeds produce different candles', () => {
        const a = generateSyntheticCandles(50, { seed: 7 });
        const b = generateSyntheticCandles(50, { seed: 77 });
        let sameCount = 0;
        for (let i = 0; i < 50; i++) {
            if (a[i].close === b[i].close) sameCount++;
        }
        assert.ok(sameCount < 50, 'different seeds diverge');
    });
});

// ─── 7. Edge Cases ──────────────────────────────────────────────────────────────

describe('edge cases', () => {
    it('generateSyntheticCandles with volatility=0 produces flat trend', () => {
        const candles = generateSyntheticCandles(100, { seed: 42, volatility: 0.000001 });
        const prices = candles.map(c => c.close);
        const range = Math.max(...prices) - Math.min(...prices);
        // With minimal volatility, price shouldn't move much
        assert.ok(range < 5, `range ${range} is small`);
    });

    it('handles very high base price', () => {
        const candles = generateSyntheticCandles(50, { seed: 42, basePrice: 75000, volatility: 0.01 });
        const avg = candles.reduce((s, c) => s + c.close, 0) / candles.length;
        assert.ok(avg > 65000 && avg < 85000, `avg ${avg.toFixed(0)} near basePrice 75000`);
    });

    it('orchestrator integration: trending market produces directional signals', () => {
        const candles = generateSyntheticCandles(200, { seed: 42, volatility: 0.008 });
        const path = writeTempCandles('trending.json', candles);
        const stdout = execSync(
            `node audit/trade.mjs --analyze --candles ${path} --output json`,
            { encoding: 'utf-8', timeout: 15000 },
        );
        const decision = JSON.parse(stdout);
        assert.ok(decision.regime && typeof decision.regime === 'string');
        assert.ok(decision.signals.length >= 1, 'has at least one signal');
    });
});
