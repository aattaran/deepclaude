/**
 * Tests for audit/scanner.mjs — multi-symbol scanner.
 *
 * Run: node --test audit/scanner.test.js
 *
 * ES module. Zero npm dependencies (node:test, node:assert).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { scanSymbols, scanResultsToTable, recommendSymbols } from './scanner.mjs';

// ─── scanSymbols ──────────────────────────────────────────────────────────────

describe('scanSymbols', () => {

  it('returns results for each symbol', async () => {
    const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD'];
    const results = await scanSymbols(symbols, { days: 5, tf: '1h', maxIterations: 1 });

    assert.equal(results.length, 3);

    const found = results.map(r => r.symbol);
    assert(found.includes('BTC-USD'));
    assert(found.includes('ETH-USD'));
    assert(found.includes('SOL-USD'));
  });

  it('results are sorted by score descending', async () => {
    const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'];
    const results = await scanSymbols(symbols, { days: 5, tf: '1h', maxIterations: 1 });

    assert.equal(results.length, 5);

    for (let i = 1; i < results.length; i++) {
      assert(
        results[i - 1].bestScore >= results[i].bestScore,
        `results[${i - 1}].bestScore (${results[i - 1].bestScore}) < results[${i}].bestScore (${results[i].bestScore})`,
      );
    }
  });

  it('results have expected shape (params, score, stats)', async () => {
    const results = await scanSymbols(['BTC-USD'], { days: 3, tf: '1h', maxIterations: 1 });
    assert.equal(results.length, 1);

    const r = results[0];

    // Top-level fields
    assert.equal(typeof r.symbol, 'string');
    assert.equal(typeof r.bestScore, 'number');
    assert.equal(typeof r.bestParams, 'object');
    assert.equal(typeof r.bestStats, 'object');
    assert.equal(typeof r.objective, 'string');
    assert.equal(typeof r.rank, 'number');

    // bestParams keys match the coarse space
    assert('stopLossPct' in r.bestParams);
    assert('takeProfitPct' in r.bestParams);
    assert('cooldownBars' in r.bestParams);
    assert('warmupBars' in r.bestParams);
    assert('minConfidence' in r.bestParams);
    assert('fusionMethod' in r.bestParams);

    // bestStats keys from orchestrator runBacktest
    assert('totalTrades' in r.bestStats);
    assert('winningTrades' in r.bestStats);
    assert('losingTrades' in r.bestStats);
    assert('winRate' in r.bestStats);
    assert('totalPnl' in r.bestStats);
    assert('avgWin' in r.bestStats);
    assert('avgLoss' in r.bestStats);
    assert('profitFactor' in r.bestStats);
    assert('perRegime' in r.bestStats);

    // score should be finite
    assert(Number.isFinite(r.bestScore), `bestScore should be finite, got ${r.bestScore}`);
  });

  it('handles empty symbols array', async () => {
    const results = await scanSymbols([], { days: 5 });
    assert.deepEqual(results, []);
  });

  it('handles single symbol', async () => {
    const results = await scanSymbols(['SOL-USD'], { days: 3, tf: '1h', maxIterations: 1 });

    assert.equal(results.length, 1);
    assert.equal(results[0].symbol, 'SOL-USD');
    assert(typeof results[0].bestScore === 'number');
    assert(results[0].bestParams !== null);
    assert(results[0].bestStats !== null);
  });

  it('works with different objective functions', async () => {
    const results = await scanSymbols(['BTC-USD'], {
      days: 3,
      tf: '1h',
      maxIterations: 1,
      objective: 'sharpe',
    });

    assert.equal(results.length, 1);
    assert.equal(results[0].objective, 'sharpe');
    assert.equal(typeof results[0].bestScore, 'number');
    assert(Number.isFinite(results[0].bestScore));
  });

  it('accepts the totalPnl objective', async () => {
    const results = await scanSymbols(['ETH-USD'], {
      days: 3,
      tf: '1h',
      maxIterations: 1,
      objective: 'totalPnl',
    });

    assert.equal(results.length, 1);
    assert.equal(results[0].objective, 'totalPnl');
  });

  it('default options produce valid results', async () => {
    // Use defaults: 30 days, 1h, volatility=0.012, maxIterations=3
    const results = await scanSymbols(['BTC-USD']);

    assert.equal(results.length, 1);
    assert(results[0].bestScore !== undefined);
    assert(Number.isFinite(results[0].bestScore));
    assert(results[0].bestStats.totalTrades >= 0);
  });

  it('options propagate correctly (different volatility)', async () => {
    const results = await scanSymbols(['BTC-USD'], {
      days: 3,
      tf: '1h',
      volatility: 0.008,
      maxIterations: 1,
    });

    assert.equal(results.length, 1);
    assert(typeof results[0].bestScore === 'number');
    assert(results[0].bestStats.totalTrades >= 0);
  });

  it('performance: 3 symbols with 1h bars completes in < 60s', async () => {
    const start = Date.now();

    const results = await scanSymbols(['BTC-USD', 'ETH-USD', 'SOL-USD'], {
      days: 10,
      tf: '1h',
      maxIterations: 1,
    });

    const elapsed = Date.now() - start;

    assert.equal(results.length, 3);
    assert(elapsed < 60000, `Took ${elapsed}ms, expected < 60000ms`);
  });

});

// ─── scanResultsToTable ───────────────────────────────────────────────────────

describe('scanResultsToTable', () => {

  it('produces expected columns', async () => {
    const results = await scanSymbols(['BTC-USD'], { days: 3, tf: '1h', maxIterations: 1 });
    const table = scanResultsToTable(results);

    assert(table.includes('RANK'));
    assert(table.includes('SYMBOL'));
    assert(table.includes('SCORE'));
    assert(table.includes('TRADES'));
    assert(table.includes('WIN%'));
    assert(table.includes('PF'));
    assert(table.includes('PNL%'));
    assert(table.includes('BEST_SL'));
    assert(table.includes('BEST_TP'));
  });

  it('returns correct column order', async () => {
    const results = await scanSymbols(['BTC-USD'], { days: 3, tf: '1h', maxIterations: 1 });
    const table = scanResultsToTable(results);
    const firstLine = table.split('\n')[0];

    // Columns should appear in order
    const rankIdx = firstLine.indexOf('RANK');
    const symbolIdx = firstLine.indexOf('SYMBOL');
    const scoreIdx = firstLine.indexOf('SCORE');

    assert(rankIdx < symbolIdx, 'RANK should come before SYMBOL');
    assert(symbolIdx < scoreIdx, 'SYMBOL should come before SCORE');
  });

  it('handles empty results', () => {
    const table = scanResultsToTable([]);
    assert.equal(table, '(no results)');
  });

  it('handles null/undefined results', () => {
    assert.equal(scanResultsToTable(null), '(no results)');
    assert.equal(scanResultsToTable(undefined), '(no results)');
  });

  it('returns a formatted string with header, separator, and data rows', async () => {
    const results = await scanSymbols(['BTC-USD'], { days: 3, tf: '1h', maxIterations: 1 });
    const table = scanResultsToTable(results);

    assert(typeof table === 'string');
    assert(table.length > 20);

    const lines = table.split('\n').filter(l => l.trim());
    // At minimum: header + separator + 1 data row
    assert(lines.length >= 3);
  });

});

// ─── recommendSymbols ─────────────────────────────────────────────────────────

describe('recommendSymbols', () => {

  it('filters by minScore', async () => {
    const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'];
    const results = await scanSymbols(symbols, { days: 5, tf: '1h', maxIterations: 1 });

    const minScore = 0.5;
    const recs = recommendSymbols(results, minScore);

    for (const r of recs) {
      assert(r.score >= minScore, `Expected score >= ${minScore}, got ${r.score}`);
    }
    assert(recs.length <= results.length);
  });

  it('returns empty array when no results meet threshold', async () => {
    const results = await scanSymbols(['BTC-USD'], { days: 3, tf: '1h', maxIterations: 1 });
    const recs = recommendSymbols(results, 999999);

    assert.equal(recs.length, 0);
  });

  it('returns empty array for empty results', () => {
    assert.deepEqual(recommendSymbols([], 1.0), []);
    assert.deepEqual(recommendSymbols(null, 1.0), []);
    assert.deepEqual(recommendSymbols(undefined, 1.0), []);
  });

  it('each recommendation has expected shape with reason', async () => {
    const results = await scanSymbols(['BTC-USD'], { days: 3, tf: '1h', maxIterations: 1 });

    // Use a very low threshold so we always get recommendations
    const recs = recommendSymbols(results, -999);
    assert(recs.length > 0);

    const rec = recs[0];
    assert.equal(typeof rec.symbol, 'string');
    assert.equal(typeof rec.score, 'number');
    assert.equal(typeof rec.reason, 'string');
    assert(rec.reason.length > 0, 'Reason should not be empty');
  });

  it('returns multiple recommendations with diverse reasons', async () => {
    const symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD'];
    const results = await scanSymbols(symbols, { days: 5, tf: '1h', maxIterations: 1 });

    const recs = recommendSymbols(results, -999);
    assert(recs.length > 0);

    for (const r of recs) {
      assert.equal(typeof r.symbol, 'string');
      assert.equal(typeof r.score, 'number');
      assert(typeof r.reason === 'string');
      // Reasons should be non-trivial
      assert(r.reason.length >= 3);
    }
  });

});
