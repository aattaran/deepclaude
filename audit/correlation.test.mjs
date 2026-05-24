/**
 * Tests for the portfolio correlation matrix module.
 *
 * Uses node:test and node:assert/strict.
 * Generates synthetic candle fixtures via trade.mjs's generateSyntheticCandles
 * with different seeds to produce correlated/uncorrelated series.
 */

import { describe, it, test } from 'node:test';
import assert from 'node:assert/strict';
import { generateSyntheticCandles } from './trade.mjs';
import {
    alignSeries,
    computeCorrelationMatrix,
    computePortfolioVariance,
    diversificationScore,
    renderCorrelationTable,
} from './correlation.mjs';

// ─── Test Fixture Helpers ──────────────────────────────────────────────────────

function makeCandles(seed, count = 100) {
    return generateSyntheticCandles(count, {
        seed,
        basePrice: 100,
        volatility: 0.003,
    });
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe('correlation module', () => {

    // ── 1 ──────────────────────────────────────────────────────────────────────

    test('computeCorrelationMatrix returns expected shape', () => {
        const data = {
            'BTC-USD': makeCandles(1),
            'ETH-USD': makeCandles(2),
            'SOL-USD': makeCandles(3),
        };
        const result = computeCorrelationMatrix(data);

        assert.ok(result.matrix, 'should have matrix');
        assert.ok(Array.isArray(result.symbols), 'symbols should be an array');
        assert.ok(Array.isArray(result.clusters), 'clusters should be an array');
        assert.ok(Array.isArray(result.warnings), 'warnings should be an array');
        assert.equal(result.symbols.length, 3);

        // Each symbol has a row in the matrix
        for (const sym of result.symbols) {
            assert.ok(result.matrix[sym], `matrix row for ${sym}`);
            assert.equal(result.matrix[sym][sym], 1.0, `diagonal for ${sym} should be 1.0`);
        }
    });

    // ── 2 ──────────────────────────────────────────────────────────────────────

    test('Perfectly correlated series → correlation near 1.0', () => {
        const data = {
            'A': makeCandles(42),
            'B': makeCandles(42), // same seed → identical prices
        };
        const result = computeCorrelationMatrix(data);
        const corr = result.matrix['A']['B'];
        assert.ok(corr > 0.9999, `expected near 1.0, got ${corr}`);
    });

    // ── 3 ──────────────────────────────────────────────────────────────────────

    test('Negatively correlated series → correlation near -1.0', () => {
        const candles = makeCandles(42);
        // Invert prices so log returns are the exact negation of the original
        const negCandles = candles.map(c => ({ ...c, close: 1 / c.close }));
        const data = { 'A': candles, 'B': negCandles };
        const result = computeCorrelationMatrix(data);
        const corr = result.matrix['A']['B'];
        assert.ok(corr < -0.9999, `expected near -1.0, got ${corr}`);
    });

    // ── 4 ──────────────────────────────────────────────────────────────────────

    test('Uncorrelated series → correlation near 0', () => {
        // Use empirically verified seed pair that produces near-zero correlation
        // with the trade.mjs mulberry32 PRNG (seed 1 vs 7777 → -0.061 at vol 0.003, n=500)
        const data = {
            'A': makeCandles(1, 500),
            'B': makeCandles(7777, 500),
        };
        const result = computeCorrelationMatrix(data);
        const corr = result.matrix['A']['B'];
        assert.ok(Math.abs(corr) < 0.3, `expected near 0, got ${corr}`);
    });

    // ── 5 ──────────────────────────────────────────────────────────────────────

    test('Aligns series of different lengths', () => {
        const data = {
            'A': makeCandles(1, 200),
            'B': makeCandles(2, 100),
            'C': makeCandles(3, 150),
        };
        const aligned = alignSeries(data);
        assert.equal(aligned['A'].length, 100);
        assert.equal(aligned['B'].length, 100);
        assert.equal(aligned['C'].length, 100);

        // Last 100 of B should match aligned B exactly
        const originalB = data['B'];
        for (let i = 0; i < 100; i++) {
            assert.equal(aligned['B'][i], originalB[i]);
        }
    });

    // ── 6 ──────────────────────────────────────────────────────────────────────

    test('Clusters correctly group high-correlation pairs', () => {
        const data = {
            'BTC': makeCandles(42),       // same seed as ETH
            'ETH': makeCandles(42),
            'SOL': makeCandles(9999),     // unrelated
        };
        const result = computeCorrelationMatrix(data);

        const clusterAB = result.clusters.find(
            c => c.includes('BTC') && c.includes('ETH')
        );
        assert.ok(clusterAB, 'BTC and ETH should share a cluster');
        assert.equal(
            clusterAB.includes('SOL'), false,
            'SOL should NOT be in the BTC/ETH cluster'
        );
    });

    // ── 7 ──────────────────────────────────────────────────────────────────────

    test('Warnings flag pairs above 0.85', () => {
        const data = {
            'BTC': makeCandles(42),
            'ETH': makeCandles(42), // same seed → near perfect corr
            'SOL': makeCandles(99),
        };
        const result = computeCorrelationMatrix(data);

        assert.ok(result.warnings.length >= 1, 'should have at least one warning');
        assert.ok(
            result.warnings.some(w => w.includes('BTC') && w.includes('ETH')),
            'warning should mention BTC/ETH pair'
        );
    });

    // ── 8 ──────────────────────────────────────────────────────────────────────

    test('diversificationScore is 1 for uncorrelated, low for highly correlated', () => {
        // Perfectly uncorrelated — all off-diagonals are 0
        const uncorrelated = {
            'A': { 'A': 1.0, 'B': 0.0, 'C': 0.0 },
            'B': { 'A': 0.0, 'B': 1.0, 'C': 0.0 },
            'C': { 'A': 0.0, 'B': 0.0, 'C': 1.0 },
        };
        assert.equal(diversificationScore(uncorrelated), 1.0);

        // Highly correlated
        const highCorr = {
            'A': { 'A': 1.0, 'B': 0.9, 'C': 0.95 },
            'B': { 'A': 0.9, 'B': 1.0, 'C': 0.85 },
            'C': { 'A': 0.95, 'B': 0.85, 'C': 1.0 },
        };
        // avg(abs(off-diag)) = (0.9 + 0.95 + 0.85) / 3 = 0.9
        // score = 1 - 0.9 = 0.1
        const score = diversificationScore(highCorr);
        assert.ok(score < 0.15, `expected low score, got ${score}`);
    });

    // ── 9 ──────────────────────────────────────────────────────────────────────

    test('computePortfolioVariance matches manual calculation for 2-asset case', () => {
        const weights = { 'A': 0.6, 'B': 0.4 };
        const corrMatrix = {
            'A': { 'A': 1.0, 'B': 0.5 },
            'B': { 'A': 0.5, 'B': 1.0 },
        };
        const vols = { 'A': 0.2, 'B': 0.15 };

        // Manual Markowitz: w1^2*s1^2 + w2^2*s2^2 + 2*w1*w2*s1*s2*rho
        const manual =
            (0.6 ** 2) * (0.2 ** 2) +
            (0.4 ** 2) * (0.15 ** 2) +
            2 * 0.6 * 0.4 * 0.2 * 0.15 * 0.5;

        const result = computePortfolioVariance(weights, corrMatrix, vols);
        // Use approximate equality: floating-point order-of-ops can differ
        const eps = 1e-12;
        assert.ok(
            Math.abs(result - manual) < eps,
            `expected ~${manual}, got ${result}`
        );
    });

    // ── 10 ─────────────────────────────────────────────────────────────────────

    test('renderCorrelationTable produces expected columns', () => {
        const matrix = {
            'BTC': { 'BTC': 1.0, 'ETH': 0.7, 'SOL': 0.3 },
            'ETH': { 'BTC': 0.7, 'ETH': 1.0, 'SOL': 0.5 },
            'SOL': { 'BTC': 0.3, 'ETH': 0.5, 'SOL': 1.0 },
        };
        const symbols = ['BTC', 'ETH', 'SOL'];
        const output = renderCorrelationTable(matrix, symbols);

        // All symbol names appear as headers / row labels
        assert.ok(output.includes('BTC'), 'output should contain BTC');
        assert.ok(output.includes('ETH'), 'output should contain ETH');
        assert.ok(output.includes('SOL'), 'output should contain SOL');

        // Diagonal values appear
        assert.ok(output.includes('1.0000'), 'output should contain 1.0000 for diagonal');

        // Off-diagonal values appear
        assert.ok(output.includes('0.7000'), 'output should contain 0.7000 (BTC-ETH)');
        assert.ok(output.includes('0.3000'), 'output should contain 0.3000 (BTC-SOL)');
    });

    // ── 11 ─────────────────────────────────────────────────────────────────────

    test('Handles empty input gracefully', () => {
        const result = computeCorrelationMatrix({});
        assert.deepEqual(result.matrix, {});
        assert.deepEqual(result.symbols, []);
        assert.deepEqual(result.clusters, []);
        assert.deepEqual(result.warnings, []);
    });

    // ── 12 ─────────────────────────────────────────────────────────────────────

    test('Single symbol → 1x1 matrix with 1.0 on diagonal', () => {
        const data = { 'BTC': makeCandles(1) };
        const result = computeCorrelationMatrix(data);

        assert.equal(result.symbols.length, 1);
        assert.equal(result.symbols[0], 'BTC');
        assert.equal(result.matrix['BTC']['BTC'], 1.0);
        assert.equal(result.clusters.length, 0, 'no clusters for single asset');
        assert.equal(result.warnings.length, 0, 'no warnings for single asset');
    });

});
