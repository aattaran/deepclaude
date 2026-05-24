/**
 * Signal Fusion Engine — unit tests
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/signal-fusion.test.js
 */

import { describe, it, mock, before, after } from 'node:test';
import assert from 'node:assert/strict';
import {
    pearsonCorrelation,
    weightedAverage,
    normalizeScores,
    fuseBayesian,
    SignalFusionEngine,
    SignalQualityAnalyzer,
    DEFAULT_WEIGHTS,
} from './signal-fusion.mjs';

// ===========================================================================
// 1. pearsonCorrelation
// ===========================================================================
describe('pearsonCorrelation', () => {

    it('returns 1.0 for identical arrays', () => {
        assert.equal(pearsonCorrelation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1);
    });

    it('returns -1.0 for perfectly negatively correlated arrays', () => {
        const r = pearsonCorrelation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]);
        assert.equal(r, -1);
    });

    it('returns near 0 for uncorrelated data', () => {
        const r = pearsonCorrelation([1, 0, -1, 0, 1], [1, 0, 1, 0, -1]);
        assert.ok(Math.abs(r) < 0.5,
            `expected |r| < 0.5 for loosely correlated data, got ${r}`);
    });

    it('returns 0 for fewer than 2 elements', () => {
        assert.equal(pearsonCorrelation([1], [2]), 0);
        assert.equal(pearsonCorrelation([], []), 0);
    });

    it('returns 0 when one array has zero variance', () => {
        assert.equal(pearsonCorrelation([1, 1, 1, 1], [2, 3, 4, 5]), 0);
        assert.equal(pearsonCorrelation([1, 2, 3, 4], [5, 5, 5, 5]), 0);
    });

    it('returns 0 for non-array input', () => {
        assert.equal(pearsonCorrelation(null, [1, 2, 3]), 0);
        assert.equal(pearsonCorrelation([1, 2, 3], undefined), 0);
    });

    it('returns 0 for mismatched length arrays', () => {
        assert.equal(pearsonCorrelation([1, 2, 3], [1, 2]), 0);
    });

    it('computes a known positive correlation correctly', () => {
        const xs = [1, 2, 3, 4, 5, 6];
        const ys = [2, 4, 6, 8, 10, 12];
        const r = pearsonCorrelation(xs, ys);
        assert.ok(Math.abs(r - 1) < 1e-10,
            `expected r ≈ 1 for perfectly correlated data, got ${r}`);
    });
});

// ===========================================================================
// 2. weightedAverage
// ===========================================================================
describe('weightedAverage', () => {

    it('computes simple mean when weights are omitted', () => {
        assert.equal(weightedAverage([2, 4, 6]), 4);
        assert.equal(weightedAverage([1, 2, 3, 4, 5]), 3);
    });

    it('computes weighted mean correctly', () => {
        // (1*0.5 + 2*0.3 + 3*0.2) / (0.5 + 0.3 + 0.2) = (0.5 + 0.6 + 0.6) / 1.0 = 1.7
        const result = weightedAverage([1, 2, 3], [0.5, 0.3, 0.2]);
        assert.ok(Math.abs(result - 1.7) < 1e-10,
            `expected 1.7, got ${result}`);
    });

    it('returns 0 for empty array', () => {
        assert.equal(weightedAverage([]), 0);
        assert.equal(weightedAverage([], []), 0);
    });

    it('returns 0 when total weight is 0', () => {
        assert.equal(weightedAverage([1, 2, 3], [0, 0, 0]), 0);
    });

    it('returns 0 for mismatched array lengths', () => {
        assert.equal(weightedAverage([1, 2], [1]), 0);
    });

    it('handles a single value with weight', () => {
        assert.equal(weightedAverage([10], [2]), 10);
    });

    it('is equivalent to simple mean when all weights equal', () => {
        const result = weightedAverage([1, 2, 3, 4], [1, 1, 1, 1]);
        assert.equal(result, 2.5);
    });
});

// ===========================================================================
// 3. normalizeScores
// ===========================================================================
describe('normalizeScores', () => {

    it('scales values to -1..1 by max absolute value', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: 2, confidence: 1 },
            { source: 'alpha', name: 'b', value: -1, confidence: 1 },
            { source: 'alpha', name: 'c', value: 0.5, confidence: 1 },
        ];
        const normalized = normalizeScores(signals);
        assert.equal(normalized[0].value, 1);    // 2/2 = 1
        assert.equal(normalized[1].value, -0.5); // -1/2 = -0.5
        assert.equal(normalized[2].value, 0.25); // 0.5/2 = 0.25
    });

    it('returns empty array for empty input', () => {
        assert.deepEqual(normalizeScores([]), []);
        assert.deepEqual(normalizeScores(null), []);
    });

    it('returns unchanged when all values are 0', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: 0, confidence: 1 },
            { source: 'beta', name: 'b', value: 0, confidence: 1 },
        ];
        const normalized = normalizeScores(signals);
        assert.equal(normalized[0].value, 0);
        assert.equal(normalized[1].value, 0);
    });

    it('preserves signal names and confidence', () => {
        const signals = [
            { source: 'alpha', name: 'momentum', value: 5, confidence: 0.8 },
        ];
        const normalized = normalizeScores(signals);
        assert.equal(normalized[0].name, 'momentum');
        assert.equal(normalized[0].confidence, 0.8);
        assert.equal(normalized[0].source, 'alpha');
    });

    it('handles values already in [-1, 1] range', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: 0.5, confidence: 1 },
            { source: 'alpha', name: 'b', value: -0.3, confidence: 1 },
        ];
        const normalized = normalizeScores(signals);
        assert.equal(normalized[0].value, 1);
        assert.equal(normalized[1].value, -0.6);
    });

    it('clamps output to [-1, 1]', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: -100, confidence: 1 },
            { source: 'alpha', name: 'b', value: 100, confidence: 1 },
        ];
        const normalized = normalizeScores(signals);
        assert.equal(normalized[0].value, -1);
        assert.equal(normalized[1].value, 1);
    });
});

// ===========================================================================
// 4. fuseBayesian
// ===========================================================================
describe('fuseBayesian', () => {

    it('returns neutral for empty signals', () => {
        const result = fuseBayesian([]);
        assert.equal(result.score, 0);
        assert.equal(result.confidence, 0);
    });

    it('returns neutral for null/undefined input', () => {
        assert.deepEqual(fuseBayesian(null), { score: 0, confidence: 0 });
        assert.deepEqual(fuseBayesian(undefined), { score: 0, confidence: 0 });
    });

    it('returns bullish score for strongly bullish signals', () => {
        const signals = [
            { source: 'alpha', name: 'mom', value: 1.0, confidence: 1.0 },
            { source: 'microstructure', name: 'flow', value: 0.8, confidence: 1.0 },
        ];
        const result = fuseBayesian(signals);
        assert.ok(result.score > 0.5,
            `expected score > 0.5 for bullish signals, got ${result.score}`);
        assert.ok(result.confidence > 0.7,
            `expected confidence > 0.7, got ${result.confidence}`);
    });

    it('returns bearish score for strongly bearish signals', () => {
        const signals = [
            { source: 'alpha', name: 'mom', value: -1.0, confidence: 1.0 },
            { source: 'microstructure', name: 'flow', value: -0.8, confidence: 1.0 },
        ];
        const result = fuseBayesian(signals);
        assert.ok(result.score < -0.5,
            `expected score < -0.5 for bearish signals, got ${result.score}`);
    });

    it('returns near 0 for perfectly conflicting signals', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: 1.0, confidence: 1.0 },
            { source: 'microstructure', name: 'b', value: -1.0, confidence: 1.0 },
        ];
        const result = fuseBayesian(signals);
        // Two opposing perfect signals cancel out
        assert.ok(Math.abs(result.score) < 0.1,
            `expected score near 0 for conflicting signals, got ${result.score}`);
    });

    it('respects custom priors', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: 0.1, confidence: 1.0 },
        ];
        const bullishPrior = fuseBayesian(signals, { bullish: 0.9, bearish: 0.1 });
        const bearishPrior = fuseBayesian(signals, { bullish: 0.1, bearish: 0.9 });
        assert.ok(bullishPrior.score > bearishPrior.score,
            `bullish prior should yield higher score (${bullishPrior.score} vs ${bearishPrior.score})`);
    });

    it('returns confidence as posterior probability of dominant hypothesis', () => {
        const signals = [
            { source: 'alpha', name: 'a', value: 0.5, confidence: 1.0 },
        ];
        const result = fuseBayesian(signals);
        assert.ok(result.confidence >= 0.5 && result.confidence <= 1.0,
            `confidence ${result.confidence} should be 0.5-1.0 for any non-neutral signal`);
    });
});

// ===========================================================================
// 5. SignalFusionEngine — weighted fusion
// ===========================================================================
describe('SignalFusionEngine (weighted method)', () => {

    it('returns 0 for empty signals', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        assert.equal(engine.getCompositeScore(), 0);
    });

    it('returns the signal value for a single signal', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignal('alpha', 'momentum', 0.7, 1.0);
        // Only alpha contributes: weight 0.30 * conf 1.0 = 0.30
        // composite = 0.7 * 0.30 / 0.30 = 0.7
        assert.equal(engine.getCompositeScore(), 0.7);
    });

    it('computes weighted average with known inputs', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignal('alpha', 'mom', 1.0, 1.0);
        engine.addSignal('microstructure', 'flow', -0.5, 1.0);

        // alpha: 1.0 * (0.30 * 1.0) = 0.30
        // microstructure: -0.5 * (0.25 * 1.0) = -0.125
        // total effective weight: 0.30 + 0.25 = 0.55
        // composite: (0.30 - 0.125) / 0.55 ≈ 0.3182
        const expected = (1.0 * 0.30 - 0.5 * 0.25) / (0.30 + 0.25);
        assert.ok(Math.abs(engine.getCompositeScore() - expected) < 1e-10,
            `expected ~${expected}, got ${engine.getCompositeScore()}`);
    });

    it('weights signals by their confidence', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        // Low-confidence bullish signal
        engine.addSignal('alpha', 'mom', 1.0, 0.1);
        // High-confidence bearish signal
        engine.addSignal('microstructure', 'flow', -1.0, 1.0);

        const score = engine.getCompositeScore();
        // Bearish signal should dominate due to higher confidence
        assert.ok(score < 0,
            `expected negative score (bearish dominates), got ${score}`);
    });

    it('handles conflicting signals of equal strength', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignal('alpha', 'a', 1.0, 1.0);
        engine.addSignal('microstructure', 'b', -1.0, 1.0);

        // alpha: 1.0 * 0.30 = 0.30
        // microstructure: -1.0 * 0.25 = -0.25
        // Composite = (0.30 - 0.25) / 0.55 ≈ 0.09 (slightly bullish due to higher alpha weight)
        assert.ok(engine.getCompositeScore() > 0,
            'should be slightly bullish due to higher alpha weight');
    });

    it('ignores invalid source names', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignal('invalid_source', 'test', 1.0, 1.0);
        assert.equal(engine.getCompositeScore(), 0);
    });

    it('supports custom weights', () => {
        const engine = new SignalFusionEngine({
            fusionMethod: 'weighted',
            weights: { alpha: 1.0, microstructure: 0, liquidity: 0, backtest: 0, sentiment: 0 },
        });
        engine.addSignal('alpha', 'mom', 1.0, 1.0);
        engine.addSignal('microstructure', 'flow', -1.0, 1.0);
        // Only alpha matters (weight 1.0), microstructure has weight 0
        assert.equal(engine.getCompositeScore(), 1.0);
    });
});

// ===========================================================================
// 6. SignalFusionEngine — bayesian fusion
// ===========================================================================
describe('SignalFusionEngine (bayesian method)', () => {

    it('returns 0 for empty signals', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'bayesian' });
        assert.equal(engine.getCompositeScore(), 0);
    });

    it('returns bullish for strongly bullish signals', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'bayesian' });
        engine.addSignal('alpha', 'mom', 0.9, 1.0);
        engine.addSignal('microstructure', 'flow', 0.8, 1.0);
        engine.addSignal('liquidity', 'depth', 0.7, 0.9);
        assert.ok(engine.getCompositeScore() > 0,
            `expected positive score for bullish signals, got ${engine.getCompositeScore()}`);
    });

    it('returns bearish for strongly bearish signals', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'bayesian' });
        engine.addSignal('alpha', 'mom', -0.9, 1.0);
        engine.addSignal('microstructure', 'flow', -0.8, 1.0);
        const score = engine.getCompositeScore();
        assert.ok(score < 0,
            `expected negative score for bearish signals, got ${score}`);
    });

    it('returns near 0 for balanced signals', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'bayesian' });
        engine.addSignal('alpha', 'a', 1.0, 1.0);
        engine.addSignal('microstructure', 'b', -1.0, 1.0);
        const score = engine.getCompositeScore();
        assert.ok(Math.abs(score) < 0.1,
            `expected score near 0 for balanced signals, got ${score}`);
    });

    it('handles a single signal', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'bayesian' });
        engine.addSignal('alpha', 'mom', 0.5, 1.0);
        const score = engine.getCompositeScore();
        assert.ok(score > 0,
            `expected positive score for bullish single signal, got ${score}`);
    });
});

// ===========================================================================
// 7. SignalFusionEngine — voting fusion
// ===========================================================================
describe('SignalFusionEngine (voting method)', () => {

    it('returns 0 for empty signals', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'voting' });
        assert.equal(engine.getCompositeScore(), 0);
    });

    it('returns positive when majority of signals are bullish', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'voting' });
        engine.addSignal('alpha', 'a', 0.8, 1.0);
        engine.addSignal('microstructure', 'b', -0.3, 1.0);
        engine.addSignal('liquidity', 'c', 0.6, 1.0);
        engine.addSignal('backtest', 'd', 0.5, 1.0);
        const score = engine.getCompositeScore();
        assert.ok(score > 0,
            `expected positive score (3 of 4 bullish), got ${score}`);
    });

    it('returns negative when majority of signals are bearish', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'voting' });
        engine.addSignal('alpha', 'a', -0.8, 1.0);
        engine.addSignal('microstructure', 'b', -0.7, 1.0);
        engine.addSignal('liquidity', 'c', 0.2, 1.0);
        const score = engine.getCompositeScore();
        assert.ok(score < 0,
            `expected negative score (2 of 3 bearish), got ${score}`);
    });

    it('returns near 0 for tied votes', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'voting' });
        engine.addSignal('alpha', 'a', 1.0, 1.0);
        engine.addSignal('microstructure', 'b', -1.0, 1.0);
        const score = engine.getCompositeScore();
        assert.ok(Math.abs(score) < 0.1,
            `expected score near 0 for tied vote, got ${score}`);
    });

    it('handles a single bullish signal', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'voting' });
        engine.addSignal('alpha', 'mom', 0.5, 1.0);
        assert.ok(engine.getCompositeScore() > 0);
    });
});

// ===========================================================================
// 8. SignalFusionEngine — getDecision
// ===========================================================================
describe('SignalFusionEngine.getDecision', () => {

    it('returns HOLD for empty signals', () => {
        const engine = new SignalFusionEngine();
        const decision = engine.getDecision();
        assert.equal(decision.action, 'HOLD');
        assert.equal(decision.confidence, 0);
    });

    it('returns BUY for strongly bullish signals', () => {
        const engine = new SignalFusionEngine({ minConfidence: 0.3 });
        engine.addSignal('alpha', 'mom', 1.0, 1.0);
        engine.addSignal('microstructure', 'flow', 1.0, 1.0);
        engine.addSignal('liquidity', 'depth', 1.0, 1.0);
        const decision = engine.getDecision();
        assert.equal(decision.action, 'BUY');
        assert.ok(decision.confidence > 0.3);
        assert.ok(typeof decision.reasoning === 'string' && decision.reasoning.length > 0);
    });

    it('returns SELL for strongly bearish signals', () => {
        const engine = new SignalFusionEngine({ minConfidence: 0.3 });
        engine.addSignal('alpha', 'mom', -1.0, 1.0);
        engine.addSignal('microstructure', 'flow', -1.0, 1.0);
        engine.addSignal('liquidity', 'depth', -1.0, 1.0);
        const decision = engine.getDecision();
        assert.equal(decision.action, 'SELL');
        assert.ok(decision.confidence > 0.3);
    });

    it('returns HOLD for weak/mixed signals below minConfidence', () => {
        const engine = new SignalFusionEngine({ minConfidence: 0.7 });
        engine.addSignal('alpha', 'mom', 0.2, 1.0);
        engine.addSignal('microstructure', 'flow', 0.3, 1.0);
        const decision = engine.getDecision();
        assert.equal(decision.action, 'HOLD');
    });

    it('respects custom minConfidence threshold', () => {
        const engine = new SignalFusionEngine({ minConfidence: 0.1 });
        engine.addSignal('alpha', 'mom', 0.5, 1.0);
        const decision = engine.getDecision();
        assert.equal(decision.action, 'BUY');
    });

    it('includes reasoning in decision', () => {
        const engine = new SignalFusionEngine();
        engine.addSignal('alpha', 'mom', 1.0, 1.0);
        const decision = engine.getDecision();
        assert.ok(typeof decision.reasoning === 'string');
        assert.ok(decision.reasoning.includes('weighted'));
        assert.ok(decision.reasoning.includes('alpha'));
    });
});

// ===========================================================================
// 9. SignalFusionEngine — getSourceBreakdown
// ===========================================================================
describe('SignalFusionEngine.getSourceBreakdown', () => {

    it('returns empty object for no signals', () => {
        const engine = new SignalFusionEngine();
        assert.deepEqual(engine.getSourceBreakdown(), {});
    });

    it('aggregates multiple signals from the same source', () => {
        const engine = new SignalFusionEngine();
        engine.addSignal('alpha', 'mom', 0.8, 0.9);
        engine.addSignal('alpha', 'senti', 0.6, 0.7);

        const breakdown = engine.getSourceBreakdown();
        assert.ok(breakdown.alpha !== undefined);
        assert.equal(breakdown.alpha.signalCount, 2);
        // value = (0.8*0.9 + 0.6*0.7) / (0.9 + 0.7) = (0.72 + 0.42) / 1.6 = 1.14 / 1.6 = 0.7125
        assert.ok(Math.abs(breakdown.alpha.score - 0.7125) < 1e-10);
        // confidence = (0.9 + 0.7) / 2 = 0.8
        assert.equal(breakdown.alpha.confidence, 0.8);
        assert.equal(breakdown.alpha.weight, 0.30);
    });

    it('includes weight from engine config', () => {
        const engine = new SignalFusionEngine({
            weights: { alpha: 0.5, microstructure: 0.5, liquidity: 0, backtest: 0, sentiment: 0 },
        });
        engine.addSignal('alpha', 'mom', 0.5, 1.0);
        const breakdown = engine.getSourceBreakdown();
        assert.equal(breakdown.alpha.weight, 0.5);
    });
});

// ===========================================================================
// 10. SignalFusionEngine — getCorrelationMatrix
// ===========================================================================
describe('SignalFusionEngine.getCorrelationMatrix', () => {

    it('returns empty object when no history', () => {
        const engine = new SignalFusionEngine();
        const matrix = engine.getCorrelationMatrix();
        assert.equal(Object.keys(matrix).length, 0);
    });

    it('returns 1.0 for identical signal patterns across rounds', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });

        // Round 1
        engine.addSignal('alpha', 'a', 0.5, 1.0);
        engine.addSignal('microstructure', 'b', 0.5, 1.0);
        engine.getCompositeScore();

        // Round 2
        engine.reset();
        engine.addSignal('alpha', 'a', 1.0, 1.0);
        engine.addSignal('microstructure', 'b', 1.0, 1.0);
        engine.getCompositeScore();

        // Round 3
        engine.reset();
        engine.addSignal('alpha', 'a', -0.5, 1.0);
        engine.addSignal('microstructure', 'b', -0.5, 1.0);
        engine.getCompositeScore();

        const matrix = engine.getCorrelationMatrix();
        assert.ok(matrix.alpha !== undefined, 'alpha should be in matrix');
        assert.ok(matrix.microstructure !== undefined, 'microstructure should be in matrix');
        // Identical patterns -> correlation should be 1.0
        assert.ok(Math.abs(matrix.alpha.microstructure - 1) < 1e-10,
            `expected alpha-microstructure correlation ≈ 1, got ${matrix.alpha.microstructure}`);
    });

    it('returns -1.0 for perfectly negatively correlated patterns', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });

        // Round 1
        engine.addSignal('alpha', 'a', 1.0, 1.0);
        engine.addSignal('microstructure', 'b', -1.0, 1.0);
        engine.getCompositeScore();

        // Round 2
        engine.reset();
        engine.addSignal('alpha', 'a', 0.5, 1.0);
        engine.addSignal('microstructure', 'b', -0.5, 1.0);
        engine.getCompositeScore();

        // Round 3
        engine.reset();
        engine.addSignal('alpha', 'a', -0.5, 1.0);
        engine.addSignal('microstructure', 'b', 0.5, 1.0);
        engine.getCompositeScore();

        const matrix = engine.getCorrelationMatrix();
        assert.equal(matrix.alpha.microstructure, -1);
    });

    it('returns 1.0 on the diagonal', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });

        engine.addSignal('alpha', 'a', 0.5, 1.0);
        engine.getCompositeScore();
        engine.reset();
        engine.addSignal('alpha', 'a', -0.3, 1.0);
        engine.getCompositeScore();

        const matrix = engine.getCorrelationMatrix();
        assert.equal(matrix.alpha.alpha, 1);
    });

    it('returns 0 for source pairs with insufficient overlapping rounds', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });

        // Only 1 round with both sources
        engine.addSignal('alpha', 'a', 0.5, 1.0);
        engine.addSignal('microstructure', 'b', 0.5, 1.0);
        engine.getCompositeScore();

        const matrix = engine.getCorrelationMatrix();
        // 1 overlapping round < 2, so correlation is 0
        assert.equal(matrix.alpha.microstructure, 0);
    });

    it('returns symmetric matrix', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });

        for (let i = 0; i < 5; i++) {
            engine.addSignal('alpha', 'a', Math.sin(i), 1.0);
            engine.addSignal('microstructure', 'b', Math.cos(i), 1.0);
            engine.getCompositeScore();
            engine.reset();
        }

        const matrix = engine.getCorrelationMatrix();

        // Both sources should be present
        assert.ok(matrix.alpha, 'alpha should be in matrix');
        assert.ok(matrix.microstructure, 'microstructure should be in matrix');

        // Matrix should be symmetric (with floating point tolerance)
        const upper = matrix.alpha.microstructure;
        const lower = matrix.microstructure.alpha;
        assert.ok(Math.abs(upper - lower) < 1e-12,
            `expected symmetric matrix: alpha.microstructure=${upper} !== microstructure.alpha=${lower}`);
    });
});

// ===========================================================================
// 11. SignalFusionEngine — addSignals (batch) and chaining
// ===========================================================================
describe('SignalFusionEngine batch operations', () => {

    it('addSignals processes multiple signals at once', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignals([
            { source: 'alpha', name: 'a', value: 1.0, confidence: 1.0 },
            { source: 'microstructure', name: 'b', value: -1.0, confidence: 1.0 },
            { source: 'liquidity', name: 'c', value: 1.0, confidence: 1.0 },
        ]);
        const breakdown = engine.getSourceBreakdown();
        assert.equal(Object.keys(breakdown).length, 3);
    });

    it('addSignals ignores null/undefined input', () => {
        const engine = new SignalFusionEngine();
        engine.addSignals(null);
        engine.addSignals(undefined);
        assert.equal(engine.getCompositeScore(), 0);
    });

    it('supports method chaining', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine
            .addSignal('alpha', 'a', 0.5, 1.0)
            .addSignal('microstructure', 'b', 0.3, 1.0);
        assert.equal(Object.keys(engine.getSourceBreakdown()).length, 2);
    });

    it('reset clears signals but preserves history for correlation', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignal('alpha', 'a', 1.0, 1.0);
        engine.getCompositeScore(); // records round 1 in history
        engine.reset();

        // Signals should be cleared
        assert.equal(engine.getCompositeScore(), 0);

        // History should still exist for correlation analysis
        const matrix = engine.getCorrelationMatrix();
        assert.ok(Object.keys(matrix).length > 0,
            'history should be preserved after reset for correlation analysis');
    });
});

// ===========================================================================
// 12. SignalQualityAnalyzer
// ===========================================================================
describe('SignalQualityAnalyzer', () => {

    // ── addOutcome + getSourceAccuracy ───────────────────────────────────

    it('returns 0 for source with no outcomes', () => {
        const analyzer = new SignalQualityAnalyzer();
        assert.equal(analyzer.getSourceAccuracy('alpha'), 0);
    });

    it('returns 1 for source with all correct outcomes', () => {
        const analyzer = new SignalQualityAnalyzer();
        analyzer.addOutcome({ source: 'alpha', name: 'mom', value: 0.5, confidence: 0.8 }, true);
        analyzer.addOutcome({ source: 'alpha', name: 'senti', value: -0.3, confidence: 0.7 }, true);
        assert.equal(analyzer.getSourceAccuracy('alpha'), 1);
    });

    it('returns 0 for source with all incorrect outcomes', () => {
        const analyzer = new SignalQualityAnalyzer();
        analyzer.addOutcome({ source: 'microstructure', name: 'flow', value: 0.5, confidence: 0.8 }, false);
        analyzer.addOutcome({ source: 'microstructure', name: 'depth', value: -0.3, confidence: 0.7 }, false);
        assert.equal(analyzer.getSourceAccuracy('microstructure'), 0);
    });

    it('computes accuracy for mixed outcomes', () => {
        const analyzer = new SignalQualityAnalyzer();
        analyzer.addOutcome({ source: 'alpha', name: 'a', value: 1, confidence: 1 }, true);
        analyzer.addOutcome({ source: 'alpha', name: 'b', value: 1, confidence: 1 }, true);
        analyzer.addOutcome({ source: 'alpha', name: 'c', value: 1, confidence: 1 }, false);
        analyzer.addOutcome({ source: 'alpha', name: 'd', value: 1, confidence: 1 }, false);
        assert.equal(analyzer.getSourceAccuracy('alpha'), 0.5);
    });

    // ── calibrateWeights ─────────────────────────────────────────────────

    it('increases weight for accurate sources and decreases for inaccurate ones', () => {
        const analyzer = new SignalQualityAnalyzer();

        // Alpha is always correct
        for (let i = 0; i < 15; i++) {
            analyzer.addOutcome({ source: 'alpha', name: 'sig', value: 0.5, confidence: 0.8 }, true);
        }
        // Microstructure is always wrong
        for (let i = 0; i < 15; i++) {
            analyzer.addOutcome({ source: 'microstructure', name: 'sig', value: -0.5, confidence: 0.8 }, false);
        }

        const base = { alpha: 0.30, microstructure: 0.25, liquidity: 0.20, backtest: 0.15, sentiment: 0.10 };
        const calibrated = analyzer.calibrateWeights(base);

        assert.ok(calibrated.alpha > calibrated.microstructure,
            `alpha (${calibrated.alpha}) should be > microstructure (${calibrated.microstructure}) after calibration`);
    });

    it('keeps weights at base values for sources with no outcomes', () => {
        const analyzer = new SignalQualityAnalyzer();
        analyzer.addOutcome({ source: 'alpha', name: 'a', value: 1, confidence: 1 }, true);

        const base = { alpha: 0.30, microstructure: 0.25, liquidity: 0.20, backtest: 0.15, sentiment: 0.10 };
        const calibrated = analyzer.calibrateWeights(base);

        // Sources with no outcomes should retain non-zero weights (adjusted by normalization)
        assert.ok(calibrated.liquidity > 0, 'liquidity weight should be > 0');
        assert.ok(calibrated.backtest > 0, 'backtest weight should be > 0');
        assert.ok(calibrated.sentiment > 0, 'sentiment weight should be > 0');
    });

    it('returns weights normalized to approximately 1', () => {
        const analyzer = new SignalQualityAnalyzer();
        for (let i = 0; i < 10; i++) {
            analyzer.addOutcome({ source: 'alpha', name: 'a', value: 1, confidence: 1 }, true);
            analyzer.addOutcome({ source: 'microstructure', name: 'b', value: 1, confidence: 1 }, false);
        }

        const calibrated = analyzer.calibrateWeights(DEFAULT_WEIGHTS);
        const total = Object.values(calibrated).reduce((s, w) => s + w, 0);
        assert.ok(Math.abs(total - 1) < 1e-6,
            `weights should sum to ~1, got ${total}`);
    });

    // ── getReliabilityReport ─────────────────────────────────────────────

    it('returns report with all source fields', () => {
        const analyzer = new SignalQualityAnalyzer();
        analyzer.addOutcome({ source: 'alpha', name: 'a', value: 0.5, confidence: 0.8 }, true);

        const report = analyzer.getReliabilityReport();

        assert.ok(report.alpha !== undefined);
        assert.equal(typeof report.alpha.accuracy, 'number');
        assert.equal(typeof report.alpha.recencyWeightedAccuracy, 'number');
        assert.equal(typeof report.alpha.signalCount, 'number');
        assert.equal(typeof report.alpha.avgConfidence, 'number');
        assert.equal(typeof report.alpha.calibrationError, 'number');
        assert.equal(report.alpha.signalCount, 1);
    });

    it('returns zeros for sources with no outcomes', () => {
        const analyzer = new SignalQualityAnalyzer();
        const report = analyzer.getReliabilityReport();

        for (const source of ['alpha', 'microstructure', 'liquidity', 'backtest', 'sentiment']) {
            assert.ok(report[source] !== undefined, `${source} should be in report`);
            assert.equal(report[source].accuracy, 0);
            assert.equal(report[source].recencyWeightedAccuracy, 0);
            assert.equal(report[source].signalCount, 0);
        }
    });

    it('computes calibrationError as |accuracy - avgConfidence|', () => {
        const analyzer = new SignalQualityAnalyzer();
        // accuracy = 2/3 ≈ 0.667, avgConfidence = (0.9 + 0.8 + 0.7)/3 = 0.8
        analyzer.addOutcome({ source: 'alpha', name: 'a', value: 0.5, confidence: 0.9 }, true);
        analyzer.addOutcome({ source: 'alpha', name: 'b', value: 0.3, confidence: 0.8 }, true);
        analyzer.addOutcome({ source: 'alpha', name: 'c', value: -0.2, confidence: 0.7 }, false);

        const report = analyzer.getReliabilityReport();
        const expectedError = Math.abs((2 / 3) - ((0.9 + 0.8 + 0.7) / 3));
        assert.ok(Math.abs(report.alpha.calibrationError - expectedError) < 1e-6,
            `expected calibrationError ~${expectedError}, got ${report.alpha.calibrationError}`);
    });

    it('includes all valid sources in the report', () => {
        const analyzer = new SignalQualityAnalyzer();
        const report = analyzer.getReliabilityReport();
        const expectedSources = ['alpha', 'microstructure', 'liquidity', 'backtest', 'sentiment', 'zone-detector', 'market-regime', 'confluence', 'smc', 'order-book', 'volume-profile'];
        assert.deepEqual(Object.keys(report).sort(), expectedSources.sort());
    });
});

// ===========================================================================
// 13. Integration: full pipeline from signals to decision
// ===========================================================================
describe('SignalFusionEngine integration', () => {

    it('confident bullish signals -> BUY via all 3 methods', () => {
        const signals = [
            { source: 'alpha', name: 'trend', value: 0.9, confidence: 0.95 },
            { source: 'microstructure', name: 'orderflow', value: 0.8, confidence: 0.85 },
            { source: 'liquidity', name: 'depth', value: 0.7, confidence: 0.8 },
            { source: 'backtest', name: 'momentum', value: 0.85, confidence: 0.9 },
        ];

        for (const method of ['weighted', 'bayesian', 'voting']) {
            const engine = new SignalFusionEngine({
                fusionMethod: method,
                minConfidence: 0.4,
            });
            engine.addSignals(signals);
            const decision = engine.getDecision();
            assert.equal(decision.action, 'BUY',
                `expected BUY for ${method}, got ${decision.action}`);
            assert.ok(decision.confidence > 0.4,
                `confidence ${decision.confidence} should exceed minConfidence for ${method}`);
        }
    });

    it('confident bearish signals -> SELL via all 3 methods', () => {
        const signals = [
            { source: 'alpha', name: 'trend', value: -0.9, confidence: 0.95 },
            { source: 'microstructure', name: 'orderflow', value: -0.8, confidence: 0.85 },
            { source: 'liquidity', name: 'depth', value: -0.7, confidence: 0.8 },
        ];

        for (const method of ['weighted', 'bayesian', 'voting']) {
            const engine = new SignalFusionEngine({
                fusionMethod: method,
                minConfidence: 0.4,
            });
            engine.addSignals(signals);
            const decision = engine.getDecision();
            assert.equal(decision.action, 'SELL',
                `expected SELL for ${method}, got ${decision.action}`);
        }
    });

    it('empty signals -> HOLD with 0 confidence', () => {
        for (const method of ['weighted', 'bayesian', 'voting']) {
            const engine = new SignalFusionEngine({ fusionMethod: method });
            const decision = engine.getDecision();
            assert.equal(decision.action, 'HOLD');
            assert.equal(decision.confidence, 0);
        }
    });

    it('single signal produces consistent results', () => {
        const engine = new SignalFusionEngine({ fusionMethod: 'weighted' });
        engine.addSignal('alpha', 'mom', 0.6, 0.8);
        const score = engine.getCompositeScore();
        assert.equal(score, 0.6);
        assert.equal(engine.getSourceBreakdown().alpha.signalCount, 1);
    });

    it('getSourceBreakdown reflects added signals', () => {
        const engine = new SignalFusionEngine();
        engine.addSignal('alpha', 'a', 0.8, 0.9);
        engine.addSignal('liquidity', 'b', 0.5, 0.7);

        const breakdown = engine.getSourceBreakdown();
        assert.equal(breakdown.alpha.signalCount, 1);
        assert.equal(breakdown.alpha.score, 0.8);
        assert.equal(breakdown.liquidity.signalCount, 1);
        assert.equal(breakdown.liquidity.score, 0.5);

        // Sources with no signals should not appear
        assert.equal(breakdown.microstructure, undefined);
        assert.equal(breakdown.backtest, undefined);
        assert.equal(breakdown.sentiment, undefined);
    });
});
