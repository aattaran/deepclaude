/**
 * Tests for audit/mtf-scanner.mjs — multi-timeframe confluence scanner.
 *
 * Run: node --test audit/mtf-scanner.test.js
 *
 * ES module. Zero npm dependencies (node:test, node:assert).
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { scanMTF, renderMTFReport, computeConviction } from './mtf-scanner.mjs';

// ─── scanMTF: Shape & Range ────────────────────────────────────────────────────────

describe('scanMTF', () => {

    it('returns expected shape', async () => {
        const result = await scanMTF('TEST-USD', ['15m', '1h'], { '15m': 1, '1h': 2 }, { seed: 42 });

        assert.ok(result);
        assert.equal(typeof result.symbol, 'string');
        assert.equal(result.symbol, 'TEST-USD');

        assert.ok(Array.isArray(result.decisions));
        assert.equal(result.decisions.length, 2);

        assert.equal(typeof result.conviction, 'number');
        assert.equal(typeof result.buys, 'number');
        assert.equal(typeof result.sells, 'number');
        assert.equal(typeof result.holds, 'number');
        assert.equal(typeof result.verdict, 'string');

        // Each decision has the expected fields
        for (const d of result.decisions) {
            assert.equal(typeof d.timeframe, 'string');
            assert.ok(['BUY', 'SELL', 'HOLD'].includes(d.action));
            assert.equal(typeof d.confidence, 'number');
            assert.equal(typeof d.compositeScore, 'number');
            assert.equal(typeof d.reasoning, 'string');
            assert.equal(typeof d.regime, 'string');
            assert.equal(typeof d.candles, 'number');
            // price can be null when action is HOLD due to insufficient data
            if (d.action !== 'HOLD' || d.reasoning !== 'insufficient candle data') {
                assert.ok(d.price === null || typeof d.price === 'number');
            }
        }
    });

    it('conviction score is in range [-1, 1]', async () => {
        // Run with all 4 default timeframes
        const result = await scanMTF('BTC-USD', ['15m', '1h', '4h', '1d'], undefined, { seed: 42 });

        assert.ok(result.conviction >= -1, `conviction ${result.conviction} < -1`);
        assert.ok(result.conviction <= 1, `conviction ${result.conviction} > 1`);
    });

    it('single timeframe produces conviction ±1 or 0', async () => {
        const result = await scanMTF('BTC-USD', ['1h'], undefined, { seed: 42 });

        assert.equal(result.decisions.length, 1);
        // With exactly one decision, conviction equals the signal value: +1, -1, or 0
        assert.ok(
            result.conviction === 1 || result.conviction === -1 || result.conviction === 0,
            `unexpected conviction ${result.conviction} for single timeframe`,
        );
    });

    it('handles unknown symbol via synthetic fallback', async () => {
        const result = await scanMTF('ZZZZZ-USD', ['1h'], undefined, { seed: 42 });

        assert.ok(result);
        assert.equal(result.symbol, 'ZZZZZ-USD');
        assert.equal(result.decisions.length, 1);
        // Should have a real decision (not just HOLD from no data)
        assert.ok(result.decisions[0].candles > 0);
    });

    it('handles empty timeframes array gracefully', async () => {
        const result = await scanMTF('TEST-USD', [], {}, { seed: 42 });

        assert.ok(result);
        assert.equal(result.symbol, 'TEST-USD');
        assert.equal(result.decisions.length, 0);
        assert.equal(result.conviction, 0);
        assert.equal(result.verdict, 'MIXED');
        assert.equal(result.error, 'no timeframes provided');
    });

    it('handles single timeframe', async () => {
        const result = await scanMTF('TEST-USD', ['4h'], undefined, { seed: 42 });

        assert.ok(result);
        assert.equal(result.decisions.length, 1);
        assert.equal(result.decisions[0].timeframe, '4h');
        assert.ok(result.decisions[0].candles > 0);
    });

});

// ─── computeConviction ──────────────────────────────────────────────────────────────

describe('computeConviction', () => {

    it('all HOLD produces conviction 0', () => {
        const decisions = [
            { timeframe: '15m', action: 'HOLD' },
            { timeframe: '1h', action: 'HOLD' },
            { timeframe: '4h', action: 'HOLD' },
        ];
        const result = computeConviction(decisions, ['15m', '1h', '4h']);

        assert.equal(result.conviction, 0);
        assert.equal(result.buys, 0);
        assert.equal(result.sells, 0);
        assert.equal(result.holds, 3);
        assert.equal(result.verdict, 'MIXED');
    });

    it('all BUY produces positive conviction', () => {
        const decisions = [
            { timeframe: '15m', action: 'BUY' },
            { timeframe: '1h', action: 'BUY' },
            { timeframe: '4h', action: 'BUY' },
        ];
        const result = computeConviction(decisions, ['15m', '1h', '4h']);

        assert.ok(result.conviction > 0, `expected positive conviction, got ${result.conviction}`);
        assert.equal(result.buys, 3);
        assert.equal(result.sells, 0);
        assert.equal(result.holds, 0);
    });

    it('all SELL produces negative conviction', () => {
        const decisions = [
            { timeframe: '15m', action: 'SELL' },
            { timeframe: '1h', action: 'SELL' },
        ];
        const result = computeConviction(decisions, ['15m', '1h']);

        assert.ok(result.conviction < 0, `expected negative conviction, got ${result.conviction}`);
        assert.equal(result.sells, 2);
        assert.equal(result.buys, 0);
        assert.equal(result.holds, 0);
    });

    it('higher TF weight is emphasized', () => {
        // 15m (weight=1) says BUY, 1d (weight=2) says SELL
        // Weighted conviction: (1*1 + (-1)*2) / (1+2) = -1/3 ≈ -0.333
        // Without weighting it would be: (1 + (-1)) / 2 = 0
        // The negative result shows the higher-TF SELL pulls harder.
        const decisions = [
            { timeframe: '15m', action: 'BUY' },
            { timeframe: '1d', action: 'SELL' },
        ];
        const result = computeConviction(decisions, ['15m', '1d']);

        assert.equal(result.conviction, -1 / 3);

        // Verify weight ordering: if we reverse the TF order, the result flips
        const reversed = computeConviction(decisions, ['1d', '15m']);
        // Now 1d (weight=1) says SELL, 15m (weight=2) says BUY
        // Weighted: ((-1)*1 + (1)*2) / (1+2) = 1/3
        assert.equal(reversed.conviction, 1 / 3);

        // The higher TF should dominate in the original ordering
        assert.ok(result.conviction < 0, 'higher-TF SELL should pull conviction negative');
    });

    it('BUY-heavy across TFs produces BULLISH_CONFLUENCE', () => {
        const decisions = [
            { timeframe: '15m', action: 'BUY' },
            { timeframe: '1h', action: 'BUY' },
            { timeframe: '4h', action: 'BUY' },
            { timeframe: '1d', action: 'HOLD' },
        ];
        const result = computeConviction(decisions, ['15m', '1h', '4h', '1d']);

        assert.ok(result.conviction > 0);
        // Weighted: (1*1 + 2*1 + 3*1 + 4*0) / (1+2+3+4) = 6/10 = 0.6
        assert.equal(result.conviction, 0.6);
        // 0.6 >= 0.5 → BULLISH_CONFLUENCE
        assert.equal(result.verdict, 'BULLISH_CONFLUENCE');
    });

    it('SELL-heavy across TFs produces BEARISH_CONFLUENCE', () => {
        const decisions = [
            { timeframe: '15m', action: 'SELL' },
            { timeframe: '1h', action: 'SELL' },
            { timeframe: '4h', action: 'SELL' },
            { timeframe: '1d', action: 'SELL' },
        ];
        const result = computeConviction(decisions, ['15m', '1h', '4h', '1d']);

        assert.ok(result.conviction < 0);
        assert.equal(result.conviction, -1); // all SELL → -1
        assert.equal(result.verdict, 'BEARISH_CONFLUENCE');
    });

    it('empty decisions returns neutral result', () => {
        const result = computeConviction([], ['15m', '1h']);

        assert.equal(result.conviction, 0);
        assert.equal(result.verdict, 'MIXED');
        assert.equal(result.buys, 0);
        assert.equal(result.sells, 0);
        assert.equal(result.holds, 0);
    });

    it('handles mixed signals near the boundary of verdict thresholds', () => {
        // Conviction = 0.4 → MIXED (below 0.5 threshold)
        const decisions = [
            { timeframe: '15m', action: 'HOLD' },
            { timeframe: '1h', action: 'BUY' },
            { timeframe: '4h', action: 'HOLD' },
        ];
        const result = computeConviction(decisions, ['15m', '1h', '4h']);

        // Weighted: (1*0 + 2*1 + 3*0) / (1+2+3) = 2/6 = 0.333...
        assert.equal(result.conviction, 1 / 3);
        assert.equal(result.verdict, 'MIXED');
    });

});

// ─── renderMTFReport ────────────────────────────────────────────────────────────────

describe('renderMTFReport', () => {

    it('includes all key column headers', async () => {
        const result = await scanMTF('TEST-USD', ['15m', '1h'], { '15m': 1, '1h': 2 }, { seed: 42 });
        const output = renderMTFReport(result);

        assert.ok(output.includes('TIMEFRAME'), 'should include TIMEFRAME column');
        assert.ok(output.includes('ACTION'), 'should include ACTION column');
        assert.ok(output.includes('SCORE'), 'should include SCORE column');
        assert.ok(output.includes('CONFIDENCE'), 'should include CONFIDENCE column');
        assert.ok(output.includes('REGIME'), 'should include REGIME column');
        assert.ok(output.includes('CANDLES'), 'should include CANDLES column');
    });

    it('includes summary line with conviction and verdict', async () => {
        const result = await scanMTF('TEST-USD', ['1h'], undefined, { seed: 42 });
        const output = renderMTFReport(result);

        assert.ok(output.includes('Conviction'), 'should include Conviction label');
        assert.ok(output.includes('Verdict'), 'should include Verdict label');
        assert.ok(output.includes('BUY') || output.includes('SELL') || output.includes('HOLD'),
            'should include action counts');
    });

    it('returns fallback for null/undefined input', () => {
        assert.equal(renderMTFReport(null), '(no data)');
        assert.equal(renderMTFReport(undefined), '(no data)');
    });

    it('displays each timeframe from the result', async () => {
        const timeframes = ['15m', '1h'];
        const result = await scanMTF('TEST-USD', timeframes, { '15m': 1, '1h': 2 }, { seed: 42 });
        const output = renderMTFReport(result);

        for (const tf of timeframes) {
            assert.ok(output.includes(tf), `should include timeframe ${tf} in report`);
        }
    });

});
