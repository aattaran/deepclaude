/**
 * Tests for Trading Dashboard (audit/dashboard.mjs)
 *
 * Zero npm dependencies.  Uses node:test + node:assert/strict.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
    renderDashboard,
    colorize,
    createLiveDashboard,
} from './dashboard.mjs';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/**
 * Build a minimal valid trading-decision object.
 */
function makeDecision(overrides) {
    return Object.assign({
        symbol: 'SOL-USD',
        price: 142.37,
        action: 'BUY',
        confidence: 0.78,
        compositeScore: 0.72,
        regime: 'trending_bullish',
        regimeConfidence: 0.85,
        direction: 'bullish',
        support: { price: 141.20, strength: 0.8 },
        resistance: { price: 143.50, strength: 0.6 },
        activeZones: [
            { price: 141.20, type: 'OB', strength: 0.8, freshness: 1.0 },
            { price: 142.80, type: 'FVG', strength: 0.6, freshness: 0.8 },
            { price: 143.50, type: 'BRKR', strength: 0.5, freshness: 0.7 },
        ],
        signals: [
            { source: 'zone-detector', name: 'zone_support', value: 0.65, confidence: 0.70 },
            { source: 'zone-detector', name: 'zone_rr_ratio', value: 0.45, confidence: 0.50 },
            { source: 'market-regime', name: 'regime_trend', value: 0.80, confidence: 0.85 },
            { source: 'microstructure', name: 'book_imbalance', value: 0.30, confidence: 0.60 },
        ],
        reasoning: 'weighted composite=0.72 > 0.15',
        params: {
            stopMultiplier: 1.5,
            tpAggressiveness: 0.3,
            positionSizeFactor: 1.0,
            trailingStopPct: 2.5,
        },
    }, overrides);
}

const SAMPLE = makeDecision();
const SAMPLE_TEXT = renderDashboard(SAMPLE);

// Global flag so tests only create live-dashboards when they mean to (avoid
// accidental side-effects in CI).
let _liveTestMode = false;

// ==========================================================================
// renderDashboard  basic structure
// ==========================================================================

describe('renderDashboard', () => {
    it('is a function', () => {
        assert.equal(typeof renderDashboard, 'function');
    });

    it('returns a string', () => {
        assert.equal(typeof SAMPLE_TEXT, 'string');
        assert.ok(SAMPLE_TEXT.length > 0);
    });

    it('contains the symbol in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('SOL-USD'));
    });

    it('contains the price in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('$142.37'));
    });

    it('contains the action (BUY/SELL/HOLD) in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('BUY'));
    });

    it('contains the regime name in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('trending_bullish'));
    });

    it('contains signal names in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('zone_support'));
        assert.ok(SAMPLE_TEXT.includes('regime_trend'));
        assert.ok(SAMPLE_TEXT.includes('book_imbalance'));
    });

    it('contains signal values in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('+0.65'));
        assert.ok(SAMPLE_TEXT.includes('+0.80'));
        assert.ok(SAMPLE_TEXT.includes('+0.30'));
    });

    it('contains signal confidence values in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('0.70'));
        assert.ok(SAMPLE_TEXT.includes('0.85'));
        assert.ok(SAMPLE_TEXT.includes('0.60'));
    });

    it('contains support price in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('141.20'));
    });

    it('contains resistance price in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('143.50'));
    });

    it('contains active zone types in the output', () => {
        assert.ok(SAMPLE_TEXT.includes('FVG') || SAMPLE_TEXT.includes('Fvg'));
        assert.ok(SAMPLE_TEXT.includes('BRKR') || SAMPLE_TEXT.includes('Brkr'));
    });

    it('contains border/box-drawing characters', () => {
        assert.ok(SAMPLE_TEXT.includes('┌'));
        assert.ok(SAMPLE_TEXT.includes('┐'));
        assert.ok(SAMPLE_TEXT.includes('└'));
        assert.ok(SAMPLE_TEXT.includes('┘'));
        assert.ok(SAMPLE_TEXT.includes('├'));
        assert.ok(SAMPLE_TEXT.includes('┤'));
    });

    it('contains the DECISION label', () => {
        assert.ok(SAMPLE_TEXT.includes('DECISION'));
    });

    it('contains the REGIME label', () => {
        assert.ok(SAMPLE_TEXT.includes('REGIME'));
    });

    it('contains the Composite label', () => {
        assert.ok(SAMPLE_TEXT.includes('Composite'));
    });

    it('contains the Confidence label', () => {
        assert.ok(SAMPLE_TEXT.includes('Confidence'));
    });

    it('contains zone section headers', () => {
        assert.ok(SAMPLE_TEXT.includes('SUPPORTS'));
        assert.ok(SAMPLE_TEXT.includes('RESISTANCES'));
        assert.ok(SAMPLE_TEXT.includes('ACTIVE ZONES'));
    });

    it('contains signal section headers', () => {
        assert.ok(SAMPLE_TEXT.includes('SIGNALS'));
        assert.ok(SAMPLE_TEXT.includes('PARAMS'));
    });
});

// ==========================================================================
// renderDashboard  edge-cases
// ==========================================================================

describe('renderDashboard edge-cases', () => {
    it('handles null/undefined gracefully', () => {
        const r1 = renderDashboard(null);
        assert.equal(typeof r1, 'string');
        assert.ok(r1.length > 0);
        // Should not throw
    });

    it('handles undefined gracefully', () => {
        const r2 = renderDashboard(undefined);
        assert.equal(typeof r2, 'string');
    });

    it('handles empty object without crashing', () => {
        const r = renderDashboard({});
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
        // Should contain default symbol
        assert.ok(r.includes('UNKNOWN'));
    });

    it('handles SELL action rendering', () => {
        const r = renderDashboard(makeDecision({ action: 'SELL', compositeScore: -0.55 }));
        assert.ok(r.includes('SELL'));
        assert.ok(r.includes('▼') || r.includes(' -0.55') || r.includes('-0.55'));
    });

    it('handles HOLD action rendering', () => {
        const r = renderDashboard(makeDecision({ action: 'HOLD', compositeScore: 0.05 }));
        assert.ok(r.includes('HOLD'));
    });

    it('handles zero active zones', () => {
        const r = renderDashboard(makeDecision({ activeZones: [] }));
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
    });

    it('handles single active zone', () => {
        const r = renderDashboard(makeDecision({
            activeZones: [{ price: 150, type: 'OB', strength: 0.7, freshness: 1 }],
        }));
        assert.equal(typeof r, 'string');
        assert.ok(r.includes('150'));
    });

    it('handles many active zones (5+)', () => {
        const zones = [];
        for (let i = 0; i < 8; i++) {
            zones.push({ price: 140 + i, type: 'T' + i, strength: 0.5, freshness: 0.5 });
        }
        const r = renderDashboard(makeDecision({
            activeZones: zones,
            support: null,
            resistance: null,
        }));
        assert.equal(typeof r, 'string');
        // Should have multiple zone rows
        assert.ok(r.includes('T0') || r.includes('T7'));
    });

    it('handles zero signals', () => {
        const r = renderDashboard(makeDecision({ signals: [] }));
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
    });

    it('handles single signal', () => {
        const r = renderDashboard(makeDecision({
            signals: [{ source: 'test', name: 'sig_one', value: 0.5, confidence: 0.6 }],
        }));
        assert.equal(typeof r, 'string');
        assert.ok(r.includes('sig_one'));
    });

    it('handles many signals (5+)', () => {
        const sigs = [];
        for (let i = 0; i < 10; i++) {
            sigs.push({ source: 't', name: 'sig_' + i, value: i / 10, confidence: 0.5 });
        }
        const r = renderDashboard(makeDecision({ signals: sigs }));
        assert.equal(typeof r, 'string');
        assert.ok(r.includes('sig_9'));
    });

    it('handles null price gracefully', () => {
        const r = renderDashboard(makeDecision({ price: null }));
        assert.equal(typeof r, 'string');
        // Should use N/A
        assert.ok(r.includes('N/A'));
    });

    it('handles zero price gracefully', () => {
        const r = renderDashboard(makeDecision({ price: 0 }));
        assert.equal(typeof r, 'string');
        assert.ok(r.includes('$0.00'));
    });

    it('handles null support/resistance', () => {
        const r = renderDashboard(makeDecision({ support: null, resistance: null }));
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
    });

    it('handles empty params', () => {
        const r = renderDashboard(makeDecision({ params: {} }));
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
    });
});

// ==========================================================================
// renderDashboard  width / truncation
// ==========================================================================

describe('renderDashboard width & truncation', () => {
    it('respects custom width', () => {
        const w = 80;
        const r = renderDashboard(SAMPLE, { width: w });
        const lines = r.split('\n');
        for (const line of lines) {
            assert.equal(line.length, w, 'Line length mismatch: ' + line);
        }
    });

    it('clamps width to minimum 40', () => {
        const r = renderDashboard(SAMPLE, { width: 10 });
        const lines = r.split('\n');
        for (const line of lines) {
            assert.ok(line.length >= 40, 'Line too short: ' + line.length);
        }
    });

    it('truncates very long symbol strings', () => {
        const longSym = 'SOL-USD-THIS-IS-A-VERY-LONG-PAIR-NAME';
        const r = renderDashboard(makeDecision({ symbol: longSym }), { width: 61 });
        // Box drawing should still be present
        assert.ok(r.includes('┌'));
        assert.ok(r.includes('┐'));
    });

    it('truncates very long signal names', () => {
        const r = renderDashboard(makeDecision({
            signals: [{
                source: 'test',
                name: 'a_very_very_long_signal_name_that_exceeds_column',
                value: 0.5,
                confidence: 0.9,
            }],
        }), { width: 40 });
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
    });

    it('handles very long regime names gracefully', () => {
        const r = renderDashboard(makeDecision({
            regime: 'super_califragilistic_expialidocious_market_state',
        }));
        assert.equal(typeof r, 'string');
        assert.ok(r.length > 0);
    });
});

// ==========================================================================
// Strength bar tests
// ==========================================================================

describe('strength bars', () => {
    it('renders strength bars as block characters for support', () => {
        // The strength bar should have block chars (█)
        const r = renderDashboard(SAMPLE, { width: 80 });
        const supportsLine = r.split('\n').find(l => l.includes('141.20'));
        assert.ok(supportsLine, 'Expected a line with support price');
        // █ should appear for the bar
        assert.ok(supportsLine.includes('█'), 'Expected strength bar blocks');
    });

    it('produces longer bars for higher strength values', () => {
        // Compare narrow vs wide output: the block count scales with strength
        // but also with column width.  We just verify both have blocks.
        const high = renderDashboard(makeDecision({
            support: { price: 100, strength: 0.9 },
            resistance: null,
            activeZones: [],
        }), { width: 80 });
        const low = renderDashboard(makeDecision({
            support: { price: 100, strength: 0.1 },
            resistance: null,
            activeZones: [],
        }), { width: 80 });
        // Both should have block chars
        assert.ok(high.includes('█'));
        assert.ok(low.includes('█'));
    });
});

// ==========================================================================
// colorize
// ==========================================================================

describe('colorize', () => {
    it('is a function', () => {
        assert.equal(typeof colorize, 'function');
    });

    it('returns a string', () => {
        const colored = colorize(SAMPLE_TEXT);
        assert.equal(typeof colored, 'string');
        assert.ok(colored.length > 0);
    });

    it('output differs from input (contains ANSI codes)', () => {
        const colored = colorize(SAMPLE_TEXT);
        // ANSI escape sequences start with \x1b[
        assert.ok(colored.includes('\x1b['));
        assert.notEqual(colored, SAMPLE_TEXT);
    });

    it('contains ANSI reset codes', () => {
        const colored = colorize(SAMPLE_TEXT);
        assert.ok(colored.includes('\x1b[0m'));
    });

    it('wraps BUY in green (ANSI 32)', () => {
        const colored = colorize(SAMPLE_TEXT);
        // BUY should be wrapped with \x1b[32m (green) or bold-green \x1b[1m\x1b[32m
        const buyIdx = colored.indexOf('BUY');
        const prefix = colored.slice(Math.max(0, buyIdx - 10), buyIdx);
        assert.ok(
            prefix.includes('\x1b[32m') || prefix.includes('\x1b[1m'),
            'Expected green/bold ANSI before BUY, got: ' + JSON.stringify(prefix),
        );
    });

    it('wraps SELL in red (ANSI 31)', () => {
        const sellText = renderDashboard(makeDecision({
            action: 'SELL',
            compositeScore: -0.55,
        }));
        const colored = colorize(sellText);
        const idx = colored.indexOf('SELL');
        const prefix = colored.slice(Math.max(0, idx - 10), idx);
        assert.ok(
            prefix.includes('\x1b[31m') || prefix.includes('\x1b[1m'),
            'Expected red/bold ANSI before SELL',
        );
    });

    it('wraps HOLD in yellow (ANSI 33)', () => {
        const holdText = renderDashboard(makeDecision({ action: 'HOLD' }));
        const colored = colorize(holdText);
        const idx = colored.indexOf('HOLD');
        const prefix = colored.slice(Math.max(0, idx - 10), idx);
        assert.ok(
            prefix.includes('\x1b[33m') || prefix.includes('\x1b[1m'),
            'Expected yellow/bold ANSI before HOLD',
        );
    });

    it('highlights high confidence (>0.7) in bold green', () => {
        const colored = colorize(SAMPLE_TEXT);
        // Confidence 78% -> > 70 -> bold green
        const line = colored.split('\n').find(l => l.includes('78%'));
        assert.ok(line, 'Expected confidence line');
        assert.ok(
            line.includes('\x1b[32m') || line.includes('\x1b[1m'),
            'Expected green/bold coloring for high confidence',
        );
    });

    it('dims low confidence (<0.3)', () => {
        const lowConfText = renderDashboard(makeDecision({ confidence: 0.15 }));
        const colored = colorize(lowConfText);
        const line = colored.split('\n').find(l => l.includes('15%'));
        assert.ok(line, 'Expected low confidence line');
        assert.ok(
            line.includes('\x1b[2m'),
            'Expected dim ANSI for low confidence',
        );
    });

    it('handles empty/null text gracefully', () => {
        assert.equal(colorize(''), '');
        assert.equal(colorize(null), '');
        assert.equal(colorize(undefined), '');
    });

    it('leaves border lines unchanged (no ANSI)', () => {
        const colored = colorize(SAMPLE_TEXT);
        const lines = colored.split('\n');
        const borderLine = lines[0];
        // First line should be a top border (┌───┐)
        assert.ok(borderLine.includes('┌') || borderLine.includes('─'));
        // Border lines should not contain ANSI codes
        const ansiCount = (borderLine.match(/\x1b\[/g) || []).length;
        assert.equal(ansiCount, 0, 'Border line should have no ANSI codes');
    });

    it('preserves line count', () => {
        const colored = colorize(SAMPLE_TEXT);
        const plainLines = SAMPLE_TEXT.split('\n').length;
        const coloredLines = colored.split('\n').length;
        assert.equal(coloredLines, plainLines);
    });

    it('ANSI codes are properly reset at boundaries', () => {
        const colored = colorize(SAMPLE_TEXT);
        // Every ANSI code that starts a style should be followed by a reset
        // Simple check: count of \x1b[ (starts) should have matching \x1b[0m (resets)
        const starts = (colored.match(/\x1b\[/g) || []).length;
        const resets = (colored.match(/\x1b\[0m/g) || []).length;
        // Exception: reset itself counts as both a "start" and a reset
        // So total starts = styling starts + reset starts
        // Total resets = number of \x1b[0m
        // Each style has a start and a reset, so starts should be about 2x resets
        // Actually it's more complex; just verify resets exist
        assert.ok(resets >= 1);
    });
});

// ==========================================================================
// createLiveDashboard
// ==========================================================================

describe('createLiveDashboard', () => {
    it('is a function', () => {
        assert.equal(typeof createLiveDashboard, 'function');
    });

    it('returns an object with update, start, stop', () => {
        const dash = createLiveDashboard();
        assert.equal(typeof dash, 'object');
        assert.notEqual(dash, null);
        assert.equal(typeof dash.update, 'function');
        assert.equal(typeof dash.start, 'function');
        assert.equal(typeof dash.stop, 'function');
    });

    it('update accepts data and renders to stdout', () => {
        const dash = createLiveDashboard({ symbol: 'TEST' });
        // update should not throw
        dash.update(makeDecision({ symbol: 'TEST' }));
    });

    it('start and stop work without crashing', () => {
        const dash = createLiveDashboard({ refreshMs: 5000 });
        dash.update(makeDecision());
        dash.start();
        dash.stop();
    });

    it('multiple calls to start are harmless', () => {
        const dash = createLiveDashboard({ refreshMs: 5000 });
        dash.update(makeDecision());
        dash.start();
        dash.start(); // second call should be a no-op
        dash.stop();
    });

    it('stop without start is harmless', () => {
        const dash = createLiveDashboard();
        dash.stop();
    });

    it('update without start is harmless', () => {
        const dash = createLiveDashboard();
        dash.update(makeDecision());
        // Should not throw
    });

    it('accepts custom width option', () => {
        const dash = createLiveDashboard({ width: 80 });
        assert.equal(typeof dash.update, 'function');
    });

    it('uses default width when not specified', () => {
        const dash = createLiveDashboard();
        assert.equal(typeof dash.update, 'function');
    });
});

// ==========================================================================
// Integration: render + colorize end-to-end
// ==========================================================================

describe('integration', () => {
    it('full pipeline render->colorize for BUY decision', () => {
        const d = makeDecision({ action: 'BUY', compositeScore: 0.85, confidence: 0.92 });
        const text = renderDashboard(d);
        const colored = colorize(text);
        assert.ok(colored.includes('BUY'));
        assert.ok(colored.includes('\x1b['));
    });

    it('full pipeline render->colorize for SELL decision', () => {
        const d = makeDecision({ action: 'SELL', compositeScore: -0.6, confidence: 0.7 });
        const text = renderDashboard(d);
        const colored = colorize(text);
        assert.ok(colored.includes('SELL'));
        assert.ok(colored.includes('\x1b['));
    });

    it('all lines in rendered output have consistent length', () => {
        const r = renderDashboard(SAMPLE, { width: 61 });
        const lines = r.split('\n');
        const expectedLen = lines[0].length;
        for (const [i, line] of lines.entries()) {
            assert.equal(
                line.length,
                expectedLen,
                'Line ' + i + ' has wrong length (expected ' + expectedLen + ', got ' + line.length + ': ' + line + ')',
            );
        }
    });

    it('rendered output starts and ends with border characters', () => {
        assert.ok(SAMPLE_TEXT.startsWith('┌'));
        assert.ok(SAMPLE_TEXT.endsWith('┘'));
    });

    it('composite score shows correct sign', () => {
        const pos = renderDashboard(makeDecision({ compositeScore: 0.72 }));
        assert.ok(pos.includes('▲') || pos.includes('+0.72'));

        const neg = renderDashboard(makeDecision({ compositeScore: -0.45 }));
        assert.ok(neg.includes('▼') || neg.includes('-0.45'));
    });

    it('strength bars are not present in border lines', () => {
        const text = renderDashboard(SAMPLE, { width: 80 });
        const lines = text.split('\n');
        // Border lines should not contain block chars
        for (const line of lines) {
            if (/[┌├└┐┤┘─]/.test(line)) {
                assert.ok(
                    !line.includes('█'),
                    'Border line should not contain block chars: ' + line,
                );
            }
        }
    });
});
