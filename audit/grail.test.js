/**
 * Grail Demo — integration tests (node:test runner)
 * Run: node --test audit/grail.test.js
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { runQuick, runOptimize, runFull } from './grail.mjs';
import { generateSyntheticCandles } from './trade.mjs';
import { renderDashboard, colorize } from './dashboard.mjs';

// ─── Helpers ────────────────────────────────────────────────────────────────────

function captureStdout(fn) {
    const logs = [];
    const orig = console.log;
    console.log = (...args) => logs.push(args.join(' '));
    try {
        fn();
    } finally {
        console.log = orig;
    }
    return logs.join('\n');
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

describe('grail demo — quick mode', () => {
    it('runs without throwing', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        assert.doesNotThrow(() => {
            const output = captureStdout(() => runQuick(args, candles));
            assert.ok(output.includes('Data Summary'), 'has data summary');
            assert.ok(output.includes('Live Analysis'), 'has live analysis');
            assert.ok(output.includes('Backtest Results'), 'has backtest results');
            assert.ok(output.includes('Signals'), 'has signals');
        });
    });

    it('shows BUY/SELL/HOLD action', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'TEST-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        const output = captureStdout(() => runQuick(args, candles));
        const hasAction = output.includes('BUY') || output.includes('SELL') || output.includes('HOLD');
        assert.ok(hasAction, 'output includes an action');
    });

    it('shows regime data', () => {
        const candles = generateSyntheticCandles(500, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 15,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        const output = captureStdout(() => runQuick(args, candles));
        // Regime info is always present in Live Analysis
        assert.ok(output.includes('Regime'), 'has regime info');
        assert.ok(output.includes('Regime Mix'), 'has regime mix bar');
    });

    it('handles small candle sets', () => {
        const candles = generateSyntheticCandles(60, { seed: 1, volatility: 0.01 });
        const args = {
            symbol: 'TINY-USD',
            days: 2,
            timeframe: '15m',
            seed: 1,
            volatility: 0.01,
            basePrice: 100,
        };

        assert.doesNotThrow(() => {
            const output = captureStdout(() => runQuick(args, candles));
            assert.ok(output.length > 0);
        });
    });

    it('works with different symbols', () => {
        for (const sym of ['SOL-USD', 'MBT', 'ETH-USD']) {
            const candles = generateSyntheticCandles(100, { seed: 42 });
            const args = {
                symbol: sym,
                days: 5,
                timeframe: '15m',
                seed: 42,
                volatility: 0.012,
                basePrice: sym === 'MBT' ? 75000 : 100,
            };
            assert.doesNotThrow(() => {
                captureStdout(() => runQuick(args, candles));
            }, `${sym} should not throw`);
        }
    });
});

describe('grail demo — optimize mode', () => {
    it('runs grid search + hill climb without throwing', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        assert.doesNotThrow(() => {
            const output = captureStdout(() => runOptimize(args, candles));
            assert.ok(output.includes('Grid Search'), 'has grid search');
            assert.ok(output.includes('Hill Climb'), 'has hill climb');
            assert.ok(output.includes('Optimization complete'), 'completes');
        });
    });

    it('shows grid best parameters', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        const output = captureStdout(() => runOptimize(args, candles));
        assert.ok(output.includes('Grid Best'), 'has grid best section');
    });
});

describe('grail demo — full mode', () => {
    it('runs full pipeline without throwing', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        assert.doesNotThrow(() => {
            const output = captureStdout(() => runFull(args, candles));
            assert.ok(output.includes('Optimized Parameters'), 'has optimized params');
            assert.ok(output.includes('Backtest Performance'), 'has backtest perf');
            assert.ok(output.includes('Verdict'), 'has verdict');
        });
    });

    it('includes a verdict (PROFITABLE, GRAIL, or NOT YIELDING)', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        const output = captureStdout(() => runFull(args, candles));
        const hasVerdict = output.includes('GRAIL') || output.includes('PROFITABLE') || output.includes('NOT YIELDING');
        assert.ok(hasVerdict, 'has a verdict');
    });

    it('renders dashboard content in output', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.012 });
        const args = {
            symbol: 'BTC-USD',
            days: 10,
            timeframe: '15m',
            seed: 42,
            volatility: 0.012,
            basePrice: 100,
        };

        const output = captureStdout(() => runFull(args, candles));
        // Dashboard should have at least the symbol and some structure
        assert.ok(output.includes('BTC-USD'), 'has symbol');
    });

    it('handles high-volatility data', () => {
        const candles = generateSyntheticCandles(300, { seed: 42, volatility: 0.03 });
        const args = {
            symbol: 'WILD-USD',
            days: 5,
            timeframe: '5m',
            seed: 42,
            volatility: 0.03,
            basePrice: 100,
        };

        assert.doesNotThrow(() => {
            const output = captureStdout(() => runFull(args, candles));
            assert.ok(output.length > 0);
        });
    });
});

describe('grail integration — dashboard rendering', () => {
    it('produces valid dashboard from backtest data', () => {
        const data = {
            symbol: 'BTC-USD',
            price: 123.45,
            action: 'BUY',
            confidence: 0.65,
            compositeScore: 0.42,
            regime: 'trending_bullish',
            regimeConfidence: 0.8,
            signals: [
                { source: 'test', name: 'backtest_winrate', value: 0.6, confidence: 0.8 },
            ],
            params: { stopLossPct: 0.02, takeProfitPct: 0.04 },
        };

        const dashboard = renderDashboard(data, { width: 61 });
        assert.ok(dashboard.includes('BTC-USD'), 'dashboard has symbol');
        assert.ok(dashboard.includes('BUY'), 'dashboard has action');
        assert.ok(dashboard.length > 50, 'dashboard has content');

        const colored = colorize(dashboard);
        assert.ok(colored.length > 0, 'colorize produces output');
    });
});

describe('grail edge cases', () => {
    it('quick mode handles empty candle array', () => {
        const args = {
            symbol: 'EMPTY-USD',
            days: 1,
            timeframe: '15m',
            seed: 1,
            volatility: 0.01,
            basePrice: 100,
        };
        // Should not throw with empty candles
        assert.doesNotThrow(() => {
            const output = captureStdout(() => runQuick(args, []));
            assert.ok(output.includes('Data Summary'), 'produces output');
        });
    });

    it('optimize mode with minimal candle count', () => {
        const candles = generateSyntheticCandles(100, { seed: 1 });
        const args = {
            symbol: 'TINY-USD',
            days: 3,
            timeframe: '15m',
            seed: 1,
            volatility: 0.01,
            basePrice: 100,
        };

        assert.doesNotThrow(() => {
            const output = captureStdout(() => runOptimize(args, candles));
            assert.ok(output.length > 0);
        });
    });
});
