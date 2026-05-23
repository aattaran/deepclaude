/**
 * Audit tools — unit tests
 * Uses Node.js built-in test runner (node:test).
 *
 * Run: node --test C:/Users/User/deepclaude/audit/audit.test.js
 */

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { validateTradeRecord, generateId } from './schemas.mjs';
import { audit } from './performance-audit-hook.mjs';
import { getExitPrice } from './backtest.mjs';

// --------------------------------------------------------------------------
// 1. schemas.mjs — validateTradeRecord, generateId
// --------------------------------------------------------------------------
describe('schemas.mjs', () => {
  describe('validateTradeRecord', () => {
    const validRecord = {
      symbol: 'MNQ',
      direction: 'long',
      entry_price: 20000,
      exit_price: 20100,
      quantity: 1,
      pnl_usd: 200,
      hold_sec: 3600,
      entry_time: 1000,
      exit_time: 5000,
    };

    it('returns valid for a well-formed trade record', () => {
      const result = validateTradeRecord(validRecord);
      assert.equal(result.valid, true);
      assert.deepEqual(result.errors, []);
    });

    it('rejects record with missing symbol', () => {
      const record = { ...validRecord, symbol: '' };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('symbol required'));
    });

    it('rejects record with missing direction', () => {
      const record = { ...validRecord, direction: '' };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('direction must be long|short'));
    });

    it('rejects record with invalid direction', () => {
      const record = { ...validRecord, direction: 'sideways' };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('direction must be long|short'));
    });

    it('rejects record with non-numeric pnl_usd', () => {
      const record = { ...validRecord, pnl_usd: 'string' };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('pnl_usd must be number'));
    });

    it('rejects record with non-numeric entry_price', () => {
      const record = { ...validRecord, entry_price: 'nan' };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('entry_price must be number'));
    });

    it('rejects record with non-numeric exit_price', () => {
      const record = { ...validRecord, exit_price: null };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('exit_price must be number'));
    });

    it('rejects record with non-numeric hold_sec', () => {
      const record = { ...validRecord, hold_sec: null };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, false);
      assert.ok(result.errors.includes('hold_sec must be number'));
    });

    it('does NOT validate quantity — negative quantity passes', () => {
      const record = { ...validRecord, quantity: -5 };
      const result = validateTradeRecord(record);
      assert.equal(result.valid, true);
    });

    it('does NOT require an id field — record without id passes', () => {
      const { id, ...recordWithoutId } = validRecord;
      const result = validateTradeRecord(recordWithoutId);
      assert.equal(result.valid, true);
    });
  });

  describe('generateId', () => {
    it('returns string matching "prefix_timestamp_rand" format', () => {
      const id = generateId('bt');
      assert.match(id, /^bt_\d+_[0-9a-z]+$/);
    });

    it('defaults to "audit" prefix when none given', () => {
      const id = generateId();
      assert.match(id, /^audit_\d+_[0-9a-z]+$/);
    });

    it('produces unique values on consecutive calls', () => {
      const a = generateId();
      const b = generateId();
      assert.notEqual(a, b);
    });
  });
});

// --------------------------------------------------------------------------
// 2. performance-audit-hook.mjs — audit()
// --------------------------------------------------------------------------
describe('performance-audit-hook.mjs', () => {
  describe('audit', () => {
    const base = Date.now();
    // Spread trades across 3 different days so daily returns have variance
    const sampleTrades = [
      {
        symbol: 'MNQ', direction: 'long', entry_price: 20000, exit_price: 20200,
        pnl_usd: 1000, quantity: 1, hold_sec: 3600,
        entry_time: base - 3 * 86400000, exit_time: base - 3 * 86400000 + 100,
      },
      {
        symbol: 'MNQ', direction: 'short', entry_price: 20200, exit_price: 20050,
        pnl_usd: 600, quantity: 1, hold_sec: 1800,
        entry_time: base - 2 * 86400000, exit_time: base - 2 * 86400000 + 200,
      },
      {
        symbol: 'MNQ', direction: 'long', entry_price: 20050, exit_price: 20300,
        pnl_usd: 500, quantity: 1, hold_sec: 2400,
        entry_time: base - 1 * 86400000, exit_time: base - 1 * 86400000 + 150,
      },
    ];

    it('returns a numeric Sharpe ratio', () => {
      const r = audit(sampleTrades, { timeframe: '30d', symbol: 'MNQ' });
      assert.equal(typeof r.sharpe_ratio, 'number');
      assert.ok(!Number.isNaN(r.sharpe_ratio));
    });

    it('returns a numeric Sortino ratio', () => {
      const r = audit(sampleTrades, { timeframe: '30d', symbol: 'MNQ' });
      assert.equal(typeof r.sortino_ratio, 'number');
      assert.ok(!Number.isNaN(r.sortino_ratio));
    });

    it('returns non-negative MaxDrawdown', () => {
      const r = audit(sampleTrades, { timeframe: '30d', symbol: 'MNQ' });
      assert.ok(r.max_drawdown >= 0);
      assert.ok(r.max_drawdown_usd >= 0);
    });

    it('uses default startingEquity of 50000', () => {
      const r = audit(sampleTrades, { timeframe: '30d', symbol: 'MNQ' });
      // All trades profitable → equity only rises → drawdown is 0
      assert.equal(r.max_drawdown, 0);
    });

    it('accepts custom startingEquity of 25000', () => {
      const r = audit(sampleTrades, {
        timeframe: '30d', symbol: 'MNQ', startingEquity: 25000,
      });
      assert.equal(typeof r.sharpe_ratio, 'number');
      assert.equal(typeof r.sortino_ratio, 'number');
      assert.ok(r.max_drawdown >= 0);
    });

    it('returns correct result shape with aggregate values', () => {
      const r = audit(sampleTrades, { timeframe: '30d', symbol: 'MNQ' });
      assert.ok(r.audit_id.startsWith('audit_'));
      assert.equal(r.timeframe, '30d');
      assert.equal(r.symbol, 'MNQ');
      assert.equal(r.total_trades, 3);
      assert.equal(r.win_rate, 1);                // all wins
      assert.equal(r.total_pnl_usd, 2100);         // 1000 + 600 + 500
      assert.equal(r.avg_pnl_per_trade, 700);      // 2100 / 3
      assert.equal(typeof r.profit_factor, 'number');
      assert.ok(Array.isArray(r.warnings));
      assert.equal(r.status, 'GREEN');
    });
  });
});

// --------------------------------------------------------------------------
// 3. backtest.mjs — getExitPrice()
// --------------------------------------------------------------------------
describe('backtest.mjs', () => {
  describe('getExitPrice', () => {
    const entryPrice = 100;
    const futureCandles = [{}, {}]; // need at least 2 so it's not end-of-data

    it('triggers stop-loss when price drops 3% below entry for a long', () => {
      const position = { direction: 'long', entryPrice };
      const candle = { open: 98, high: 99, low: 95, close: 96, timestamp: 2000 };

      // stopPrice = 100 * (1 - 0.03) = 97; candle.low (95) <= 97 -> trigger
      const exit = getExitPrice(position, candle, futureCandles, [], 0.03, 0.05);
      assert.equal(exit.price, 97);
      assert.equal(exit.reason, 'stop_loss');
    });

    it('triggers take-profit when price rises 5% above entry for a long', () => {
      const position = { direction: 'long', entryPrice };
      const candle = { open: 100, high: 107, low: 99, close: 106, timestamp: 2000 };

      // stopPrice = 97, candle.low (99) > 97 -> stop skipped
      // targetPrice = 105, candle.high (107) >= 105 -> trigger
      const exit = getExitPrice(position, candle, futureCandles, [], 0.03, 0.05);
      assert.equal(exit.price, 105);
      assert.equal(exit.reason, 'take_profit');
    });

    it('returns null (hold) when no exit condition is met', () => {
      const position = { direction: 'long', entryPrice };
      const candle = { open: 100, high: 102, low: 98.5, close: 101, timestamp: 2000 };

      // stopPrice = 97, candle.low (98.5) > 97 -> no stop
      // targetPrice = 105, candle.high (102) < 105 -> no TP
      // history.length < 30 -> no SMA crossover -> hold
      const exit = getExitPrice(position, candle, futureCandles, [], 0.03, 0.05);
      assert.strictEqual(exit, null);
    });

    it('forces exit at end of data with end_of_data reason', () => {
      const position = { direction: 'long', entryPrice };
      const candle = { open: 100, high: 102, low: 98.5, close: 101, timestamp: 2000 };

      // Only 1 futureCandle signals end of data -> force close
      const exit = getExitPrice(position, candle, [{}], [], 0.03, 0.05);
      assert.equal(exit.price, 101);
      assert.equal(exit.reason, 'end_of_data');
    });
  });
});
