// audit/presets.test.js — Tests for trading parameter presets
//
// Run: node --test audit/presets.test.js

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  MBT_PRESET,
  BTC_USD_PRESET,
  SOL_USD_PRESET,
  ETH_USD_PRESET,
  DEFAULT_PRESET,
  AGGRESSIVE_PRESET,
  SCALPING_PRESET,
  getPreset,
  applyPreset,
} from './presets.mjs';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ALL_PRESETS = [
  { name: 'MBT_PRESET', preset: MBT_PRESET },
  { name: 'BTC_USD_PRESET', preset: BTC_USD_PRESET },
  { name: 'SOL_USD_PRESET', preset: SOL_USD_PRESET },
  { name: 'ETH_USD_PRESET', preset: ETH_USD_PRESET },
  { name: 'DEFAULT_PRESET', preset: DEFAULT_PRESET },
  { name: 'AGGRESSIVE_PRESET', preset: AGGRESSIVE_PRESET },
  { name: 'SCALPING_PRESET', preset: SCALPING_PRESET },
];

const REQUIRED_FIELDS = [
  'stopLossPct',
  'takeProfitPct',
  'cooldownBars',
  'warmupBars',
  'minConfidence',
  'fusionMethod',
  'zoneThreshold',
  'positionSize',
  'bestTimeframe',
  'regimePreference',
  'avoidRegimes',
  'maxDailyRisk',
  'commissionNote',
];

function isNonEmptyString(v) {
  return typeof v === 'string' && v.length > 0;
}

function isPositiveNumber(v) {
  return typeof v === 'number' && Number.isFinite(v) && v > 0;
}

function isNonNegativeNumber(v) {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0;
}

// ---------------------------------------------------------------------------
// 1. Every preset has all required fields
// ---------------------------------------------------------------------------

describe('all presets — structural integrity', () => {
  for (const { name, preset } of ALL_PRESETS) {
    it(`${name} has all required fields`, () => {
      for (const field of REQUIRED_FIELDS) {
        assert.ok(
          Object.hasOwn(preset, field),
          `${name} missing required field '${field}'`
        );
      }
    });

    it(`${name} has exactly the required fields (no extra)`, () => {
      const keys = Object.keys(preset);
      for (const key of keys) {
        assert.ok(
          REQUIRED_FIELDS.includes(key),
          `${name} has unexpected field '${key}'`
        );
      }
      assert.equal(keys.length, REQUIRED_FIELDS.length);
    });
  }
});

// ---------------------------------------------------------------------------
// 2. Field type / format validation
// ---------------------------------------------------------------------------

describe('all presets — field type validation', () => {
  for (const { name, preset } of ALL_PRESETS) {
    it(`${name} stopLossPct is a positive number`, () => {
      assert.ok(isPositiveNumber(preset.stopLossPct), `${name} stopLossPct invalid`);
    });

    it(`${name} takeProfitPct > stopLossPct`, () => {
      assert.ok(
        preset.takeProfitPct > preset.stopLossPct,
        `${name} takeProfitPct (${preset.takeProfitPct}) must be > stopLossPct (${preset.stopLossPct})`
      );
    });

    it(`${name} cooldownBars is a positive integer`, () => {
      assert.ok(Number.isInteger(preset.cooldownBars) && preset.cooldownBars > 0);
    });

    it(`${name} warmupBars is a positive integer`, () => {
      assert.ok(Number.isInteger(preset.warmupBars) && preset.warmupBars > 0);
    });

    it(`${name} minConfidence is in [0, 1]`, () => {
      assert.ok(
        preset.minConfidence >= 0 && preset.minConfidence <= 1,
        `${name} minConfidence ${preset.minConfidence} out of range`
      );
    });

    it(`${name} positionSize is a non-empty string`, () => {
      assert.ok(isNonEmptyString(preset.positionSize));
    });

    it(`${name} bestTimeframe is a non-empty string`, () => {
      assert.ok(isNonEmptyString(preset.bestTimeframe));
    });

    it(`${name} regimePreference is a non-empty array of strings`, () => {
      assert.ok(Array.isArray(preset.regimePreference) && preset.regimePreference.length > 0);
      for (const r of preset.regimePreference) {
        assert.ok(isNonEmptyString(r));
      }
    });

    it(`${name} avoidRegimes is a non-empty array of strings`, () => {
      assert.ok(Array.isArray(preset.avoidRegimes) && preset.avoidRegimes.length > 0);
      for (const r of preset.avoidRegimes) {
        assert.ok(isNonEmptyString(r));
      }
    });

    it(`${name} maxDailyRisk is a non-negative number`, () => {
      assert.ok(isNonNegativeNumber(preset.maxDailyRisk));
    });

    it(`${name} commissionNote is a non-empty string`, () => {
      assert.ok(isNonEmptyString(preset.commissionNote));
    });

    it(`${name} fusionMethod is a string`, () => {
      assert.ok(isNonEmptyString(preset.fusionMethod));
    });

    it(`${name} zoneThreshold is a positive number`, () => {
      assert.ok(isPositiveNumber(preset.zoneThreshold));
    });
  }
});

// ---------------------------------------------------------------------------
// 3. Numerical value range validation
// ---------------------------------------------------------------------------

describe('all presets — numerical ranges', () => {
  for (const { name, preset } of ALL_PRESETS) {
    it(`${name} stopLossPct in range [0.005, 0.10]`, () => {
      assert.ok(
        preset.stopLossPct >= 0.005 && preset.stopLossPct <= 0.10,
        `${name} stopLossPct ${preset.stopLossPct} outside [0.005, 0.10]`
      );
    });

    it(`${name} takeProfitPct in range [0.01, 0.20]`, () => {
      assert.ok(
        preset.takeProfitPct >= 0.01 && preset.takeProfitPct <= 0.20,
        `${name} takeProfitPct ${preset.takeProfitPct} outside [0.01, 0.20]`
      );
    });

    it(`${name} cooldownBars in range [1, 100]`, () => {
      assert.ok(
        preset.cooldownBars >= 1 && preset.cooldownBars <= 100,
        `${name} cooldownBars ${preset.cooldownBars} outside [1, 100]`
      );
    });

    it(`${name} warmupBars in range [10, 500]`, () => {
      assert.ok(
        preset.warmupBars >= 10 && preset.warmupBars <= 500,
        `${name} warmupBars ${preset.warmupBars} outside [10, 500]`
      );
    });
  }
});

// ---------------------------------------------------------------------------
// 4. MBT micro-contract invariants (research-backed)
// ---------------------------------------------------------------------------

describe('MBT_PRESET — micro-contract specialisation', () => {
  it('MBT stopLossPct is tighter than BTC_USD stopLossPct', () => {
    assert.ok(
      MBT_PRESET.stopLossPct < BTC_USD_PRESET.stopLossPct,
      `MBT stop ${MBT_PRESET.stopLossPct} should be < BTC stop ${BTC_USD_PRESET.stopLossPct}`
    );
  });

  it('MBT takeProfitPct is tighter than BTC_USD takeProfitPct', () => {
    assert.ok(
      MBT_PRESET.takeProfitPct < BTC_USD_PRESET.takeProfitPct,
      `MBT TP ${MBT_PRESET.takeProfitPct} should be < BTC TP ${BTC_USD_PRESET.takeProfitPct}`
    );
  });

  it('MBT cooldownBars is shorter than BTC_USD cooldownBars (frequent setups)', () => {
    assert.ok(
      MBT_PRESET.cooldownBars < BTC_USD_PRESET.cooldownBars
    );
  });

  it('MBT warmupBars is shorter than BTC_USD warmupBars', () => {
    assert.ok(
      MBT_PRESET.warmupBars < BTC_USD_PRESET.warmupBars
    );
  });

  it('MBT minConfidence is lower than BTC_USD minConfidence (more signals)', () => {
    assert.ok(
      MBT_PRESET.minConfidence < BTC_USD_PRESET.minConfidence
    );
  });

  it('MBT maxDailyRisk is smaller than BTC_USD maxDailyRisk', () => {
    assert.ok(
      MBT_PRESET.maxDailyRisk < BTC_USD_PRESET.maxDailyRisk
    );
  });

  it('MBT bestTimeframe is 5m (mean reversion sweet spot)', () => {
    assert.equal(MBT_PRESET.bestTimeframe, '5m');
  });

  it('MBT zoneThreshold is tighter than BTC_USD zoneThreshold', () => {
    assert.ok(
      MBT_PRESET.zoneThreshold < BTC_USD_PRESET.zoneThreshold
    );
  });
});

// ---------------------------------------------------------------------------
// 5. Scalping invariants
// ---------------------------------------------------------------------------

describe('SCALPING_PRESET — scalping specialisation', () => {
  it('has the tightest stopLossPct of all presets', () => {
    const others = ALL_PRESETS.filter((p) => p.name !== 'SCALPING_PRESET');
    for (const { name, preset } of others) {
      assert.ok(
        SCALPING_PRESET.stopLossPct <= preset.stopLossPct,
        `SCALPING stop ${SCALPING_PRESET.stopLossPct} should be <= ${name} stop ${preset.stopLossPct}`
      );
    }
  });

  it('has the shortest cooldownBars of all presets', () => {
    const others = ALL_PRESETS.filter((p) => p.name !== 'SCALPING_PRESET');
    for (const { name, preset } of others) {
      assert.ok(
        SCALPING_PRESET.cooldownBars <= preset.cooldownBars,
        `SCALPING cooldown ${SCALPING_PRESET.cooldownBars} should be <= ${name} cooldown ${preset.cooldownBars}`
      );
    }
  });

  it('has the smallest zoneThreshold of all presets', () => {
    const others = ALL_PRESETS.filter((p) => p.name !== 'SCALPING_PRESET');
    for (const { name, preset } of others) {
      assert.ok(
        SCALPING_PRESET.zoneThreshold <= preset.zoneThreshold,
        `SCALPING zone ${SCALPING_PRESET.zoneThreshold} should be <= ${name} zone ${preset.zoneThreshold}`
      );
    }
  });

  it('bestTimeframe is 1m', () => {
    assert.equal(SCALPING_PRESET.bestTimeframe, '1m');
  });
});

// ---------------------------------------------------------------------------
// 6. Aggressive invariants
// ---------------------------------------------------------------------------

describe('AGGRESSIVE_PRESET — aggressive specialisation', () => {
  it('has the widest stopLossPct', () => {
    const others = ALL_PRESETS.filter((p) => p.name !== 'AGGRESSIVE_PRESET');
    for (const { name, preset } of others) {
      assert.ok(
        AGGRESSIVE_PRESET.stopLossPct >= preset.stopLossPct,
        `AGGRESSIVE stop ${AGGRESSIVE_PRESET.stopLossPct} should be >= ${name} stop ${preset.stopLossPct}`
      );
    }
  });

  it('bestTimeframe is 1h', () => {
    assert.equal(AGGRESSIVE_PRESET.bestTimeframe, '1h');
  });
});

// ---------------------------------------------------------------------------
// 7. getPreset — symbol matching
// ---------------------------------------------------------------------------

describe('getPreset — symbol matching', () => {
  it('returns MBT_PRESET for "MBT"', () => {
    assert.equal(getPreset('MBT'), MBT_PRESET);
  });

  it('returns MBT_PRESET for "MBT-USD"', () => {
    assert.equal(getPreset('MBT-USD'), MBT_PRESET);
  });

  it('returns MBT_PRESET for "mbt" (case-insensitive)', () => {
    assert.equal(getPreset('mbt'), MBT_PRESET);
  });

  it('returns MBT_PRESET for "MICRO_BTC" (underscore variant)', () => {
    assert.equal(getPreset('MICRO_BTC'), MBT_PRESET);
  });

  it('returns BTC_USD_PRESET for "BTC-USD"', () => {
    assert.equal(getPreset('BTC-USD'), BTC_USD_PRESET);
  });

  it('returns BTC_USD_PRESET for "bitcoin" (case-insensitive)', () => {
    assert.equal(getPreset('bitcoin'), BTC_USD_PRESET);
  });

  it('returns SOL_USD_PRESET for "SOL"', () => {
    assert.equal(getPreset('SOL'), SOL_USD_PRESET);
  });

  it('returns SOL_USD_PRESET for "sol_usd" (underscore variant)', () => {
    assert.equal(getPreset('sol_usd'), SOL_USD_PRESET);
  });

  it('returns ETH_USD_PRESET for "ETH-USD"', () => {
    assert.equal(getPreset('ETH-USD'), ETH_USD_PRESET);
  });

  it('returns ETH_USD_PRESET for "ethereum" (full name)', () => {
    assert.equal(getPreset('ethereum'), ETH_USD_PRESET);
  });

  it('returns SCALPING_PRESET for "SCALP"', () => {
    assert.equal(getPreset('SCALP'), SCALPING_PRESET);
  });

  it('returns AGGRESSIVE_PRESET for "AGGRESSIVE"', () => {
    assert.equal(getPreset('AGGRESSIVE'), AGGRESSIVE_PRESET);
  });

  it('falls back to DEFAULT_PRESET for unknown symbol', () => {
    assert.equal(getPreset('DOGE-USD'), DEFAULT_PRESET);
  });

  it('falls back to DEFAULT_PRESET for empty string', () => {
    assert.equal(getPreset(''), DEFAULT_PRESET);
  });

  it('falls back to DEFAULT_PRESET for null', () => {
    assert.equal(getPreset(null), DEFAULT_PRESET);
  });

  it('falls back to DEFAULT_PRESET for undefined', () => {
    assert.equal(getPreset(undefined), DEFAULT_PRESET);
  });
});

// ---------------------------------------------------------------------------
// 8. applyPreset — merge / override behaviour
// ---------------------------------------------------------------------------

describe('applyPreset — merge and override', () => {
  it('returns a full config with all preset values', () => {
    const config = applyPreset(MBT_PRESET);
    for (const field of REQUIRED_FIELDS) {
      assert.ok(Object.hasOwn(config, field), `config missing '${field}'`);
    }
  });

  it('does not mutate the original preset', () => {
    const original = { ...MBT_PRESET };
    applyPreset(MBT_PRESET, { stopLossPct: 0.99 });
    assert.equal(MBT_PRESET.stopLossPct, original.stopLossPct);
  });

  it('overrides a single numeric field', () => {
    const config = applyPreset(MBT_PRESET, { stopLossPct: 0.05 });
    assert.equal(config.stopLossPct, 0.05);
    // Other fields unchanged
    assert.equal(config.takeProfitPct, MBT_PRESET.takeProfitPct);
  });

  it('overrides a string field', () => {
    const config = applyPreset(MBT_PRESET, { positionSize: '5 micros' });
    assert.equal(config.positionSize, '5 micros');
  });

  it('overrides regimePreference array', () => {
    const overrides = { regimePreference: ['trending_bearish'] };
    const config = applyPreset(MBT_PRESET, overrides);
    assert.deepEqual(config.regimePreference, ['trending_bearish']);
    // Original not mutated
    assert.deepEqual(MBT_PRESET.regimePreference, ['ranging', 'trending_bullish']);
  });

  it('overrides avoidRegimes array', () => {
    const overrides = { avoidRegimes: ['all'] };
    const config = applyPreset(MBT_PRESET, overrides);
    assert.deepEqual(config.avoidRegimes, ['all']);
  });

  it('adds extra keys from overrides that are not in the preset', () => {
    const config = applyPreset(MBT_PRESET, { customParam: 'test', enabled: true });
    assert.equal(config.customParam, 'test');
    assert.equal(config.enabled, true);
  });

  it('merges multiple overrides at once', () => {
    const config = applyPreset(DEFAULT_PRESET, {
      stopLossPct: 0.01,
      takeProfitPct: 0.03,
      cooldownBars: 5,
      maxDailyRisk: 100,
    });
    assert.equal(config.stopLossPct, 0.01);
    assert.equal(config.takeProfitPct, 0.03);
    assert.equal(config.cooldownBars, 5);
    assert.equal(config.maxDailyRisk, 100);
    assert.equal(config.warmupBars, DEFAULT_PRESET.warmupBars); // unchanged
  });

  it('produces config compatible with createOrchestrator expectation', () => {
    const config = applyPreset(MBT_PRESET);
    // Typical fields an orchestrator would expect
    assert.ok(typeof config.stopLossPct === 'number');
    assert.ok(typeof config.takeProfitPct === 'number');
    assert.ok(typeof config.cooldownBars === 'number');
    assert.ok(typeof config.minConfidence === 'number');
    assert.ok(typeof config.fusionMethod === 'string');
    assert.ok(Array.isArray(config.regimePreference));
    assert.ok(Array.isArray(config.avoidRegimes));
  });
});

// ---------------------------------------------------------------------------
// 9. Integration: getPreset + applyPreset pipeline
// ---------------------------------------------------------------------------

describe('getPreset + applyPreset integration', () => {
  it('getPreset result feeds into applyPreset', () => {
    const preset = getPreset('MBT');
    const config = applyPreset(preset, { stopLossPct: 0.02 });
    assert.equal(config.stopLossPct, 0.02);
    assert.equal(config.takeProfitPct, MBT_PRESET.takeProfitPct);
    assert.equal(config.bestTimeframe, '5m');
  });

  it('getPreset + applyPreset for unknown symbol uses DEFAULT_PRESET', () => {
    const preset = getPreset('UNKNOWN');
    assert.equal(preset, DEFAULT_PRESET);
    const config = applyPreset(preset);
    assert.equal(config.stopLossPct, DEFAULT_PRESET.stopLossPct);
  });
});

// ---------------------------------------------------------------------------
// 10. Edge: ensure regime arrays are not shared by reference
// ---------------------------------------------------------------------------

describe('preset immutability — regime arrays', () => {
  it('regimePreference arrays are distinct objects per preset', () => {
    const seen = new Set();
    for (const { preset } of ALL_PRESETS) {
      assert.ok(!seen.has(preset.regimePreference), 'regimePreference reference shared');
      seen.add(preset.regimePreference);
    }
  });

  it('avoidRegimes arrays are distinct objects per preset', () => {
    const seen = new Set();
    for (const { preset } of ALL_PRESETS) {
      assert.ok(!seen.has(preset.avoidRegimes), 'avoidRegimes reference shared');
      seen.add(preset.avoidRegimes);
    }
  });

  it('applyPreset copies regime arrays (not by reference)', () => {
    const config = applyPreset(MBT_PRESET);
    config.regimePreference.push('test-entry');
    assert.equal(MBT_PRESET.regimePreference.length, 2); // original unchanged
  });
});
