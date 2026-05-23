// audit/presets.mjs — Trading parameter presets for different instruments
//
// Research-backed presets tuned for mean-reversion farming, scalping, and trend strategies.
// All values are validated by audit/presets.test.js.

// ---------------------------------------------------------------------------
// Micro Bitcoin (MBT) — primary research target
// ---------------------------------------------------------------------------
export const MBT_PRESET = {
  stopLossPct: 0.015,
  takeProfitPct: 0.025,
  cooldownBars: 8,
  warmupBars: 50,
  minConfidence: 0.2,
  fusionMethod: 'weighted',
  zoneThreshold: 0.01,
  positionSize: '1-2 micros',
  bestTimeframe: '5m',
  regimePreference: ['ranging', 'trending_bullish'],
  avoidRegimes: ['volatile', 'breakdown'],
  maxDailyRisk: 250,
  commissionNote: 'Sub-$0.50/contract round-turn required for viability',
};

// ---------------------------------------------------------------------------
// Standard Bitcoin (BTC-USD) — wider stops, longer holds
// ---------------------------------------------------------------------------
export const BTC_USD_PRESET = {
  stopLossPct: 0.035,
  takeProfitPct: 0.06,
  cooldownBars: 12,
  warmupBars: 100,
  minConfidence: 0.35,
  fusionMethod: 'weighted',
  zoneThreshold: 0.02,
  positionSize: '0.01-0.05 BTC',
  bestTimeframe: '15m',
  regimePreference: ['trending_bullish', 'trending_bearish', 'ranging'],
  avoidRegimes: ['breakdown'],
  maxDailyRisk: 500,
  commissionNote: 'Sub-$2/contract round-turn required for viability',
};

// ---------------------------------------------------------------------------
// Solana (SOL-USD) — higher volatility, tighter stops
// ---------------------------------------------------------------------------
export const SOL_USD_PRESET = {
  stopLossPct: 0.04,
  takeProfitPct: 0.07,
  cooldownBars: 10,
  warmupBars: 80,
  minConfidence: 0.3,
  fusionMethod: 'weighted',
  zoneThreshold: 0.025,
  positionSize: '5-20 SOL',
  bestTimeframe: '5m',
  regimePreference: ['trending_bullish', 'ranging'],
  avoidRegimes: ['volatile', 'breakdown'],
  maxDailyRisk: 300,
  commissionNote: 'Sub-$0.10/contract round-turn required for viability',
};

// ---------------------------------------------------------------------------
// Ethereum (ETH-USD) — moderate settings
// ---------------------------------------------------------------------------
export const ETH_USD_PRESET = {
  stopLossPct: 0.03,
  takeProfitPct: 0.05,
  cooldownBars: 10,
  warmupBars: 80,
  minConfidence: 0.3,
  fusionMethod: 'weighted',
  zoneThreshold: 0.015,
  positionSize: '0.1-0.5 ETH',
  bestTimeframe: '15m',
  regimePreference: ['trending_bullish', 'ranging'],
  avoidRegimes: ['volatile', 'breakdown'],
  maxDailyRisk: 400,
  commissionNote: 'Sub-$1/contract round-turn required for viability',
};

// ---------------------------------------------------------------------------
// Conservative defaults — safe starting point for any instrument
// ---------------------------------------------------------------------------
export const DEFAULT_PRESET = {
  stopLossPct: 0.02,
  takeProfitPct: 0.04,
  cooldownBars: 12,
  warmupBars: 100,
  minConfidence: 0.4,
  fusionMethod: 'weighted',
  zoneThreshold: 0.015,
  positionSize: '1 unit',
  bestTimeframe: '15m',
  regimePreference: ['ranging', 'trending_bullish'],
  avoidRegimes: ['volatile', 'breakdown'],
  maxDailyRisk: 200,
  commissionNote: 'Adjust based on instrument commission structure',
};

// ---------------------------------------------------------------------------
// Aggressive — wide stops, high conviction, fewer trades
// ---------------------------------------------------------------------------
export const AGGRESSIVE_PRESET = {
  stopLossPct: 0.05,
  takeProfitPct: 0.12,
  cooldownBars: 20,
  warmupBars: 150,
  minConfidence: 0.6,
  fusionMethod: 'weighted',
  zoneThreshold: 0.03,
  positionSize: '2-3 units',
  bestTimeframe: '1h',
  regimePreference: ['trending_bullish', 'trending_bearish'],
  avoidRegimes: ['ranging', 'volatile', 'breakdown'],
  maxDailyRisk: 600,
  commissionNote: 'Wide stops/profits tolerate higher commissions',
};

// ---------------------------------------------------------------------------
// Scalping — tight stops, quick profits, short cooldown
// ---------------------------------------------------------------------------
export const SCALPING_PRESET = {
  stopLossPct: 0.008,
  takeProfitPct: 0.015,
  cooldownBars: 3,
  warmupBars: 30,
  minConfidence: 0.25,
  fusionMethod: 'weighted',
  zoneThreshold: 0.005,
  positionSize: '1 unit',
  bestTimeframe: '1m',
  regimePreference: ['ranging'],
  avoidRegimes: ['trending_bearish', 'volatile', 'breakdown'],
  maxDailyRisk: 150,
  commissionNote: 'Ultra-low commissions essential; sub-$0.10/contract',
};

// ---------------------------------------------------------------------------
// Registry: symbol → preset lookup
// ---------------------------------------------------------------------------
const PRESET_REGISTRY = [
  { symbols: ['MBT', 'MBT-USD', 'MBTUSD', 'MICRO-BTC', 'MICRO_BTC'], preset: MBT_PRESET },
  { symbols: ['BTC', 'BTC-USD', 'BTCUSD', 'BITCOIN', 'XBT'], preset: BTC_USD_PRESET },
  { symbols: ['SOL', 'SOL-USD', 'SOLUSD'], preset: SOL_USD_PRESET },
  { symbols: ['ETH', 'ETH-USD', 'ETHUSD', 'ETHEREUM'], preset: ETH_USD_PRESET },
  { symbols: ['SCALP'], preset: SCALPING_PRESET },
  { symbols: ['AGGRESSIVE'], preset: AGGRESSIVE_PRESET },
];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Returns the best preset for a given symbol string.
 *
 * Matching is case-insensitive and supports partial / dash / underscore variants.
 * Falls back to DEFAULT_PRESET for unknown symbols.
 *
 * @param {string} symbol — e.g. 'MBT', 'BTC-USD', 'sol', 'eth_usd'
 * @returns {object} one of the preset objects
 */
export function getPreset(symbol) {
  if (!symbol || typeof symbol !== 'string') {
    return DEFAULT_PRESET;
  }

  const normalized = symbol.toUpperCase().replace(/[_-]/g, '-').trim();

  for (const entry of PRESET_REGISTRY) {
    const exactMatch = entry.symbols.some((s) => s === normalized);
    if (exactMatch) return entry.preset;

    // Partial / prefix match: e.g. 'MBT' matches 'MBT-USD'
    const prefixMatch = entry.symbols.some((s) => s.startsWith(normalized) || normalized.startsWith(s));
    if (prefixMatch) return entry.preset;
  }

  return DEFAULT_PRESET;
}

/**
 * Returns a full config object suitable for createOrchestrator() or runBacktest()
 * with preset values, overridden by any provided overrides.
 *
 * @param {object} preset — one of the preset objects
 * @param {object} [overrides={}] — key/value pairs to override
 * @returns {object} merged config
 */
export function applyPreset(preset, overrides = {}) {
  const config = {
    // Core trading parameters
    stopLossPct: preset.stopLossPct,
    takeProfitPct: preset.takeProfitPct,
    positionSize: preset.positionSize,
    cooldownBars: preset.cooldownBars,
    warmupBars: preset.warmupBars,
    minConfidence: preset.minConfidence,
    fusionMethod: preset.fusionMethod,
    zoneThreshold: preset.zoneThreshold,

    // Strategy hints
    bestTimeframe: preset.bestTimeframe,
    regimePreference: [...preset.regimePreference],
    avoidRegimes: [...preset.avoidRegimes],
    maxDailyRisk: preset.maxDailyRisk,

    // Instrument metadata
    commissionNote: preset.commissionNote,
  };

  // Apply overrides — spread overrides so consumers can also spread
  // individual keys. Deep-merge regime arrays if provided.
  for (const [key, value] of Object.entries(overrides)) {
    if ((key === 'regimePreference' || key === 'avoidRegimes') && Array.isArray(value)) {
      config[key] = [...value];
    } else {
      config[key] = value;
    }
  }

  return config;
}
