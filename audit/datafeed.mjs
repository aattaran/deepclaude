/**
 * Real Market Data Pipeline — fetch, cache, and normalize OHLCV data.
 *
 * Sources: Binance public REST API (no auth required).
 * Falls back to synthetic generation when offline.
 *
 * Usage:
 *   import { fetchCandles, loadOrFetch, getCacheAge } from './datafeed.mjs';
 *   const candles = await loadOrFetch('BTC-USDT', '15m', 30);
 *
 * ES module. Zero npm dependencies. Uses Node built-in fetch (18+) or https.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { generateSyntheticCandles } from './trade.mjs';

// ─── Constants ──────────────────────────────────────────────────────────────────

const BINANCE_BASE = 'https://api.binance.com';
const BINANCE_API = `${BINANCE_BASE}/api/v3`;
const DEFAULT_CACHE_DIR = join(import.meta.dirname ?? '.', '.cache');
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;
const CACHE_MAX_AGE_MS = 60 * 60 * 1000; // 1 hour for intraday timeframes

const BINANCE_TIMEFRAMES = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w',
};

const TIMEFRAME_BARS_PER_DAY = {
    '1m': 1440, '5m': 288, '15m': 96, '30m': 48,
    '1h': 24, '4h': 6, '1d': 1, '1w': 1 / 7,
};

/** Max candles per Binance request (API limit is 1000). */
const BINANCE_MAX_LIMIT = 1000;

// ─── 1. Symbol Normalization ────────────────────────────────────────────────────

/**
 * Normalize a human symbol to Binance format.
 * "BTC-USD" → "BTCUSDT", "SOL-USD" → "SOLUSDT", "MBT" → "BTCUSDT"
 */
function toBinanceSymbol(symbol) {
    const upper = symbol.toUpperCase().replace(/[^A-Z0-9]/g, '');
    // MBT is Micro Bitcoin → use BTCUSDT
    if (upper === 'MBT' || upper === 'MBTUSD' || upper === 'MBTUSDT') return 'BTCUSDT';
    // If already Binance format, return as-is
    if (/^[A-Z0-9]{5,12}$/.test(upper) && upper.endsWith('USDT')) return upper;
    // Strip -USD suffix and append USDT
    if (upper.endsWith('USD')) return upper.replace(/USD$/, 'USDT');
    // Default: append USDT
    return `${upper}USDT`;
}

/**
 * Reverse: Binance symbol → human display symbol.
 */
function fromBinanceSymbol(binanceSymbol) {
    return binanceSymbol.replace(/USDT$/, '-USD');
}

// ─── 2. API Fetching ────────────────────────────────────────────────────────────

/**
 * Fetch kline/candlestick data from Binance public API.
 *
 * @param {string} symbol — e.g., "BTCUSDT", "SOLUSDT"
 * @param {string} timeframe — "1m"|"5m"|"15m"|"30m"|"1h"|"4h"|"1d"|"1w"
 * @param {number} [limit=500] — number of candles (max 1000)
 * @returns {Promise<Array>} — normalized candles
 */
export async function fetchCandles(symbol, timeframe = '15m', limit = 500) {
    const binanceSymbol = toBinanceSymbol(symbol);
    const binanceTf = BINANCE_TIMEFRAMES[timeframe];
    if (!binanceTf) {
        throw new Error(`Unsupported timeframe: ${timeframe}. Use: ${Object.keys(BINANCE_TIMEFRAMES).join(', ')}`);
    }

    const actualLimit = Math.min(limit, BINANCE_MAX_LIMIT);
    const url = `${BINANCE_API}/klines?symbol=${binanceSymbol}&interval=${binanceTf}&limit=${actualLimit}`;

    let lastError = null;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 15000);

            const response = await fetch(url, { signal: controller.signal, headers: { 'Accept': 'application/json' } });
            clearTimeout(timeout);

            if (!response.ok) {
                const body = await response.text().catch(() => '');
                const err = new Error(`Binance API ${response.status}: ${body.slice(0, 200)}`);
                // 451 = geo-restricted — permanent, don't retry
                if (response.status === 451) err.noRetry = true;
                throw err;
            }

            const raw = await response.json();
            if (!Array.isArray(raw)) {
                throw new Error(`Unexpected Binance response: ${JSON.stringify(raw).slice(0, 200)}`);
            }

            return normalizeBinanceKlines(raw, fromBinanceSymbol(binanceSymbol));
        } catch (err) {
            lastError = err;
            if (err.name === 'AbortError') {
                lastError = new Error('Binance API request timed out');
            }
            if (err.noRetry) break;
            if (attempt < MAX_RETRIES - 1) {
                await sleep(RETRY_DELAY_MS * (attempt + 1));
            }
        }
    }

    throw lastError ?? new Error('Failed to fetch candles');
}

/**
 * Normalize Binance kline format to system candle format.
 *
 * Binance kline: [openTime, open, high, low, close, volume, closeTime, ...]
 */
function normalizeBinanceKlines(klines, symbol) {
    return klines.map(k => ({
        timestamp: k[0],
        open: +k[1],
        high: +k[2],
        low: +k[3],
        close: +k[4],
        volume: +k[5],
        symbol,
    }));
}

// ─── 3. Caching ─────────────────────────────────────────────────────────────────

/**
 * Build a cache file path for a given symbol/timeframe/days combination.
 */
function cachePath(symbol, timeframe, days, cacheDir) {
    const binanceSymbol = toBinanceSymbol(symbol);
    const safeSymbol = binanceSymbol.replace(/[^A-Za-z0-9]/g, '_');
    const dir = cacheDir ?? DEFAULT_CACHE_DIR;
    return join(dir, `${safeSymbol}_${timeframe}_${days}d.json`);
}

/**
 * Read candles from cache if available and fresh.
 * Returns null if cache miss or stale.
 */
function readCache(symbol, timeframe, days, cacheDir, maxAgeMs) {
    const path = cachePath(symbol, timeframe, days, cacheDir);
    try {
        const raw = readFileSync(path, 'utf-8');
        const cached = JSON.parse(raw);

        // Check freshness
        const age = Date.now() - (cached._cachedAt ?? 0);
        const maxAge = maxAgeMs ?? CACHE_MAX_AGE_MS;
        if (age > maxAge) return null;

        // Validate structure
        if (!Array.isArray(cached.candles) || cached.candles.length === 0) return null;
        if (typeof cached.candles[0].close !== 'number') return null;

        return cached.candles;
    } catch (_) {
        return null;
    }
}

/**
 * Write candles to cache.
 */
function writeCache(symbol, timeframe, days, candles, cacheDir) {
    const path = cachePath(symbol, timeframe, days, cacheDir);
    try {
        const dir = cacheDir ?? DEFAULT_CACHE_DIR;
        if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
        writeFileSync(path, JSON.stringify({
            _cachedAt: Date.now(),
            _symbol: symbol,
            _timeframe: timeframe,
            _days: days,
            _count: candles.length,
            candles,
        }, null, 2));
    } catch (_) {
        // Cache write failures are non-fatal
    }
}

/**
 * Get the age of cached data in milliseconds.
 * Returns -1 if no cache exists.
 */
export function getCacheAge(symbol, timeframe, days, cacheDir) {
    const path = cachePath(symbol, timeframe, days, cacheDir);
    try {
        const raw = readFileSync(path, 'utf-8');
        const cached = JSON.parse(raw);
        return Date.now() - (cached._cachedAt ?? 0);
    } catch (_) {
        return -1;
    }
}

// ─── 4. Main API: load or fetch ─────────────────────────────────────────────────

/**
 * Load candles from cache if fresh, otherwise fetch from API.
 * Falls back to synthetic generation when offline.
 *
 * @param {string} symbol — e.g., "BTC-USD", "SOL-USDT", "MBT"
 * @param {string} timeframe — "1m"|"5m"|"15m"|"1h"|"4h"|"1d"
 * @param {number} days — approximate calendar days of data
 * @param {object} [opts]
 * @param {string} [opts.cacheDir] — cache directory path
 * @param {number} [opts.maxAgeMs] — max cache age in ms (default: 1 hour)
 * @param {boolean} [opts.forceFetch=false] — skip cache, always fetch
 * @param {boolean} [opts.allowSynthetic=true] — fall back to synthetic if offline
 * @param {number} [opts.seed] — PRNG seed for synthetic fallback
 * @param {number} [opts.basePrice] — base price for synthetic fallback
 * @returns {Promise<Array>} — normalized candles
 */
export async function loadOrFetch(symbol, timeframe = '15m', days = 30, opts = {}) {
    const cacheDir = opts.cacheDir ?? DEFAULT_CACHE_DIR;
    const maxAgeMs = opts.maxAgeMs ?? CACHE_MAX_AGE_MS;
    const forceFetch = opts.forceFetch ?? false;
    const allowSynthetic = opts.allowSynthetic ?? true;

    // 1. Check cache
    if (!forceFetch) {
        const cached = readCache(symbol, timeframe, days, cacheDir, maxAgeMs);
        if (cached) return cached;
    }

    // 2. Try fetching from API
    const barsPerDay = TIMEFRAME_BARS_PER_DAY[timeframe] ?? 96;
    const neededBars = Math.min(days * barsPerDay, BINANCE_MAX_LIMIT);

    try {
        const candles = await fetchCandles(symbol, timeframe, neededBars);
        // Cache the result
        writeCache(symbol, timeframe, days, candles, cacheDir);
        return candles;
    } catch (err) {
        // 3. Try stale cache as fallback
        const staleCache = readCache(symbol, timeframe, days, cacheDir, Infinity);
        if (staleCache) return staleCache;

        // 4. Synthetic fallback
        if (allowSynthetic) {
            const basePrice = opts.basePrice ?? (symbol.toUpperCase().includes('BTC') || symbol.toUpperCase().includes('MBT') ? 75000 : 100);
            const volatility = opts.volatility ?? 0.012;
            const seed = opts.seed ?? Math.floor(Math.random() * 2147483647);
            const candles = generateSyntheticCandles(Math.floor(neededBars), { seed, basePrice, volatility });
            writeCache(symbol, timeframe, days, candles, cacheDir);
            return candles;
        }

        throw err;
    }
}

// ─── 5. Multi-Symbol Fetch ──────────────────────────────────────────────────────

/**
 * Fetch data for multiple symbols in parallel.
 *
 * @param {string[]} symbols
 * @param {string} timeframe
 * @param {number} days
 * @param {object} [opts]
 * @returns {Promise<Object<string, Array>>} — { symbol: candles }
 */
export async function fetchMultiple(symbols, timeframe = '15m', days = 30, opts = {}) {
    const results = {};
    const errors = {};

    const promises = symbols.map(async (symbol) => {
        try {
            results[symbol] = await loadOrFetch(symbol, timeframe, days, opts);
        } catch (err) {
            errors[symbol] = err.message;
        }
    });

    await Promise.allSettled(promises);

    if (Object.keys(errors).length > 0) {
        // Attach errors to result for inspection
        results._errors = errors;
    }

    return results;
}

// ─── 6. Metadata ────────────────────────────────────────────────────────────────

/**
 * Get exchange info for a symbol (status, tick size, etc.).
 *
 * @param {string} symbol
 * @returns {Promise<object|null>}
 */
export async function getSymbolInfo(symbol) {
    const binanceSymbol = toBinanceSymbol(symbol);
    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 10000);
        const response = await fetch(
            `${BINANCE_API}/exchangeInfo?symbol=${binanceSymbol}`,
            { signal: controller.signal },
        );
        clearTimeout(timeout);

        if (!response.ok) return null;
        const data = await response.json();
        const sym = data.symbols?.[0];
        if (!sym) return null;

        return {
            symbol: fromBinanceSymbol(sym.symbol),
            baseAsset: sym.baseAsset,
            quoteAsset: sym.quoteAsset,
            status: sym.status,
            tickSize: sym.filters?.find(f => f.filterType === 'PRICE_FILTER')?.tickSize,
            minQty: sym.filters?.find(f => f.filterType === 'LOT_SIZE')?.minQty,
        };
    } catch (_) {
        return null;
    }
}

// ─── 7. Utilities ───────────────────────────────────────────────────────────────

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Get a quick summary of candle data.
 */
export function summarizeCandles(candles) {
    if (!candles || candles.length === 0) return null;
    const closes = candles.map(c => c.close);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const volumes = candles.map(c => c.volume ?? 0);
    const startPrice = closes[0];
    const endPrice = closes[closes.length - 1];
    const change = (endPrice - startPrice) / startPrice;

    return {
        count: candles.length,
        start: candles[0].timestamp,
        end: candles[candles.length - 1].timestamp,
        startPrice: +startPrice.toFixed(2),
        endPrice: +endPrice.toFixed(2),
        changePct: +(change * 100).toFixed(2),
        high: +Math.max(...highs).toFixed(2),
        low: +Math.min(...lows).toFixed(2),
        avgVolume: +(volumes.reduce((a, b) => a + b, 0) / volumes.length).toFixed(2),
    };
}

// ─── Type Definitions ────────────────────────────────────────────────────────────

/**
 * @typedef {object} Candle
 * @property {number} timestamp — unix ms
 * @property {number} open
 * @property {number} high
 * @property {number} low
 * @property {number} close
 * @property {number} volume
 * @property {string} [symbol]
 */
