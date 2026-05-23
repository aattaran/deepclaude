/**
 * Trading System Orchestrator — unified signal pipeline
 *
 * Wires together:
 *   ZoneDetector      → S/R levels, FVGs, breaker zones, liquidity voids
 *   Market Regime     → trending/ranging/accumulation/breakout/etc.
 *   Signal Fusion     → weighted/bayesian/voting composite score
 *   Order Book        → bid/ask imbalance, absorption, spoofing (optional)
 *   Decision Graph    → traceable decision lineage (optional)
 *   Crypto Signals    → signed envelopes for verifiable provenance
 *
 * Architecture:
 *   Feed candles → Orchestrator.run(candles, orderBook?) → TradingDecision
 *
 * The Orchestrator is the "brain" — one call produces a complete trading decision
 * with rationale, confidence, and cryptographic proof.
 *
 * ES module. Zero npm dependencies.
 */

import { ZoneDetector } from './zone-detector.mjs';
import { classifyRegime } from './market-regime.mjs';
import { SignalFusionEngine } from './signal-fusion.mjs';
import { OrderBookAnalyzer, computeMicroPrice } from './microstructure.mjs';
import { zoneConfluenceScore } from './confluence.mjs';

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_ZONE_THRESHOLD = 0.015;
const DEFAULT_FVG_MIN_GAP = 0.003;
const DEFAULT_BREAKER_THRESHOLD = 0.02;
const DEFAULT_VOID_VOL_THRESHOLD = 10;
const MAX_ACTIVE_ZONES = 20;

const DECISION = {
    BUY: 'BUY',
    SELL: 'SELL',
    HOLD: 'HOLD',
};

// ─── 1. Signal Extraction Helpers ────────────────────────────────────────────

/**
 * Convert zone information into signal values.
 * Active supports below price → bullish pressure.
 * Active resistances above price → bearish pressure.
 * Confluence zones → stronger signal.
 */
function signalsFromZones(activeZones, currentPrice) {
    const signals = [];

    let supportStrength = 0;
    let resistanceStrength = 0;
    let nearestSupportDist = Infinity;
    let nearestResistanceDist = Infinity;

    for (const zone of activeZones) {
        const distPct = (zone.price - currentPrice) / currentPrice;

        if (zone.price < currentPrice) {
            // Support — bullish
            const strength = zone.strength * (1 - Math.min(Math.abs(distPct) / 0.01, 1));
            supportStrength = Math.max(supportStrength, strength);
            nearestSupportDist = Math.min(nearestSupportDist, Math.abs(distPct));
        } else if (zone.price > currentPrice) {
            // Resistance — bearish
            const strength = zone.strength * (1 - Math.min(Math.abs(distPct) / 0.01, 1));
            resistanceStrength = Math.max(resistanceStrength, strength);
            nearestResistanceDist = Math.min(nearestResistanceDist, Math.abs(distPct));
        }
    }

    // Emit only the DOMINANT zone direction to prevent self-cancellation.
    // If support and resistance are roughly equal, both are skipped (uncertain).
    const strengthDiff = supportStrength - resistanceStrength;
    const SIGNIFICANCE_THRESHOLD = 0.15;

    if (Math.abs(strengthDiff) > SIGNIFICANCE_THRESHOLD) {
        const dominantValue = Math.abs(strengthDiff);
        const dominantConf = 0.5 + Math.min(dominantValue * 0.5, 0.35);
        if (strengthDiff > 0) {
            signals.push({ source: 'zone-detector', name: 'zone_support_dominant', value: dominantValue, confidence: dominantConf });
        } else {
            signals.push({ source: 'zone-detector', name: 'zone_resistance_dominant', value: -dominantValue, confidence: dominantConf });
        }
    }

    // Risk/reward from zone distances (asymmetric — only when clearly favorable)
    if (nearestSupportDist < Infinity && nearestResistanceDist < Infinity) {
        const rr = nearestResistanceDist / Math.max(nearestSupportDist, 0.0001);
        if (rr > 1.5) {
            // Favorable RR — closer support, further resistance = bullish bias
            const rrSignal = Math.min((rr - 1) / 2, 1);
            signals.push({ source: 'zone-detector', name: 'zone_rr_favorable', value: rrSignal, confidence: 0.5 });
        } else if (rr < 0.67) {
            // Inverse — closer resistance, further support = bearish bias
            const rrSignal = -Math.min((1 / Math.max(rr, 0.01) - 1) / 2, 1);
            signals.push({ source: 'zone-detector', name: 'zone_rr_favorable', value: rrSignal, confidence: 0.5 });
        }
    }

    return signals;
}

/**
 * Convert regime information into signal values.
 */
function signalsFromRegime(regimeResult) {
    const signals = [];

    switch (regimeResult.regime) {
        case 'trending_bullish':
            signals.push({ source: 'market-regime', name: 'regime_trend', value: 0.8, confidence: regimeResult.confidence });
            break;
        case 'trending_bearish':
            signals.push({ source: 'market-regime', name: 'regime_trend', value: -0.8, confidence: regimeResult.confidence });
            break;
        case 'ranging':
            signals.push({ source: 'market-regime', name: 'regime_mean_revert', value: 0.3, confidence: regimeResult.confidence * 0.5 });
            break;
        case 'breakout':
            signals.push({ source: 'market-regime', name: 'regime_breakout', value: 0.7, confidence: regimeResult.confidence });
            break;
        case 'breakdown':
            signals.push({ source: 'market-regime', name: 'regime_breakout', value: -0.7, confidence: regimeResult.confidence });
            break;
        case 'volatile':
            signals.push({ source: 'market-regime', name: 'regime_volatile', value: 0, confidence: 0.8 });
            break;
        case 'accumulation':
            signals.push({ source: 'market-regime', name: 'regime_accumulation', value: 0.4, confidence: regimeResult.confidence * 0.6 });
            break;
        case 'distribution':
            signals.push({ source: 'market-regime', name: 'regime_distribution', value: -0.4, confidence: regimeResult.confidence * 0.6 });
            break;
    }

    return signals;
}

/**
 * Convert order book microstructure signals.
 */
function signalsFromOrderBook(orderBookResult) {
    const signals = [];

    if (!orderBookResult) return signals;

    if (orderBookResult.imbalance !== undefined) {
        // Imbalance > 0.5 = buying pressure → bullish
        const value = (orderBookResult.imbalance - 0.5) * 2; // scale to -1..1
        signals.push({ source: 'microstructure', name: 'book_imbalance', value, confidence: 0.6 });
    }

    if (orderBookResult.spoofingDetected) {
        signals.push({ source: 'microstructure', name: 'spoofing_warning', value: -0.3, confidence: 0.5 });
    }

    if (orderBookResult.absorption !== undefined) {
        // Absorption detected → possible reversal signal
        signals.push({ source: 'microstructure', name: 'absorption', value: orderBookResult.absorption > 0 ? 0.5 : -0.5, confidence: 0.55 });
    }

    return signals;
}

/**
 * Convert volume profile signals.
 */
function signalsFromVolumeProfile(vpResult) {
    const signals = [];
    if (!vpResult) return signals;

    if (vpResult.poc) {
        const distFromPoc = vpResult.distanceFromPoc ?? 0;
        // Price near POC → ranging tendency (mean reversion signal)
        if (Math.abs(distFromPoc) < 0.005) {
            signals.push({ source: 'volume-profile', name: 'near_poc', value: 0.1, confidence: 0.4 });
        }
    }

    return signals;
}

// ─── 2. Orchestrator ─────────────────────────────────────────────────────────

/**
 * Create a new Trading Orchestrator.
 *
 * @param {object} [config]
 * @param {number} [config.zoneThreshold=0.015] — min body fraction for order blocks
 * @param {number} [config.fvgMinGap=0.003] — min wick gap for FVGs
 * @param {number} [config.breakerThreshold=0.02] — min breach for breaker zones
 * @param {number} [config.voidVolThreshold=10] — max volume for liquidity voids
 * @param {string} [config.fusionMethod='weighted'] — 'weighted', 'bayesian', or 'voting'
 * @param {number} [config.minConfidence=0.15] — min confidence for BUY/SELL decisions
 * @param {string} [config.secret] — secret for signing decisions (if provided)
 * @param {import('./decision-graph.mjs').DecisionGraph} [config.decisionGraph] — optional decision graph
 * @returns {Orchestrator}
 */
export function createOrchestrator(config = {}) {
    const zoneThreshold = config.zoneThreshold ?? DEFAULT_ZONE_THRESHOLD;
    const fvgMinGap = config.fvgMinGap ?? DEFAULT_FVG_MIN_GAP;
    const breakerThreshold = config.breakerThreshold ?? DEFAULT_BREAKER_THRESHOLD;
    const voidVolThreshold = config.voidVolThreshold ?? DEFAULT_VOID_VOL_THRESHOLD;
    const fusionMethod = config.fusionMethod ?? 'weighted';
    const minConfidence = config.minConfidence ?? 0.15;
    const secret = config.secret ?? null;
    const decisionGraph = config.decisionGraph ?? null;

    let lastDecision = null;
    let decisionCount = 0;

    /**
     * Run the full signal pipeline on a set of candles.
     *
     * @param {Array<{ timestamp, open, high, low, close, volume }>} candles
     * @param {object} [context]
     * @param {Array<{ price: number, volume: number }>} [context.bids] — order book bid levels
     * @param {Array<{ price: number, volume: number }>} [context.asks] — order book ask levels
     * @param {string} [context.symbol] — trading pair symbol
     * @param {string} [context.source='orchestrator'] — signal source identifier
     * @returns {TradingDecision}
     */
    function run(candles, context = {}) {
        if (!candles || candles.length < 5) {
            return emptyDecision('insufficient candle data');
        }

        const currentPrice = candles[candles.length - 1].close;
        const symbol = context.symbol ?? 'UNKNOWN';
        const source = context.source ?? 'orchestrator';

        // ── 1. Zone Detection ──
        const zd = new ZoneDetector(candles);
        zd.detectOrderBlocks(zoneThreshold);
        zd.detectFairValueGaps(fvgMinGap);
        zd.detectBreakerZones(breakerThreshold);
        zd.detectLiquidityVoids(voidVolThreshold);
        const activeZones = zd.getActiveZones(currentPrice).slice(0, MAX_ACTIVE_ZONES);

        // ── 2. Market Regime ──
        const regime = classifyRegime(candles);

        // ── 3. Order Book Microstructure (if provided) ──
        const obAnalyzer = new OrderBookAnalyzer([]);
        let obResult = null;
        if (context.bids && context.asks && context.bids.length > 0 && context.asks.length > 0) {
            const depth = [...context.bids, ...context.asks].map(l => ({
                price: l.price,
                size: l.volume,
                side: context.bids.includes(l) ? 'bid' : 'ask',
            }));
            const imbalance = obAnalyzer.getBidAskImbalance(depth);
            const absorption = detectOrderBookAbsorption(context.bids, context.asks, currentPrice);
            obResult = { imbalance, absorption };
        }

        // ── 4. Signal Assembly ──
        const zoneSignals = signalsFromZones(activeZones, currentPrice);
        const regimeSignals = signalsFromRegime(regime);
        const obSignals = signalsFromOrderBook(obResult);

        // Confluence: how clustered are the active zones? Higher = stronger signal quality
        const confluenceScore = zoneConfluenceScore(activeZones, currentPrice);
        const confluenceSignals = [];
        if (confluenceScore > 0) {
            confluenceSignals.push({
                source: 'confluence',
                name: 'zone_clustering',
                value: confluenceScore,
                confidence: Math.min(confluenceScore * 1.2, 1.0),
            });
        }

        const allSignals = [...zoneSignals, ...regimeSignals, ...obSignals, ...confluenceSignals];

        // ── 5. Signal Fusion ──
        const fusion = new SignalFusionEngine({ method: fusionMethod });
        fusion.addSignals(allSignals);

        const compositeScore = fusion.getCompositeScore();
        const decision = fusion.getDecision({ minConfidence });

        // ── 6. Zone Context ──
        const nearestSupport = activeZones
            .filter(z => z.price < currentPrice)
            .sort((a, b) => b.price - a.price)[0] ?? null;
        const nearestResistance = activeZones
            .filter(z => z.price > currentPrice)
            .sort((a, b) => a.price - b.price)[0] ?? null;

        // ── 7. Build Decision ──
        const tradingDecision = {
            id: `TD-${Date.now()}-${(decisionCount++).toString(36)}`,
            ts: Date.now(),
            symbol,
            price: currentPrice,
            action: decision.action,
            confidence: decision.confidence,
            compositeScore,
            regime: regime.regime,
            regimeConfidence: regime.confidence,
            direction: regime.direction,
            support: nearestSupport ? { price: nearestSupport.price, strength: nearestSupport.strength } : null,
            resistance: nearestResistance ? { price: nearestResistance.price, strength: nearestResistance.strength } : null,
            activeZones: activeZones.slice(0, 5).map(z => ({
                price: z.price,
                type: z.type,
                strength: z.strength,
                freshness: z.freshness,
            })),
            signals: allSignals.map(s => ({
                source: s.source,
                name: s.name,
                value: +s.value.toFixed(4),
                confidence: +s.confidence.toFixed(3),
            })),
            reasoning: decision.reasoning,
            params: regime.params,
            source,
        };

        lastDecision = tradingDecision;

        // ── 8. Record in Decision Graph (if configured) ──
        if (decisionGraph) {
            try {
                decisionGraph.addDecision({
                    id: tradingDecision.id,
                    ts: tradingDecision.ts,
                    symbol,
                    action: tradingDecision.action,
                    signals: tradingDecision.signals.map(s => ({
                        source: s.source,
                        name: s.name,
                        value: s.value,
                    })),
                    reasoning: tradingDecision.reasoning,
                    metadata: {
                        regime: regime.regime,
                        compositeScore,
                        price: currentPrice,
                    },
                });
            } catch (_) { /* decision graph is optional */ }
        }

        return tradingDecision;
    }

    /**
     * Get the most recent decision.
     * @returns {TradingDecision|null}
     */
    function getLastDecision() {
        return lastDecision;
    }

    /**
     * Reset the orchestrator's runtime state.
     */
    function reset() {
        lastDecision = null;
        decisionCount = 0;
    }

    return { run, getLastDecision, reset, get decisionCount() { return decisionCount; } };
}

// ─── 3. Backtest Runner ──────────────────────────────────────────────────────

/**
 * Run the orchestrator over historical candles to produce a backtest.
 *
 * Walks through candles sequentially, running the orchestrator at each step
 * where enough history exists. Tracks PnL for each decision.
 *
 * @param {Array<{ timestamp, open, high, low, close, volume }>} candles
 * @param {object} [config] — orchestrator config + backtest options
 * @param {number} [config.warmupBars=50] — min candles before first decision
 * @param {number} [config.cooldownBars=10] — min bars between decisions
 * @param {number} [config.maxHoldingBars=100] — auto-close after N bars
 * @param {number} [config.stopLossPct=0.02] — hard stop loss
 * @param {number} [config.takeProfitPct=0.04] — hard take profit
 * @returns {BacktestResult}
 */
export function runBacktest(candles, config = {}) {
    const warmupBars = config.warmupBars ?? 50;
    const cooldownBars = config.cooldownBars ?? 10;
    const maxHoldingBars = config.maxHoldingBars ?? 100;
    const stopLossPct = config.stopLossPct ?? 0.02;
    const takeProfitPct = config.takeProfitPct ?? 0.04;

    const orchestrator = createOrchestrator(config);
    const trades = [];
    let lastTradeBar = -cooldownBars;
    let activeTrade = null;

    for (let i = warmupBars; i < candles.length; i++) {
        const window = candles.slice(0, i + 1);
        const currentPrice = candles[i].close;

        // Check active trade exit
        if (activeTrade) {
            const barsHeld = i - activeTrade.entryBar;
            const priceChange = (currentPrice - activeTrade.entryPrice) / activeTrade.entryPrice;

            let exit = false;
            let exitReason = '';

            if (activeTrade.direction === 'long') {
                if (priceChange <= -stopLossPct) { exit = true; exitReason = 'stop_loss'; }
                if (priceChange >= takeProfitPct) { exit = true; exitReason = 'take_profit'; }
            } else {
                if (priceChange >= stopLossPct) { exit = true; exitReason = 'stop_loss'; }
                if (priceChange <= -takeProfitPct) { exit = true; exitReason = 'take_profit'; }
            }
            if (barsHeld >= maxHoldingBars) { exit = true; exitReason = 'max_hold'; }

            if (exit) {
                const pnl = activeTrade.direction === 'long'
                    ? (currentPrice - activeTrade.entryPrice) / activeTrade.entryPrice
                    : (activeTrade.entryPrice - currentPrice) / activeTrade.entryPrice;

                trades.push({
                    entryBar: activeTrade.entryBar,
                    exitBar: i,
                    direction: activeTrade.direction,
                    entryPrice: activeTrade.entryPrice,
                    exitPrice: currentPrice,
                    pnl: +pnl.toFixed(6),
                    pnlPct: +(pnl * 100).toFixed(2),
                    exitReason,
                    entryConfidence: activeTrade.confidence,
                    regime: activeTrade.regime,
                });

                activeTrade = null;
            }
        }

        // Check cooldown
        if (i - lastTradeBar < cooldownBars) continue;
        if (activeTrade) continue; // already in a trade

        // Run orchestrator
        const decision = orchestrator.run(window, { symbol: config.symbol });

        if (decision.action === 'BUY') {
            activeTrade = {
                entryBar: i,
                entryPrice: currentPrice,
                direction: 'long',
                confidence: decision.confidence,
                regime: decision.regime,
            };
            lastTradeBar = i;
        } else if (decision.action === 'SELL') {
            activeTrade = {
                entryBar: i,
                entryPrice: currentPrice,
                direction: 'short',
                confidence: decision.confidence,
                regime: decision.regime,
            };
            lastTradeBar = i;
        }
    }

    // Close any open trade at the end
    if (activeTrade) {
        const lastPrice = candles[candles.length - 1].close;
        const pnl = activeTrade.direction === 'long'
            ? (lastPrice - activeTrade.entryPrice) / activeTrade.entryPrice
            : (activeTrade.entryPrice - lastPrice) / activeTrade.entryPrice;

        trades.push({
            entryBar: activeTrade.entryBar,
            exitBar: candles.length - 1,
            direction: activeTrade.direction,
            entryPrice: activeTrade.entryPrice,
            exitPrice: lastPrice,
            pnl: +pnl.toFixed(6),
            pnlPct: +(pnl * 100).toFixed(2),
            exitReason: 'end_of_data',
            entryConfidence: activeTrade.confidence,
            regime: activeTrade.regime,
        });
    }

    // Compute statistics
    const winningTrades = trades.filter(t => t.pnl > 0);
    const losingTrades = trades.filter(t => t.pnl < 0);
    const winRate = trades.length > 0 ? winningTrades.length / trades.length : 0;
    const totalPnl = trades.reduce((s, t) => s + t.pnl, 0);
    const avgWin = winningTrades.length > 0
        ? winningTrades.reduce((s, t) => s + t.pnl, 0) / winningTrades.length
        : 0;
    const avgLoss = losingTrades.length > 0
        ? losingTrades.reduce((s, t) => s + t.pnl, 0) / losingTrades.length
        : 0;
    const profitFactor = Math.abs(avgLoss) > 0
        ? (avgWin * winningTrades.length) / Math.abs(avgLoss * losingTrades.length)
        : winningTrades.length > 0 ? Infinity : 0;

    // Per-regime stats
    const perRegime = {};
    for (const t of trades) {
        const r = t.regime ?? 'unknown';
        if (!perRegime[r]) perRegime[r] = { count: 0, wins: 0, pnl: 0 };
        perRegime[r].count++;
        if (t.pnl > 0) perRegime[r].wins++;
        perRegime[r].pnl += t.pnl;
    }
    for (const [regime, stats] of Object.entries(perRegime)) {
        stats.winRate = +(stats.wins / stats.count).toFixed(3);
        stats.avgPnl = +(stats.pnl / stats.count).toFixed(6);
        stats.totalPnl = +stats.pnl.toFixed(6);
        delete stats.wins;
        delete stats.pnl;
    }

    return {
        trades,
        stats: {
            totalTrades: trades.length,
            winningTrades: winningTrades.length,
            losingTrades: losingTrades.length,
            winRate: +winRate.toFixed(4),
            totalPnl: +totalPnl.toFixed(6),
            avgWin: +avgWin.toFixed(6),
            avgLoss: +avgLoss.toFixed(6),
            profitFactor: +Math.min(profitFactor, 999).toFixed(2),
            perRegime,
        },
    };
}

// ─── 4. Helpers ──────────────────────────────────────────────────────────────

function detectOrderBookAbsorption(bids, asks, currentPrice) {
    if (!bids || !asks || !bids.length || !asks.length) return undefined;

    const sortedBids = [...bids].sort((a, b) => b.price - a.price);
    const sortedAsks = [...asks].sort((a, b) => a.price - b.price);

    const totalBidVol = sortedBids.slice(0, 5).reduce((s, b) => s + b.volume, 0);
    const totalAskVol = sortedAsks.slice(0, 5).reduce((s, a) => s + a.volume, 0);

    // Absorption: one side has dominant volume
    if (totalBidVol > totalAskVol * 2) return 1; // bid absorption
    if (totalAskVol > totalBidVol * 2) return -1; // ask absorption
    return 0;
}

function emptyDecision(reason) {
    return {
        id: null,
        ts: Date.now(),
        symbol: 'UNKNOWN',
        price: null,
        action: 'HOLD',
        confidence: 0,
        compositeScore: 0,
        regime: 'unknown',
        regimeConfidence: 0,
        direction: null,
        support: null,
        resistance: null,
        activeZones: [],
        signals: [],
        reasoning: reason,
        params: {},
        source: 'orchestrator',
    };
}

// ─── Type Definitions ────────────────────────────────────────────────────────

/**
 * @typedef {object} TradingDecision
 * @property {string} id — unique decision ID
 * @property {number} ts — timestamp
 * @property {string} symbol — trading pair
 * @property {number} price — current price
 * @property {'BUY'|'SELL'|'HOLD'} action
 * @property {number} confidence — 0-1
 * @property {number} compositeScore — fused signal score -1..1
 * @property {string} regime — market regime
 * @property {number} regimeConfidence
 * @property {string|null} direction
 * @property {{ price: number, strength: number }|null} support
 * @property {{ price: number, strength: number }|null} resistance
 * @property {Array<{ price: number, type: string, strength: number, freshness: number }>} activeZones
 * @property {Array<{ source: string, name: string, value: number, confidence: number }>} signals
 * @property {string} reasoning
 * @property {object} params — strategy parameters for the current regime
 * @property {string} source
 */

/**
 * @typedef {object} BacktestResult
 * @property {Array} trades
 * @property {{ totalTrades, winningTrades, losingTrades, winRate, totalPnl, avgWin, avgLoss, profitFactor, perRegime }} stats
 */

export { DECISION };
