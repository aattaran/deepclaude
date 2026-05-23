import { createServer } from 'http';
import { request as httpsRequest } from 'https';
import { URL } from 'url';
import { Transform } from 'stream';
import { translateRequest, GeminiStreamTranslator, translateNonStreamingResponse } from './gemini-translator.js';

const ANTHROPIC_FALLBACK = 'https://api.anthropic.com';
const GEMINI_BASE = 'https://generativelanguage.googleapis.com';
const MODEL_PATHS = ['/v1/messages'];
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000; // 5 min per request
const NON_ANTHROPIC_BACKENDS = new Set(['gemini']);
const HAIKU_PATTERN = /^claude-haiku/;

// Auto-routing: Haiku-tier → Gemini, Opus/Sonnet → DeepSeek
const AUTO_ROUTE = {
    haiku:       { primary: 'gemini',   fallback: 'deepseek', fallbackModel: 'deepseek-v4-flash' },
    sonnet_opus: { primary: 'deepseek', fallback: 'gemini',   fallbackModel: 'gemini-3.5-flash' },
};

function isHaikuModel(model) { return HAIKU_PATTERN.test(model); }

function resolveAutoBackend(model, backends) {
    const tier = isHaikuModel(model) ? 'haiku' : 'sonnet_opus';
    const route = AUTO_ROUTE[tier];
    const primary = backends[route.primary];
    const fallback = backends[route.fallback];
    if (!primary?.apiKey && !fallback?.apiKey) return null;

    // Build context for the chosen primary (or fallback if primary missing)
    let ctx = null;
    let fb = null;

    const buildCtx = (name, cfg, remapModel) => ({
        name,
        target: cfg.target,
        apiKey: cfg.apiKey,
        useBearer: cfg.useBearer,
        isNonAnthropic: NON_ANTHROPIC_BACKENDS.has(name),
        model: remapModel,
    });

    if (primary?.apiKey) {
        const remap = (MODEL_REMAP[route.primary]?.[model] || model);
        ctx = buildCtx(route.primary, primary, remap);
        if (fallback?.apiKey) {
            fb = buildCtx(route.fallback, fallback, route.fallbackModel);
        }
    } else if (fallback?.apiKey) {
        const remap = (MODEL_REMAP[route.fallback]?.[model] || route.fallbackModel);
        ctx = buildCtx(route.fallback, fallback, remap);
    }

    return { ctx, fallback: fb, tier };
}

const MODEL_REMAP = {
    deepseek: {
        'claude-opus-4-6':    'deepseek-v4-pro',
        'claude-opus-4-7':    'deepseek-v4-pro',
        'claude-sonnet-4-6':  'deepseek-v4-flash',
        'claude-sonnet-4-5-20250929': 'deepseek-v4-flash',
        'claude-haiku-4-5-20251001':  'deepseek-v4-flash',
    },
    openrouter: {
        'claude-opus-4-6':    'deepseek/deepseek-v4-pro',
        'claude-opus-4-7':    'deepseek/deepseek-v4-pro',
        'claude-sonnet-4-6':  'deepseek/deepseek-v4-flash',
        'claude-sonnet-4-5-20250929': 'deepseek/deepseek-v4-flash',
        'claude-haiku-4-5-20251001':  'deepseek/deepseek-v4-flash',
    },
    gemini: {
        'claude-opus-4-6':    'gemini-3.5-flash',
        'claude-opus-4-7':    'gemini-3.5-flash',
        'claude-sonnet-4-6':  'gemini-3.5-flash',
        'claude-sonnet-4-5-20250929': 'gemini-3.5-flash',
        'claude-haiku-4-5-20251001':  'gemini-3.5-flash',
    },
};

const PRICING_PER_M = {
    deepseek:   { input: 0.44,  output: 0.87 },
    openrouter: { input: 0.44,  output: 0.87 },
    fireworks:  { input: 1.74,  output: 3.48 },
    gemini:     { input: 1.50,  output: 9.00 },
    anthropic:  { input: 3.00,  output: 15.00 },
    _single:    { input: 0.44,  output: 0.87 },
};

/**
 * Transform stream that intercepts SSE events and injects missing `usage`
 * fields. DeepSeek/OpenRouter may omit `usage` in message_start or
 * message_delta, which crashes Claude Code ("$.input_tokens" is undefined).
 */
class UsageNormalizer extends Transform {
    constructor(onUsage) {
        super();
        this._buf = '';
        this._onUsage = onUsage;
        this._inputTokens = 0;
        this._outputTokens = 0;
    }

    _transform(chunk, _enc, cb) {
        this._buf += chunk.toString();
        const parts = this._buf.split('\n\n');
        this._buf = parts.pop();
        for (const part of parts) {
            this.push(this._fix(part) + '\n\n');
        }
        cb();
    }

    _fix(event) {
        const m = event.match(/^data: (.+)$/m);
        if (!m) return event;
        try {
            const d = JSON.parse(m[1]);
            let changed = false;
            if (d.type === 'message_start' && d.message) {
                if (d.message.usage) {
                    this._inputTokens = d.message.usage.input_tokens || 0;
                } else {
                    d.message.usage = { input_tokens: 0, output_tokens: 0 };
                    changed = true;
                }
            }
            if (d.type === 'message_delta') {
                if (d.usage) {
                    this._outputTokens = d.usage.output_tokens || 0;
                } else {
                    d.usage = { output_tokens: 0 };
                    changed = true;
                }
            }
            if (changed) return event.replace(m[1], () => JSON.stringify(d));
        } catch { /* not JSON, pass through */ }
        return event;
    }

    _flush(cb) {
        if (this._buf.trim()) this.push(this._fix(this._buf) + '\n\n');
        if (this._onUsage) this._onUsage(this._inputTokens, this._outputTokens);
        cb();
    }
}

/**
 * For non-streaming JSON responses, ensure `usage` exists.
 */
function normalizeJsonBody(buf) {
    try {
        const obj = JSON.parse(buf);
        if (obj.type === 'message' && !obj.usage) {
            obj.usage = { input_tokens: 0, output_tokens: 0 };
            return Buffer.from(JSON.stringify(obj));
        }
    } catch { /* not JSON */ }
    return buf;
}

function stripAllThinkingBlocks(body) {
    if (!body?.messages) return;
    for (const msg of body.messages) {
        if (!Array.isArray(msg.content)) continue;
        msg.content = msg.content.filter(b => b.type !== 'thinking');
    }
}

function stripUnsignedThinkingBlocks(body) {
    if (!body?.messages) return;
    for (const msg of body.messages) {
        if (!Array.isArray(msg.content)) continue;
        msg.content = msg.content.filter(
            block => block.type !== 'thinking' || block.signature
        );
    }
}

// ---------------------------------------------------------------------------
// Anthropic API compatibility fixes (May 2026 breaking changes)
// ---------------------------------------------------------------------------

/**
 * Convert old `thinking: {"type": "enabled", "budget_tokens": N}` to new
 * `thinking: {"type": "adaptive"}` with a top-level `effort` parameter.
 * Budget mapping: ≤1024 → low, ≤4096 → medium, ≤8192 → high, >8192 → max.
 */
function normalizeThinkingBlocks(body) {
    if (!body?.thinking) return;
    if (body.thinking.type === 'enabled' && typeof body.thinking.budget_tokens === 'number') {
        const budget = body.thinking.budget_tokens;
        let effort = 'max';
        if (budget <= 1024) effort = 'low';
        else if (budget <= 4096) effort = 'medium';
        else if (budget <= 8192) effort = 'high';
        body.thinking = { type: 'adaptive' };
        body.effort = effort;
    }
}

/**
 * Strip sampling parameters (temperature, top_p, top_k) when thinking is
 * active, because Anthropic rejects these alongside extended thinking.
 */
function stripSamplingParamsOnThinking(body) {
    if (!body?.thinking) return;
    if (body.thinking.type === 'adaptive') {
        delete body.temperature;
        delete body.top_p;
        delete body.top_k;
    }
}

/**
 * Normalize the renamed `output_format` → `output_config` field.  The old API
 * used a top-level `output_format` string; the new API expects
 * `output_config: { format: "..." }`.
 */
function normalizeOutputConfig(body) {
    if (!body) return;
    if ('output_format' in body) {
        if (!('output_config' in body)) {
            body.output_config = { format: body.output_format };
        }
        delete body.output_format;
    }
}

/**
 * Filter out assistant messages with empty content arrays (prefill blocks).
 * Anthropic now returns 400 for such messages.
 */
function filterAssistantPrefill(body) {
    if (!body?.messages) return;
    body.messages = body.messages.filter(msg => {
        if (msg.role === 'assistant') {
            return Array.isArray(msg.content) && msg.content.length > 0;
        }
        return true;
    });
}

/**
 * Warn if an OAuth token (sk-ant-oat*) is detected. OAuth tokens are blocked
 * from 3rd-party tools; users must switch to sk-ant-api03-* keys.
 */
function warnOAuthToken(apiKey) {
    if (typeof apiKey === 'string' && apiKey.startsWith('sk-ant-oat')) {
        console.warn('[MODEL-PROXY] WARNING: Detected OAuth API key (sk-ant-oat*). OAuth tokens are blocked from 3rd-party tools. Use sk-ant-api03-* keys instead.');
    }
}

/**
 * Apply all Anthropic API compatibility fixes to a parsed request body.
 */
function normalizeAnthropicRequestBody(body) {
    if (!body) return;
    normalizeOutputConfig(body);
    filterAssistantPrefill(body);
    normalizeThinkingBlocks(body);
    stripSamplingParamsOnThinking(body);
}

export function startModelProxy({ targetUrl, apiKey, startPort = 3200, backends, defaultMode }) {
    return new Promise((resolve, reject) => {
        const initialTarget = new URL(targetUrl);
        const initialBearer = targetUrl.includes('openrouter') || targetUrl.includes('fireworks');

        const allBackends = {};
        if (backends) {
            for (const [name, cfg] of Object.entries(backends)) {
                allBackends[name] = {
                    target: new URL(cfg.url),
                    apiKey: cfg.apiKey,
                    useBearer: cfg.url.includes('openrouter') || cfg.url.includes('fireworks'),
                };
            }
        }
        const initialName = defaultMode || (backends ? 'anthropic' : null);
        const startBackend = initialName && initialName !== 'anthropic' && allBackends[initialName];

        const state = {
            mode: initialName || '_single',
            target: startBackend ? startBackend.target : initialTarget,
            apiKey: startBackend ? startBackend.apiKey : apiKey,
            useBearer: startBackend ? startBackend.useBearer : initialBearer,
            hadNonAnthropicSession: !!startBackend,
        };

        // Warn about deprecated OAuth tokens
        warnOAuthToken(apiKey);
        if (backends) {
            for (const [, cfg] of Object.entries(backends)) {
                warnOAuthToken(cfg.apiKey);
            }
        }

        // Auto-detect Gemini legacy mode — targetUrl contains generativelanguage
        if (state.mode === '_single' && state.target.hostname.includes('generativelanguage')) {
            state.mode = 'gemini';
        }

        let reqCount = 0;
        const t0Global = Date.now();
        const costs = {};

        function recordUsage(backend, inputTokens, outputTokens) {
            if (!costs[backend]) costs[backend] = { input: 0, output: 0, requests: 0 };
            costs[backend].input += inputTokens || 0;
            costs[backend].output += outputTokens || 0;
            costs[backend].requests++;
        }

        function getCostSummary() {
            const summary = {};
            let totalActual = 0;
            let totalAnthropic = 0;
            for (const [backend, tokens] of Object.entries(costs)) {
                const p = PRICING_PER_M[backend] || PRICING_PER_M._single;
                const ap = PRICING_PER_M.anthropic;
                const actual = (tokens.input * p.input + tokens.output * p.output) / 1_000_000;
                const anthropicEq = (tokens.input * ap.input + tokens.output * ap.output) / 1_000_000;
                totalActual += actual;
                totalAnthropic += anthropicEq;
                summary[backend] = {
                    input_tokens: tokens.input,
                    output_tokens: tokens.output,
                    requests: tokens.requests,
                    cost: +actual.toFixed(4),
                    anthropic_equivalent: +anthropicEq.toFixed(4),
                };
            }
            return {
                backends: summary,
                total_cost: +totalActual.toFixed(4),
                anthropic_equivalent: +totalAnthropic.toFixed(4),
                savings: +((totalAnthropic - totalActual).toFixed(4)),
            };
        }

        function switchMode(name) {
            if (name === 'anthropic') {
                const prev = state.mode;
                state.mode = 'anthropic';
                state.target = new URL(ANTHROPIC_FALLBACK);
                state.apiKey = null;
                state.useBearer = false;
                state.hadNonAnthropicSession = false;
                return { mode: 'anthropic', previous: prev };
            }
            if (name === 'auto') {
                const available = [];
                if (allBackends.deepseek?.apiKey) available.push('deepseek');
                if (allBackends.gemini?.apiKey) available.push('gemini');
                if (available.length === 0) return { error: 'Auto mode requires at least one backend (deepseek or gemini) with API key set.' };
                const prev = state.mode;
                state.mode = 'auto';
                state.target = allBackends.deepseek?.target || allBackends.gemini?.target || new URL('https://api.deepseek.com/anthropic');
                state.apiKey = allBackends.deepseek?.apiKey || allBackends.gemini?.apiKey;
                state.useBearer = false;
                state.hadNonAnthropicSession = true;
                return { mode: 'auto', previous: prev, available };
            }
            const b = allBackends[name];
            if (!b) return { error: `Unknown backend: ${name}. Valid: anthropic, auto, ${Object.keys(allBackends).join(', ')}` };
            if (!b.apiKey) return { error: `API key not set for ${name}` };
            const prev = state.mode;
            state.mode = name;
            state.target = b.target;
            state.apiKey = b.apiKey;
            state.useBearer = b.useBearer;
            state.hadNonAnthropicSession = true;
            return { mode: name, previous: prev };
        }

        const server = createServer((clientReq, clientRes) => {
            const urlPath = clientReq.url.split('?')[0];

            // Control endpoints — /_proxy/* (never collides with /v1/*)
            if (urlPath.startsWith('/_proxy/')) {
                if (urlPath === '/_proxy/status') {
                    const status = {
                        mode: state.mode,
                        uptime: Math.round((Date.now() - t0Global) / 1000),
                        requests: reqCount,
                    };
                    if (state.mode === 'auto') {
                        status.routing = {
                            haiku: { primary: 'gemini', fallback: 'deepseek' },
                            sonnet_opus: { primary: 'deepseek', fallback: 'gemini' },
                        };
                        status.backends = {};
                        if (allBackends.deepseek?.apiKey) status.backends.deepseek = 'available';
                        if (allBackends.gemini?.apiKey) status.backends.gemini = 'available';
                    }
                    clientRes.writeHead(200, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify(status));
                    return;
                }
                if (urlPath === '/_proxy/cost') {
                    clientRes.writeHead(200, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify(getCostSummary()));
                    return;
                }
                if (urlPath === '/_proxy/mode' && clientReq.method === 'POST') {
                    const origin = clientReq.headers['origin'] || '';
                    if (origin && !origin.startsWith('http://127.0.0.1') && !origin.startsWith('http://localhost')) {
                        clientRes.writeHead(403, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify({ error: 'Forbidden' }));
                        return;
                    }
                    const chunks = [];
                    let bodySize = 0;
                    clientReq.on('data', c => {
                        bodySize += c.length;
                        if (bodySize > 1024) { clientReq.destroy(); return; }
                        chunks.push(c);
                    });
                    clientReq.on('end', () => {
                        const body = Buffer.concat(chunks).toString();
                        const m = body.match(/backend=([a-z]+)/);
                        if (!m) {
                            clientRes.writeHead(400, { 'content-type': 'application/json' });
                            clientRes.end(JSON.stringify({ error: 'Missing backend= in body' }));
                            return;
                        }
                        const result = switchMode(m[1]);
                        if (result.error) {
                            clientRes.writeHead(400, { 'content-type': 'application/json' });
                            clientRes.end(JSON.stringify(result));
                            return;
                        }
                        console.log(`[MODEL-PROXY] Mode switched: ${result.previous} → ${result.mode}`);
                        clientRes.writeHead(200, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify(result));
                    });
                    return;
                }
                if (urlPath === '/_proxy/mode' && clientReq.method !== 'POST') {
                    clientRes.writeHead(405, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({ error: 'Use POST' }));
                    return;
                }
                clientRes.writeHead(404, { 'content-type': 'application/json' });
                clientRes.end(JSON.stringify({ error: 'Not found' }));
                return;
            }

            const reqId = ++reqCount;
            const t0 = Date.now();

            // --- Routing context (overridden for auto mode after body parse) ---
            const isAnthropicMode = state.mode === 'anthropic';
            const isModelCall = !isAnthropicMode && MODEL_PATHS.includes(urlPath);
            const isAutoMode = state.mode === 'auto';
            let isNonAnthropic = false;
            let dest = null;
            let fullPath = null;

            // For non-auto modes, resolve routing now; auto mode resolves after body parse
            if (!isAutoMode) {
                isNonAnthropic = NON_ANTHROPIC_BACKENDS.has(state.mode);
                dest = isModelCall ? state.target : new URL(ANTHROPIC_FALLBACK);

                if (isModelCall) {
                    const base = state.target.pathname.replace(/\/$/, '');
                    let overlap = '';
                    for (let i = 1; i <= Math.min(base.length, urlPath.length); i++) {
                        if (base.endsWith(urlPath.substring(0, i))) overlap = urlPath.substring(0, i);
                    }
                    fullPath = overlap ? base + urlPath.substring(overlap.length) : base + urlPath;
                } else {
                    fullPath = clientReq.url;
                }
            }

            // Deferred for auto mode: backendCtx set after body parse
            let backendCtx = null;
            let fallbackCtx = null;

            // Headers (auth set after backend resolution for auto mode)
            const headers = { ...clientReq.headers };
            delete headers['content-length'];

            const chunks = [];
            clientReq.on('data', c => chunks.push(c));
            clientReq.on('end', () => {
                let body = Buffer.concat(chunks);

                // ── Auto mode: resolve backend from model tier ──
                if (isAutoMode && isModelCall) {
                    try {
                        const parsed = JSON.parse(body);
                        const route = resolveAutoBackend(parsed.model || '', allBackends);
                        if (!route) {
                            clientRes.writeHead(502, { 'content-type': 'application/json' });
                            clientRes.end(JSON.stringify({ error: { message: 'Auto mode: no backend available. Set DEEPSEEK_API_KEY and/or GEMINI_API_KEY.' } }));
                            return;
                        }
                        backendCtx = route.ctx;
                        fallbackCtx = route.fallback;
                        dest = backendCtx.target;
                        isNonAnthropic = backendCtx.isNonAnthropic;
                        // Build fullPath for this backend
                        const base = dest.pathname.replace(/\/$/, '');
                        let overlap = '';
                        for (let i = 1; i <= Math.min(base.length, urlPath.length); i++) {
                            if (base.endsWith(urlPath.substring(0, i))) overlap = urlPath.substring(0, i);
                        }
                        fullPath = overlap ? base + urlPath.substring(overlap.length) : base + urlPath;
                        // Auth
                        headers.host = dest.host;
                        delete headers['authorization'];
                        delete headers['x-api-key'];
                        delete headers['x-goog-api-key'];
                        if (isNonAnthropic) {
                            headers['x-goog-api-key'] = backendCtx.apiKey;
                        } else if (backendCtx.useBearer) {
                            headers['authorization'] = `Bearer ${backendCtx.apiKey}`;
                        } else {
                            headers['x-api-key'] = backendCtx.apiKey;
                        }
                        console.log(`[MODEL-PROXY] #${reqId} auto → ${backendCtx.name} (${route.tier}, ${parsed.model} → ${backendCtx.model})${fallbackCtx ? ' | fallback: ' + fallbackCtx.name : ''}`);
                    } catch (e) {
                        console.error(`[MODEL-PROXY] #${reqId} auto resolve error: ${e.message}`);
                        clientRes.writeHead(502, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify({ error: { message: 'Auto mode: failed to parse request' } }));
                        return;
                    }
                }

                // For non-auto model calls, set auth now
                if (isModelCall && !isAutoMode) {
                    headers.host = dest.host;
                    delete headers['authorization'];
                    delete headers['x-api-key'];
                    delete headers['x-goog-api-key'];
                    if (isNonAnthropic) {
                        headers['x-goog-api-key'] = state.apiKey;
                    } else if (state.useBearer) {
                        headers['authorization'] = `Bearer ${state.apiKey}`;
                    } else {
                        headers['x-api-key'] = state.apiKey;
                    }
                }

                const backendName = backendCtx ? backendCtx.name : state.mode;

                // Remap Anthropic model names to backend-specific names
                let remappedModel = null;
                if (backendCtx) {
                    try {
                        const parsed = JSON.parse(body);
                        console.log(`[MODEL-PROXY] #${reqId} auto model: ${parsed.model} → ${backendCtx.model}`);
                        parsed.model = backendCtx.model;
                        remappedModel = backendCtx.model;
                        body = Buffer.from(JSON.stringify(parsed));
                    } catch { /* pass */ }
                } else if (isModelCall && MODEL_REMAP[state.mode]) {
                    try {
                        const parsed = JSON.parse(body);
                        const mapped = MODEL_REMAP[state.mode][parsed.model];
                        if (mapped) {
                            console.log(`[MODEL-PROXY] #${reqId} model remap: ${parsed.model} → ${mapped}`);
                            parsed.model = mapped;
                            remappedModel = mapped;
                            body = Buffer.from(JSON.stringify(parsed));
                        }
                    } catch { /* not JSON or parse error, pass through */ }
                }

                // Translate request body for non-Anthropic backends (Gemini)
                if (isModelCall && isNonAnthropic) {
                    try {
                        const parsed = JSON.parse(body);
                        const { geminiBody, geminiModel } = translateRequest(parsed);
                        remappedModel = geminiModel;
                        body = Buffer.from(JSON.stringify(geminiBody));
                        const streamParam = parsed.stream !== false ? 'streamGenerateContent?alt=sse' : 'generateContent';
                        fullPath = `/v1beta/models/${geminiModel}:${streamParam}`;
                        console.log(`[MODEL-PROXY] #${reqId} Gemini: ${geminiModel} → ${fullPath}`);
                    } catch(e) {
                        console.error(`[MODEL-PROXY] #${reqId} Gemini translate error: ${e.message}`);
                    }
                } else {
                    // Apply Anthropic API compatibility fixes (May 2026 breaking changes)
                    if (MODEL_PATHS.includes(urlPath)) {
                        try {
                            const parsed = JSON.parse(body);
                            normalizeAnthropicRequestBody(parsed);
                            body = Buffer.from(JSON.stringify(parsed));
                        } catch { /* pass through */ }
                    }

                    // Strip thinking blocks before forwarding
                    if (isAnthropicMode && MODEL_PATHS.includes(urlPath)) {
                        try {
                            const parsed = JSON.parse(body);
                            if (state.hadNonAnthropicSession) {
                                stripAllThinkingBlocks(parsed);
                            } else {
                                stripUnsignedThinkingBlocks(parsed);
                            }
                            body = Buffer.from(JSON.stringify(parsed));
                        } catch { /* pass through */ }
                    }
                    if (isModelCall) {
                        try {
                            const parsed = JSON.parse(body);
                            stripAllThinkingBlocks(parsed);
                            body = Buffer.from(JSON.stringify(parsed));
                        } catch { /* pass through */ }
                    }
                }

                if (isModelCall && !isAutoMode) {
                    console.log(`[MODEL-PROXY] #${reqId} → ${dest.hostname}${fullPath}`);
                }

                // ── Send upstream (with failover for auto mode) ──
                sendUpstream(body);

                function sendUpstream(bodyToSend, isRetry) {
                    const useCtx = isRetry && fallbackCtx ? fallbackCtx : backendCtx;
                    const useDest = useCtx ? useCtx.target : dest;
                    const useNonAnthropic = useCtx ? useCtx.isNonAnthropic : isNonAnthropic;
                    const useName = useCtx ? useCtx.name : backendName;
                    let usePath = fullPath;
                    let useBody = bodyToSend;

                    if (isRetry && fallbackCtx) {
                        // Rebuild path and body for fallback
                        const fbBase = fallbackCtx.target.pathname.replace(/\/$/, '');
                        let fbOverlap = '';
                        for (let i = 1; i <= Math.min(fbBase.length, urlPath.length); i++) {
                            if (fbBase.endsWith(urlPath.substring(0, i))) fbOverlap = urlPath.substring(0, i);
                        }
                        usePath = fbOverlap ? fbBase + urlPath.substring(fbOverlap.length) : fbBase + urlPath;
                        // Translate if fallback is Gemini
                        if (fallbackCtx.isNonAnthropic) {
                            try {
                                const parsed = JSON.parse(bodyToSend);
                                parsed.model = fallbackCtx.model;
                                const { geminiBody, geminiModel } = translateRequest(parsed);
                                const streamParam = parsed.stream !== false ? 'streamGenerateContent?alt=sse' : 'generateContent';
                                usePath = `/v1beta/models/${geminiModel}:${streamParam}`;
                                useBody = Buffer.from(JSON.stringify(geminiBody));
                            } catch { /* use as-is */ }
                        } else {
                            try {
                                const parsed = JSON.parse(bodyToSend);
                                parsed.model = fallbackCtx.model;
                                useBody = Buffer.from(JSON.stringify(parsed));
                            } catch { /* use as-is */ }
                        }
                        // Rebuild headers for fallback
                        headers.host = fallbackCtx.target.host;
                        delete headers['authorization'];
                        delete headers['x-api-key'];
                        delete headers['x-goog-api-key'];
                        if (fallbackCtx.isNonAnthropic) {
                            headers['x-goog-api-key'] = fallbackCtx.apiKey;
                        } else if (fallbackCtx.useBearer) {
                            headers['authorization'] = `Bearer ${fallbackCtx.apiKey}`;
                        } else {
                            headers['x-api-key'] = fallbackCtx.apiKey;
                        }
                        console.log(`[MODEL-PROXY] #${reqId} FAILOVER → ${fallbackCtx.name} (${fallbackCtx.model})`);
                    }

                    const opts = {
                        hostname: useDest.hostname,
                        port: useDest.port || 443,
                        path: usePath,
                        method: clientReq.method,
                        headers: { ...headers, 'content-length': useBody.length },
                        timeout: REQUEST_TIMEOUT_MS,
                    };

                    const proxyReq = httpsRequest(opts, (proxyRes) => {
                        if (isModelCall) {
                            const ttfb = Date.now() - t0;
                            console.log(`[MODEL-PROXY] #${reqId} TTFB ${ttfb}ms (status ${proxyRes.statusCode})${isRetry ? ' [retry]' : ''}`);
                        }

                        const ct = proxyRes.headers['content-type'] || '';
                        const isSSE = ct.includes('text/event-stream');

                        if (isModelCall && isSSE) {
                            if (useNonAnthropic) {
                                const translator = new GeminiStreamTranslator();
                                const norm = new UsageNormalizer((inp, out) => recordUsage(useName, inp, out));
                                clientRes.writeHead(proxyRes.statusCode, {
                                    'content-type': 'text/event-stream',
                                    'cache-control': 'no-cache',
                                    'connection': 'keep-alive',
                                });
                                proxyRes.pipe(translator).pipe(norm).pipe(clientRes);
                                proxyRes.on('end', () => {
                                    const totalOut = translator.outputTokens || norm._outputTokens;
                                    console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${useName}, ${translator.inputTokens}in/${totalOut}out)${isRetry ? ' [retry]' : ''}`);
                                });
                            } else {
                                clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
                                const norm = new UsageNormalizer((inp, out) => recordUsage(useName, inp, out));
                                proxyRes.pipe(norm).pipe(clientRes);
                                proxyRes.on('end', () => {
                                    console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${norm._inputTokens}in/${norm._outputTokens}out)${isRetry ? ' [retry]' : ''}`);
                                });
                            }
                        } else if (isModelCall && ct.includes('application/json')) {
                            const respChunks = [];
                            proxyRes.on('data', c => respChunks.push(c));
                            proxyRes.on('end', () => {
                                const raw = Buffer.concat(respChunks);
                                if (useNonAnthropic) {
                                    const translated = translateNonStreamingResponse(raw);
                                    recordUsage(useName, translated.usage.input_tokens, translated.usage.output_tokens);
                                    const fixed = Buffer.from(JSON.stringify(translated));
                                    const outHeaders = { 'content-type': 'application/json', 'content-length': fixed.length };
                                    clientRes.writeHead(proxyRes.statusCode, outHeaders);
                                    clientRes.end(fixed);
                                    console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${useName}-json, ${fixed.length}b)${isRetry ? ' [retry]' : ''}`);
                                } else {
                                    const fixed = normalizeJsonBody(raw);
                                    try {
                                        const j = JSON.parse(fixed);
                                        if (j.usage) recordUsage(useName, j.usage.input_tokens, j.usage.output_tokens);
                                    } catch {}
                                    const outHeaders = { ...proxyRes.headers, 'content-length': fixed.length };
                                    clientRes.writeHead(proxyRes.statusCode, outHeaders);
                                    clientRes.end(fixed);
                                    console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (json, ${fixed.length}b)${isRetry ? ' [retry]' : ''}`);
                                }
                            });
                        } else {
                            clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
                            proxyRes.pipe(clientRes);
                            if (isModelCall) {
                                proxyRes.on('end', () => {
                                    console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
                                });
                            }
                        }
                    });

                    proxyReq.on('timeout', () => {
                        console.error(`[MODEL-PROXY] #${reqId} TIMEOUT after ${REQUEST_TIMEOUT_MS / 1000}s (${useName})`);
                        if (!isRetry && fallbackCtx && isAutoMode) {
                            console.log(`[MODEL-PROXY] #${reqId} failover on timeout`);
                            sendUpstream(bodyToSend, true);
                        } else {
                            proxyReq.destroy(new Error('Request timeout'));
                        }
                    });

                    proxyReq.on('error', (err) => {
                        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
                        console.error(`[MODEL-PROXY] #${reqId} ERROR after ${elapsed}s: ${err.message} (${useName})`);
                        if (!isRetry && fallbackCtx && isAutoMode) {
                            console.log(`[MODEL-PROXY] #${reqId} failover on error`);
                            sendUpstream(bodyToSend, true);
                        } else if (!clientRes.headersSent) {
                            clientRes.writeHead(502, { 'content-type': 'application/json' });
                            clientRes.end(JSON.stringify({ error: { message: 'Upstream connection error' } }));
                        }
                    });

                    proxyReq.end(useBody);
                }
            });
        });

        function tryListen(port) {
            server.once('error', (err) => {
                if (err.code === 'EADDRINUSE' && port < startPort + 20) {
                    tryListen(port + 1);
                } else {
                    reject(err);
                }
            });
            server.listen(port, '127.0.0.1', () => {
                const actualPort = server.address().port;
                console.log(`[MODEL-PROXY] Listening on 127.0.0.1:${actualPort} → ${targetUrl} (mode: ${state.mode})`);
                resolve({ port: actualPort, close: () => server.close(), switchMode });
            });
        }

        tryListen(startPort);
    });
}

export {
    MODEL_REMAP,
    PRICING_PER_M,
    NON_ANTHROPIC_BACKENDS,
    AUTO_ROUTE,
    isHaikuModel,
    resolveAutoBackend,
    stripAllThinkingBlocks,
    stripUnsignedThinkingBlocks,
    normalizeThinkingBlocks,
    stripSamplingParamsOnThinking,
    normalizeOutputConfig,
    filterAssistantPrefill,
    warnOAuthToken,
    normalizeAnthropicRequestBody,
};
