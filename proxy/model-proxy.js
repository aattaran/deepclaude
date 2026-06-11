import { createServer, request as httpRequest } from 'http';
import { request as httpsRequest } from 'https';
import { URL } from 'url';
import { Transform } from 'stream';
import {
    buildResponsesRequest,
    parseResult,
    synthesizeSSE,
    ssePing,
    isTerminalStatus,
    pollIntervalForWindow,
    toolsErrorIndicatesUnsupported,
} from './responses-adapter.js';

const ANTHROPIC_FALLBACK = 'https://api.anthropic.com';
const MODEL_PATHS = ['/v1/messages'];
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;
const PING_INTERVAL_MS = 25_000;

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
};

function loadPricing() {
    const sIn = parseFloat(process.env.SFERENCE_INPUT_PRICE || '0');
    const sOut = parseFloat(process.env.SFERENCE_OUTPUT_PRICE || '0');
    const bgIn = parseFloat(process.env.BG_PROVIDER_INPUT_PRICE || '0');
    const bgOut = parseFloat(process.env.BG_PROVIDER_OUTPUT_PRICE || '0');
    return {
        deepseek:   { input: 0.44,  output: 0.87 },
        openrouter: { input: 0.44,  output: 0.87 },
        fireworks:  { input: 1.74,  output: 3.48 },
        anthropic:  { input: 3.00,  output: 15.00 },
        sference:   { input: sIn,   output: sOut },
        bg:         { input: bgIn,  output: bgOut },
        _single:    { input: 0.44,  output: 0.87 },
    };
}

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
        } catch { /* not JSON */ }
        return event;
    }

    _flush(cb) {
        if (this._buf.trim()) this.push(this._fix(this._buf) + '\n\n');
        if (this._onUsage) this._onUsage(this._inputTokens, this._outputTokens);
        cb();
    }
}

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

function normalizeBackendConfig(name, cfg) {
    const url = cfg.url || cfg.target;
    const useBearer = cfg.authStyle === 'bearer'
        || (cfg.authStyle !== 'x-api-key' && (url.includes('openrouter') || url.includes('fireworks')));
    return {
        name,
        target: new URL(url),
        apiKey: cfg.apiKey,
        type: cfg.type || 'anthropic-compatible',
        useBearer,
        model: cfg.model || null,
        window: cfg.window || null,
        pricing: cfg.pricing || null,
    };
}

function authHeaders(backend) {
    const h = { 'content-type': 'application/json' };
    if (!backend.apiKey) return h;
    if (backend.useBearer) {
        h.authorization = `Bearer ${backend.apiKey}`;
    } else {
        h['x-api-key'] = backend.apiKey;
    }
    return h;
}

function responsesPath(baseUrl) {
    const base = baseUrl.pathname.replace(/\/$/, '');
    return base.endsWith('/v1') ? `${base}/responses` : `${base}/v1/responses`;
}

function httpsJson(method, url, headers, body) {
    return new Promise((resolve, reject) => {
        const u = new URL(url);
        const isHttp = u.protocol === 'http:';
        const requestFn = isHttp ? httpRequest : httpsRequest;
        const payload = body != null ? JSON.stringify(body) : undefined;
        const opts = {
            hostname: u.hostname,
            port: u.port || (isHttp ? 80 : 443),
            path: u.pathname + u.search,
            method,
            headers: {
                ...headers,
                ...(payload ? { 'content-length': Buffer.byteLength(payload) } : {}),
            },
            timeout: REQUEST_TIMEOUT_MS,
        };

        const req = requestFn(opts, (res) => {
            const chunks = [];
            res.on('data', c => chunks.push(c));
            res.on('end', () => {
                const raw = Buffer.concat(chunks).toString();
                let json = null;
                try { json = raw ? JSON.parse(raw) : null; } catch { json = { _raw: raw }; }
                resolve({ status: res.statusCode, json, raw });
            });
        });
        req.on('timeout', () => req.destroy(new Error('Request timeout')));
        req.on('error', reject);
        if (payload) req.end(payload);
        else req.end();
    });
}

function applyBackendToState(state, backend) {
    state.mode = backend.name;
    state.target = backend.target;
    state.apiKey = backend.apiKey;
    state.useBearer = backend.useBearer;
    state.backendType = backend.type;
    state.model = backend.model;
    state.window = backend.window;
    if (backend.type !== 'anthropic-compatible') state.hadNonAnthropicSession = true;
}

export function startModelProxy({ targetUrl, apiKey, startPort = 3200, backends, defaultMode, initialBackend }) {
    return new Promise((resolve, reject) => {
        const PRICING_PER_M = loadPricing();

        const allBackends = {};
        if (backends) {
            for (const [name, cfg] of Object.entries(backends)) {
                if (cfg?.url || cfg?.target) {
                    allBackends[name] = normalizeBackendConfig(name, cfg);
                }
            }
        }

        if (defaultMode && initialBackend && !allBackends[defaultMode]) {
            allBackends[defaultMode] = normalizeBackendConfig(defaultMode, initialBackend);
        }

        let singleBackend = null;
        if (initialBackend) {
            singleBackend = normalizeBackendConfig('_single', initialBackend);
        } else if (targetUrl && apiKey) {
            singleBackend = normalizeBackendConfig('_single', {
                url: targetUrl,
                apiKey,
                type: 'anthropic-compatible',
            });
        }

        const initialName = defaultMode || (Object.keys(allBackends).length ? 'anthropic' : null);
        const startBackend = initialName && initialName !== 'anthropic' && allBackends[initialName];

        const state = {
            mode: initialName || '_single',
            target: startBackend ? startBackend.target : (singleBackend?.target || new URL(targetUrl || ANTHROPIC_FALLBACK)),
            apiKey: startBackend ? startBackend.apiKey : (singleBackend?.apiKey || apiKey),
            useBearer: startBackend ? startBackend.useBearer : (singleBackend?.useBearer || false),
            backendType: startBackend ? startBackend.type : (singleBackend?.type || 'anthropic-compatible'),
            model: startBackend?.model || singleBackend?.model || null,
            window: startBackend?.window || singleBackend?.window || null,
            hadNonAnthropicSession: !!(startBackend || (singleBackend?.type === 'responses-bg')),
        };

        const toolModes = {};
        let reqCount = 0;
        const t0Global = Date.now();
        const costs = {};

        function activeBackend() {
            if (state.mode === 'anthropic') return null;
            if (state.mode === '_single') return singleBackend;
            return allBackends[state.mode] || null;
        }

        function getToolMode(name) {
            return toolModes[name] || 'native';
        }

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
                const b = allBackends[backend];
                const p = b?.pricing || PRICING_PER_M[backend] || PRICING_PER_M._single;
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
                state.backendType = 'anthropic-compatible';
                state.model = null;
                return { mode: 'anthropic', previous: prev };
            }
            const b = allBackends[name];
            if (!b) return { error: `Unknown backend: ${name}. Valid: anthropic, ${Object.keys(allBackends).join(', ')}` };
            if (!b.apiKey) return { error: `API key not set for ${name}` };
            const prev = state.mode;
            applyBackendToState(state, b);
            return { mode: name, previous: prev, type: b.type, window: b.window };
        }

        function setWindow(window) {
            const prev = state.window;
            state.window = window;
            const b = activeBackend();
            if (b) b.window = window;
            return { window, previous: prev };
        }

        async function submitBackgroundRequest(backend, anthropicBody, toolMode, reqId) {
            const base = `${backend.target.protocol}//${backend.target.host}`;
            const createUrl = `${base}${responsesPath(backend.target)}`;
            const headers = authHeaders(backend);

            const body = buildResponsesRequest(anthropicBody, {
                model: backend.model || anthropicBody.model,
                window: backend.window || state.window,
                toolMode,
            });

            console.error(`[MODEL-PROXY] #${reqId} bg submit (${toolMode}) → ${createUrl}`);
            let res = await httpsJson('POST', createUrl, headers, body);

            if (res.status === 400 && toolMode === 'native' && anthropicBody.tools?.length
                && toolsErrorIndicatesUnsupported(res.raw || res.json)) {
                console.error(`[MODEL-PROXY] #${reqId} tools rejected, falling back to emulate`);
                toolModes[backend.name || state.mode] = 'emulate';
                const emulated = buildResponsesRequest(anthropicBody, {
                    model: backend.model || anthropicBody.model,
                    window: backend.window || state.window,
                    toolMode: 'emulate',
                });
                res = await httpsJson('POST', createUrl, headers, emulated);
            } else if (res.status >= 200 && res.status < 300 && toolMode === 'native') {
                toolModes[backend.name || state.mode] = 'native';
            }

            return res;
        }

        async function pollBackgroundResponse(backend, responseId, reqId, onPing, signal) {
            const base = `${backend.target.protocol}//${backend.target.host}`;
            const pollUrl = `${base}${responsesPath(backend.target)}/${responseId}`;
            const headers = authHeaders(backend);
            const interval = pollIntervalForWindow(backend.window || state.window || '1h');

            while (!signal.aborted) {
                const res = await httpsJson('GET', pollUrl, headers);
                if (res.status !== 200) {
                    throw new Error(`Poll failed (${res.status}): ${res.raw?.slice(0, 200)}`);
                }
                const status = res.json?.status;
                console.error(`[MODEL-PROXY] #${reqId} poll status=${status}`);
                if (isTerminalStatus(status)) return res.json;
                await new Promise(r => setTimeout(r, interval));
                if (onPing) onPing();
            }
            throw new Error('Poll aborted');
        }

        async function cancelBackgroundResponse(backend, responseId) {
            const base = `${backend.target.protocol}//${backend.target.host}`;
            const url = `${base}${responsesPath(backend.target)}/${responseId}`;
            try {
                await httpsJson('DELETE', url, authHeaders(backend));
                console.error(`[MODEL-PROXY] cancelled bg response ${responseId}`);
            } catch (e) {
                console.error(`[MODEL-PROXY] cancel failed: ${e.message}`);
            }
        }

        async function handleResponsesBg(clientReq, clientRes, bodyBuf, reqId, t0) {
            const backend = activeBackend();
            if (!backend) {
                clientRes.writeHead(502, { 'content-type': 'application/json' });
                clientRes.end(JSON.stringify({ error: { message: 'No active background backend' } }));
                return;
            }

            let anthropicBody;
            try {
                anthropicBody = JSON.parse(bodyBuf);
                stripAllThinkingBlocks(anthropicBody);
            } catch {
                clientRes.writeHead(400, { 'content-type': 'application/json' });
                clientRes.end(JSON.stringify({ error: { message: 'Invalid JSON body' } }));
                return;
            }

            // Anthropic semantics: stream only when explicitly requested.
            const wantsStream = anthropicBody.stream === true;
            const usageKey = state.mode;
            const toolMode = getToolMode(backend.name || state.mode);
            const abort = { aborted: false };
            let responseId = null;
            let pingTimer = null;

            const canWrite = () => !abort.aborted && !clientRes.writableEnded && !clientRes.destroyed;

            clientRes.on('close', () => {
                if (!clientRes.writableEnded) {
                    abort.aborted = true;
                    if (pingTimer) clearInterval(pingTimer);
                    if (responseId) cancelBackgroundResponse(backend, responseId);
                }
            });

            try {
                const submit = await submitBackgroundRequest(backend, anthropicBody, toolMode, reqId);
                if (submit.status < 200 || submit.status >= 300) {
                    clientRes.writeHead(submit.status, { 'content-type': 'application/json' });
                    clientRes.end(submit.raw || JSON.stringify(submit.json));
                    return;
                }

                responseId = submit.json?.id;
                if (!responseId) {
                    clientRes.writeHead(502, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({ error: { message: 'No response id from provider' } }));
                    return;
                }

                if (wantsStream) {
                    clientRes.writeHead(200, {
                        'content-type': 'text/event-stream',
                        'cache-control': 'no-cache',
                        connection: 'keep-alive',
                    });
                    clientRes.write(ssePing());
                    pingTimer = setInterval(() => {
                        if (canWrite()) clientRes.write(ssePing());
                    }, PING_INTERVAL_MS);
                }

                const result = await pollBackgroundResponse(backend, responseId, reqId, () => {
                    if (wantsStream && canWrite()) clientRes.write(ssePing());
                }, abort);

                if (abort.aborted) return;
                if (pingTimer) clearInterval(pingTimer);

                const mode = getToolMode(backend.name || state.mode);
                const parsed = parseResult(result, mode);
                recordUsage(usageKey, parsed.usage?.input_tokens, parsed.usage?.output_tokens);

                if (wantsStream) {
                    clientRes.end(synthesizeSSE(parsed));
                } else if (parsed.error) {
                    clientRes.writeHead(502, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: parsed.message } }));
                } else {
                    clientRes.writeHead(200, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({
                        id: parsed.id,
                        type: 'message',
                        role: 'assistant',
                        model: parsed.model,
                        content: parsed.content,
                        stop_reason: parsed.stop_reason,
                        stop_sequence: null,
                        usage: parsed.usage,
                    }));
                }

                console.error(`[MODEL-PROXY] #${reqId} bg done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
            } catch (err) {
                if (pingTimer) clearInterval(pingTimer);
                console.error(`[MODEL-PROXY] #${reqId} bg ERROR: ${err.message}`);
                if (abort.aborted || clientRes.destroyed) return;
                if (clientRes.headersSent) {
                    // Headers already sent as SSE — emit an Anthropic error event.
                    if (!clientRes.writableEnded) {
                        clientRes.end(`event: error\ndata: ${JSON.stringify({ type: 'error', error: { type: 'api_error', message: err.message } })}\n\n`);
                    }
                } else {
                    clientRes.writeHead(502, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: err.message } }));
                }
            }
        }

        // CSRF guard: browser pages must not be able to flip the backend.
        function isForbiddenOrigin(clientReq) {
            const origin = clientReq.headers['origin'] || '';
            return !!origin && !origin.startsWith('http://127.0.0.1') && !origin.startsWith('http://localhost');
        }

        function readControlBody(clientReq, maxSize, onDone) {
            const chunks = [];
            let bodySize = 0;
            clientReq.on('data', c => {
                bodySize += c.length;
                if (bodySize > maxSize) { clientReq.destroy(); return; }
                chunks.push(c);
            });
            clientReq.on('end', () => onDone(Buffer.concat(chunks).toString()));
        }

        const server = createServer((clientReq, clientRes) => {
            const urlPath = clientReq.url.split('?')[0];

            if (urlPath.startsWith('/_proxy/')) {
                if (urlPath === '/_proxy/status') {
                    const b = activeBackend();
                    clientRes.writeHead(200, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({
                        mode: state.mode,
                        backend_type: state.backendType,
                        window: state.window,
                        tool_mode: getToolMode(state.mode),
                        model: state.model || b?.model,
                        uptime: Math.round((Date.now() - t0Global) / 1000),
                        requests: reqCount,
                    }));
                    return;
                }
                if (urlPath === '/_proxy/cost') {
                    clientRes.writeHead(200, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify(getCostSummary()));
                    return;
                }
                if (urlPath === '/_proxy/mode' && clientReq.method === 'POST') {
                    if (isForbiddenOrigin(clientReq)) {
                        clientRes.writeHead(403, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify({ error: 'Forbidden' }));
                        return;
                    }
                    readControlBody(clientReq, 1024, (body) => {
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
                        console.error(`[MODEL-PROXY] Mode switched: ${result.previous} → ${result.mode}`);
                        clientRes.writeHead(200, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify(result));
                    });
                    return;
                }
                if (urlPath === '/_proxy/window' && clientReq.method === 'POST') {
                    if (isForbiddenOrigin(clientReq)) {
                        clientRes.writeHead(403, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify({ error: 'Forbidden' }));
                        return;
                    }
                    readControlBody(clientReq, 1024, (body) => {
                        const m = body.match(/window=([0-9a-z]+)/);
                        if (!m) {
                            clientRes.writeHead(400, { 'content-type': 'application/json' });
                            clientRes.end(JSON.stringify({ error: 'Missing window= in body (15m|1h|6h|24h|48h)' }));
                            return;
                        }
                        const result = setWindow(m[1]);
                        console.error(`[MODEL-PROXY] Window: ${result.previous} → ${result.window}`);
                        clientRes.writeHead(200, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify(result));
                    });
                    return;
                }
                if ((urlPath === '/_proxy/mode' || urlPath === '/_proxy/window') && clientReq.method !== 'POST') {
                    clientRes.writeHead(405, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({ error: 'Use POST' }));
                    return;
                }
                clientRes.writeHead(404, { 'content-type': 'application/json' });
                clientRes.end(JSON.stringify({ error: 'Not found' }));
                return;
            }

            const isAnthropicMode = state.mode === 'anthropic';
            const isModelCall = !isAnthropicMode && MODEL_PATHS.includes(urlPath);
            const isBgBackend = state.backendType === 'responses-bg';
            const dest = isModelCall && !isBgBackend ? state.target : new URL(ANTHROPIC_FALLBACK);

            let fullPath;
            if (isModelCall && !isBgBackend) {
                const base = state.target.pathname.replace(/\/$/, '');
                let overlap = '';
                for (let i = 1; i <= Math.min(base.length, urlPath.length); i++) {
                    if (base.endsWith(urlPath.substring(0, i))) overlap = urlPath.substring(0, i);
                }
                fullPath = overlap ? base + urlPath.substring(overlap.length) : base + urlPath;
            } else {
                fullPath = clientReq.url;
            }

            const reqId = ++reqCount;
            const t0 = Date.now();

            const chunks = [];
            clientReq.on('data', c => chunks.push(c));
            clientReq.on('end', () => {
                const body = Buffer.concat(chunks);

                if (isBgBackend && isModelCall && clientReq.method === 'POST') {
                    handleResponsesBg(clientReq, clientRes, body, reqId, t0);
                    return;
                }

                let bodyMut = body;
                if (isModelCall) {
                    console.error(`[MODEL-PROXY] #${reqId} → ${dest.hostname}${fullPath}`);
                }

                const headers = { ...clientReq.headers, host: dest.host };
                delete headers['content-length'];

                if (isModelCall) {
                    delete headers['authorization'];
                    delete headers['x-api-key'];
                    if (state.useBearer) {
                        headers['authorization'] = `Bearer ${state.apiKey}`;
                    } else {
                        headers['x-api-key'] = state.apiKey;
                    }
                }

                if (isModelCall && MODEL_REMAP[state.mode]) {
                    try {
                        const parsed = JSON.parse(bodyMut);
                        const mapped = MODEL_REMAP[state.mode][parsed.model];
                        if (mapped) {
                            console.error(`[MODEL-PROXY] #${reqId} model remap: ${parsed.model} → ${mapped}`);
                            parsed.model = mapped;
                            bodyMut = Buffer.from(JSON.stringify(parsed));
                        }
                    } catch { /* pass through */ }
                }

                if (isAnthropicMode && MODEL_PATHS.includes(urlPath)) {
                    try {
                        const parsed = JSON.parse(bodyMut);
                        if (state.hadNonAnthropicSession) {
                            stripAllThinkingBlocks(parsed);
                        } else {
                            stripUnsignedThinkingBlocks(parsed);
                        }
                        bodyMut = Buffer.from(JSON.stringify(parsed));
                    } catch { /* pass through */ }
                }
                if (isModelCall && !isBgBackend) {
                    try {
                        const parsed = JSON.parse(bodyMut);
                        stripAllThinkingBlocks(parsed);
                        bodyMut = Buffer.from(JSON.stringify(parsed));
                    } catch { /* pass through */ }
                }

                const opts = {
                    hostname: dest.hostname,
                    port: dest.port || 443,
                    path: fullPath,
                    method: clientReq.method,
                    headers: { ...headers, 'content-length': bodyMut.length },
                    timeout: REQUEST_TIMEOUT_MS,
                };

                const proxyReq = httpsRequest(opts, (proxyRes) => {
                    if (isModelCall) {
                        const ttfb = Date.now() - t0;
                        console.error(`[MODEL-PROXY] #${reqId} TTFB ${ttfb}ms (status ${proxyRes.statusCode})`);
                    }

                    const ct = proxyRes.headers['content-type'] || '';
                    const isSSE = ct.includes('text/event-stream');

                    if (isModelCall && isSSE) {
                        clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
                        const norm = new UsageNormalizer((inp, out) => recordUsage(state.mode, inp, out));
                        proxyRes.pipe(norm).pipe(clientRes);
                        proxyRes.on('end', () => {
                            console.error(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${norm._inputTokens}in/${norm._outputTokens}out)`);
                        });
                    } else if (isModelCall && ct.includes('application/json')) {
                        const respChunks = [];
                        proxyRes.on('data', c => respChunks.push(c));
                        proxyRes.on('end', () => {
                            const raw = Buffer.concat(respChunks);
                            const fixed = normalizeJsonBody(raw);
                            try {
                                const j = JSON.parse(fixed);
                                if (j.usage) recordUsage(state.mode, j.usage.input_tokens, j.usage.output_tokens);
                            } catch {}
                            const outHeaders = { ...proxyRes.headers, 'content-length': fixed.length };
                            clientRes.writeHead(proxyRes.statusCode, outHeaders);
                            clientRes.end(fixed);
                            console.error(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (json, ${fixed.length}b)`);
                        });
                    } else {
                        clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
                        proxyRes.pipe(clientRes);
                        if (isModelCall) {
                            proxyRes.on('end', () => {
                                console.error(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
                            });
                        }
                    }
                });

                proxyReq.on('timeout', () => {
                    console.error(`[MODEL-PROXY] #${reqId} TIMEOUT after ${REQUEST_TIMEOUT_MS / 1000}s`);
                    proxyReq.destroy(new Error('Request timeout'));
                });

                proxyReq.on('error', (err) => {
                    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
                    console.error(`[MODEL-PROXY] #${reqId} ERROR after ${elapsed}s: ${err.message}`);
                    if (!clientRes.headersSent) {
                        clientRes.writeHead(502, { 'content-type': 'application/json' });
                    }
                    clientRes.end(JSON.stringify({ error: { message: 'Upstream connection error' } }));
                });

                proxyReq.end(bodyMut);
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
                console.error(`[MODEL-PROXY] Listening on 127.0.0.1:${actualPort} (mode: ${state.mode}, type: ${state.backendType})`);
                resolve({ port: actualPort, close: () => server.close(), switchMode });
            });
        }

        tryListen(startPort);
    });
}
