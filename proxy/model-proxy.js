import { createServer } from 'http';
import { request as httpsRequest } from 'https';
import { URL } from 'url';
import { Transform } from 'stream';

const ANTHROPIC_FALLBACK = 'https://api.anthropic.com';
const MODEL_PATHS = ['/v1/messages'];
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000; // 5 min per request
const IMAGE_FALLBACK_ENABLED = (process.env.DEEPCLAUDE_IMAGE_FALLBACK || 'anthropic') !== 'off';

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

// Many-to-one collisions in MODEL_REMAP collapse last-write-wins.
const REVERSE_MODEL_REMAP = {};
for (const [backend, table] of Object.entries(MODEL_REMAP)) {
    REVERSE_MODEL_REMAP[backend] = {};
    for (const [claudeName, backendName] of Object.entries(table)) {
        REVERSE_MODEL_REMAP[backend][backendName] = claudeName;
    }
}

const PRICING_PER_M = {
    deepseek:      { input: 0.44,  output: 0.87 },
    openrouter:    { input: 0.44,  output: 0.87 },
    fireworks:     { input: 1.74,  output: 3.48 },
    anthropic:     { input: 3.00,  output: 15.00 },
    // Max OAuth burns subscription quota, not per-token cost.
    anthropic_max: { input: 0,     output: 0 },
    _single:       { input: 0.44,  output: 0.87 },
};

/**
 * Transform stream that intercepts SSE events and injects missing `usage`
 * fields. DeepSeek/OpenRouter may omit `usage` in message_start or
 * message_delta, which crashes Claude Code ("$.input_tokens" is undefined).
 *
 * Optionally rewrites `message.model` in `message_start` events — used on
 * the image-fallback path so Claude Code sees the backend model name on
 * the response side even though Anthropic served the claude-* name on
 * the wire.
 */
class UsageNormalizer extends Transform {
    constructor(onUsage, { modelRewrite } = {}) {
        super();
        this._buf = '';
        this._onUsage = onUsage;
        this._inputTokens = 0;
        this._outputTokens = 0;
        this._modelRewrite = modelRewrite || null;
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
                if (this._modelRewrite && d.message.model === this._modelRewrite.from) {
                    d.message.model = this._modelRewrite.to;
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

// Recurses into tool_result.content[] because Claude Code's Read tool
// wraps a returned PNG there rather than at the top of the message.
function containsImageBlock(messages) {
    if (!Array.isArray(messages)) return false;
    const blockHasImage = (block) => {
        if (!block) return false;
        if (block.type === 'image') return true;
        if (block.type === 'tool_result' && Array.isArray(block.content)) {
            return block.content.some(blockHasImage);
        }
        return false;
    };
    for (const msg of messages) {
        if (!Array.isArray(msg.content)) continue;
        if (msg.content.some(blockHasImage)) return true;
    }
    return false;
}

// Replaces image blocks with a text placeholder, recursing into
// tool_result.content[]. Used on non-Anthropic routes so a single image
// turn earlier in history doesn't pin the rest of the conversation to
// Anthropic — which would silently spend Max quota while the TUI still
// shows the cheap backend. Returns the count of images replaced.
function stripImagesFromMessages(messages) {
    if (!Array.isArray(messages)) return 0;
    let count = 0;
    const strip = (block) => {
        if (!block) return block;
        if (block.type === 'image') {
            count++;
            return { type: 'text', text: '[image omitted]' };
        }
        if (block.type === 'tool_result' && Array.isArray(block.content)) {
            return { ...block, content: block.content.map(strip) };
        }
        return block;
    };
    for (const msg of messages) {
        if (!Array.isArray(msg.content)) continue;
        msg.content = msg.content.map(strip);
    }
    return count;
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
                return { mode: 'anthropic', previous: prev };
            }
            const b = allBackends[name];
            if (!b) return { error: `Unknown backend: ${name}. Valid: anthropic, ${Object.keys(allBackends).join(', ')}` };
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
                    clientRes.writeHead(200, { 'content-type': 'application/json' });
                    clientRes.end(JSON.stringify({
                        mode: state.mode,
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

            // Routing is deferred to the end-of-body handler so the body
            // can be inspected for image content blocks before the
            // dest/headers are decided.
            const chunks = [];
            clientReq.on('data', c => chunks.push(c));
            clientReq.on('end', () => {
                let body = Buffer.concat(chunks);
                const isMessagesPath = MODEL_PATHS.includes(urlPath);

                let parsed = null;
                if (isMessagesPath) {
                    try { parsed = JSON.parse(body); } catch {}
                }

                // Only the LATEST message triggers the Anthropic reroute
                // (a fresh attachment, or a Read tool_result that just came
                // back). Stale images in older history are stripped below
                // so the conversation can return to the backend on text-only
                // follow-ups instead of silently pinning to Max quota.
                const lastMsg = parsed?.messages?.[parsed.messages.length - 1];
                const forceAnthropicForImage = (
                    IMAGE_FALLBACK_ENABLED &&
                    state.mode !== 'anthropic' &&
                    !!lastMsg &&
                    containsImageBlock([lastMsg])
                );

                const isAnthropicMode = state.mode === 'anthropic' || forceAnthropicForImage;
                const isModelCall = !isAnthropicMode && isMessagesPath;
                const trackUsage = isModelCall || forceAnthropicForImage;
                const dest = isModelCall ? state.target : new URL(ANTHROPIC_FALLBACK);
                const effectiveMode = forceAnthropicForImage ? 'anthropic_max' : state.mode;

                // Two-sided swap: outbound to Anthropic, reversed on the
                // response so Claude Code never sees a non-backend name.
                let imageRewrite = null;
                if (forceAnthropicForImage && parsed?.model) {
                    const canonical = REVERSE_MODEL_REMAP[state.mode]?.[parsed.model];
                    if (canonical) {
                        imageRewrite = { backend: parsed.model, canonical };
                    }
                }

                // Build upstream path. target.pathname may overlap with
                // clientReq.url (e.g. OpenRouter /api/v1 + /v1/messages).
                // Strip the shared prefix to avoid /api/v1/v1/messages.
                let fullPath;
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

                if (trackUsage) {
                    const tag = forceAnthropicForImage
                        ? ` [image→anthropic${imageRewrite ? `, ${imageRewrite.backend}→${imageRewrite.canonical}` : ''}]`
                        : '';
                    console.log(`[MODEL-PROXY] #${reqId} → ${dest.hostname}${fullPath}${tag}`);
                }

                const headers = { ...clientReq.headers, host: dest.host };
                delete headers['content-length'];
                // Force plain bytes upstream — the proxy mutates response
                // bodies (UsageNormalizer toString'es bytes, normalizeJsonBody
                // reparses) and would otherwise emit a content-encoding: gzip
                // header followed by non-gzip bytes, breaking the client with
                // "Decompression error: ZlibError".
                delete headers['accept-encoding'];

                if (isModelCall) {
                    delete headers['authorization'];
                    delete headers['x-api-key'];
                    if (state.useBearer) {
                        headers['authorization'] = `Bearer ${state.apiKey}`;
                    } else {
                        headers['x-api-key'] = state.apiKey;
                    }
                }
                // forceAnthropicForImage path leaves the client auth header
                // intact — whatever Claude Code is carrying authenticates
                // at Anthropic.

                if (parsed) {
                    // clear_thinking_* requires thinking enabled; Anthropic
                    // 400s on the mismatch.
                    if (forceAnthropicForImage) {
                        delete parsed.thinking;
                        delete parsed.context_management;
                        if (imageRewrite) {
                            parsed.model = imageRewrite.canonical;
                        }
                    }

                    // Strip stale images from history on non-Anthropic
                    // routes. Without this, every text follow-up on an
                    // image-bearing conversation would still detect the
                    // image and re-route to Anthropic — burning Max quota
                    // while the TUI advertises the cheap backend.
                    if (isModelCall) {
                        const stripped = stripImagesFromMessages(parsed.messages);
                        if (stripped > 0) {
                            console.log(`[MODEL-PROXY] #${reqId} stripped ${stripped} stale image block${stripped === 1 ? '' : 's'} from history`);
                        }
                    }

                    if (isModelCall && MODEL_REMAP[state.mode]) {
                        const mapped = MODEL_REMAP[state.mode][parsed.model];
                        if (mapped) {
                            console.log(`[MODEL-PROXY] #${reqId} model remap: ${parsed.model} → ${mapped}`);
                            parsed.model = mapped;
                        }
                    }

                    // Foreign backends emit signed-but-invalid thinking
                    // blocks; strip ALL when crossing into or out of one.
                    // Pure Anthropic sessions strip only unsigned, preserving
                    // valid signed blocks for continuity.
                    if (isAnthropicMode) {
                        if (state.hadNonAnthropicSession || forceAnthropicForImage) {
                            stripAllThinkingBlocks(parsed);
                        } else {
                            stripUnsignedThinkingBlocks(parsed);
                        }
                    } else if (isModelCall) {
                        stripAllThinkingBlocks(parsed);
                    }

                    body = Buffer.from(JSON.stringify(parsed));
                }

                const opts = {
                    hostname: dest.hostname,
                    port: dest.port || 443,
                    path: fullPath,
                    method: clientReq.method,
                    headers: { ...headers, 'content-length': body.length },
                    timeout: REQUEST_TIMEOUT_MS,
                };

                const proxyReq = httpsRequest(opts, (proxyRes) => {
                    if (trackUsage) {
                        const ttfb = Date.now() - t0;
                        console.log(`[MODEL-PROXY] #${reqId} TTFB ${ttfb}ms (status ${proxyRes.statusCode})`);
                    }

                    const ct = proxyRes.headers['content-type'] || '';
                    const isSSE = ct.includes('text/event-stream');

                    if (trackUsage && isSSE) {
                        // Mirror of the accept-encoding strip on the request
                        // side — the response body is being mutated, so the
                        // forwarded headers must not advertise gzip.
                        const { 'content-encoding': _ce1, ...sseHeaders } = proxyRes.headers;
                        clientRes.writeHead(proxyRes.statusCode, sseHeaders);
                        const modelRewrite = imageRewrite
                            ? { from: imageRewrite.canonical, to: imageRewrite.backend }
                            : null;
                        const norm = new UsageNormalizer(
                            (inp, out) => recordUsage(effectiveMode, inp, out),
                            { modelRewrite },
                        );
                        proxyRes.pipe(norm).pipe(clientRes);
                        proxyRes.on('end', () => {
                            console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${norm._inputTokens}in/${norm._outputTokens}out)`);
                        });
                    } else if (trackUsage && ct.includes('application/json')) {
                        const respChunks = [];
                        proxyRes.on('data', c => respChunks.push(c));
                        proxyRes.on('end', () => {
                            const raw = Buffer.concat(respChunks);
                            let fixed = normalizeJsonBody(raw);
                            try {
                                const j = JSON.parse(fixed);
                                if (j.usage) recordUsage(effectiveMode, j.usage.input_tokens, j.usage.output_tokens);
                                if (imageRewrite && j.model === imageRewrite.canonical) {
                                    j.model = imageRewrite.backend;
                                    fixed = Buffer.from(JSON.stringify(j));
                                }
                            } catch {}
                            const { 'content-encoding': _ce2, ...jsonHeaders } = proxyRes.headers;
                            const outHeaders = { ...jsonHeaders, 'content-length': fixed.length };
                            clientRes.writeHead(proxyRes.statusCode, outHeaders);
                            clientRes.end(fixed);
                            console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (json, ${fixed.length}b)`);
                        });
                    } else {
                        // Non-model or unknown content-type: pass through unchanged.
                        clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
                        proxyRes.pipe(clientRes);
                        if (trackUsage) {
                            proxyRes.on('end', () => {
                                console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
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

                proxyReq.end(body);
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
