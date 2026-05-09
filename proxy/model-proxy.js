import { createServer } from 'http';
import { request as httpsRequest } from 'https';
import { URL } from 'url';
import { Transform } from 'stream';

const ANTHROPIC_FALLBACK = 'https://api.anthropic.com';
const MODEL_PATHS = ['/v1/messages'];
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000; // 5 min per request

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

const PRICING_PER_M = {
    deepseek:   { input: 0.44,  output: 0.87 },
    openrouter: { input: 0.44,  output: 0.87 },
    fireworks:  { input: 1.74,  output: 3.48 },
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

// ─────────────────────────────────────────────────────────────────────────────
// PER-PATH REQUEST PROCESSORS
//
// Each function owns the full request-side transformation for its path. There
// is NO shared "if (mode === X)" sanitization logic outside these functions —
// the dispatcher picks one and runs it. Adding a new backend means adding a
// new processor + wiring it into the dispatcher; existing paths stay untouched.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * PATH A — Anthropic (state.mode === 'anthropic')
 *
 * Forwarding to api.anthropic.com.
 *  • If the conversation history contains turns from a non-Anthropic backend
 *    (state.hadNonAnthropicSession), strip ALL thinking blocks. Foreign
 *    backends generate signed-but-invalid blocks that pass the unsigned
 *    filter and trip Anthropic 400s.
 *  • Otherwise, strip only UNSIGNED thinking blocks (Anthropic's normal
 *    requirement — its own signed blocks are valid and must be passed back).
 *  • No model remap.
 */
function processAnthropicRequest(body, state, reqId) {
    try {
        const parsed = JSON.parse(body);
        if (state.hadNonAnthropicSession) {
            stripAllThinkingBlocks(parsed);
        } else {
            stripUnsignedThinkingBlocks(parsed);
        }
        return Buffer.from(JSON.stringify(parsed));
    } catch {
        // Body wasn't JSON — pass through unchanged.
        return body;
    }
}

/**
 * PATH B — DeepSeek (state.mode === 'deepseek')
 *
 * Forwarding to DeepSeek's Anthropic-compat endpoint.
 *  • Apply MODEL_REMAP['deepseek'] to the `model` field.
 *  • DO NOT strip thinking blocks. DeepSeek's endpoint REQUIRES prior
 *    thinking blocks to be passed back when the request includes
 *    thinking-mode parameters; absence returns 400 with
 *    "content[].thinking ... must be passed back".
 *  • The JSONL Claude Code persists will reflect real DeepSeek content
 *    (foreign-signed thinking, deepseek-v4-pro model name). That is by
 *    design — a DeepSeek session is a DeepSeek session.
 */
function processDeepseekRequest(body, state, reqId) {
    try {
        const parsed = JSON.parse(body);
        const remap = MODEL_REMAP.deepseek;
        if (remap && parsed.model && remap[parsed.model]) {
            const mapped = remap[parsed.model];
            console.log(`[MODEL-PROXY] #${reqId} model remap: ${parsed.model} → ${mapped}`);
            parsed.model = mapped;
            return Buffer.from(JSON.stringify(parsed));
        }
        return body;
    } catch {
        // Body wasn't JSON — pass through unchanged.
        return body;
    }
}

/**
 * PATH C — Other non-Anthropic backends (openrouter, fireworks, _single)
 *
 * Forwarding to the configured backend.
 *  • Apply MODEL_REMAP[mode] if defined for this mode.
 *  • Strip ALL thinking blocks. These backends are assumed to reject
 *    foreign-signed thinking blocks (the original deepclaude assumption).
 *    We have NOT validated whether any of them require continuity like
 *    DeepSeek does. If a future backend turns out to need pass-through,
 *    promote it to its own dedicated path (do not weaken this one).
 */
function processOtherBackendRequest(body, state, reqId) {
    try {
        const parsed = JSON.parse(body);
        let changed = false;

        const remap = MODEL_REMAP[state.mode];
        if (remap && parsed.model && remap[parsed.model]) {
            const mapped = remap[parsed.model];
            console.log(`[MODEL-PROXY] #${reqId} model remap: ${parsed.model} → ${mapped}`);
            parsed.model = mapped;
            changed = true;
        }

        // Strip foreign thinking blocks before forwarding.
        const before = JSON.stringify(parsed.messages || null);
        stripAllThinkingBlocks(parsed);
        if (JSON.stringify(parsed.messages || null) !== before) changed = true;

        return changed ? Buffer.from(JSON.stringify(parsed)) : body;
    } catch {
        // Body wasn't JSON — pass through unchanged.
        return body;
    }
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

        // If the proxy boots in anthropic mode AND ANTHROPIC_API_KEY is set in
        // the env, populate apiKey from there so the auth-substitution path
        // works from the first request without needing a /_proxy/mode flip.
        const anthropicBootKey = (initialName === 'anthropic' && process.env.ANTHROPIC_API_KEY) || null;

        const state = {
            mode: initialName || '_single',
            target: startBackend ? startBackend.target : initialTarget,
            apiKey: startBackend ? startBackend.apiKey : (anthropicBootKey || apiKey),
            // Anthropic uses x-api-key (NOT Bearer); only OpenRouter/Fireworks
            // use Bearer. So if booting in anthropic mode, useBearer is false.
            useBearer: startBackend ? startBackend.useBearer : (anthropicBootKey ? false : initialBearer),
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

        // Anthropic-side auth: optional. If ANTHROPIC_API_KEY is set in the
        // proxy's process env, the proxy will INJECT it as the auth header on
        // requests in anthropic mode (same substitution as for non-anthropic
        // backends, completing the path-isolation story for auth too). If
        // unset, the proxy passes the client's existing auth header through
        // unchanged (transparent passthrough — useful for OAuth bridge).
        const anthropicApiKey = process.env.ANTHROPIC_API_KEY || null;

        function switchMode(name) {
            if (name === 'anthropic') {
                const prev = state.mode;
                state.mode = 'anthropic';
                state.target = new URL(ANTHROPIC_FALLBACK);
                state.apiKey = anthropicApiKey;
                // Anthropic auth uses the x-api-key header (NOT Bearer).
                // Bearer is reserved for OAuth bridge sessions, which are
                // never set up via ANTHROPIC_API_KEY env var.
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

            // ─────────────────────────────────────────────────────────────
            // Decide ONCE which path this request takes. After this point,
            // no shared sanitization branches exist — each path is handled
            // by its own processor function.
            // ─────────────────────────────────────────────────────────────
            const isAnthropicMode = state.mode === 'anthropic';
            // A "model call" is a /v1/messages request to a NON-Anthropic
            // backend (i.e. PATH B or PATH C). Anthropic-mode /v1/messages
            // calls are still proxied to api.anthropic.com but flagged
            // separately because they use a different upstream (the
            // ANTHROPIC_FALLBACK target, not state.target) and skip the
            // backend auth header / model remap entirely.
            const isModelCall = !isAnthropicMode && MODEL_PATHS.includes(urlPath);
            const isAnthropicModelCall = isAnthropicMode && MODEL_PATHS.includes(urlPath);
            const dest = isModelCall ? state.target : new URL(ANTHROPIC_FALLBACK);

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

            const reqId = ++reqCount;
            const t0 = Date.now();

            if (isModelCall) {
                console.log(`[MODEL-PROXY] #${reqId} → ${dest.hostname}${fullPath}`);
            }

            const headers = { ...clientReq.headers, host: dest.host };
            delete headers['content-length'];

            // Auth substitution per active path. If state.apiKey is set
            // (because the user configured DEEPSEEK_API_KEY / ANTHROPIC_API_KEY
            // / OPENROUTER_API_KEY / FIREWORKS_API_KEY in the proxy's env),
            // strip the client's auth headers and inject the path-appropriate
            // key. This makes mode switching seamless: Claude Code's
            // ANTHROPIC_AUTH_TOKEN can be any value (even a wrong one); the
            // proxy substitutes the right key per mode.
            //
            // If state.apiKey is null (typically anthropic mode without
            // ANTHROPIC_API_KEY set), the client's auth header is forwarded
            // unchanged — useful for OAuth bridge mode where Claude Code
            // already holds a valid Anthropic session token.
            if (MODEL_PATHS.includes(urlPath) && state.apiKey) {
                delete headers['authorization'];
                delete headers['x-api-key'];
                if (state.useBearer) {
                    headers['authorization'] = `Bearer ${state.apiKey}`;
                } else {
                    headers['x-api-key'] = state.apiKey;
                }
            }

            const chunks = [];
            clientReq.on('data', c => chunks.push(c));
            clientReq.on('end', () => {
                let body = Buffer.concat(chunks);

                // ─────────────────────────────────────────────────────────
                // Dispatch to ONE per-path request processor. Each processor
                // owns all sanitization for its path — no shared logic.
                //
                //   PATH A — Anthropic         : strip thinking (signed?
                //                                depends on prior session),
                //                                no model remap.
                //   PATH B — DeepSeek          : remap model only,
                //                                NEVER strip thinking
                //                                (endpoint requires it).
                //   PATH C — Other non-Anthr.  : remap model + strip ALL
                //                                thinking blocks.
                //   (non-/v1/messages traffic in non-Anthropic mode and all
                //    non-/v1/messages traffic in Anthropic mode bypasses
                //    request-body processing entirely — pure passthrough.)
                // ─────────────────────────────────────────────────────────
                if (isAnthropicModelCall) {
                    body = processAnthropicRequest(body, state, reqId);
                } else if (isModelCall && state.mode === 'deepseek') {
                    body = processDeepseekRequest(body, state, reqId);
                } else if (isModelCall) {
                    body = processOtherBackendRequest(body, state, reqId);
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
                    if (isModelCall) {
                        const ttfb = Date.now() - t0;
                        console.log(`[MODEL-PROXY] #${reqId} TTFB ${ttfb}ms (status ${proxyRes.statusCode})`);
                    }

                    // Backend can drop the connection mid-stream (own timeout,
                    // restart, network glitch). Without this handler, Node's
                    // EventEmitter throws 'Unhandled error' and crashes the proxy.
                    proxyRes.on('error', (err) => {
                        console.error(`[MODEL-PROXY] #${reqId} proxyRes error: ${err.message}`);
                        if (!clientRes.destroyed) clientRes.destroy();
                    });

                    const ct = proxyRes.headers['content-type'] || '';
                    const isSSE = ct.includes('text/event-stream');

                    // Response path: pass-through for ALL paths. UsageNormalizer
                    // for SSE (Claude-Code crash guard); normalizeJsonBody for
                    // non-streaming JSON (same guard). No model un-mapping. No
                    // thinking-block stripping. The JSONL Claude Code persists
                    // for a backend session reflects that backend's actual
                    // output — that's by design.
                    if (isModelCall && isSSE) {
                        clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
                        const norm = new UsageNormalizer((inp, out) => recordUsage(state.mode, inp, out));
                        proxyRes.pipe(norm).pipe(clientRes);
                        proxyRes.on('end', () => {
                            console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (${norm._inputTokens}in/${norm._outputTokens}out)`);
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
                            console.log(`[MODEL-PROXY] #${reqId} done in ${((Date.now() - t0) / 1000).toFixed(1)}s (json, ${fixed.length}b)`);
                        });
                    } else {
                        // Non-model or unknown content-type: pass through
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
                    console.error(`[MODEL-PROXY] #${reqId} TIMEOUT after ${REQUEST_TIMEOUT_MS / 1000}s`);
                    proxyReq.destroy(new Error('Request timeout'));
                });

                proxyReq.on('error', (err) => {
                    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
                    console.error(`[MODEL-PROXY] #${reqId} ERROR after ${elapsed}s: ${err.message}`);
                    if (!clientRes.headersSent) {
                        // Headers not sent — safe to write a clean 502 JSON body.
                        clientRes.writeHead(502, { 'content-type': 'application/json' });
                        clientRes.end(JSON.stringify({ error: { message: 'Upstream connection error' } }));
                    } else if (!clientRes.destroyed) {
                        // Headers already sent — likely SSE streaming. Appending
                        // JSON would corrupt the stream and crash Claude Code's
                        // SSE parser. Close the socket cleanly instead.
                        clientRes.destroy();
                    }
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
