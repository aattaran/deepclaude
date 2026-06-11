#!/usr/bin/env node
import { startModelProxy } from './model-proxy.js';

const SFERENCE_MODEL = process.env.SFERENCE_MODEL || 'moonshotai/Kimi-K2.6';
const SFERENCE_WINDOW = process.env.SFERENCE_COMPLETION_WINDOW || '1h';
const SFERENCE_URL = process.env.SFERENCE_BASE_URL || 'https://api.sference.com';

const BACKEND_DEFS = {
    deepseek: {
        url: 'https://api.deepseek.com/anthropic',
        keyEnv: 'DEEPSEEK_API_KEY',
        type: 'anthropic-compatible',
    },
    openrouter: {
        url: 'https://openrouter.ai/api/v1',
        keyEnv: 'OPENROUTER_API_KEY',
        type: 'anthropic-compatible',
    },
    fireworks: {
        url: 'https://api.fireworks.ai/inference/v1',
        keyEnv: 'FIREWORKS_API_KEY',
        type: 'anthropic-compatible',
    },
    sference: {
        url: SFERENCE_URL,
        keyEnv: 'SFERENCE_API_KEY',
        type: 'responses-bg',
        model: SFERENCE_MODEL,
        window: SFERENCE_WINDOW,
        authStyle: 'x-api-key',
    },
};

if (process.env.BG_PROVIDER_URL && process.env.BG_PROVIDER_API_KEY && process.env.BG_PROVIDER_MODEL) {
    BACKEND_DEFS.bg = {
        url: process.env.BG_PROVIDER_URL,
        keyEnv: 'BG_PROVIDER_API_KEY',
        type: 'responses-bg',
        model: process.env.BG_PROVIDER_MODEL,
        window: process.env.BG_PROVIDER_WINDOW || '1h',
        authStyle: process.env.BG_PROVIDER_AUTH_STYLE || 'x-api-key',
    };
}

function buildBackendEntry(name, def, keyOverride) {
    const key = keyOverride ?? process.env[def.keyEnv];
    if (!key && def.type === 'responses-bg') return null;
    return {
        url: def.url,
        apiKey: key || null,
        type: def.type || 'anthropic-compatible',
        model: def.model,
        window: def.window,
        authStyle: def.authStyle,
    };
}

function buildAllBackends() {
    const backends = {};
    for (const [name, def] of Object.entries(BACKEND_DEFS)) {
        const entry = buildBackendEntry(name, def);
        if (entry?.apiKey || def.type === 'anthropic-compatible') {
            backends[name] = entry;
        }
    }
    return backends;
}

function parseArgs(argv) {
    const args = { window: process.env.SFERENCE_COMPLETION_WINDOW || '1h' };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--window' && argv[i + 1]) {
            args.window = argv[++i];
        } else if (argv[i] === '--mode' && argv[i + 1]) {
            args.mode = argv[++i];
        } else if (argv[i] === '--port' && argv[i + 1]) {
            args.port = parseInt(argv[++i], 10);
        } else if (!args.targetUrl && argv[i].startsWith('http')) {
            args.targetUrl = argv[i];
        } else if (!args.apiKey && args.targetUrl && !argv[i].startsWith('-')) {
            args.apiKey = argv[i];
        } else if (!args.backendName && !argv[i].startsWith('-')) {
            args.backendName = argv[i];
        }
    }
    return args;
}

const cli = parseArgs(process.argv.slice(2));
const targetUrl = cli.targetUrl || process.env.CHEAPCLAUDE_TARGET_URL;
const apiKey = cli.apiKey || process.env.CHEAPCLAUDE_API_KEY;
const backendName = cli.backendName || process.env.CHEAPCLAUDE_DEFAULT_MODE;

if (targetUrl && apiKey) {
    const backends = buildAllBackends();
    const mode = backendName || process.env.CHEAPCLAUDE_DEFAULT_BACKEND;

    let initialBackend = null;
    if (mode && BACKEND_DEFS[mode]) {
        const def = BACKEND_DEFS[mode];
        initialBackend = buildBackendEntry(mode, def, apiKey);
        if (initialBackend && cli.window) initialBackend.window = cli.window;
        if (initialBackend?.apiKey) backends[mode] = initialBackend;
    } else if (backendName === 'responses-bg' || process.env.CHEAPCLAUDE_BACKEND_TYPE === 'responses-bg') {
        initialBackend = {
            url: targetUrl,
            apiKey,
            type: 'responses-bg',
            model: process.env.SFERENCE_MODEL || process.env.BG_PROVIDER_MODEL,
            window: cli.window,
            authStyle: 'x-api-key',
        };
    }

    const hasBackends = Object.keys(backends).length > 0;
    const defaultMode = mode && backends[mode] ? mode : (initialBackend?.type === 'responses-bg' ? mode : undefined);

    const { port } = await startModelProxy({
        targetUrl,
        apiKey,
        backends: hasBackends ? backends : undefined,
        defaultMode,
        initialBackend: initialBackend || undefined,
    });
    console.log(port);
} else {
    const backends = buildAllBackends();
    for (const [name, def] of Object.entries(BACKEND_DEFS)) {
        if (!backends[name]) {
            backends[name] = buildBackendEntry(name, def) || { url: def.url, apiKey: null, type: def.type };
        }
    }

    const defaultMode = cli.mode || 'anthropic';
    const port = cli.port || 3200;

    if (cli.window && backends.sference) backends.sference.window = cli.window;

    const fallback = backends.sference?.apiKey ? backends.sference : backends.deepseek;
    const fallbackUrl = fallback?.url || 'https://api.deepseek.com/anthropic';
    const fallbackKey = fallback?.apiKey || 'unused';

    const proxy = await startModelProxy({
        targetUrl: fallbackUrl,
        apiKey: fallbackKey,
        startPort: port,
        backends,
        defaultMode,
    });

    console.log(`Proxy on :${proxy.port} (mode: ${defaultMode})`);
    console.log(`Switch: curl -sX POST http://127.0.0.1:${proxy.port}/_proxy/mode -d backend=sference`);
    console.log(`Window: curl -sX POST http://127.0.0.1:${proxy.port}/_proxy/window -d window=1h`);
    console.log(`Status: curl -s http://127.0.0.1:${proxy.port}/_proxy/status`);
}
