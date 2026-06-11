# Model Proxy

The proxy sits between Claude Code and inference backends. It handles two backend types:

| Type | Behavior |
|---|---|
| `anthropic-compatible` | Passthrough `/v1/messages` to DeepSeek, OpenRouter, Fireworks, etc. |
| `responses-bg` | Translate Anthropic `/v1/messages` → async `POST /v1/responses` with `background: true`, poll, synthesize SSE |

## Remote control split

When using `claude remote-control`, the bridge authentication must go to Anthropic while model API calls go to the configured backend:

```
claude remote-control
  ├── Bridge WebSocket → wss://bridge.claudeusercontent.com (Anthropic, hardcoded)
  └── Model API calls  → http://localhost:3200 (this proxy)
                            ├── /v1/messages → active backend
                            └── everything else → api.anthropic.com (passthrough)
```

## Background provider flow (`responses-bg`)

Used by sference and any provider implementing the OpenAI Responses API with `background: true`:

```
Claude Code  →  POST /v1/messages (Anthropic, stream=true)
Proxy        →  POST /v1/responses { background: true, metadata: { completion_window: "1h" } }
Proxy        ←  { id, status: "in_progress" }
Proxy        →  SSE pings every 25s (keeps Claude Code connection alive)
Proxy        →  poll GET /v1/responses/{id} until terminal
Proxy        ←  { status: "completed", output[], usage }
Proxy        →  synthesize Anthropic SSE (message_start … message_stop)
Client abort →  DELETE /v1/responses/{id}
```

Tool calling uses auto mode: native OpenAI function tools first; if the provider rejects them (400), falls back to prompt-based tool emulation for the session.

## Control endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/_proxy/status` | GET | Current mode, backend type, window, tool_mode, uptime |
| `/_proxy/mode` | POST | Switch backend: `backend=deepseek\|openrouter\|fireworks\|sference\|anthropic` |
| `/_proxy/window` | POST | Set completion window: `window=15m\|1h\|6h\|24h\|48h` |
| `/_proxy/cost` | GET | Token usage and cost tracking |

## Usage

```javascript
import { startModelProxy } from './model-proxy.js';

const proxy = await startModelProxy({
    targetUrl: 'https://api.sference.com',
    apiKey: process.env.SFERENCE_API_KEY,
    defaultMode: 'sference',
    initialBackend: {
        url: 'https://api.sference.com',
        apiKey: process.env.SFERENCE_API_KEY,
        type: 'responses-bg',
        model: 'moonshotai/Kimi-K2.6',
        window: '1h',
        authStyle: 'x-api-key',
    },
    backends: { /* all registered backends */ },
});

// Claude Code env:
// ANTHROPIC_BASE_URL=http://127.0.0.1:${proxy.port}
// (do NOT set ANTHROPIC_AUTH_TOKEN for remote control)
```

Or via CLI:

```bash
SFERENCE_API_KEY=sk_... node proxy/start-proxy.js https://api.sference.com sk_... sference --window 1h
```

## Why a proxy?

Claude Code's remote control uses two separate channels:
1. **Bridge** (WebSocket) — hardcoded Anthropic OAuth
2. **Model API** (HTTP to `ANTHROPIC_BASE_URL`) — configurable

Background providers add a third concern: Claude Code can't submit async jobs or poll. The proxy hides the entire Responses API lifecycle behind familiar Anthropic streaming.
