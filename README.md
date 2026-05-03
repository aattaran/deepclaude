# cheapclaude

Use Claude Code's autonomous agent loop with **DeepSeek V4 Pro**, **OpenRouter**, or any Anthropic-compatible backend. Same UX, 17x cheaper.

## What this does

Claude Code is the best autonomous coding agent — but it costs $200/month with usage caps. DeepSeek V4 Pro scores 96.4% on LiveCodeBench and costs $0.87/M output tokens.

**cheapclaude** swaps the brain while keeping the body:

```
Your terminal
  └── Claude Code CLI (tool loop, file editing, bash, git — unchanged)
        └── API calls → DeepSeek V4 Pro ($0.87/M) instead of Anthropic ($15/M)
```

Everything works: file reading, editing, bash execution, subagent spawning, autonomous multi-step coding loops. The only difference is which model thinks.

## Quick start (2 minutes)

### 1. Get a DeepSeek API key

Sign up at [platform.deepseek.com](https://platform.deepseek.com), add $5 credit, copy your API key.

### 2. Set environment variables

**Windows (PowerShell):**
```powershell
setx DEEPSEEK_API_KEY "sk-your-key-here"
```

**macOS/Linux:**
```bash
echo 'export DEEPSEEK_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 3. Install

**Windows:**
```powershell
# Copy the script to a directory in your PATH
Copy-Item cheapclaude.ps1 "$env:USERPROFILE\.local\bin\cheapclaude.ps1"

# Or add the repo directory to PATH
setx PATH "$env:PATH;C:\path\to\cheapclaude"
```

**macOS/Linux:**
```bash
chmod +x cheapclaude.sh
sudo ln -s "$(pwd)/cheapclaude.sh" /usr/local/bin/cheapclaude
```

### 4. Use it

```bash
cheapclaude                  # Launch Claude Code with DeepSeek V4 Pro
cheapclaude --status         # Show available backends and keys
cheapclaude --backend or     # Use OpenRouter (cheapest, $0.44/M input)
cheapclaude --backend fw     # Use Fireworks AI (fastest, US servers)
cheapclaude --backend anthropic  # Normal Claude Code (when you need Opus)
cheapclaude --cost           # Show pricing comparison
cheapclaude --benchmark      # Latency test across all providers
```

## How it works

Claude Code reads these environment variables to determine where to send API calls:

| Variable | What it does |
|---|---|
| `ANTHROPIC_BASE_URL` | API endpoint (default: api.anthropic.com) |
| `ANTHROPIC_AUTH_TOKEN` | API key for the backend |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Model name for Opus-tier tasks |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Model name for Sonnet-tier tasks |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Model name for Haiku-tier (subagents) |
| `CLAUDE_CODE_SUBAGENT_MODEL` | Model for spawned subagents |

**cheapclaude** sets these per-session (not permanently), launches Claude Code, then restores your original settings on exit.

## Supported backends

| Backend | Flag | Input/M | Output/M | Servers | Notes |
|---|---|---|---|---|---|
| **DeepSeek** (default) | `--backend ds` | $0.44 | $0.87 | China | Auto context caching (120x cheaper on repeat turns) |
| **OpenRouter** | `--backend or` | $0.44 | $0.87 | US | Cheapest, lowest latency from US/EU |
| **Fireworks AI** | `--backend fw` | $1.74 | $3.48 | US | Fastest inference |
| **Anthropic** | `--backend anthropic` | $3.00 | $15.00 | US | Original Claude Opus (for hard problems) |

### Setup per backend

**DeepSeek** (default — just needs `DEEPSEEK_API_KEY`):
```bash
setx DEEPSEEK_API_KEY "sk-..."           # Windows
export DEEPSEEK_API_KEY="sk-..."         # macOS/Linux
```

**OpenRouter** (optional):
```bash
setx OPENROUTER_API_KEY "sk-or-..."      # Windows
export OPENROUTER_API_KEY="sk-or-..."    # macOS/Linux
```

**Fireworks AI** (optional):
```bash
setx FIREWORKS_API_KEY "fw_..."          # Windows
export FIREWORKS_API_KEY="fw_..."        # macOS/Linux
```

## Cost comparison

| Usage level | Anthropic Max | cheapclaude (DeepSeek) | Savings |
|---|---|---|---|
| Light (10 days/mo) | $200/mo (capped) | ~$20/mo | 90% |
| Heavy (25 days/mo) | $200/mo (capped) | ~$50/mo | 75% |
| With auto loops | $200/mo (capped) | ~$80/mo | 60% |

DeepSeek's automatic context caching makes agent loops extremely cheap — after the first request, the system prompt and file context are cached at $0.004/M (vs $0.44/M uncached).

## What works and what doesn't

### Works
- File reading, writing, editing (Read/Write/Edit tools)
- Bash/PowerShell execution
- Glob and Grep search
- Multi-step autonomous tool loops
- Subagent spawning
- Git operations
- Project initialization (`/init`)
- Thinking mode (enabled by default)

### Doesn't work or degraded
| Feature | Reason |
|---|---|
| Image/vision input | DeepSeek's Anthropic endpoint doesn't support images |
| Parallel tool use | Disabled — tools execute one at a time |
| MCP server tools | Not supported through compatibility layer |
| Prompt caching savings | DeepSeek has its own caching (automatic), but Anthropic's `cache_control` is ignored |

### Intelligence difference
- **Routine tasks** (80% of work): DeepSeek V4 Pro is comparable to Claude Opus
- **Complex reasoning** (20%): Claude Opus is stronger — switch with `--backend anthropic`

## VS Code / Cursor integration

Add terminal profiles so you can launch cheapclaude from the IDE:

**Settings > JSON:**
```json
{
  "terminal.integrated.profiles.windows": {
    "DeepSeek Agent": {
      "path": "powershell.exe",
      "args": ["-ExecutionPolicy", "Bypass", "-NoExit", "-File", "C:\\path\\to\\cheapclaude.ps1"]
    }
  }
}
```

Or on macOS/Linux:
```json
{
  "terminal.integrated.profiles.linux": {
    "DeepSeek Agent": {
      "path": "/usr/local/bin/cheapclaude"
    }
  }
}
```

## Remote control (`--remote`)

Open a Claude Code session in any browser — with DeepSeek as the brain:

```bash
cheapclaude --remote                # Remote control + DeepSeek
cheapclaude --remote -b or          # Remote control + OpenRouter
cheapclaude --remote -b anthropic   # Remote control + Anthropic (normal)
```

This prints a `https://claude.ai/code/session_...` URL you can open on your phone, tablet, or any browser.

### How it works

Remote control needs Anthropic's bridge for the WebSocket connection, but model calls can go elsewhere. cheapclaude starts a local proxy that splits the traffic:

```
claude remote-control
  ├── Bridge WebSocket → wss://bridge.claudeusercontent.com (Anthropic, hardcoded)
  └── Model API calls  → http://localhost:3200 (proxy)
                            ├── /v1/messages → DeepSeek ($0.87/M)
                            └── everything else → Anthropic (passthrough)
```

### Prerequisites
- Must be logged into Claude Code: `claude auth login`
- Must have a claude.ai subscription (the bridge is Anthropic infrastructure)
- Node.js 18+ (for the proxy)

The proxy starts automatically and stops when the session ends. See [proxy/README.md](proxy/README.md) for technical details.

## License

MIT
