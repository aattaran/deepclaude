# deepclaude

Use Claude Code's autonomous agent loop with **DeepSeek V4 Pro**, **OpenRouter**, or any Anthropic-compatible backend. Same UX, 17x cheaper.

![Remote control running DeepSeek V4 Pro in the browser](screenshots/remote-control-deepseek.png)

## What this does

Claude Code is the best autonomous coding agent — but it costs $200/month with usage caps. DeepSeek V4 Pro scores 96.4% on LiveCodeBench and costs $0.87/M output tokens.

**deepclaude** swaps the brain while keeping the body:

```
Your terminal
  +-- Claude Code CLI (tool loop, file editing, bash, git - unchanged)
        +-- API calls -> DeepSeek V4 Pro ($0.87/M) instead of Anthropic ($15/M)
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
Copy-Item deepclaude.ps1 "$env:USERPROFILE\.local\bin\deepclaude.ps1"

# Or add the repo directory to PATH
setx PATH "$env:PATH;C:\path\to\deepclaude"
```

**macOS/Linux:**
```bash
chmod +x deepclaude.sh
sudo ln -s "$(pwd)/deepclaude.sh" /usr/local/bin/deepclaude
```

### 4. Use it

```bash
deepclaude                  # Launch Claude Code with DeepSeek V4 Pro
deepclaude --status         # Show available backends and keys
deepclaude --backend or     # Use OpenRouter (cheapest, $0.44/M input)
deepclaude --backend fw     # Use Fireworks AI (fastest, US servers)
deepclaude --backend anthropic  # Normal Claude Code (when you need Opus)
deepclaude --cost           # Show pricing comparison
deepclaude --benchmark      # Latency test across all providers
deepclaude --switch ds      # Switch backend mid-session (no restart)
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

**deepclaude** sets these per-session (not permanently), launches Claude Code, then restores your original settings on exit.

## Supported backends

| Backend | Flag | Input/M | Output/M | Servers | Notes |
|---|---|---|---|---|---|
| **DeepSeek** (default) | `--backend ds` | $0.44 | $0.87 | China | Auto context caching (120x cheaper on repeat turns) |
| **OpenRouter** | `--backend or` | $0.44 | $0.87 | US | Cheapest, lowest latency from US/EU |
| **Fireworks AI** | `--backend fw` | $1.74 | $3.48 | US | Fastest inference |
| **Anthropic** | `--backend anthropic` | $3.00 | $15.00 | US | Original Claude Opus (for hard problems) |

### Setup per backend

**DeepSeek** (default - just needs `DEEPSEEK_API_KEY`):
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

| Usage level | Anthropic Max | deepclaude (DeepSeek) | Savings |
|---|---|---|---|
| Light (10 days/mo) | $200/mo (capped) | ~$20/mo | 90% |
| Heavy (25 days/mo) | $200/mo (capped) | ~$50/mo | 75% |
| With auto loops | $200/mo (capped) | ~$80/mo | 60% |

DeepSeek's automatic context caching makes agent loops extremely cheap - after the first request, the system prompt and file context are cached at $0.004/M (vs $0.44/M uncached).

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
| Parallel tool use | Disabled - tools execute one at a time |
| MCP server tools | Not supported through compatibility layer |
| Prompt caching savings | DeepSeek has its own caching (automatic), but Anthropic's `cache_control` is ignored |

### Intelligence difference
- **Routine tasks** (80% of work): DeepSeek V4 Pro is comparable to Claude Opus
- **Complex reasoning** (20%): Claude Opus is stronger - switch with `--backend anthropic`

## Live switching (no restart)

Switch between Anthropic and DeepSeek **mid-session** - from inside Claude Code itself. No restart, no terminal commands. Just type a slash command.

**In Claude Code terminal:**

![/deepseek in Claude Code CLI](screenshots/terminal%20for%20terminal%20embed2.PNG)

**In Claude Code VS Code extension:**

![/deepseek in VS Code extension](screenshots/terminal%20for%20vscode%20embed2.PNG)

### How it works

The proxy runs on `localhost:3200` and intercepts all API calls. A control endpoint (`/_proxy/mode`) lets you switch the active backend instantly:

```
Claude Code -> localhost:3200 (proxy)
                 |
                 +-- /_proxy/mode POST -> switch backend
                 +-- /_proxy/status GET -> current backend + uptime
                 +-- /_proxy/cost GET -> token usage + cost savings
                 |
                 +-- /v1/messages -> active backend (DeepSeek/OpenRouter/Anthropic)
                 +-- everything else -> Anthropic (passthrough)
```

### Option 1: Slash commands (recommended)

Add these files to `~/.claude/commands/`:

**`deepseek.md`:**
```
Switch the model proxy to DeepSeek. Run this command silently and report the result:
curl -sX POST http://127.0.0.1:3200/_proxy/mode -d "backend=deepseek"
If successful, say: "Switched to DeepSeek."
```

**`anthropic.md`:**
```
Switch the model proxy back to Anthropic. Run this command silently and report the result:
curl -sX POST http://127.0.0.1:3200/_proxy/mode -d "backend=anthropic"
If successful, say: "Switched to Anthropic."
```

**`openrouter.md`:**
```
Switch the model proxy to OpenRouter. Run this command silently and report the result:
curl -sX POST http://127.0.0.1:3200/_proxy/mode -d "backend=openrouter"
If successful, say: "Switched to OpenRouter."
```

Then type `/deepseek`, `/anthropic`, or `/openrouter` in any Claude Code session to switch instantly.

### Option 2: CLI flag

```bash
deepclaude --switch deepseek    # or: ds, or, fw, anthropic
deepclaude -s anthropic
```

### Option 3: VS Code keyboard shortcuts

Add to `.vscode/tasks.json`:
```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Proxy: Switch to DeepSeek",
      "type": "shell",
      "command": "Invoke-RestMethod -Uri http://127.0.0.1:3200/_proxy/mode -Method Post -Body 'backend=deepseek'",
      "presentation": { "reveal": "always" },
      "problemMatcher": []
    },
    {
      "label": "Proxy: Switch to Anthropic",
      "type": "shell",
      "command": "Invoke-RestMethod -Uri http://127.0.0.1:3200/_proxy/mode -Method Post -Body 'backend=anthropic'",
      "presentation": { "reveal": "always" },
      "problemMatcher": []
    }
  ]
}
```

Then bind in `keybindings.json`:
```json
{ "key": "ctrl+alt+d", "command": "workbench.action.tasks.runTask", "args": "Proxy: Switch to DeepSeek" },
{ "key": "ctrl+alt+a", "command": "workbench.action.tasks.runTask", "args": "Proxy: Switch to Anthropic" }
```

### Cost tracking

The proxy tracks token usage and calculates savings vs Anthropic pricing:

```bash
curl -s http://127.0.0.1:3200/_proxy/cost
```

Returns:
```json
{
  "backends": {
    "deepseek": {
      "input_tokens": 125000,
      "output_tokens": 45000,
      "requests": 12,
      "cost": 0.0941,
      "anthropic_equivalent": 1.05
    }
  },
  "total_cost": 0.0941,
  "anthropic_equivalent": 1.05,
  "savings": 0.9559
}
```

## VS Code / Cursor integration

Add terminal profiles so you can launch deepclaude from the IDE:

**Settings > JSON:**
```json
{
  "terminal.integrated.profiles.windows": {
    "DeepSeek Agent": {
      "path": "powershell.exe",
      "args": ["-ExecutionPolicy", "Bypass", "-NoExit", "-File", "C:\\path\\to\\deepclaude.ps1"]
    }
  }
}
```

Or on macOS/Linux:
```json
{
  "terminal.integrated.profiles.linux": {
    "DeepSeek Agent": {
      "path": "/usr/local/bin/deepclaude"
    }
  }
}
```

## Remote control (`--remote`)

Open a Claude Code session in any browser - with DeepSeek as the brain:

```bash
deepclaude --remote                # Remote control + DeepSeek
deepclaude --remote -b or          # Remote control + OpenRouter
deepclaude --remote -b anthropic   # Remote control + Anthropic (normal)
```

This prints a `https://claude.ai/code/session_...` URL you can open on your phone, tablet, or any browser.

### How it works

Remote control needs Anthropic's bridge for the WebSocket connection, but model calls can go elsewhere. deepclaude starts a local proxy that splits the traffic:

```
claude remote-control
  +-- Bridge WebSocket -> wss://bridge.claudeusercontent.com (Anthropic, hardcoded)
  +-- Model API calls  -> http://localhost:3200 (proxy)
                            +-- /v1/messages -> DeepSeek ($0.87/M)
                            +-- everything else -> Anthropic (passthrough)
```

### Prerequisites
- Must be logged into Claude Code: `claude auth login`
- Must have a claude.ai subscription (the bridge is Anthropic infrastructure)
- Node.js 18+ (for the proxy)

The proxy starts automatically and stops when the session ends. See [proxy/README.md](proxy/README.md) for technical details.

## License

MIT
