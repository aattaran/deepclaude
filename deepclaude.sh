#!/usr/bin/env bash
# deepclaude — Use Claude Code with DeepSeek V4 Pro or other cheap backends
# Usage: deepclaude [--backend ds|or|fw|anthropic] [--remote] [--status] [--cost] [--benchmark]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ---
DEEPSEEK_URL="https://api.deepseek.com/anthropic"
OPENROUTER_URL="https://openrouter.ai/api"
FIREWORKS_URL="https://api.fireworks.ai/inference"

BACKEND="${CHEAPCLAUDE_DEFAULT_BACKEND:-ds}"
ACTION="launch"
SWITCH_BACKEND=""
PROXY_PID=""

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend|-b) BACKEND="$2"; shift 2 ;;
        --switch|-s)  ACTION="switch"; SWITCH_BACKEND="$2"; shift 2 ;;
        --remote|-r)  ACTION="remote"; shift ;;
        --status)     ACTION="status"; shift ;;
        --cost)       ACTION="cost"; shift ;;
        --benchmark)  ACTION="benchmark"; shift ;;
        --help|-h)    ACTION="help"; shift ;;
        *)            break ;;
    esac
done

cleanup_proxy() {
    if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
        kill "$PROXY_PID" 2>/dev/null || true
        echo "  Proxy stopped."
    fi
}
trap cleanup_proxy EXIT

mask_key() {
    local k="$1"
    if [[ -z "$k" ]]; then echo "MISSING"; else echo "set (****${k: -4})"; fi
}

resolve_backend() {
    local url="" key="" opus="" sonnet="" haiku="" subagent=""
    case "$BACKEND" in
        ds|deepseek)
            key="${DEEPSEEK_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: DEEPSEEK_API_KEY not set" >&2; exit 1; }
            url="$DEEPSEEK_URL"
            opus="deepseek-v4-pro"; sonnet="deepseek-v4-pro"
            haiku="deepseek-v4-flash"; subagent="deepseek-v4-flash"
            ;;
        or|openrouter)
            key="${OPENROUTER_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: OPENROUTER_API_KEY not set" >&2; exit 1; }
            url="$OPENROUTER_URL"
            opus="deepseek/deepseek-v4-pro"; sonnet="deepseek/deepseek-v4-pro"
            haiku="deepseek/deepseek-v4-pro"; subagent="deepseek/deepseek-v4-pro"
            ;;
        fw|fireworks)
            key="${FIREWORKS_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: FIREWORKS_API_KEY not set" >&2; exit 1; }
            url="$FIREWORKS_URL"
            opus="accounts/fireworks/models/deepseek-v4-pro"
            sonnet="accounts/fireworks/models/deepseek-v4-pro"
            haiku="accounts/fireworks/models/deepseek-v4-pro"
            subagent="accounts/fireworks/models/deepseek-v4-pro"
            ;;
        anthropic) ;;
        *) echo "ERROR: Unknown backend '$BACKEND'. Use: ds, or, fw, anthropic" >&2; exit 1 ;;
    esac
    RESOLVED_URL="$url"; RESOLVED_KEY="$key"
    RESOLVED_OPUS="$opus"; RESOLVED_SONNET="$sonnet"
    RESOLVED_HAIKU="$haiku"; RESOLVED_SUBAGENT="$subagent"
}

set_model_env() {
    # Use canonical Claude model names. Proxy MODEL_REMAP forward-maps to
    # backend-native IDs (deepseek-v4-pro etc) for non-anthropic modes;
    # anthropic mode passes through unchanged. This makes mid-session
    # /anthropic /deepseek /openrouter /fireworks switches all work.
    export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-7"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4-5"
    export CLAUDE_CODE_SUBAGENT_MODEL="claude-opus-4-7"
    export CLAUDE_CODE_EFFORT_LEVEL="max"
}

show_status() {
    echo ""
    echo "  deepclaude — Backend Status"
    echo "  ============================"
    echo ""
    echo "  Keys:"
    echo "    DEEPSEEK_API_KEY:    $(mask_key "${DEEPSEEK_API_KEY:-}")"
    echo "    OPENROUTER_API_KEY:  $(mask_key "${OPENROUTER_API_KEY:-}")"
    echo "    FIREWORKS_API_KEY:   $(mask_key "${FIREWORKS_API_KEY:-}")"
    echo ""
    echo "  Backends:"
    echo "    deepclaude                  # DeepSeek V4 Pro (default)"
    echo "    deepclaude -b or            # OpenRouter (cheapest)"
    echo "    deepclaude -b fw            # Fireworks AI (fastest)"
    echo "    deepclaude -b anthropic     # Normal Claude Code"
    echo "    deepclaude --remote         # Remote control + DeepSeek"
    echo "    deepclaude --remote -b or   # Remote control + OpenRouter"
    echo ""
    local proxy_status
    proxy_status=$(curl -s http://127.0.0.1:3200/_proxy/status 2>/dev/null) || proxy_status=""
    if [[ -n "$proxy_status" ]]; then
        echo "  Proxy: running"
        echo "    $proxy_status"
    else
        echo "  Proxy: not running"
    fi
    echo ""
}

show_cost() {
    echo ""
    echo "  DeepSeek V4 Pro Pricing"
    echo "  ======================="
    echo ""
    echo "  Provider        Input/M    Output/M   Cache Hit/M"
    echo "  ----------      --------   --------   -----------"
    echo "  DeepSeek        \$0.44      \$0.87      \$0.004"
    echo "  OpenRouter      \$0.44      \$0.87      (provider)"
    echo "  Fireworks       \$1.74      \$3.48      (provider)"
    echo "  Anthropic       \$3.00      \$15.00     \$0.30"
    echo ""
    echo "  Monthly estimate (heavy use, 25 days): \$30-80"
    echo ""
}

show_help() {
    echo "deepclaude — Claude Code with cheap backends"
    echo ""
    echo "Usage: deepclaude [options] [-- claude-args...]"
    echo ""
    echo "Options:"
    echo "  -b, --backend <ds|or|fw|anthropic>  Backend (default: ds)"
    echo "  -r, --remote                        Remote control mode (browser URL)"
    echo "  --status                             Show keys and backends"
    echo "  --cost                               Pricing comparison"
    echo "  --benchmark                          Latency test"
    echo "  -s, --switch <backend>               Switch proxy mid-session"
    echo "  -h, --help                           This help"
    echo ""
    echo "Environment variables:"
    echo "  DEEPSEEK_API_KEY      DeepSeek API key (required for ds)"
    echo "  OPENROUTER_API_KEY    OpenRouter API key (required for or)"
    echo "  FIREWORKS_API_KEY     Fireworks API key (required for fw)"
    echo "  CHEAPCLAUDE_DEFAULT_BACKEND  Default backend (default: ds)"
}

do_switch() {
    local backend="$SWITCH_BACKEND"
    case "$backend" in
        ds|deepseek)   backend="deepseek" ;;
        or|openrouter) backend="openrouter" ;;
        fw|fireworks)  backend="fireworks" ;;
        anthropic)     backend="anthropic" ;;
        *) echo "ERROR: Unknown backend '$backend'. Use: ds, or, fw, anthropic" >&2; exit 1 ;;
    esac
    local resp
    resp=$(curl -sX POST http://127.0.0.1:3200/_proxy/mode -d "backend=$backend" 2>/dev/null) || {
        echo "  Proxy not running. Start with: deepclaude" >&2; exit 1
    }
    echo "  $resp"
}

run_benchmark() {
    echo ""
    echo "  Latency Benchmark (1 request each)"
    echo "  ==================================="
    for name in deepseek openrouter fireworks; do
        local url="" key="" model=""
        case "$name" in
            deepseek)   url="$DEEPSEEK_URL"; key="${DEEPSEEK_API_KEY:-}"; model="deepseek-v4-pro" ;;
            openrouter) url="$OPENROUTER_URL"; key="${OPENROUTER_API_KEY:-}"; model="deepseek/deepseek-v4-pro" ;;
            fireworks)  url="$FIREWORKS_URL"; key="${FIREWORKS_API_KEY:-}"; model="accounts/fireworks/models/deepseek-v4-pro" ;;
        esac
        if [[ -z "$key" ]]; then echo "  $name: SKIP (no key)"; continue; fi
        local start_ms=$(date +%s%3N 2>/dev/null || python3 -c 'import time;print(int(time.time()*1000))')
        local status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$url/v1/messages" \
            -H "x-api-key: $key" -H "content-type: application/json" -H "anthropic-version: 2023-06-01" \
            -d "{\"model\":\"$model\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply: ok\"}]}" \
            --max-time 30 2>/dev/null || echo "timeout")
        local end_ms=$(date +%s%3N 2>/dev/null || python3 -c 'import time;print(int(time.time()*1000))')
        local elapsed=$((end_ms - start_ms))
        if [[ "$status" == "200" ]]; then
            echo "  $name: OK (${elapsed}ms)"
        else
            echo "  $name: FAIL ($status, ${elapsed}ms)"
        fi
    done
    echo ""
}

launch_claude() {
    if [[ "$BACKEND" == "anthropic" ]]; then
        echo "  Launching Claude Code (normal Anthropic backend)..."
        unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN
        unset ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
        unset ANTHROPIC_DEFAULT_HAIKU_MODEL CLAUDE_CODE_SUBAGENT_MODEL
        unset CLAUDE_CODE_EFFORT_LEVEL
        exec claude "$@"
    fi

    resolve_backend

    # Path 2: cheap backend via proxy (subscription-friendly).
    # Spawns the model proxy on 3200 (auto-increments if busy).
    # CRITICAL: do NOT export ANTHROPIC_AUTH_TOKEN. Let the user's
    # existing auth (subscription OAuth or ANTHROPIC_API_KEY) flow
    # through. The proxy substitutes the backend's API key per request,
    # so client auth is irrelevant for non-anthropic /v1/messages calls.
    echo "  Starting model proxy for $BACKEND..."
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "  Note: /anthropic mid-session switch unavailable (no ANTHROPIC_API_KEY)."
        echo "        For anthropic mode, exit and run: deepclaude -b anthropic"
    fi

    local port_file
    port_file=$(mktemp)
    # Tee proxy stdout to /tmp/proxy.log so we can diagnose failures after the fact.
    node "$SCRIPT_DIR/proxy/start-proxy.js" "$RESOLVED_URL" "$RESOLVED_KEY" 2>&1 | tee /tmp/proxy.log > "$port_file" &
    PROXY_PID=$!

    local tries=0 proxy_port=""
    while [[ -z "$proxy_port" ]] && [[ $tries -lt 30 ]]; do
        sleep 0.2
        proxy_port=$(grep -oE '^[0-9]+$' "$port_file" 2>/dev/null | head -1 || true)
        tries=$((tries + 1))
    done
    rm -f "$port_file"
    if [[ -z "$proxy_port" ]]; then
        echo "ERROR: Proxy failed to start (port not detected)" >&2
        exit 1
    fi

    # Switch proxy to chosen backend (legacy startup defaults to anthropic).
    case "$BACKEND" in
        ds) curl -sX POST "http://127.0.0.1:$proxy_port/_proxy/mode" -d "backend=deepseek" >/dev/null 2>&1 ;;
        or) curl -sX POST "http://127.0.0.1:$proxy_port/_proxy/mode" -d "backend=openrouter" >/dev/null 2>&1 ;;
        fw) curl -sX POST "http://127.0.0.1:$proxy_port/_proxy/mode" -d "backend=fireworks" >/dev/null 2>&1 ;;
    esac

    echo "  Proxy on :$proxy_port -> $RESOLVED_URL ($BACKEND)"
    echo "  Model: $RESOLVED_OPUS (main) + $RESOLVED_HAIKU (subagents)"
    echo "  Auth: passthrough (subscription OAuth or ANTHROPIC_API_KEY)"
    echo ""

    export ANTHROPIC_BASE_URL="http://127.0.0.1:$proxy_port"
    set_model_env
    # No export of ANTHROPIC_AUTH_TOKEN. No unset of ANTHROPIC_API_KEY:
    # claude will use whichever it has; proxy already inherited the env.

    # NOT exec: the trap needs to fire on exit to kill the proxy.
    claude "$@"
}

launch_remote() {
    if [[ "$BACKEND" == "anthropic" ]]; then
        echo "  Launching remote control (Anthropic)..."
        unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN
        unset ANTHROPIC_DEFAULT_OPUS_MODEL ANTHROPIC_DEFAULT_SONNET_MODEL
        unset ANTHROPIC_DEFAULT_HAIKU_MODEL CLAUDE_CODE_SUBAGENT_MODEL
        unset CLAUDE_CODE_EFFORT_LEVEL ANTHROPIC_API_KEY
        exec claude remote-control "$@"
    fi

    resolve_backend

    echo "  Starting model proxy for $BACKEND..."

    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "  WARN: ANTHROPIC_API_KEY not set -- /anthropic mid-session switch will 401."
        echo "        Set with: export ANTHROPIC_API_KEY=sk-ant-..."
    fi

    local port_file
    port_file=$(mktemp)
    node "$SCRIPT_DIR/proxy/start-proxy.js" "$RESOLVED_URL" "$RESOLVED_KEY" > "$port_file" &
    PROXY_PID=$!

    local tries=0 proxy_port=""
    while [[ -z "$proxy_port" ]] && [[ $tries -lt 30 ]]; do
        sleep 0.2
        # Proxy stdout contains banner first, then port number on its own line.
        # Match a line containing ONLY digits, take first such line.
        proxy_port=$(grep -oE '^[0-9]+$' "$port_file" 2>/dev/null | head -1 || true)
        tries=$((tries + 1))
    done
    rm -f "$port_file"
    if [[ -z "$proxy_port" ]]; then
        echo "ERROR: Proxy failed to start (port not detected)" >&2
        exit 1
    fi

    echo "  Proxy on :$proxy_port -> $RESOLVED_URL"
    echo "  Launching remote control via $BACKEND..."
    echo ""

    export ANTHROPIC_BASE_URL="http://127.0.0.1:$proxy_port"
    set_model_env
    unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN

    claude remote-control "$@"
}

# --- Main ---
case "$ACTION" in
    status)    show_status ;;
    cost)      show_cost ;;
    benchmark) run_benchmark ;;
    help)      show_help ;;
    switch)    do_switch ;;
    remote)    launch_remote "$@" ;;
    launch)    launch_claude "$@" ;;
esac
