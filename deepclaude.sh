#!/usr/bin/env bash
# deepclaude — Use Claude Code with DeepSeek V4 Pro or other cheap backends
# Usage: deepclaude [--backend ds|or|fw|anthropic] [--remote] [--status] [--cost] [--benchmark]

set -euo pipefail

# Resolve SCRIPT_DIR through any symlink chain (e.g. /usr/local/bin/deepclaude
# -> /path/to/repo/deepclaude.sh) so $SCRIPT_DIR/proxy/... works regardless of
# how the script was invoked.
_source="${BASH_SOURCE[0]}"
while [ -L "$_source" ]; do
    _dir="$(cd "$(dirname "$_source")" && pwd)"
    _source="$(readlink "$_source")"
    [[ "$_source" != /* ]] && _source="$_dir/$_source"
done
SCRIPT_DIR="$(cd "$(dirname "$_source")" && pwd)"
unset _source _dir

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
            opus="deepseek-v4-pro"; sonnet="deepseek-v4-flash"
            haiku="deepseek-v4-flash"; subagent="deepseek-v4-flash"
            ;;
        or|openrouter)
            key="${OPENROUTER_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: OPENROUTER_API_KEY not set" >&2; exit 1; }
            url="$OPENROUTER_URL"
            opus="deepseek/deepseek-v4-pro"; sonnet="deepseek/deepseek-v4-flash"
            haiku="deepseek/deepseek-v4-flash"; subagent="deepseek/deepseek-v4-flash"
            ;;
        fw|fireworks)
            key="${FIREWORKS_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: FIREWORKS_API_KEY not set" >&2; exit 1; }
            url="$FIREWORKS_URL"
            opus="accounts/fireworks/models/deepseek-v4-pro"
            sonnet="accounts/fireworks/models/deepseek-v4-flash"
            haiku="accounts/fireworks/models/deepseek-v4-flash"
            subagent="accounts/fireworks/models/deepseek-v4-flash"
            ;;
        anthropic) ;;
        *) echo "ERROR: Unknown backend '$BACKEND'. Use: ds, or, fw, anthropic" >&2; exit 1 ;;
    esac
    RESOLVED_URL="$url"; RESOLVED_KEY="$key"
    RESOLVED_OPUS="$opus"; RESOLVED_SONNET="$sonnet"
    RESOLVED_HAIKU="$haiku"; RESOLVED_SUBAGENT="$subagent"
}

set_model_env() {
    # Use canonical Claude model names so Claude Code's permission gate
    # (auto / bypassPermissions) treats the session as a Claude session.
    # The local proxy translates these names back to backend-specific names
    # on the wire via MODEL_REMAP in proxy/model-proxy.js.
    # IMPORTANT: these names must remain keys in every MODEL_REMAP backend
    # block, otherwise unmapped requests will 404 against the backend.
    export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-7"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4-5-20251001"
    export CLAUDE_CODE_SUBAGENT_MODEL="claude-haiku-4-5-20251001"
    export CLAUDE_CODE_EFFORT_LEVEL="max"
}

backend_long_name() {
    case "$1" in
        ds|deepseek)   echo "deepseek" ;;
        or|openrouter) echo "openrouter" ;;
        fw|fireworks)  echo "fireworks" ;;
        anthropic)     echo "anthropic" ;;
        *) echo "ERROR: Unknown backend '$1'. Use: ds, or, fw, anthropic" >&2; return 1 ;;
    esac
}

# Starts proxy/start-proxy.js in the background and waits for it to bind a
# port. Sets PROXY_PID, PROXY_PORT, and PROXY_LOG as script globals so the
# EXIT trap (cleanup_proxy) can see the pid. Must be called WITHOUT command
# substitution — $(start_proxy) would run in a subshell and the globals
# would never reach the parent.
# PROXY_LOG defaults to /tmp/deepclaude-proxy.$$.log so concurrent invocations
# don't truncate each other's logs; override with PROXY_LOG=<path>.
# Requires: RESOLVED_URL, RESOLVED_KEY, BACKEND already set (call resolve_backend first).
start_proxy() {
    local backend_long
    backend_long=$(backend_long_name "$BACKEND") || exit 1

    PROXY_LOG="${PROXY_LOG:-/tmp/deepclaude-proxy.$$.log}"
    : > "$PROXY_LOG"
    node "$SCRIPT_DIR/proxy/start-proxy.js" "$RESOLVED_URL" "$RESOLVED_KEY" "$backend_long" >> "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!

    # Wait for a line that is a bare integer (the port emitted by start-proxy.js
    # after startModelProxy resolves). The proxy also writes a human-readable
    # "[MODEL-PROXY] Listening on ..." banner first, so we can't just read line 1.
    local proxy_port=""
    local tries=0
    while [[ -z "$proxy_port" ]] && [[ $tries -lt 30 ]]; do
        if kill -0 "$PROXY_PID" 2>/dev/null; then
            proxy_port=$(grep -E '^[0-9]+$' "$PROXY_LOG" 2>/dev/null | head -1)
        else
            echo "ERROR: Proxy process died during startup" >&2
            echo "  Log: $PROXY_LOG" >&2
            tail -20 "$PROXY_LOG" >&2 2>/dev/null
            exit 1
        fi
        [[ -z "$proxy_port" ]] && sleep 0.2
        tries=$((tries + 1))
    done

    if [[ -z "$proxy_port" ]]; then
        echo "ERROR: Proxy failed to report a port within 6s" >&2
        echo "  Log: $PROXY_LOG" >&2
        tail -20 "$PROXY_LOG" >&2 2>/dev/null
        exit 1
    fi

    PROXY_PORT="$proxy_port"
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

# Run claude and surface the tail of $PROXY_LOG if it exits abnormally.
# Skips signal-induced exits (>=128, e.g. 130 SIGINT, 143 SIGTERM) so
# Ctrl+C doesn't dump a noisy log on intentional quit.
run_claude_with_log_tail() {
    local exit_code=0
    "$@" || exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -lt 128 ]]; then
        echo "" >&2
        echo "  claude exited with status $exit_code. Last 20 lines of $PROXY_LOG:" >&2
        tail -20 "$PROXY_LOG" >&2 2>/dev/null || true
    fi
    return $exit_code
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

    echo "  Starting model proxy for $BACKEND..."
    # Call directly (not via $()): start_proxy sets PROXY_PID/PROXY_PORT/PROXY_LOG
    # as script globals, which a subshell would never propagate to the parent
    # — the EXIT trap needs PROXY_PID to actually clean up the node process.
    start_proxy
    echo "  Proxy log: $PROXY_LOG"

    echo "  Launching Claude Code via $BACKEND..."
    echo "  Proxy on :$PROXY_PORT -> $RESOLVED_URL"
    echo "  Model: $RESOLVED_OPUS (main) + $RESOLVED_HAIKU (subagents)"
    echo ""

    # Route through local proxy so the model name remap fires and Claude Code
    # sees a Claude model (unlocking auto-mode and bypassPermissions).
    export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
    set_model_env
    # Proxy injects auth itself; the client must not also send credentials.
    unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN

    # Don't exec — we want the EXIT trap to clean up the proxy.
    run_claude_with_log_tail claude "$@"
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
    start_proxy
    echo "  Proxy log: $PROXY_LOG"

    echo "  Proxy on :$PROXY_PORT -> $RESOLVED_URL"
    echo "  Launching remote control via $BACKEND..."
    echo ""

    export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
    set_model_env
    unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN

    run_claude_with_log_tail claude remote-control "$@"
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
