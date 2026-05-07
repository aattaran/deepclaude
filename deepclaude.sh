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
AUTO_MODE=0

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend|-b) BACKEND="$2"; shift 2 ;;
        --switch|-s)  ACTION="switch"; SWITCH_BACKEND="$2"; shift 2 ;;
        --remote|-r)  ACTION="remote"; shift ;;
        --auto)       AUTO_MODE=1; shift ;;
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
    if [[ "$AUTO_MODE" == "1" ]]; then
        # Claude Code's auto / bypassPermissions modes are gated on the env
        # var being a `claude-*` name; the proxy translates back to the
        # backend name on the wire via MODEL_REMAP. The TUI welcome chip
        # will display the canonical name — that's the explicit tradeoff
        # surfaced by print_auto_mode_tip at launch.
        export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-7"
        export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"
        export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4-5-20251001"
        export CLAUDE_CODE_SUBAGENT_MODEL="claude-haiku-4-5-20251001"
    else
        export ANTHROPIC_DEFAULT_OPUS_MODEL="$RESOLVED_OPUS"
        export ANTHROPIC_DEFAULT_SONNET_MODEL="$RESOLVED_SONNET"
        export ANTHROPIC_DEFAULT_HAIKU_MODEL="$RESOLVED_HAIKU"
        export CLAUDE_CODE_SUBAGENT_MODEL="$RESOLVED_SUBAGENT"
    fi
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

# Sets PROXY_PID, PROXY_PORT, PROXY_LOG as script globals so the EXIT trap
# can clean up the node child. Must be called WITHOUT command substitution
# — $(start_proxy) runs in a subshell and globals never reach the parent.
# Requires: RESOLVED_URL, RESOLVED_KEY, BACKEND already set.
start_proxy() {
    local backend_long
    backend_long=$(backend_long_name "$BACKEND") || exit 1

    PROXY_LOG="${PROXY_LOG:-/tmp/deepclaude-proxy.$$.log}"
    : > "$PROXY_LOG"
    node "$SCRIPT_DIR/proxy/start-proxy.js" "$RESOLVED_URL" "$RESOLVED_KEY" "$backend_long" >> "$PROXY_LOG" 2>&1 &
    PROXY_PID=$!

    # The proxy emits a banner line, then a bare-numeric port line on a
    # successful bind. Match the bare integer to skip the banner; do not
    # introduce other numeric-only stdout in proxy startup.
    local proxy_port=""
    local tries=0
    while [[ -z "$proxy_port" ]] && [[ $tries -lt 30 ]]; do
        if kill -0 "$PROXY_PID" 2>/dev/null; then
            # `|| true`: with `set -o pipefail`, grep no-match (exit 1)
            # would otherwise exit the script; we expect zero matches on
            # early iterations before the proxy has emitted its port.
            proxy_port=$(grep -E '^[0-9]+$' "$PROXY_LOG" 2>/dev/null | head -1 || true)
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
    echo "  --auto                               Unlock auto/bypassPermissions modes"
    echo "                                       (TUI shows claude-* names; wire still routes to backend)"
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

print_auto_mode_tip() {
    if [[ "$AUTO_MODE" == "1" ]]; then
        echo "  Auto mode: ON"
        echo "    TUI will display 'claude-opus-4-7' (auto/bypassPermissions unlocked)."
        echo "    Actual routing: $RESOLVED_OPUS via $RESOLVED_URL."
        echo "    Verify with: curl -s http://127.0.0.1:\$PROXY_PORT/_proxy/cost | jq"
    else
        echo "  Auto mode: OFF (TUI shows '$RESOLVED_OPUS')"
        echo "    Pass --auto to unlock auto/bypassPermissions modes — TUI will then show 'claude-opus-4-7'."
    fi
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
    start_proxy
    echo "  Proxy log: $PROXY_LOG"

    echo "  Launching Claude Code via $BACKEND..."
    echo "  Proxy on :$PROXY_PORT -> $RESOLVED_URL"
    echo "  Model: $RESOLVED_OPUS (main) + $RESOLVED_HAIKU (subagents)"
    print_auto_mode_tip
    echo ""

    export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
    set_model_env

    # Don't `exec` — the EXIT trap needs to fire to stop the proxy.
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

    local port_file
    port_file=$(mktemp)
    node "$SCRIPT_DIR/proxy/start-proxy.js" "$RESOLVED_URL" "$RESOLVED_KEY" > "$port_file" &
    PROXY_PID=$!

    local tries=0
    while [[ ! -s "$port_file" ]] && [[ $tries -lt 30 ]]; do
        sleep 0.2
        tries=$((tries + 1))
    done

    if [[ ! -s "$port_file" ]]; then
        echo "ERROR: Proxy failed to start" >&2
        rm -f "$port_file"
        exit 1
    fi

    local proxy_port
    proxy_port=$(head -1 "$port_file")
    rm -f "$port_file"

    echo "  Proxy on :$proxy_port -> $RESOLVED_URL"
    echo "  Launching remote control via $BACKEND..."
    print_auto_mode_tip
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
