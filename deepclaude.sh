#!/usr/bin/env bash
# deepclaude — Use Claude Code with DeepSeek V4 Pro or other cheap backends
# Usage: deepclaude [--backend ds|or|fw|sf|anthropic] [--window 1h] [--remote] [--status] [--cost] [--benchmark]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ---
DEEPSEEK_URL="https://api.deepseek.com/anthropic"
OPENROUTER_URL="https://openrouter.ai/api"
FIREWORKS_URL="https://api.fireworks.ai/inference"
SFERENCE_URL="${SFERENCE_BASE_URL:-https://api.sference.com}"
SFERENCE_MODEL="${SFERENCE_MODEL:-moonshotai/Kimi-K2.6}"

BACKEND="${CHEAPCLAUDE_DEFAULT_BACKEND:-ds}"
ACTION="launch"
SWITCH_BACKEND=""
COMPLETION_WINDOW="${SFERENCE_COMPLETION_WINDOW:-1h}"
PROXY_PID=""
RESOLVED_TYPE="anthropic-compatible"
RESOLVED_BACKEND_NAME=""

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend|-b) BACKEND="$2"; shift 2 ;;
        --window|-w)  COMPLETION_WINDOW="$2"; shift 2 ;;
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
    local url="" key="" opus="" sonnet="" haiku="" subagent="" btype="anthropic-compatible" bname=""
    case "$BACKEND" in
        ds|deepseek)
            key="${DEEPSEEK_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: DEEPSEEK_API_KEY not set" >&2; exit 1; }
            url="$DEEPSEEK_URL"; bname="deepseek"
            opus="deepseek-v4-pro"; sonnet="deepseek-v4-pro"
            haiku="deepseek-v4-flash"; subagent="deepseek-v4-flash"
            ;;
        or|openrouter)
            key="${OPENROUTER_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: OPENROUTER_API_KEY not set" >&2; exit 1; }
            url="$OPENROUTER_URL"; bname="openrouter"
            opus="deepseek/deepseek-v4-pro"; sonnet="deepseek/deepseek-v4-pro"
            haiku="deepseek/deepseek-v4-pro"; subagent="deepseek/deepseek-v4-pro"
            ;;
        fw|fireworks)
            key="${FIREWORKS_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: FIREWORKS_API_KEY not set" >&2; exit 1; }
            url="$FIREWORKS_URL"; bname="fireworks"
            opus="accounts/fireworks/models/deepseek-v4-pro"
            sonnet="accounts/fireworks/models/deepseek-v4-pro"
            haiku="accounts/fireworks/models/deepseek-v4-pro"
            subagent="accounts/fireworks/models/deepseek-v4-pro"
            ;;
        sf|sference)
            key="${SFERENCE_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: SFERENCE_API_KEY not set" >&2; exit 1; }
            url="$SFERENCE_URL"; bname="sference"; btype="responses-bg"
            opus="$SFERENCE_MODEL"; sonnet="$SFERENCE_MODEL"
            haiku="$SFERENCE_MODEL"; subagent="$SFERENCE_MODEL"
            ;;
        bg)
            key="${BG_PROVIDER_API_KEY:-}"
            [[ -z "$key" ]] && { echo "ERROR: BG_PROVIDER_API_KEY not set" >&2; exit 1; }
            url="${BG_PROVIDER_URL:-}"; bname="bg"; btype="responses-bg"
            [[ -z "$url" ]] && { echo "ERROR: BG_PROVIDER_URL not set" >&2; exit 1; }
            opus="${BG_PROVIDER_MODEL:-}"; sonnet="$opus"; haiku="$opus"; subagent="$opus"
            [[ -z "$opus" ]] && { echo "ERROR: BG_PROVIDER_MODEL not set" >&2; exit 1; }
            COMPLETION_WINDOW="${BG_PROVIDER_WINDOW:-$COMPLETION_WINDOW}"
            ;;
        anthropic) ;;
        *) echo "ERROR: Unknown backend '$BACKEND'. Use: ds, or, fw, sf, bg, anthropic" >&2; exit 1 ;;
    esac
    RESOLVED_URL="$url"; RESOLVED_KEY="$key"
    RESOLVED_OPUS="$opus"; RESOLVED_SONNET="$sonnet"
    RESOLVED_HAIKU="$haiku"; RESOLVED_SUBAGENT="$subagent"
    RESOLVED_TYPE="$btype"; RESOLVED_BACKEND_NAME="$bname"
}

set_model_env() {
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$RESOLVED_OPUS"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$RESOLVED_SONNET"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$RESOLVED_HAIKU"
    export CLAUDE_CODE_SUBAGENT_MODEL="$RESOLVED_SUBAGENT"
    export CLAUDE_CODE_EFFORT_LEVEL="max"
}

start_proxy() {
    echo "  Starting model proxy for $BACKEND..."

    local port_file
    port_file=$(mktemp)
    SFERENCE_COMPLETION_WINDOW="$COMPLETION_WINDOW" \
    CHEAPCLAUDE_DEFAULT_MODE="$RESOLVED_BACKEND_NAME" \
    node "$SCRIPT_DIR/proxy/start-proxy.js" "$RESOLVED_URL" "$RESOLVED_KEY" "$RESOLVED_BACKEND_NAME" --window "$COMPLETION_WINDOW" > "$port_file" &
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

    PROXY_PORT=$(head -1 "$port_file")
    rm -f "$port_file"

    echo "  Proxy on :$PROXY_PORT -> $RESOLVED_URL"
    if [[ "$RESOLVED_TYPE" == "responses-bg" ]]; then
        echo "  Background mode: window=$COMPLETION_WINDOW model=$RESOLVED_OPUS"
    fi
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
    echo "    SFERENCE_API_KEY:    $(mask_key "${SFERENCE_API_KEY:-}")"
    echo "    BG_PROVIDER_API_KEY: $(mask_key "${BG_PROVIDER_API_KEY:-}")"
    echo ""
    echo "  Backends:"
    echo "    deepclaude                  # DeepSeek V4 Pro (default)"
    echo "    deepclaude -b or            # OpenRouter (cheapest)"
    echo "    deepclaude -b fw            # Fireworks AI (fastest)"
    echo "    deepclaude -b sf -w 1h      # sference background (Kimi-K2.6)"
    echo "    deepclaude -b bg            # Custom background provider"
    echo "    deepclaude -b anthropic     # Normal Claude Code"
    echo "    deepclaude --remote         # Remote control + DeepSeek"
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
    echo "  Provider Pricing"
    echo "  ================"
    echo ""
    echo "  Provider        Input/M    Output/M   Notes"
    echo "  ----------      --------   --------   -----"
    echo "  DeepSeek        \$0.44      \$0.87      sync + cache"
    echo "  OpenRouter      \$0.44      \$0.87      sync"
    echo "  Fireworks       \$1.74      \$3.48      sync"
    echo "  sference        (varies)   (varies)   async bg, up to 75% off"
    echo "  Anthropic       \$3.00      \$15.00     sync"
    echo ""
}

show_help() {
    echo "deepclaude — Claude Code with cheap backends"
    echo ""
    echo "Usage: deepclaude [options] [-- claude-args...]"
    echo ""
    echo "Options:"
    echo "  -b, --backend <ds|or|fw|sf|bg|anthropic>  Backend (default: ds)"
    echo "  -w, --window <15m|1h|6h|24h|48h>           Completion window (bg providers)"
    echo "  -r, --remote                              Remote control mode"
    echo "  --status                                   Show keys and backends"
    echo "  --cost                                     Pricing comparison"
    echo "  --benchmark                                Latency test"
    echo "  -s, --switch <backend>                     Switch proxy mid-session"
    echo "  -h, --help                                 This help"
    echo ""
    echo "Environment variables:"
    echo "  DEEPSEEK_API_KEY              DeepSeek API key"
    echo "  OPENROUTER_API_KEY            OpenRouter API key"
    echo "  FIREWORKS_API_KEY             Fireworks API key"
    echo "  SFERENCE_API_KEY              sference API key"
    echo "  SFERENCE_MODEL                Model (default: moonshotai/Kimi-K2.6)"
    echo "  SFERENCE_COMPLETION_WINDOW    Window (default: 1h)"
    echo "  BG_PROVIDER_URL/API_KEY/MODEL Custom background provider"
    echo "  CHEAPCLAUDE_DEFAULT_BACKEND   Default backend (default: ds)"
}

do_switch() {
    local backend="$SWITCH_BACKEND"
    case "$backend" in
        ds|deepseek)   backend="deepseek" ;;
        or|openrouter) backend="openrouter" ;;
        fw|fireworks)  backend="fireworks" ;;
        sf|sference)   backend="sference" ;;
        bg)            backend="bg" ;;
        anthropic)     backend="anthropic" ;;
        *) echo "ERROR: Unknown backend '$backend'" >&2; exit 1 ;;
    esac
    local resp
    resp=$(curl -sX POST http://127.0.0.1:3200/_proxy/mode -d "backend=$backend" 2>/dev/null) || {
        echo "  Proxy not running. Start with: deepclaude" >&2; exit 1
    }
    echo "  $resp"
    if [[ -n "$COMPLETION_WINDOW" && ( "$backend" == "sference" || "$backend" == "bg" ) ]]; then
        local wresp
        wresp=$(curl -sX POST http://127.0.0.1:3200/_proxy/window -d "window=$COMPLETION_WINDOW" 2>/dev/null) || true
        [[ -n "$wresp" ]] && echo "  $wresp"
    fi
}

run_benchmark() {
    now_ms() { python3 -c 'import time; print(int(time.time()*1000))'; }

    echo ""
    echo "  Latency Benchmark"
    echo "  ================="
    for name in deepseek openrouter fireworks sference; do
        local url="" key="" model="" btype="anthropic-compatible"
        case "$name" in
            deepseek)   url="$DEEPSEEK_URL"; key="${DEEPSEEK_API_KEY:-}"; model="deepseek-v4-pro" ;;
            openrouter) url="$OPENROUTER_URL"; key="${OPENROUTER_API_KEY:-}"; model="deepseek/deepseek-v4-pro" ;;
            fireworks)  url="$FIREWORKS_URL"; key="${FIREWORKS_API_KEY:-}"; model="accounts/fireworks/models/deepseek-v4-pro" ;;
            sference)   url="$SFERENCE_URL"; key="${SFERENCE_API_KEY:-}"; model="$SFERENCE_MODEL"; btype="responses-bg" ;;
        esac
        if [[ -z "$key" ]]; then echo "  $name: SKIP (no key)"; continue; fi
        local start_ms=$(now_ms)
        if [[ "$btype" == "responses-bg" ]]; then
            local status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$url/v1/responses" \
                -H "X-API-Key: $key" -H "content-type: application/json" \
                -d "{\"model\":\"$model\",\"input\":\"Reply: ok\",\"background\":true,\"metadata\":{\"completion_window\":\"1h\"}}" \
                --max-time 30 2>/dev/null || echo "timeout")
            local end_ms=$(now_ms)
            local elapsed=$((end_ms - start_ms))
            if [[ "$status" == "200" || "$status" == "201" || "$status" == "202" ]]; then
                echo "  $name: submit OK (${elapsed}ms, async — completion time varies)"
            else
                echo "  $name: FAIL ($status, ${elapsed}ms)"
            fi
        else
            local status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$url/v1/messages" \
                -H "x-api-key: $key" -H "content-type: application/json" -H "anthropic-version: 2023-06-01" \
                -d "{\"model\":\"$model\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Reply: ok\"}]}" \
                --max-time 30 2>/dev/null || echo "timeout")
            local end_ms=$(now_ms)
            local elapsed=$((end_ms - start_ms))
            if [[ "$status" == "200" ]]; then
                echo "  $name: OK (${elapsed}ms)"
            else
                echo "  $name: FAIL ($status, ${elapsed}ms)"
            fi
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

    if [[ "$RESOLVED_TYPE" == "responses-bg" ]]; then
        start_proxy
        echo "  Launching Claude Code via $BACKEND (background)..."
        echo ""
        export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
        set_model_env
        unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
        # No exec: the EXIT trap must stay alive to stop the proxy on exit.
        claude "$@"
        return
    fi

    echo "  Launching Claude Code via $BACKEND..."
    echo "  Endpoint: $RESOLVED_URL"
    echo "  Model: $RESOLVED_OPUS (main) + $RESOLVED_HAIKU (subagents)"
    echo ""

    export ANTHROPIC_BASE_URL="$RESOLVED_URL"
    export ANTHROPIC_AUTH_TOKEN="$RESOLVED_KEY"
    set_model_env
    unset ANTHROPIC_API_KEY

    exec claude "$@"
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
    start_proxy

    echo "  Launching remote control via $BACKEND..."
    echo ""

    export ANTHROPIC_BASE_URL="http://127.0.0.1:$PROXY_PORT"
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
