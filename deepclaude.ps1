<#
.SYNOPSIS
    deepclaude — Use Claude Code with DeepSeek V4 Pro or other cheap backends.

.USAGE
    deepclaude                      # DeepSeek via proxy (default; subscription-friendly)
    deepclaude --backend or         # OpenRouter via proxy
    deepclaude --backend fw         # Fireworks via proxy
    deepclaude --backend anthropic  # Direct Claude Code (no proxy)
    deepclaude --remote             # Remote control + DeepSeek (browser URL)
    deepclaude --remote -b or       # Remote control + OpenRouter
    deepclaude --status             # Show keys and backends
    deepclaude --cost               # Pricing comparison
    deepclaude --benchmark          # Latency test

.NOTES
    For mid-session /anthropic switching, set ANTHROPIC_API_KEY in env BEFORE launching.
    The proxy reads it at startup and substitutes per-mode. Without it, /anthropic
    is refused with a clear error directing you to relaunch in -b anthropic.
#>

param(
    [Alias("b")]
    [string]$Backend,
    [Alias("r")]
    [switch]$Remote,
    [switch]$Status,
    [switch]$Cost,
    [switch]$Benchmark,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if (-not $Backend -and -not $Status -and -not $Cost -and -not $Benchmark -and -not $Help) {
    $Backend = if ($env:CHEAPCLAUDE_DEFAULT_BACKEND) { $env:CHEAPCLAUDE_DEFAULT_BACKEND } else { "ds" }
}

# --- Config: resolve all keys from process env or User scope ---
$DeepSeekKey = if ($env:DEEPSEEK_API_KEY) { $env:DEEPSEEK_API_KEY } else {
    [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
}
$OpenRouterKey = if ($env:OPENROUTER_API_KEY) { $env:OPENROUTER_API_KEY } else {
    [Environment]::GetEnvironmentVariable("OPENROUTER_API_KEY", "User")
}
$FireworksKey = if ($env:FIREWORKS_API_KEY) { $env:FIREWORKS_API_KEY } else {
    [Environment]::GetEnvironmentVariable("FIREWORKS_API_KEY", "User")
}
$AnthropicApiKey = if ($env:ANTHROPIC_API_KEY) { $env:ANTHROPIC_API_KEY } else {
    [Environment]::GetEnvironmentVariable("ANTHROPIC_API_KEY", "User")
}

$Providers = @{
    ds = @{
        name = "DeepSeek (via proxy)"
        url = "https://api.deepseek.com/anthropic"
        key = $DeepSeekKey; keyName = "DEEPSEEK_API_KEY"
        backendId = "deepseek"
        opus = "deepseek-v4-pro"; haiku = "deepseek-v4-flash"
    }
    or = @{
        name = "OpenRouter (via proxy)"
        url = "https://openrouter.ai/api"
        key = $OpenRouterKey; keyName = "OPENROUTER_API_KEY"
        backendId = "openrouter"
        opus = "deepseek/deepseek-v4-pro"; haiku = "deepseek/deepseek-v4-pro"
    }
    fw = @{
        name = "Fireworks AI (via proxy)"
        url = "https://api.fireworks.ai/inference"
        key = $FireworksKey; keyName = "FIREWORKS_API_KEY"
        backendId = "fireworks"
        opus = "accounts/fireworks/models/deepseek-v4-pro"; haiku = "accounts/fireworks/models/deepseek-v4-pro"
    }
}

function Get-KeyDisplay($k) {
    if (-not $k) { return "MISSING" }
    return "set (****" + $k.Substring($k.Length - [Math]::Min(4, $k.Length)) + ")"
}

# Set canonical Claude model names so the proxy's MODEL_REMAP can forward-map
# per backend mode. Anthropic mode passes through unchanged. This makes
# mid-session /anthropic /deepseek /openrouter /fireworks switches all work.
function Set-ClaudeModelEnv {
    $env:ANTHROPIC_DEFAULT_OPUS_MODEL = "claude-opus-4-7"
    $env:ANTHROPIC_DEFAULT_SONNET_MODEL = "claude-sonnet-4-6"
    $env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "claude-haiku-4-5"
    $env:CLAUDE_CODE_SUBAGENT_MODEL = "claude-opus-4-7"
    $env:CLAUDE_CODE_EFFORT_LEVEL = "max"
}

# Spawn proxy, detect port robustly (numeric-only line — proxy banner prints
# first, the port number after; old code's Select-Object -First 1 grabbed
# the banner). Auto-switch to chosen backend mode after spawn.
function Start-ProxyAndDetectPort {
    param([string]$Url, [string]$Key, [string]$BackendId, [string]$ScriptDir)

    $proxyScript = Join-Path $ScriptDir "proxy\start-proxy.js"
    $portFile = Join-Path $env:TEMP "deepclaude-proxy-port.txt"
    $logFile = Join-Path $env:TEMP "deepclaude-proxy.log"

    if (Test-Path $portFile) { Remove-Item $portFile -ErrorAction SilentlyContinue }

    # ANTHROPIC_API_KEY in env is inherited by the child node process. Proxy
    # reads it at startup for /anthropic mode-substitution.
    $proxyProc = Start-Process -FilePath "node" `
        -ArgumentList $proxyScript,$Url,$Key `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $portFile `
        -RedirectStandardError $logFile

    $proxyPort = $null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Milliseconds 200
        if (Test-Path $portFile) {
            $content = Get-Content $portFile -ErrorAction SilentlyContinue
            $proxyPort = $content | Where-Object { $_ -match '^\d+$' } | Select-Object -First 1
            if ($proxyPort) { break }
        }
    }
    Remove-Item $portFile -ErrorAction SilentlyContinue

    if (-not $proxyPort) {
        Write-Host "ERROR: Proxy failed to start (port not detected)" -ForegroundColor Red
        if ($proxyProc -and -not $proxyProc.HasExited) {
            Stop-Process -Id $proxyProc.Id -Force -ErrorAction SilentlyContinue
        }
        return $null
    }

    # Switch proxy to chosen backend (legacy startup defaults to anthropic).
    if ($BackendId -and $BackendId -ne "anthropic") {
        try {
            Invoke-WebRequest -Uri "http://127.0.0.1:$proxyPort/_proxy/mode" `
                -Method POST -Body "backend=$BackendId" `
                -UseBasicParsing -TimeoutSec 5 | Out-Null
        } catch {
            Write-Host "  WARN: failed to set initial proxy mode to $BackendId" -ForegroundColor Yellow
        }
    }

    return @{ Port = $proxyPort; Process = $proxyProc; LogFile = $logFile }
}

# --- Status ---
if ($Status) {
    Write-Host "`n  deepclaude - Backend Status" -ForegroundColor Cyan
    Write-Host "  ============================" -ForegroundColor DarkGray
    Write-Host "`n  Keys:" -ForegroundColor Yellow
    Write-Host "    DEEPSEEK_API_KEY:    $(Get-KeyDisplay $DeepSeekKey)"
    Write-Host "    OPENROUTER_API_KEY:  $(Get-KeyDisplay $OpenRouterKey)"
    Write-Host "    FIREWORKS_API_KEY:   $(Get-KeyDisplay $FireworksKey)"
    Write-Host "    ANTHROPIC_API_KEY:   $(Get-KeyDisplay $AnthropicApiKey)  (for /anthropic mid-session)"
    Write-Host "`n  Backends:" -ForegroundColor Yellow
    Write-Host "    deepclaude              # DeepSeek V4 Pro via proxy (default)"
    Write-Host "    deepclaude -b or        # OpenRouter via proxy"
    Write-Host "    deepclaude -b fw        # Fireworks AI via proxy"
    Write-Host "    deepclaude -b anthropic # Direct Claude (no proxy)"
    Write-Host ""
    exit 0
}

# --- Cost ---
if ($Cost) {
    Write-Host "`n  DeepSeek V4 Pro Pricing" -ForegroundColor Cyan
    Write-Host "  =======================" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Provider        Input/M    Output/M   Cache Hit/M" -ForegroundColor Yellow
    Write-Host "  ----------      --------   --------   -----------"
    Write-Host "  DeepSeek        `$0.44      `$0.87      `$0.004" -ForegroundColor Green
    Write-Host "  OpenRouter      `$0.44      `$0.87      (provider)"
    Write-Host "  Fireworks       `$1.74      `$3.48      (provider)"
    Write-Host "  Anthropic       `$3.00      `$15.00     `$0.30"
    Write-Host ""
    exit 0
}

# --- Help ---
if ($Help) {
    Write-Host "deepclaude - Claude Code with cheap backends (subscription-friendly)"
    Write-Host ""
    Write-Host "Usage: deepclaude [-b backend] [--remote] [--status] [--cost] [--benchmark]"
    Write-Host ""
    Write-Host "  -b, --backend   ds (default), or, fw, anthropic"
    Write-Host "  -r, --remote    Remote control mode (browser URL)"
    Write-Host ""
    Write-Host "Mid-session switching: set ANTHROPIC_API_KEY in env BEFORE launching"
    Write-Host "  to enable /anthropic /deepseek /openrouter /fireworks slash commands."
    exit 0
}

# --- Benchmark ---
if ($Benchmark) {
    Write-Host "`n  Latency Benchmark" -ForegroundColor Cyan
    foreach ($id in @("ds","or","fw")) {
        $p = $Providers[$id]
        Write-Host "  $($p.name)..." -NoNewline
        if (-not $p.key) { Write-Host " SKIP (no key)" -ForegroundColor DarkGray; continue }
        $useBearer = $id -in @("or","fw")
        $headers = if ($useBearer) {
            @{ "Authorization" = "Bearer $($p.key)"; "content-type" = "application/json"; "anthropic-version" = "2023-06-01" }
        } else {
            @{ "x-api-key" = $p.key; "content-type" = "application/json"; "anthropic-version" = "2023-06-01" }
        }
        $body = @{ model = $p.opus; max_tokens = 32; messages = @(@{ role = "user"; content = "Reply: ok" }) } | ConvertTo-Json -Depth 5
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $null = Invoke-RestMethod -Uri "$($p.url)/v1/messages" -Method POST -Headers $headers -Body $body -TimeoutSec 30
            $sw.Stop()
            Write-Host " OK ($($sw.ElapsedMilliseconds)ms)" -ForegroundColor Green
        } catch {
            $sw.Stop()
            Write-Host " FAIL ($($sw.ElapsedMilliseconds)ms)" -ForegroundColor Red
        }
    }
    Write-Host ""
    exit 0
}

# --- Remote ---
if ($Remote) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ($Backend -eq "anthropic") {
        Write-Host "`n  Launching remote control (Anthropic)...`n" -ForegroundColor Cyan
        foreach ($v in @("ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_DEFAULT_OPUS_MODEL",
            "ANTHROPIC_DEFAULT_SONNET_MODEL","ANTHROPIC_DEFAULT_HAIKU_MODEL",
            "CLAUDE_CODE_SUBAGENT_MODEL","CLAUDE_CODE_EFFORT_LEVEL")) {
            Remove-Item "Env:$v" -ErrorAction SilentlyContinue
        }
        Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
        & claude remote-control @Args
        exit 0
    }

    $p = $Providers[$Backend]
    if (-not $p) { Write-Host "ERROR: Unknown backend '$Backend'" -ForegroundColor Red; exit 1 }
    if (-not $p.key) { Write-Host "ERROR: $($p.keyName) not set" -ForegroundColor Red; exit 1 }

    Write-Host "`n  Starting model proxy for $($p.name)..." -ForegroundColor Cyan
    if (-not $AnthropicApiKey) {
        Write-Host "  WARN: ANTHROPIC_API_KEY not set -- /anthropic mid-session will 401." -ForegroundColor Yellow
    }

    $proxy = Start-ProxyAndDetectPort -Url $p.url -Key $p.key -BackendId $p.backendId -ScriptDir $ScriptDir
    if (-not $proxy) { exit 1 }

    Write-Host "  Proxy on :$($proxy.Port) -> $($p.url) ($Backend)" -ForegroundColor DarkGray
    Write-Host "  Log: $($proxy.LogFile)" -ForegroundColor DarkGray
    Write-Host ""

    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$($proxy.Port)"
    Set-ClaudeModelEnv
    Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue

    try {
        & claude remote-control @Args
    } finally {
        if ($proxy.Process -and -not $proxy.Process.HasExited) {
            Stop-Process -Id $proxy.Process.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  Proxy stopped." -ForegroundColor DarkGray
        }
    }
    exit 0
}

# --- Launch (non-remote) ---
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Path 1: anthropic mode = plain claude under whatever auth is configured.
if ($Backend -eq "anthropic") {
    foreach ($v in @("ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL","ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL","CLAUDE_CODE_EFFORT_LEVEL")) {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
    Write-Host "`n  Launching Claude Code (normal Anthropic backend)...`n" -ForegroundColor Cyan
    & claude @Args
    exit 0
}

# Path 2: cheap backend via proxy (subscription-friendly).
# CRITICAL: do NOT set ANTHROPIC_AUTH_TOKEN. Let the user's existing auth
# (subscription OAuth or ANTHROPIC_API_KEY) flow through. The proxy
# substitutes the backend's API key per request.
$p = $Providers[$Backend]
if (-not $p) { Write-Host "ERROR: Unknown backend '$Backend'. Use: ds, or, fw, anthropic" -ForegroundColor Red; exit 1 }
if (-not $p.key) { Write-Host "ERROR: $($p.keyName) not set" -ForegroundColor Red; exit 1 }

Write-Host "`n  Starting model proxy for $Backend..." -ForegroundColor Cyan
if (-not $AnthropicApiKey) {
    Write-Host "  Note: /anthropic mid-session switch unavailable (no ANTHROPIC_API_KEY)." -ForegroundColor Yellow
    Write-Host "        For anthropic mode, exit and run: deepclaude -b anthropic" -ForegroundColor Yellow
}

$proxy = Start-ProxyAndDetectPort -Url $p.url -Key $p.key -BackendId $p.backendId -ScriptDir $ScriptDir
if (-not $proxy) { exit 1 }

Write-Host "  Proxy on :$($proxy.Port) -> $($p.url) ($Backend)" -ForegroundColor DarkGray
Write-Host "  Model: $($p.opus) (main) + $($p.haiku) (subagents)" -ForegroundColor DarkGray
Write-Host "  Auth: passthrough (subscription OAuth or ANTHROPIC_API_KEY)" -ForegroundColor DarkGray
Write-Host "  Log: $($proxy.LogFile)" -ForegroundColor DarkGray
Write-Host ""

$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$($proxy.Port)"
Set-ClaudeModelEnv
# NOTE: not setting ANTHROPIC_AUTH_TOKEN — preserves subscription OAuth flow.
# NOTE: not unsetting ANTHROPIC_API_KEY — proxy already inherited it; claude
# uses it if set, else falls back to OAuth.

try {
    & claude @Args
} finally {
    if ($proxy.Process -and -not $proxy.Process.HasExited) {
        Stop-Process -Id $proxy.Process.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  Proxy stopped." -ForegroundColor DarkGray
    }
    foreach ($v in @("ANTHROPIC_BASE_URL","ANTHROPIC_DEFAULT_OPUS_MODEL","ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL","CLAUDE_CODE_SUBAGENT_MODEL","CLAUDE_CODE_EFFORT_LEVEL")) {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
}
