<#
.SYNOPSIS
    deepclaude — Use Claude Code with DeepSeek V4 Pro or other cheap backends.

.USAGE
    deepclaude                      # DeepSeek V4 Pro (default)
    deepclaude --backend or         # OpenRouter (cheapest)
    deepclaude --backend fw         # Fireworks AI (fastest)
    deepclaude --backend anthropic  # Normal Claude Code
    deepclaude --remote             # Remote control + DeepSeek (browser URL)
    deepclaude --remote -b or       # Remote control + OpenRouter
    deepclaude --status             # Show keys and backends
    deepclaude --cost               # Pricing comparison
    deepclaude --benchmark          # Latency test
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

# --- Config ---
$DeepSeekKey = if ($env:DEEPSEEK_API_KEY) { $env:DEEPSEEK_API_KEY } else {
    [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
}
$OpenRouterKey = if ($env:OPENROUTER_API_KEY) { $env:OPENROUTER_API_KEY } else {
    [Environment]::GetEnvironmentVariable("OPENROUTER_API_KEY", "User")
}
$FireworksKey = if ($env:FIREWORKS_API_KEY) { $env:FIREWORKS_API_KEY } else {
    [Environment]::GetEnvironmentVariable("FIREWORKS_API_KEY", "User")
}

$Providers = @{
    ds = @{
        name = "DeepSeek (direct)"
        url = "https://api.deepseek.com/anthropic"
        key = $DeepSeekKey; keyName = "DEEPSEEK_API_KEY"
        opus = "deepseek-v4-pro"; sonnet = "deepseek-v4-pro"
        haiku = "deepseek-v4-flash"; subagent = "deepseek-v4-flash"
    }
    or = @{
        name = "OpenRouter"
        url = "https://openrouter.ai/api"
        key = $OpenRouterKey; keyName = "OPENROUTER_API_KEY"
        opus = "deepseek/deepseek-v4-pro"; sonnet = "deepseek/deepseek-v4-pro"
        haiku = "deepseek/deepseek-v4-pro"; subagent = "deepseek/deepseek-v4-pro"
    }
    fw = @{
        name = "Fireworks AI"
        url = "https://api.fireworks.ai/inference"
        key = $FireworksKey; keyName = "FIREWORKS_API_KEY"
        opus = "accounts/fireworks/models/deepseek-v4-pro"
        sonnet = "accounts/fireworks/models/deepseek-v4-pro"
        haiku = "accounts/fireworks/models/deepseek-v4-pro"
        subagent = "accounts/fireworks/models/deepseek-v4-pro"
    }
}

function Get-KeyDisplay($k) {
    if (-not $k) { return "MISSING" }
    return "set (****" + $k.Substring($k.Length - [Math]::Min(4, $k.Length)) + ")"
}

# --- Status ---
if ($Status) {
    Write-Host "`n  deepclaude - Backend Status" -ForegroundColor Cyan
    Write-Host "  ============================" -ForegroundColor DarkGray
    Write-Host "`n  Keys:" -ForegroundColor Yellow
    Write-Host "    DEEPSEEK_API_KEY:    $(Get-KeyDisplay $DeepSeekKey)"
    Write-Host "    OPENROUTER_API_KEY:  $(Get-KeyDisplay $OpenRouterKey)"
    Write-Host "    FIREWORKS_API_KEY:   $(Get-KeyDisplay $FireworksKey)"
    Write-Host "`n  Backends:" -ForegroundColor Yellow
    Write-Host "    deepclaude              # DeepSeek V4 Pro (default)"
    Write-Host "    deepclaude -b or        # OpenRouter (cheapest)"
    Write-Host "    deepclaude -b fw        # Fireworks AI (fastest)"
    Write-Host "    deepclaude -b anthropic # Normal Claude Code"
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
    Write-Host "  Monthly estimate (heavy use): `$30-80 vs `$200 Anthropic" -ForegroundColor Green
    Write-Host ""
    exit 0
}

# --- Help ---
if ($Help) {
    Write-Host "deepclaude - Claude Code with cheap backends"
    Write-Host ""
    Write-Host "Usage: deepclaude [-b backend] [--status] [--cost] [--benchmark]"
    Write-Host ""
    Write-Host "  -b, --backend   ds (default), or, fw, anthropic"
    Write-Host "  --status        Show keys and backends"
    Write-Host "  --cost          Pricing comparison"
    Write-Host "  --benchmark     Latency test"
    exit 0
}

# --- Benchmark ---
if ($Benchmark) {
    Write-Host "`n  Latency Benchmark" -ForegroundColor Cyan
    Write-Host "  ==================" -ForegroundColor DarkGray
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
            $code = if ($_.Exception.Response) { $_.Exception.Response.StatusCode.value__ } else { "timeout" }
            Write-Host " FAIL ($code, $($sw.ElapsedMilliseconds)ms)" -ForegroundColor Red
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

    $proxyScript = Join-Path $ScriptDir "proxy\start-proxy.js"
    $proxyProc = Start-Process -FilePath "node" -ArgumentList $proxyScript,$p.url,$p.key -PassThru -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\deepclaude-proxy-port.txt"

    $tries = 0
    while ($tries -lt 30) {
        Start-Sleep -Milliseconds 200
        $tries++
        if (Test-Path "$env:TEMP\deepclaude-proxy-port.txt") {
            $content = Get-Content "$env:TEMP\deepclaude-proxy-port.txt" -ErrorAction SilentlyContinue
            if ($content) { break }
        }
    }

    $proxyPort = (Get-Content "$env:TEMP\deepclaude-proxy-port.txt" -ErrorAction SilentlyContinue | Select-Object -First 1)
    Remove-Item "$env:TEMP\deepclaude-proxy-port.txt" -ErrorAction SilentlyContinue

    if (-not $proxyPort) {
        Write-Host "ERROR: Proxy failed to start" -ForegroundColor Red
        if ($proxyProc -and -not $proxyProc.HasExited) { Stop-Process -Id $proxyProc.Id -Force }
        exit 1
    }

    Write-Host "  Proxy on :$proxyPort -> $($p.url)" -ForegroundColor DarkGray
    Write-Host "  Launching remote control via $($p.name)...`n" -ForegroundColor Cyan

    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$proxyPort"
    $env:ANTHROPIC_DEFAULT_OPUS_MODEL = $p.opus
    $env:ANTHROPIC_DEFAULT_SONNET_MODEL = $p.sonnet
    $env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $p.haiku
    $env:CLAUDE_CODE_SUBAGENT_MODEL = $p.subagent
    $env:CLAUDE_CODE_EFFORT_LEVEL = "max"
    Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue

    try {
        & claude remote-control @Args
    } finally {
        if ($proxyProc -and -not $proxyProc.HasExited) {
            Stop-Process -Id $proxyProc.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  Proxy stopped." -ForegroundColor DarkGray
        }
    }
    exit 0
}

# --- Launch ---
if ($Backend -eq "anthropic") {
    foreach ($v in @("ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL","ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL","CLAUDE_CODE_EFFORT_LEVEL")) {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
    Write-Host "`n  Launching Claude Code (normal Anthropic)...`n" -ForegroundColor Cyan
    & claude @Args
    exit 0
}

$p = $Providers[$Backend]
if (-not $p) { Write-Host "ERROR: Unknown backend '$Backend'. Use: ds, or, fw, anthropic" -ForegroundColor Red; exit 1 }
if (-not $p.key) { Write-Host "ERROR: $($p.keyName) not set" -ForegroundColor Red; exit 1 }

Write-Host "`n  Launching Claude Code via $($p.name)..." -ForegroundColor Cyan
Write-Host "  Endpoint: $($p.url)" -ForegroundColor DarkGray
Write-Host "  Model: $($p.opus) (main) + $($p.haiku) (subagents)" -ForegroundColor DarkGray
Write-Host ""

$env:ANTHROPIC_BASE_URL = $p.url
$env:ANTHROPIC_AUTH_TOKEN = $p.key
$env:ANTHROPIC_MODEL = $p.opus
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = $p.opus
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = $p.sonnet
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $p.haiku
$env:CLAUDE_CODE_SUBAGENT_MODEL = $p.subagent
$env:CLAUDE_CODE_EFFORT_LEVEL = "max"
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue

try {
    & claude @Args
} finally {
    foreach ($v in @("ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL","ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL","CLAUDE_CODE_SUBAGENT_MODEL","CLAUDE_CODE_EFFORT_LEVEL")) {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
}
