<#
.SYNOPSIS
    deepclaude — Use Claude Code with DeepSeek V4 Pro or other cheap backends.

.USAGE
    deepclaude                      # DeepSeek V4 Pro (default)
    deepclaude --backend or         # OpenRouter (cheapest)
    deepclaude --backend fw         # Fireworks AI (fastest)
    deepclaude --backend sf -w 1h   # sference background (Kimi-K2.6)
    deepclaude --backend anthropic  # Normal Claude Code
    deepclaude --remote             # Remote control + DeepSeek (browser URL)
    deepclaude --status             # Show keys and backends
    deepclaude --cost               # Pricing comparison
    deepclaude --benchmark          # Latency test
#>

param(
    [Alias("b")]
    [string]$Backend,
    [Alias("w")]
    [string]$Window,
    [Alias("r")]
    [switch]$Remote,
    [Alias("s")]
    [string]$Switch,
    [switch]$Status,
    [switch]$Cost,
    [switch]$Benchmark,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if (-not $Backend -and -not $Status -and -not $Cost -and -not $Benchmark -and -not $Help -and -not $Switch) {
    $Backend = if ($env:CHEAPCLAUDE_DEFAULT_BACKEND) { $env:CHEAPCLAUDE_DEFAULT_BACKEND } else { "ds" }
}

if (-not $Window) {
    $Window = if ($env:SFERENCE_COMPLETION_WINDOW) { $env:SFERENCE_COMPLETION_WINDOW } else { "1h" }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SferenceModel = if ($env:SFERENCE_MODEL) { $env:SFERENCE_MODEL } else { "moonshotai/Kimi-K2.6" }
$SferenceUrl = if ($env:SFERENCE_BASE_URL) { $env:SFERENCE_BASE_URL } else { "https://api.sference.com" }

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
$SferenceKey = if ($env:SFERENCE_API_KEY) { $env:SFERENCE_API_KEY } else {
    [Environment]::GetEnvironmentVariable("SFERENCE_API_KEY", "User")
}
$BgProviderKey = if ($env:BG_PROVIDER_API_KEY) { $env:BG_PROVIDER_API_KEY } else {
    [Environment]::GetEnvironmentVariable("BG_PROVIDER_API_KEY", "User")
}

$Providers = @{
    ds = @{
        name = "DeepSeek (direct)"
        url = "https://api.deepseek.com/anthropic"
        key = $DeepSeekKey; keyName = "DEEPSEEK_API_KEY"
        backendName = "deepseek"
        type = "anthropic-compatible"
        opus = "deepseek-v4-pro"; sonnet = "deepseek-v4-pro"
        haiku = "deepseek-v4-flash"; subagent = "deepseek-v4-flash"
    }
    or = @{
        name = "OpenRouter"
        url = "https://openrouter.ai/api"
        key = $OpenRouterKey; keyName = "OPENROUTER_API_KEY"
        backendName = "openrouter"
        type = "anthropic-compatible"
        opus = "deepseek/deepseek-v4-pro"; sonnet = "deepseek/deepseek-v4-pro"
        haiku = "deepseek/deepseek-v4-pro"; subagent = "deepseek/deepseek-v4-pro"
    }
    fw = @{
        name = "Fireworks AI"
        url = "https://api.fireworks.ai/inference"
        key = $FireworksKey; keyName = "FIREWORKS_API_KEY"
        backendName = "fireworks"
        type = "anthropic-compatible"
        opus = "accounts/fireworks/models/deepseek-v4-pro"
        sonnet = "accounts/fireworks/models/deepseek-v4-pro"
        haiku = "accounts/fireworks/models/deepseek-v4-pro"
        subagent = "accounts/fireworks/models/deepseek-v4-pro"
    }
    sf = @{
        name = "sference (background)"
        url = $SferenceUrl
        key = $SferenceKey; keyName = "SFERENCE_API_KEY"
        backendName = "sference"
        type = "responses-bg"
        opus = $SferenceModel; sonnet = $SferenceModel
        haiku = $SferenceModel; subagent = $SferenceModel
    }
}
$Providers["sference"] = $Providers["sf"]

if ($BgProviderKey -and $env:BG_PROVIDER_URL -and $env:BG_PROVIDER_MODEL) {
    $Providers["bg"] = @{
        name = "Custom background provider"
        url = $env:BG_PROVIDER_URL
        key = $BgProviderKey; keyName = "BG_PROVIDER_API_KEY"
        backendName = "bg"
        type = "responses-bg"
        opus = $env:BG_PROVIDER_MODEL; sonnet = $env:BG_PROVIDER_MODEL
        haiku = $env:BG_PROVIDER_MODEL; subagent = $env:BG_PROVIDER_MODEL
    }
}

function Get-KeyDisplay($k) {
    if (-not $k) { return "MISSING" }
    return "set (****" + $k.Substring($k.Length - [Math]::Min(4, $k.Length)) + ")"
}

function Start-DeepClaudeProxy($Provider) {
    Write-Host "`n  Starting model proxy for $($Provider.name)..." -ForegroundColor Cyan

    $proxyScript = Join-Path $ScriptDir "proxy\start-proxy.js"
    $portFile = Join-Path $env:TEMP "deepclaude-proxy-port.txt"
    if (Test-Path $portFile) { Remove-Item $portFile -Force }

    $env:SFERENCE_COMPLETION_WINDOW = $Window
    $env:CHEAPCLAUDE_DEFAULT_MODE = $Provider.backendName

    $proxyProc = Start-Process -FilePath "node" `
        -ArgumentList @($proxyScript, $Provider.url, $Provider.key, $Provider.backendName, "--window", $Window) `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $portFile

    $tries = 0
    while ($tries -lt 30) {
        Start-Sleep -Milliseconds 200
        $tries++
        if (Test-Path $portFile) {
            $content = Get-Content $portFile -ErrorAction SilentlyContinue
            if ($content) { break }
        }
    }

    $proxyPort = (Get-Content $portFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    Remove-Item $portFile -ErrorAction SilentlyContinue

    if (-not $proxyPort) {
        Write-Host "ERROR: Proxy failed to start" -ForegroundColor Red
        if ($proxyProc -and -not $proxyProc.HasExited) { Stop-Process -Id $proxyProc.Id -Force }
        exit 1
    }

    Write-Host "  Proxy on :$proxyPort -> $($Provider.url)" -ForegroundColor DarkGray
    if ($Provider.type -eq "responses-bg") {
        Write-Host "  Background mode: window=$Window model=$($Provider.opus)" -ForegroundColor DarkGray
    }

    return @{ Port = $proxyPort; Process = $proxyProc }
}

function Set-ModelEnv($Provider) {
    $env:ANTHROPIC_DEFAULT_OPUS_MODEL = $Provider.opus
    $env:ANTHROPIC_DEFAULT_SONNET_MODEL = $Provider.sonnet
    $env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $Provider.haiku
    $env:CLAUDE_CODE_SUBAGENT_MODEL = $Provider.subagent
    $env:CLAUDE_CODE_EFFORT_LEVEL = "max"
}

function Clear-AnthropicEnv {
    foreach ($v in @("ANTHROPIC_BASE_URL","ANTHROPIC_AUTH_TOKEN","ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL","ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL","CLAUDE_CODE_EFFORT_LEVEL","ANTHROPIC_API_KEY")) {
        Remove-Item "Env:$v" -ErrorAction SilentlyContinue
    }
}

# --- Status ---
if ($Status) {
    Write-Host "`n  deepclaude - Backend Status" -ForegroundColor Cyan
    Write-Host "  ============================" -ForegroundColor DarkGray
    Write-Host "`n  Keys:" -ForegroundColor Yellow
    Write-Host "    DEEPSEEK_API_KEY:    $(Get-KeyDisplay $DeepSeekKey)"
    Write-Host "    OPENROUTER_API_KEY:  $(Get-KeyDisplay $OpenRouterKey)"
    Write-Host "    FIREWORKS_API_KEY:   $(Get-KeyDisplay $FireworksKey)"
    Write-Host "    SFERENCE_API_KEY:    $(Get-KeyDisplay $SferenceKey)"
    Write-Host "    BG_PROVIDER_API_KEY: $(Get-KeyDisplay $BgProviderKey)"
    Write-Host "`n  Backends:" -ForegroundColor Yellow
    Write-Host "    deepclaude              # DeepSeek V4 Pro (default)"
    Write-Host "    deepclaude -b or        # OpenRouter"
    Write-Host "    deepclaude -b fw        # Fireworks AI"
    Write-Host "    deepclaude -b sf -w 1h  # sference background"
    Write-Host "    deepclaude -b anthropic # Normal Claude Code"
    Write-Host ""
    exit 0
}

# --- Cost ---
if ($Cost) {
    Write-Host "`n  Provider Pricing" -ForegroundColor Cyan
    Write-Host "  ================" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "  Provider        Input/M    Output/M   Notes" -ForegroundColor Yellow
    Write-Host "  ----------      --------   --------   -----"
    Write-Host "  DeepSeek        `$0.44      `$0.87      sync + cache" -ForegroundColor Green
    Write-Host "  OpenRouter      `$0.44      `$0.87      sync"
    Write-Host "  Fireworks       `$1.74      `$3.48      sync"
    Write-Host "  sference        (varies)   (varies)   async bg"
    Write-Host "  Anthropic       `$3.00      `$15.00     sync"
    Write-Host ""
    exit 0
}

# --- Help ---
if ($Help) {
    Write-Host "deepclaude - Claude Code with cheap backends"
    Write-Host ""
    Write-Host "Usage: deepclaude [-b backend] [-w window] [--status] [--cost] [--benchmark]"
    Write-Host ""
    Write-Host "  -b, --backend   ds (default), or, fw, sf, bg, anthropic"
    Write-Host "  -w, --window    15m|1h|6h|24h|48h (background providers)"
    Write-Host "  --status        Show keys and backends"
    Write-Host "  --cost          Pricing comparison"
    Write-Host "  --benchmark     Latency test"
    exit 0
}

# --- Switch ---
if ($Switch) {
    $map = @{ ds="deepseek"; or="openrouter"; fw="fireworks"; sf="sference"; sference="sference"; bg="bg"; anthropic="anthropic" }
    $backend = $map[$Switch]
    if (-not $backend) { Write-Host "ERROR: Unknown backend '$Switch'" -ForegroundColor Red; exit 1 }
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:3200/_proxy/mode" -Method Post -Body "backend=$backend"
        Write-Host "  $($resp | ConvertTo-Json -Compress)"
        if ($backend -in @("sference","bg")) {
            $wresp = Invoke-RestMethod -Uri "http://127.0.0.1:3200/_proxy/window" -Method Post -Body "window=$Window"
            Write-Host "  $($wresp | ConvertTo-Json -Compress)"
        }
    } catch {
        Write-Host "  Proxy not running. Start with: deepclaude" -ForegroundColor Red
        exit 1
    }
    exit 0
}

# --- Benchmark ---
if ($Benchmark) {
    Write-Host "`n  Latency Benchmark" -ForegroundColor Cyan
    Write-Host "  =================" -ForegroundColor DarkGray
    foreach ($id in @("ds","or","fw","sf")) {
        $p = $Providers[$id]
        if (-not $p) { continue }
        Write-Host "  $($p.name)..." -NoNewline
        if (-not $p.key) { Write-Host " SKIP (no key)" -ForegroundColor DarkGray; continue }
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            if ($p.type -eq "responses-bg") {
                $body = @{
                    model = $p.opus
                    input = "Reply: ok"
                    background = $true
                    metadata = @{ completion_window = "1h" }
                } | ConvertTo-Json -Depth 5
                $headers = @{ "X-API-Key" = $p.key; "content-type" = "application/json" }
                $null = Invoke-RestMethod -Uri "$($p.url)/v1/responses" -Method POST -Headers $headers -Body $body -TimeoutSec 30
                $sw.Stop()
                Write-Host " submit OK ($($sw.ElapsedMilliseconds)ms, async)" -ForegroundColor Green
            } else {
                $useBearer = $id -in @("or","fw")
                $headers = if ($useBearer) {
                    @{ "Authorization" = "Bearer $($p.key)"; "content-type" = "application/json"; "anthropic-version" = "2023-06-01" }
                } else {
                    @{ "x-api-key" = $p.key; "content-type" = "application/json"; "anthropic-version" = "2023-06-01" }
                }
                $body = @{ model = $p.opus; max_tokens = 32; messages = @(@{ role = "user"; content = "Reply: ok" }) } | ConvertTo-Json -Depth 5
                $null = Invoke-RestMethod -Uri "$($p.url)/v1/messages" -Method POST -Headers $headers -Body $body -TimeoutSec 30
                $sw.Stop()
                Write-Host " OK ($($sw.ElapsedMilliseconds)ms)" -ForegroundColor Green
            }
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
    if ($Backend -eq "anthropic") {
        Write-Host "`n  Launching remote control (Anthropic)...`n" -ForegroundColor Cyan
        Clear-AnthropicEnv
        & claude remote-control @Args
        exit 0
    }

    $p = $Providers[$Backend]
    if (-not $p) { Write-Host "ERROR: Unknown backend '$Backend'" -ForegroundColor Red; exit 1 }
    if (-not $p.key) { Write-Host "ERROR: $($p.keyName) not set" -ForegroundColor Red; exit 1 }

    $proxy = Start-DeepClaudeProxy $p
    Write-Host "  Launching remote control via $($p.name)...`n" -ForegroundColor Cyan

    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$($proxy.Port)"
    Set-ModelEnv $p
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

# --- Launch ---
if ($Backend -eq "anthropic") {
    Clear-AnthropicEnv
    Write-Host "`n  Launching Claude Code (normal Anthropic)...`n" -ForegroundColor Cyan
    & claude @Args
    exit 0
}

$p = $Providers[$Backend]
if (-not $p) { Write-Host "ERROR: Unknown backend '$Backend'. Use: ds, or, fw, sf, bg, anthropic" -ForegroundColor Red; exit 1 }
if (-not $p.key) { Write-Host "ERROR: $($p.keyName) not set" -ForegroundColor Red; exit 1 }

if ($p.type -eq "responses-bg") {
    $proxy = Start-DeepClaudeProxy $p
    Write-Host "  Launching Claude Code via $($p.name)...`n" -ForegroundColor Cyan
    $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$($proxy.Port)"
    Set-ModelEnv $p
    Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:ANTHROPIC_AUTH_TOKEN -ErrorAction SilentlyContinue
    try {
        & claude @Args
    } finally {
        if ($proxy.Process -and -not $proxy.Process.HasExited) {
            Stop-Process -Id $proxy.Process.Id -Force -ErrorAction SilentlyContinue
            Write-Host "  Proxy stopped." -ForegroundColor DarkGray
        }
    }
    exit 0
}

Write-Host "`n  Launching Claude Code via $($p.name)..." -ForegroundColor Cyan
Write-Host "  Endpoint: $($p.url)" -ForegroundColor DarkGray
Write-Host "  Model: $($p.opus) (main) + $($p.haiku) (subagents)" -ForegroundColor DarkGray
Write-Host ""

$env:ANTHROPIC_BASE_URL = $p.url
$env:ANTHROPIC_AUTH_TOKEN = $p.key
Set-ModelEnv $p
Remove-Item Env:ANTHROPIC_API_KEY -ErrorAction SilentlyContinue

& claude @Args

Clear-AnthropicEnv
