<#
.SYNOPSIS
    Kite MCP Server Failover Utility (PowerShell).
.DESCRIPTION
    Switches KITE_MCP_URL between hosted https://mcp.kite.trade/mcp and local http://localhost:8080/mcp,
    enforcing mandatory EXCLUDED_TOOLS=place_order,modify_order,cancel_order.
.PARAMETER Action
    One of: status, switch-local, switch-hosted, check-health
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet("status", "switch-local", "switch-hosted", "check-health")]
    [string]$Action = "status",

    [Parameter()]
    [string]$CustomUrl = "http://localhost:8080/mcp"
)

$HostUrl = "https://mcp.kite.trade/mcp"
$LocalUrl = $CustomUrl
$MandatoryExclusions = "place_order,modify_order,cancel_order"
$EnvPath = Join-Path $PSScriptRoot "..\.env"

function Get-ActiveUrl {
    if ($env:KITE_MCP_URL) {
        return $env:KITE_MCP_URL
    }
    if (Test-Path $EnvPath) {
        $lines = Get-Content $EnvPath
        foreach ($line in $lines) {
            if ($line -match "^KITE_MCP_URL\s*=\s*(.*)") {
                return $matches[1].Trim().Trim("'`"")
            }
        }
    }
    return $HostUrl
}

function Set-EnvUrl([string]$NewUrl) {
    if (-not (Test-Path $EnvPath)) {
        "KITE_MCP_URL=$NewUrl" | Out-File -FilePath $EnvPath -Encoding utf8
        Write-Host "[OK] Created .env with KITE_MCP_URL=$NewUrl" -ForegroundColor Green
        return
    }

    $lines = Get-Content $EnvPath
    $updated = $false
    $newLines = @()
    foreach ($line in $lines) {
        if ($line -match "^KITE_MCP_URL\s*=") {
            $newLines += "KITE_MCP_URL=$NewUrl"
            $updated = $true
        } else {
            $newLines += $line
        }
    }
    if (-not $updated) {
        $newLines += "KITE_MCP_URL=$NewUrl"
    }
    $newLines | Out-File -FilePath $EnvPath -Encoding utf8
    Write-Host "[OK] Updated .env with KITE_MCP_URL=$NewUrl" -ForegroundColor Green
}

switch ($Action) {
    "status" {
        $current = Get-ActiveUrl
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Zerodha Kite MCP Configuration Status" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Active KITE_MCP_URL : $current"
        Write-Host "Local Mandatory Env : EXCLUDED_TOOLS=$MandatoryExclusions"
        Write-Host "============================================================" -ForegroundColor Cyan
    }
    "switch-local" {
        Write-Host "[*] Switching to local kite-mcp-server ($LocalUrl)..." -ForegroundColor Yellow
        Set-EnvUrl $LocalUrl
        Write-Host "SAFETY NOTICE: Start local Go binary with:" -ForegroundColor Red
        Write-Host "`$env:EXCLUDED_TOOLS='$MandatoryExclusions'" -ForegroundColor Yellow
        Write-Host "`$env:PORT=8080" -ForegroundColor Yellow
        Write-Host ".\kite-mcp-server.exe" -ForegroundColor Yellow
    }
    "switch-hosted" {
        Write-Host "[*] Switching to hosted mcp.kite.trade ($HostUrl)..." -ForegroundColor Green
        Set-EnvUrl $HostUrl
    }
    "check-health" {
        $current = Get-ActiveUrl
        Write-Host "[*] Probing $current..."
        try {
            $response = Invoke-WebRequest -Uri $current -Method Get -TimeoutSec 3 -ErrorAction Stop
            Write-Host "[+] Server reachable: HTTP $($response.StatusCode)" -ForegroundColor Green
        } catch {
            if ($_.Exception.Response) {
                Write-Host "[+] Server reachable (HTTP $($_.Exception.Response.StatusCode))" -ForegroundColor Green
            } else {
                Write-Host "[-] Connection error: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }
}
