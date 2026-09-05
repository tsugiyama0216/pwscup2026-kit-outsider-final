#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $projectDir
$uvExe = Join-Path $repoRoot '.tools\uv\uv.exe'

if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "uv was not found. Run setup-local.ps1 from the repository root first: $uvExe"
}

# Keep the Python runtime and download cache inside the repository.
$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools\python'
$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache'

& $uvExe sync --project $projectDir --python 3.12 --frozen
if ($LASTEXITCODE -ne 0) {
    throw "OpenDP environment sync failed with exit code $LASTEXITCODE"
}

Write-Host ''
Write-Host 'OpenDP AIM environment is ready.'
& $uvExe run --project $projectDir --frozen python --version
if ($LASTEXITCODE -ne 0) {
    throw "Python check failed with exit code $LASTEXITCODE"
}

Write-Host 'Run the AIM smoke test with:'
Write-Host '  powershell -ExecutionPolicy Bypass -File .\opendp_aim\run-smoke.ps1'
