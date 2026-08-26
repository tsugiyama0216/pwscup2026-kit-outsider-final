#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $projectDir
$uvExe = Join-Path $repoRoot '.tools\uv\uv.exe'
$smokeScript = Join-Path $projectDir 'scripts\smoke_test.py'

if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "uv was not found. Run setup-local.ps1 from the repository root first: $uvExe"
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools\python'
$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache'

& $uvExe run --project $projectDir --frozen python $smokeScript
if ($LASTEXITCODE -ne 0) {
    throw "AIM smoke test failed with exit code $LASTEXITCODE"
}
