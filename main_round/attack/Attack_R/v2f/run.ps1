[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$shared = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
& (Join-Path $shared 'prepare_attack_m.ps1')
$python = Join-Path $shared '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run main_round/attack/setup.ps1 first.' }
$env:PYTHONUTF8 = '1'
& $python -B (Join-Path $PSScriptRoot 'run.py')
if ($LASTEXITCODE -ne 0) { throw 'R v2f generation failed; inspect local diagnostics.' }
