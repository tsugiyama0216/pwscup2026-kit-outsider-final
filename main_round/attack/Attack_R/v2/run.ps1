param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$python = Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot 'run.py'
if ($OutputDirectory) { & $python -B $runner --output $OutputDirectory }
else { & $python -B $runner }
if ($LASTEXITCODE -ne 0) { throw 'R v2 generation or validation failed' }
