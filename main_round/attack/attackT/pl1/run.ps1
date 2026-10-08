param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$pl1Python = Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'
$pl1Args = @('-B', (Join-Path $PSScriptRoot 'run.py'))
if ($OutputDirectory) { $pl1Args += @('--output', $OutputDirectory) }
& $pl1Python @pl1Args
if ($LASTEXITCODE -ne 0) { throw 'pl1 generation or validation failed' }
