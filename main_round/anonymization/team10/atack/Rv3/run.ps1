param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$rv3Root = (Resolve-Path "$PSScriptRoot/../../../../..").Path
$env:PYTHONUTF8 = '1'
$rv3Args = @('-B', "$PSScriptRoot/run.py")
if ($OutputDirectory) { $rv3Args += @('--output', $OutputDirectory) }
& "$rv3Root/main_round/attack/.venv/Scripts/python.exe" @rv3Args
if ($LASTEXITCODE -ne 0) { throw 'Rv3 generation or validation failed' }
