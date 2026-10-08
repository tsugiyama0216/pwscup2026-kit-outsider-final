[CmdletBinding()]
param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$rv4Root = $PSScriptRoot
while (-not (Test-Path -LiteralPath (Join-Path $rv4Root 'main_round/attack/Attack_R/v2/model.py'))) {
    $rv4Parent = Split-Path -Parent $rv4Root
    if (-not $rv4Parent -or $rv4Parent -eq $rv4Root) { throw 'Run from the saved repository copy.' }
    $rv4Root = $rv4Parent
}
& (Join-Path $rv4Root 'main_round/attack/prepare_attack_m.ps1')
$rv4Args = @('-B', (Join-Path $PSScriptRoot 'run.py'))
if ($OutputDirectory) { $rv4Args += @('--output', [IO.Path]::GetFullPath($OutputDirectory)) }
$env:PYTHONUTF8 = '1'
& (Join-Path $rv4Root 'main_round/attack/.venv/Scripts/python.exe') @rv4Args
if ($LASTEXITCODE -ne 0) { throw "Rv4 failed: $LASTEXITCODE" }
