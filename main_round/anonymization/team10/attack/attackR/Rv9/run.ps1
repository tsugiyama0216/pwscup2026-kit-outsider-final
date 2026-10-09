[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$seriesPython=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../../../attack/.venv/Scripts/python.exe'))
$env:PYTHONUTF8='1'
& $seriesPython -B (Join-Path $PSScriptRoot '../rv6_series/infer.py') --model 'Rv9' --output ([IO.Path]::GetFullPath($OutputDirectory))
if ($LASTEXITCODE -ne 0) {throw 'Rv9 inference failed'}
