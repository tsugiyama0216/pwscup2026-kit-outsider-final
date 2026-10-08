[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$pl2Python=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../.venv/Scripts/python.exe'))
$env:PYTHONUTF8='1'
& $pl2Python -B (Join-Path $PSScriptRoot 'infer.py') --output ([IO.Path]::GetFullPath($OutputDirectory))
if ($LASTEXITCODE -ne 0) {throw 'pl2 inference failed'}
