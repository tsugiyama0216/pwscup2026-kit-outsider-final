[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$rv5Python=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'))
$env:PYTHONUTF8='1'
& $rv5Python -B (Join-Path $PSScriptRoot 'infer.py') --output ([IO.Path]::GetFullPath($OutputDirectory))
if ($LASTEXITCODE -ne 0) {throw 'Rv5 inference failed'}
