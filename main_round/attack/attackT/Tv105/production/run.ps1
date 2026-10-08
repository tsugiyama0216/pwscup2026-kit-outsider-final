[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$Tv105Python=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../../.venv/Scripts/python.exe'))
$env:PYTHONUTF8='1'
& $Tv105Python -B (Join-Path $PSScriptRoot 'infer.py') --output ([IO.Path]::GetFullPath($OutputDirectory))
if ($LASTEXITCODE -ne 0) {throw 'Tv105 inference failed'}
