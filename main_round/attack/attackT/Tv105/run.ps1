[CmdletBinding()]
param([ValidateSet('development','confirmation')][string]$Phase='development',[int]$Jobs=3)
$ErrorActionPreference='Stop'
$Tv105Python=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'))
$env:PYTHONUTF8='1'
& $Tv105Python -B (Join-Path $PSScriptRoot 'experiment.py') $Phase --jobs $Jobs
if ($LASTEXITCODE -ne 0) {throw 'Tv105 experiment failed'}
