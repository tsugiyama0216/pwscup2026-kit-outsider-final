[CmdletBinding()]
param([ValidateSet('development','confirmation')][string]$Phase='development',[int]$Jobs=3)
$ErrorActionPreference='Stop'
$pl2Python=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../.venv/Scripts/python.exe'))
$env:PYTHONUTF8='1'
& $pl2Python -B (Join-Path $PSScriptRoot 'experiment.py') $Phase --jobs $Jobs
if ($LASTEXITCODE -ne 0) {throw 'pl2 experiment failed'}
