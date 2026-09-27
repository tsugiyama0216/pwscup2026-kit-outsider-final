[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('check', 'baseline', 'validate')]
    [string]$Action = 'check',
    [Parameter(Position = 1)]
    [string]$SubmissionZip
)

$ErrorActionPreference = 'Stop'
$attackPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $attackPython)) {
    throw 'Run setup.ps1 in this directory first.'
}
if ($Action -eq 'validate' -and -not $SubmissionZip) {
    throw 'Usage: ./attack.ps1 validate <submission.zip>'
}
$oldUtf8 = $env:PYTHONUTF8
$oldMpl = $env:MPLCONFIGDIR
try {
    $env:PYTHONUTF8 = '1'
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'local-output/matplotlib'
    New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
    $attackArgs = @((Join-Path $PSScriptRoot 'attack_workspace.py'), $Action)
    if ($Action -eq 'validate') { $attackArgs += $SubmissionZip }
    & $attackPython @attackArgs
    if ($LASTEXITCODE -ne 0) { throw "Attack workspace command failed (exit $LASTEXITCODE)." }
} finally {
    $env:PYTHONUTF8 = $oldUtf8
    $env:MPLCONFIGDIR = $oldMpl
}
