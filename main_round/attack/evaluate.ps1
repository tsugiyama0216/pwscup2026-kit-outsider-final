[CmdletBinding()]
param(
    [ValidateSet('dev', 'holdout')][string]$Split = 'dev',
    [string[]]$Methods = @('official', 'multiview', 'constant', 'background_only'),
    [string]$Plugin,
    [string]$Output,
    [ValidateRange(0, 10000)][int]$NullRepetitions = 100
)
$ErrorActionPreference = 'Stop'
$evalPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $evalPython)) { throw 'Run setup.ps1 first.' }
$evalArgs = @('-m', 'local_eval', '--split', $Split, '--methods', ($Methods -join ','), '--null-reps', "$NullRepetitions")
if ($Plugin) { $evalArgs += @('--plugin', (Resolve-Path -LiteralPath $Plugin).Path) }
if ($Output) { $evalArgs += @('--output', [System.IO.Path]::GetFullPath($Output)) }
$evalOldUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    Push-Location $PSScriptRoot
    try {
        & $evalPython @evalArgs
        if ($LASTEXITCODE -ne 0) { throw "Local evaluation failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
} finally { $env:PYTHONUTF8 = $evalOldUtf8 }
