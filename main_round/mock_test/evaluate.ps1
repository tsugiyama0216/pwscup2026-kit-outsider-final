[CmdletBinding()]
param(
    [ValidateSet('dev', 'holdout')][string]$Split = 'dev',
    [string[]]$Methods = @('official', 'multiview', 'constant', 'background_only'),
    [string[]]$Plugin,
    [string[]]$CaseDirectory,
    [string]$Output,
    [ValidateRange(0, 10000)][int]$NullRepetitions = 100
)
$ErrorActionPreference = 'Stop'
$evalPython = Join-Path $PSScriptRoot '../attack/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $evalPython)) { throw 'Run ../attack/setup.ps1 first.' }
if ($CaseDirectory) {
    if ($PSBoundParameters.ContainsKey('Split') -or $PSBoundParameters.ContainsKey('Methods')) {
        throw 'CaseDirectory uses the saved splits and compares official plus Plugin; omit Split and Methods.'
    }
    $evalArgs = @('-B', '-m', 'local_eval.evaluate_case', '--null-reps', "$NullRepetitions")
    foreach ($casePath in $CaseDirectory) { $evalArgs += @('--case', (Resolve-Path -LiteralPath $casePath).Path) }
} else {
    $evalArgs = @('-B', '-m', 'local_eval', '--split', $Split, '--methods', ($Methods -join ','), '--null-reps', "$NullRepetitions")
}
foreach ($pluginPath in $Plugin) { $evalArgs += @('--plugin', (Resolve-Path -LiteralPath $pluginPath).Path) }
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
