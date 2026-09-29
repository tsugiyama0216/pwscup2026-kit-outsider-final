[CmdletBinding()]
param(
    [ValidateSet('dev', 'holdout')][string]$Split = 'dev',
    [ValidateRange(0, 4294967295)][long]$Seed = 20260927,
    [ValidateSet('copy', 'rounded', 'jitter', 'bootstrap', 'independent', 'strong_jitter', 'copula')]
    [string[]]$Style,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$evalPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $evalPython)) { throw 'Run setup.ps1 first.' }
$generateArgs = @('-m', 'local_eval.generate', '--split', $Split, '--seed', "$Seed")
foreach ($name in $Style) { $generateArgs += @('--style', $name) }
if ($Output) { $generateArgs += @('--output', [System.IO.Path]::GetFullPath($Output)) }
$oldUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    Push-Location $PSScriptRoot
    try {
        & $evalPython @generateArgs
        if ($LASTEXITCODE -ne 0) { throw "Mock generation failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
} finally {
    $env:PYTHONUTF8 = $oldUtf8
}
