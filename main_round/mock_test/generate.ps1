[CmdletBinding()]
param(
    [ValidateSet('dev', 'holdout')][string]$Split = 'dev',
    [ValidateRange(0, 4294967295)][long]$Seed = 20260927,
    [ValidateRange(1, 1000)][int]$Count = 3,
    [ValidateSet('copy', 'rounded', 'jitter', 'bootstrap', 'independent', 'strong_jitter', 'copula')]
    [string[]]$Style,
    [string]$Output
)

$ErrorActionPreference = 'Stop'
$evalPython = Join-Path $PSScriptRoot '../attack/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $evalPython)) { throw 'Run ../attack/setup.ps1 first.' }
$generateArgs = @('-B', '-m', 'local_eval.generate', '--split', $Split, '--seed', "$Seed", '--count', "$Count")
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
