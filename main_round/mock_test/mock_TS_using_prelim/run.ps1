[CmdletBinding()]
param(
    [ValidateSet('Verify','Replay','Export','Attack','Score','Run')]
    [string]$Action = 'Replay',
    [string]$Plugin,
    [string]$Output,
    [string]$BundleDirectory,
    [string]$PredictionDirectory,
    [int]$AttackerId = 24,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $Python = Join-Path $PSScriptRoot '..\..\attack\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python環境がありません。main_round/attack/setup.ps1で準備するか、-Pythonで実行ファイルを指定してください。'
}
$runArguments = @('-I', '-B', (Join-Path $PSScriptRoot 'cli.py'), $Action, '--attacker-id', "$AttackerId")
if ($Plugin) { $runArguments += @('--plugin', $Plugin) }
if ($Output) { $runArguments += @('--output', $Output) }
if ($BundleDirectory) { $runArguments += @('--bundle', $BundleDirectory) }
if ($PredictionDirectory) { $runArguments += @('--predictions', $PredictionDirectory) }
$savedEncoding = $env:PYTHONIOENCODING
try {
    $env:PYTHONIOENCODING = 'utf-8'
    & $Python @runArguments
    if ($LASTEXITCODE -ne 0) { throw "予備戦評価が失敗しました (exit=$LASTEXITCODE)" }
} finally {
    $env:PYTHONIOENCODING = $savedEncoding
}
