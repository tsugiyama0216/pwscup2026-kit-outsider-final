#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Verify', 'Reproduce', 'Test', 'Generate')]
    [string]$Mode = 'Verify',
    [string]$Python
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$modelDirectory = $PSScriptRoot
$repositoryDirectory = [IO.Path]::GetFullPath((Join-Path $modelDirectory '..\..\..\..\..'))
if (-not $Python) {
    $Python = Join-Path $repositoryDirectory 'prelim\workspace\starter\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Scoring Python not found: $Python"
}
$env:PYTHONUTF8 = '1'
$pythonArguments = @('-X', 'utf8')
if ($Mode -eq 'Test') {
    $pythonArguments += @('-m', 'unittest', 'discover', '-s', $modelDirectory, '-p', 'test_*.py', '-v')
} else {
    $pythonArguments += (Join-Path $modelDirectory 'generate.py')
    if ($Mode -eq 'Verify') { $pythonArguments += '--verify-only' }
    if ($Mode -eq 'Reproduce') { $pythonArguments += '--reproduce-check' }
}
& $Python @pythonArguments
if ($LASTEXITCODE -ne 0) {
    throw "R5 operation failed with exit code $LASTEXITCODE"
}
