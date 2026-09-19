#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Verify','Reproduce','Generate','Test')]
    [string]$Mode = 'Verify',
    [string]$Python
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repositoryDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..\..'))
if (-not $Python) { $Python = Join-Path $repositoryDirectory 'prelim\workspace\starter\.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Scoring Python not found' }
$pythonArguments = @('-X','utf8')
if ($Mode -eq 'Test') {
    $pythonArguments += @('-m','unittest','discover','-s',$PSScriptRoot,'-p','test_*.py','-v')
} else {
    $pythonArguments += (Join-Path $PSScriptRoot 'generate.py')
    if ($Mode -eq 'Verify') { $pythonArguments += '--verify-only' }
    if ($Mode -eq 'Reproduce') { $pythonArguments += '--reproduce-check' }
}
& $Python @pythonArguments
if ($LASTEXITCODE -ne 0) { throw "R11 operation failed: $LASTEXITCODE" }
