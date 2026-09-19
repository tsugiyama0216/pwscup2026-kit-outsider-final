#requires -Version 5.1
[CmdletBinding()]
param([ValidateSet('Verify','Generate','Test','Research')][string]$Mode='Verify')
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repositoryDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..\..'))
$pythonExecutable = Join-Path $repositoryDirectory 'prelim\workspace\starter\.venv\Scripts\python.exe'
$pythonArguments = @('-X','utf8')
if ($Mode -eq 'Test') {
    $pythonArguments += @('-m','unittest','discover','-s',$PSScriptRoot,'-p','test_*.py','-v')
} elseif ($Mode -eq 'Research') {
    $pythonArguments += (Join-Path $PSScriptRoot 'research.py')
} else {
    $pythonArguments += (Join-Path $PSScriptRoot 'generate.py')
    if ($Mode -eq 'Verify') { $pythonArguments += '--verify-only' }
}
& $pythonExecutable @pythonArguments
if ($LASTEXITCODE -ne 0) { throw "R12 operation failed: $LASTEXITCODE" }
