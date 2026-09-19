#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Verify','Reproduce','Generate')]
    [string]$Mode = 'Verify',
    [string]$Python
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repositoryDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\..\..'))
if (-not $Python) {
    $Python = Join-Path $repositoryDirectory 'prelim\workspace\starter\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Scoring Python not found' }
$pythonArguments = @('-X','utf8',(Join-Path $PSScriptRoot 'r10.py'))
if ($Mode -eq 'Verify') { $pythonArguments += '--verify' }
if ($Mode -eq 'Reproduce') { $pythonArguments += '--reproduce' }
& $Python @pythonArguments
if ($LASTEXITCODE -ne 0) { throw "R10 operation failed: $LASTEXITCODE" }
