[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('check', 'baseline', 'validate', 'v1-diagnose', 'v1-evaluate', 'v1-predict', 'v1-test')]
    [string]$Action = 'check',
    [Parameter(Position = 1)]
    [string]$SubmissionZip,
    [ValidateSet('dev', 'holdout')]
    [string]$Stage = 'dev',
    [string]$Selected = 'ensemble_nearest',
    [string]$OutputDirectory
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
$oldThreads = @{}
foreach ($threadKey in @('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')) {
    $oldThreads[$threadKey] = [Environment]::GetEnvironmentVariable($threadKey, 'Process')
}
try {
    $env:PYTHONUTF8 = '1'
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot 'local-output/matplotlib'
    New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
    $attackArgs = @((Join-Path $PSScriptRoot 'attack_workspace.py'), $Action)
    if ($Action -eq 'validate') { $attackArgs += $SubmissionZip }
    if ($Action.StartsWith('v1-')) {
        foreach ($threadKey in $oldThreads.Keys) { [Environment]::SetEnvironmentVariable($threadKey, '1', 'Process') }
        $scriptName = switch ($Action) {
            'v1-diagnose' { 'diagnose.py' }
            'v1-evaluate' { 'evaluate.py' }
            'v1-predict' { 'run.py' }
            'v1-test' { 'test_v1.py' }
        }
        $attackArgs = @('-B', (Join-Path $PSScriptRoot "v1/$scriptName"))
        if ($Action -eq 'v1-evaluate') { $attackArgs += @('--stage', $Stage, '--selected', $Selected) }
        if ($OutputDirectory) { $attackArgs += @('--output', $OutputDirectory) }
    }
    & $attackPython @attackArgs
    if ($LASTEXITCODE -ne 0) { throw "Attack workspace command failed (exit $LASTEXITCODE)." }
} finally {
    $env:PYTHONUTF8 = $oldUtf8
    $env:MPLCONFIGDIR = $oldMpl
    foreach ($threadKey in $oldThreads.Keys) {
        [Environment]::SetEnvironmentVariable($threadKey, $oldThreads[$threadKey], 'Process')
    }
}
