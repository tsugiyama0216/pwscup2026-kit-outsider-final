#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Generate', 'PackageExisting', 'Score')]
    [string]$Mode = 'PackageExisting',
    [string]$AimPython,
    [string]$ScorePython,
    [string]$KitRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$modelDir = $PSScriptRoot
$dataDir = [IO.Path]::GetFullPath((Join-Path $modelDir '..\..\participant_data'))
$workspaceDir = [IO.Path]::GetFullPath((Join-Path $modelDir '..\..\..\..\..\..'))
if (-not $AimPython) { $AimPython = Join-Path $workspaceDir 'pwscup2026-kit\opendp_aim\.venv\Scripts\python.exe' }
if (-not $ScorePython) { $ScorePython = Join-Path $workspaceDir 'pwscup2026-kit\starter\.venv\Scripts\python.exe' }
if (-not $KitRoot) { $KitRoot = Join-Path $modelDir 'local-output\official-kit\pwscup2026-kit-3dd71bd3436f2abfdbb64da97e860f22729ea054' }
$env:PYTHONUTF8 = '1'

function Invoke-Python {
    param([string]$Python, [string[]]$PythonArgs)
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
    & $Python @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw "Python operation failed (exit=$LASTEXITCODE)." }
}

if ($Mode -in @('Preflight', 'Generate')) {
    $configPath = Join-Path $modelDir 'config\aim_main_team10_S1.json'
    Invoke-Python $ScorePython @(
        (Join-Path $modelDir 'scripts\build_public_config.py'),
        '--base-config', (Join-Path $modelDir 'config\prelim_submitted_v5_base.json'),
        '--background', (Join-Path $dataDir 'A_bg.csv'), '--schema', (Join-Path $dataDir 'schema.json'),
        '--output', $configPath, '--config-version', 'pwscup-aim-main-team10-S1-prelim-v5-method'
    )
    $generateArgs = @(
        (Join-Path $modelDir 'scripts\generate_pwscup.py'),
        '--input', (Join-Path $dataDir 'B.csv'), '--background', (Join-Path $dataDir 'A_bg.csv'),
        '--schema', (Join-Path $dataDir 'schema.json'), '--config', $configPath,
        '--output', (Join-Path $modelDir 'C.csv'), '--privacy-report', (Join-Path $modelDir 'privacy_report.json')
    )
    Invoke-Python $AimPython ($generateArgs + '--validate-only')
    if ($Mode -eq 'Preflight') { return }
    Invoke-Python $AimPython ($generateArgs + @('--release-ledger', (Join-Path $modelDir 'release_ledger.json')))
}
if ($Mode -in @('Generate', 'PackageExisting')) {
    Invoke-Python $ScorePython @((Join-Path $modelDir 'scripts\package_submission.py'), '--kit-root', $KitRoot)
}
if ($Mode -in @('Generate', 'Score')) {
    Invoke-Python $ScorePython @((Join-Path $modelDir 'scripts\score_submission.py'), '--kit-root', $KitRoot)
}
