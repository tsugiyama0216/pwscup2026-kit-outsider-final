#requires -Version 5.1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $projectDir
$uvExe = Join-Path $repoRoot '.tools\uv\uv.exe'
$starterDir = Join-Path $repoRoot 'starter'
$distDir = Join-Path $repoRoot 'participant_data'
$outputDir = Join-Path $repoRoot 'local-output\opendp_aim\v5_uspec'
$parentOutput = Join-Path $repoRoot 'local-output\opendp_aim\C.csv'
$parentReport = Join-Path $repoRoot 'local-output\opendp_aim\privacy_report.json'
$config = Join-Path $projectDir 'config\pwscup_aim_v5.json'
$candidate = Join-Path $outputDir 'C.csv'
$report = Join-Path $outputDir 'postprocess_report.json'
$score = Join-Path $outputDir 'score.json'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools\python'
$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache'
$env:VIRTUAL_ENV = $null

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(Mandatory)] [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

Write-Host '[1/3] Reapplying the public outcome-donor postprocess...'
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $projectDir, '--frozen',
    'python', (Join-Path $projectDir 'scripts\repostprocess_public_outcomes.py'),
    '--input', $parentOutput,
    '--background', (Join-Path $distDir 'A_bg.csv'),
    '--schema', (Join-Path $distDir 'schema.json'),
    '--config', $config,
    '--parent-privacy-report', $parentReport,
    '--output', $candidate,
    '--report', $report
)

Write-Host '[2/3] Validating the standalone C.csv...'
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $starterDir, '--frozen',
    'python', (Join-Path $projectDir 'scripts\validate_c_csv.py'),
    $candidate, '--dist', $distDir
)

Write-Host '[3/3] Running the current local utility score...'
$jsonLines = & $uvExe @(
    'run', '--project', $starterDir, '--frozen',
    'python', '-m', 'pwscup2026.kit.selfscore',
    $candidate, '--dist', $distDir, '--json'
)
if ($LASTEXITCODE -ne 0) {
    throw "Self-score failed with exit code $LASTEXITCODE"
}
[System.IO.File]::WriteAllText(
    $score,
    ($jsonLines -join [Environment]::NewLine) + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)
$scoreObject = ($jsonLines -join [Environment]::NewLine) | ConvertFrom-Json

Write-Host ''
Write-Host 'v5 U_spec candidate completed.'
Write-Host ("  U:       {0:F4}" -f [double]$scoreObject.utility)
Write-Host ("  U_spec:  {0:F4}" -f [double]$scoreObject.facets.U_spec)
Write-Host "  C.csv:   $candidate"
Write-Host "  Score:   $score"
Write-Host "  Report:  $report"
