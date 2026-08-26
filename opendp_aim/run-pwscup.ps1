#requires -Version 5.1

[CmdletBinding()]
param(
    [string]$Config,
    [string]$Token,
    [switch]$SkipGenerate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $projectDir
$uvExe = Join-Path $repoRoot '.tools\uv\uv.exe'
$starterDir = Join-Path $repoRoot 'starter'
$distDir = Join-Path $repoRoot 'participant_data'
$outputDir = Join-Path $repoRoot 'local-output\opendp_aim'
$generator = Join-Path $projectDir 'scripts\generate_pwscup.py'

if (-not $Config) {
    $Config = Join-Path $projectDir 'config\pwscup_aim_v4.json'
}

$cPath = Join-Path $outputDir 'C.csv'
$privacyReport = Join-Path $outputDir 'privacy_report.json'
$scoreJson = Join-Path $outputDir 'score.json'
$tokenPath = Join-Path $outputDir 'token.txt'
$defenseZip = Join-Path $outputDir 'defense_submission.zip'

if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "uv was not found. Run setup-local.ps1 first: $uvExe"
}
if (-not (Test-Path -LiteralPath $Config)) {
    throw "AIM config was not found: $Config"
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools\python'
$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache'
# Ignore an unrelated activated venv in the parent shell.
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

function New-FlatZip {
    param(
        [Parameter(Mandatory)] [string[]]$Files,
        [Parameter(Mandatory)] [string]$Destination
    )

    Compress-Archive `
        -LiteralPath $Files `
        -DestinationPath $Destination `
        -CompressionLevel Optimal `
        -Force
}

if (-not (Test-Path -LiteralPath (Join-Path $projectDir '.venv'))) {
    Write-Host '[setup] Preparing the OpenDP environment...'
    & (Join-Path $projectDir 'setup.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw "OpenDP setup failed with exit code $LASTEXITCODE"
    }
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

if (-not $SkipGenerate) {
    Write-Host '[1/4] Generating C.csv with one OpenDP AIM release...'
    Invoke-NativeCommand -FilePath $uvExe -Arguments @(
        'run', '--project', $projectDir, '--frozen',
        'python', $generator,
        '--input', (Join-Path $distDir 'B_practice.csv'),
        '--background', (Join-Path $distDir 'A_bg.csv'),
        '--schema', (Join-Path $distDir 'schema.json'),
        '--config', $Config,
        '--output', $cPath,
        '--privacy-report', $privacyReport
    )
} else {
    Write-Host '[1/4] Reusing the existing C.csv (-SkipGenerate)...'
    if (-not (Test-Path -LiteralPath $cPath)) {
        throw "C.csv was not found: $cPath"
    }
    if (-not (Test-Path -LiteralPath $privacyReport)) {
        throw "Privacy report was not found; rerun without -SkipGenerate: $privacyReport"
    }

    try {
        $existingReport = Get-Content -LiteralPath $privacyReport -Raw | ConvertFrom-Json
    } catch {
        throw "Privacy report is invalid; rerun without -SkipGenerate: $privacyReport"
    }
    $expectedConfigHash = (Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToLowerInvariant()
    $actualConfigHash = [string]$existingReport.config_sha256
    if ($actualConfigHash.ToLowerInvariant() -ne $expectedConfigHash) {
        throw 'Existing C.csv was generated with a different config; rerun without -SkipGenerate.'
    }
    if (-not $existingReport.PSObject.Properties['output_sha256']) {
        throw 'Privacy report has no C.csv hash; rerun once without -SkipGenerate.'
    }
    $expectedOutputHash = (Get-FileHash -LiteralPath $cPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $actualOutputHash = [string]$existingReport.output_sha256
    if ($actualOutputHash.ToLowerInvariant() -ne $expectedOutputHash) {
        throw 'Existing C.csv does not match the privacy report; rerun without -SkipGenerate.'
    }
}

if ($PSBoundParameters.ContainsKey('Token')) {
    $tokenValue = $Token.Trim()
} else {
    $sampleToken = Join-Path $distDir 'sample_submissions\token.txt'
    $tokenValue = (Get-Content -LiteralPath $sampleToken -Raw).Trim()
    Write-Warning 'Using the practice token. Pass -Token for a real submission.'
}
if ($tokenValue -notmatch '^[A-Za-z0-9]{16,}$') {
    throw 'Token must contain only letters and digits and be at least 16 characters.'
}
[System.IO.File]::WriteAllText(
    $tokenPath,
    $tokenValue + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host '[2/4] Packaging and validating the defense submission...'
New-FlatZip -Files @($cPath, $tokenPath) -Destination $defenseZip
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $starterDir, '--frozen',
    'python', '-m', 'pwscup2026.kit.validate',
    $defenseZip, '--kind', 'defense', '--dist', $distDir
)

Write-Host '[3/4] Running the official local utility score...'
$scoreArguments = @(
    'run', '--project', $starterDir, '--frozen',
    'python', '-m', 'pwscup2026.kit.selfscore',
    $cPath, '--dist', $distDir
)
Invoke-NativeCommand -FilePath $uvExe -Arguments $scoreArguments

Write-Host '[4/4] Saving the score breakdown as JSON...'
$jsonArguments = $scoreArguments + '--json'
$jsonLines = & $uvExe @jsonArguments
if ($LASTEXITCODE -ne 0) {
    throw "JSON self-score failed with exit code $LASTEXITCODE"
}
[System.IO.File]::WriteAllText(
    $scoreJson,
    ($jsonLines -join [Environment]::NewLine) + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host ''
Write-Host 'OpenDP AIM workflow completed.'
Write-Host "  C.csv:          $cPath"
Write-Host "  Privacy report: $privacyReport"
Write-Host "  Score JSON:     $scoreJson"
Write-Host "  Defense ZIP:    $defenseZip"
