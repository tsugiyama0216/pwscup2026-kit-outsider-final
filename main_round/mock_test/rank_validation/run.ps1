param(
    [Parameter(Mandatory=$true)][string]$Handoff,
    [string]$Output,
    [string]$Python,
    [string]$OfficialFeedback = (Join-Path $PSScriptRoot 'evidence/official_totals_20261009.json')
)
$ErrorActionPreference = 'Stop'
$mockRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path $mockRoot -Parent) -Parent
if (-not $Python) {
    $Python = Join-Path $repoRoot 'main_round/attack/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $commonGit = & git -C $repoRoot rev-parse --path-format=absolute --git-common-dir
        if ($LASTEXITCODE -ne 0) { throw 'Cannot locate existing repository Python; pass -Python.' }
        $primaryRoot = Split-Path $commonGit -Parent
        $Python = Join-Path $primaryRoot 'main_round/attack/.venv/Scripts/python.exe'
    }
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Existing attack Python not found; pass -Python.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
$Handoff = (Resolve-Path -LiteralPath $Handoff).Path
$OfficialFeedback = (Resolve-Path -LiteralPath $OfficialFeedback).Path
if (-not $Output) { $Output = Join-Path $mockRoot ('local-output/rank-validation/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$Output = [IO.Path]::GetFullPath($Output)
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = '1'
Push-Location $mockRoot
try {
    & $Python -B -m rank_validation --handoff $Handoff --official-feedback $OfficialFeedback --check-public-profile --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Rank validation failed.' }
} finally {
    Pop-Location
    $env:PYTHONUTF8 = $oldUtf8
}
