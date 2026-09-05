param(
    [string]$Dist = "",
    [string]$Targets = "",
    [string]$OutputDir = "",
    [string]$CandidateDir = "",
    [string]$PythonExe = "",
    [switch]$MainOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($Dist)) {
    $Dist = Join-Path $repoRoot "..\pwscup2026-kit\work\prelim\team24\runtime_dist"
}
if ([string]::IsNullOrWhiteSpace($Targets)) {
    $Targets = Join-Path $repoRoot "team-shared\prelim\attack-targets"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\prelim\team24\attack-daybreak-independent"
}
if ([string]::IsNullOrWhiteSpace($CandidateDir)) {
    $CandidateDir = Join-Path $repoRoot "submissions\prelim\team24\attack-candidate-daybreak-independent"
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $pythonCandidates = @(
        (Join-Path $repoRoot "starter\.venv\Scripts\python.exe"),
        (Join-Path $repoRoot "..\pwscup2026-kit\starter\.venv\Scripts\python.exe")
    )
    $PythonExe = $pythonCandidates | Where-Object {
        Test-Path -LiteralPath $_
    } | Select-Object -First 1
}
if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Compatible Python was not found. Specify -PythonExe."
}

$arguments = @(
    (Join-Path $PSScriptRoot "generate_submission_daybreak_independent.py"),
    "--dist", (Resolve-Path -LiteralPath $Dist).Path,
    "--targets", (Resolve-Path -LiteralPath $Targets).Path,
    "--out-dir", $OutputDir,
    "--candidate-dir", $CandidateDir,
    "--self-target-id", "24",
    "--exclude-targets", "24",
    "--mia-total-shared", "508",
    "--acknowledge-prelim-private-block-rule"
)
if ($MainOnly) {
    $arguments += "--main-only"
}

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Daybreak independent attack generation failed (exit code $LASTEXITCODE)"
}
