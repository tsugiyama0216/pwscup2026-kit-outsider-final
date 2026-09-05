param(
    [string]$Dist = "",
    [string]$Targets = "",
    [string]$OutputDir = "",
    [string]$CandidateDir = "",
    [string]$PythonExe = "",
    [switch]$SkipDomias,
    [switch]$MainOnly
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

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
if ([string]::IsNullOrWhiteSpace($Dist)) {
    $Dist = Join-Path $repoRoot "..\pwscup2026-kit\work\prelim\team24\runtime_dist"
}
if ([string]::IsNullOrWhiteSpace($Targets)) {
    $Targets = Join-Path $repoRoot "team-shared\prelim\attack-targets"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\prelim\team24\attack-phase-v4"
}
if ([string]::IsNullOrWhiteSpace($CandidateDir)) {
    $CandidateDir = Join-Path $repoRoot "submissions\prelim\team24\attack-candidate-v4"
}

$arguments = @(
    (Join-Path $PSScriptRoot "generate_submission_v4.py"),
    "--dist", (Resolve-Path -LiteralPath $Dist).Path,
    "--targets", (Resolve-Path -LiteralPath $Targets).Path,
    "--out-dir", $OutputDir,
    "--candidate-dir", $CandidateDir,
    "--self-target-id", "24",
    "--exclude-targets", "24",
    "--mia-total-shared", "508"
)
if ($SkipDomias) { $arguments += "--skip-domias" }
if ($MainOnly) { $arguments += "--main-only" }

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "attack v4 generation failed (exit code $LASTEXITCODE)"
}
