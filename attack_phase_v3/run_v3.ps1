param(
    [Parameter(Mandatory = $true)]
    [string]$Dist,
    [Parameter(Mandatory = $true)]
    [string]$SelfTargetId,
    [string]$Targets = "",
    [string]$OutputDir = "",
    [string]$CandidateDir = "",
    [string]$PythonExe = "",
    [double]$AiaActiveFraction = 0.5,
    [double]$AiaLinkTrustDistance = 0.20,
    [double]$AiaAnomalousEventRate = 0.50,
    [double]$AiaAnomalousLinkTrustDistance = 0.08,
    [string]$ExcludeTargets = "",
    [int]$MiaMinPerTarget = 25,
    [int]$MiaMaxPerTarget = 27,
    [int]$MiaTotalShared = 736,
    [double]$MiaTransportEpsilon = 0.35,
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
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $pythonCommand) {
            $PythonExe = $pythonCommand.Source
        }
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe) -or -not (Test-Path -LiteralPath $PythonExe)) {
    throw "Compatible Python was not found. Specify -PythonExe."
}

$resolvedDist = (Resolve-Path -LiteralPath $Dist).Path
if ([string]::IsNullOrWhiteSpace($Targets)) {
    $resolvedTargets = Join-Path $resolvedDist "targets"
}
else {
    $resolvedTargets = (Resolve-Path -LiteralPath $Targets).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\attack-phase-v3"
}
if ([string]::IsNullOrWhiteSpace($ExcludeTargets)) {
    $ExcludeTargets = $SelfTargetId
}

$arguments = @(
    (Join-Path $PSScriptRoot "generate_submission_v3.py"),
    "--dist", $resolvedDist,
    "--targets", $resolvedTargets,
    "--out-dir", $OutputDir,
    "--self-target-id", $SelfTargetId,
    "--exclude-targets", $ExcludeTargets,
    "--aia-active-fraction", $AiaActiveFraction,
    "--aia-link-trust-distance", $AiaLinkTrustDistance,
    "--aia-anomalous-event-rate", $AiaAnomalousEventRate,
    "--aia-anomalous-link-trust-distance", $AiaAnomalousLinkTrustDistance,
    "--mia-min-per-target", $MiaMinPerTarget,
    "--mia-max-per-target", $MiaMaxPerTarget,
    "--mia-total-shared", $MiaTotalShared,
    "--mia-transport-epsilon", $MiaTransportEpsilon
)
if (-not [string]::IsNullOrWhiteSpace($CandidateDir)) {
    $arguments += @("--candidate-dir", $CandidateDir)
}
if ($SkipDomias) {
    $arguments += "--skip-domias"
}
if ($MainOnly) {
    $arguments += "--main-only"
}

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "attack v3 generation failed (exit code $LASTEXITCODE)"
}
