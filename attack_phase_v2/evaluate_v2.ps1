param(
    [string]$Dist = "",
    [string]$OutputDir = "",
    [string]$PythonExe = "",
    [string]$Seeds = "20260826,20260827",
    [string]$Scenarios = "raw,noise_0p03,noise_0p10,time_shuffled,independent",
    [double]$AiaLinkTrustDistance = 0.20,
    [double]$AiaAnomalousEventRate = 0.50,
    [double]$AiaAnomalousLinkTrustDistance = 0.08,
    [switch]$SkipDomias
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
if ([string]::IsNullOrWhiteSpace($Dist)) {
    $Dist = Join-Path $repoRoot "participant_data"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\attack-shadow-evaluation-v2"
}
$resolvedDist = (Resolve-Path -LiteralPath $Dist).Path

$arguments = @(
    (Join-Path $PSScriptRoot "evaluate_shadow_v2.py"),
    "--dist", $resolvedDist,
    "--out-dir", $OutputDir,
    "--seeds", $Seeds,
    "--scenarios", $Scenarios,
    "--aia-link-trust-distance", $AiaLinkTrustDistance,
    "--aia-anomalous-event-rate", $AiaAnomalousEventRate,
    "--aia-anomalous-link-trust-distance", $AiaAnomalousLinkTrustDistance
)
if ($SkipDomias) {
    $arguments += "--skip-domias"
}

& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "attack v2 shadow evaluation failed (exit code $LASTEXITCODE)"
}
