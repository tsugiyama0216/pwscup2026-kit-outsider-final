param(
    [Parameter(Mandatory = $true)]
    [string]$Dist,
    [string]$Targets = "",
    [string]$OutputDir = "",
    [double]$AiaActiveFraction = 0.5,
    [string]$ExcludeTargets = "",
    [int]$MiaMinPerTarget = 25,
    [int]$MiaMaxPerTarget = 27,
    [int]$MiaTotalShared = 736,
    [double]$MiaTransportEpsilon = 0.35,
    [switch]$SkipDomias
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$uv = Join-Path $repoRoot ".tools\uv\uv.exe"
if (-not (Test-Path -LiteralPath $uv)) {
    throw "uv が見つかりません: $uv"
}
$resolvedDist = (Resolve-Path -LiteralPath $Dist).Path
if ([string]::IsNullOrWhiteSpace($Targets)) {
    $resolvedTargets = Join-Path $resolvedDist "targets"
}
else {
    $resolvedTargets = (Resolve-Path -LiteralPath $Targets).Path
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\attack-phase"
}

$oldPythonInstallDir = $env:UV_PYTHON_INSTALL_DIR
$oldCacheDir = $env:UV_CACHE_DIR
$oldVirtualEnv = $env:VIRTUAL_ENV

try {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot ".tools\python"
    $env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
    $env:VIRTUAL_ENV = $null
    $arguments = @(
        "run",
        "--project", (Join-Path $repoRoot "starter"),
        "--frozen",
        "python", (Join-Path $PSScriptRoot "generate_submission.py"),
        "--dist", $resolvedDist,
        "--targets", $resolvedTargets,
        "--out-dir", $OutputDir,
        "--aia-active-fraction", $AiaActiveFraction,
        "--mia-min-per-target", $MiaMinPerTarget,
        "--mia-max-per-target", $MiaMaxPerTarget,
        "--mia-total-shared", $MiaTotalShared,
        "--mia-transport-epsilon", $MiaTransportEpsilon
    )
    if (-not [string]::IsNullOrWhiteSpace($ExcludeTargets)) {
        $arguments += @("--exclude-targets", $ExcludeTargets)
    }
    if ($SkipDomias) {
        $arguments += "--skip-domias"
    }
    & $uv @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "privacy competition analysis failed (exit code $LASTEXITCODE)"
    }
}
finally {
    $env:UV_PYTHON_INSTALL_DIR = $oldPythonInstallDir
    $env:UV_CACHE_DIR = $oldCacheDir
    $env:VIRTUAL_ENV = $oldVirtualEnv
}
