param(
    [string]$Dist = "",
    [string]$OutputDir = "",
    [string]$Seeds = "20260826,20260827",
    [string]$Scenarios = "raw,noise_0p03,noise_0p10,time_shuffled,independent",
    [switch]$SkipDomias
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$uv = Join-Path $repoRoot ".tools\uv\uv.exe"
if (-not (Test-Path -LiteralPath $uv)) {
    throw "uv が見つかりません: $uv"
}
if ([string]::IsNullOrWhiteSpace($Dist)) {
    $Dist = Join-Path $repoRoot "participant_data"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\attack-shadow-evaluation"
}
$resolvedDist = (Resolve-Path -LiteralPath $Dist).Path

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
        "python", (Join-Path $PSScriptRoot "evaluate_shadow.py"),
        "--dist", $resolvedDist,
        "--out-dir", $OutputDir,
        "--seeds", $Seeds,
        "--scenarios", $Scenarios
    )
    if ($SkipDomias) {
        $arguments += "--skip-domias"
    }
    & $uv @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "shadow attack evaluation failed (exit code $LASTEXITCODE)"
    }
}
finally {
    $env:UV_PYTHON_INSTALL_DIR = $oldPythonInstallDir
    $env:UV_CACHE_DIR = $oldCacheDir
    $env:VIRTUAL_ENV = $oldVirtualEnv
}
