param(
    [string]$CPath = "",
    [string]$OutputDir = "",
    [string]$DefenseConfig = "",
    [string]$LineageReport = "",
    [string]$TargetName = "v4",
    [int]$Seed = 7,
    [int]$Permutations = 2000
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$uv = Join-Path $repoRoot ".tools\uv\uv.exe"
if (-not (Test-Path -LiteralPath $uv)) {
    throw "uv が見つかりません: $uv"
}
if ([string]::IsNullOrWhiteSpace($CPath)) {
    $CPath = Join-Path $repoRoot "local-output\opendp_aim\C.csv"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "local-output\opendp_aim\toy-attack-v4"
}
if ([string]::IsNullOrWhiteSpace($DefenseConfig)) {
    $DefenseConfig = Join-Path $repoRoot "opendp_aim\config\pwscup_aim_v4.json"
}
if ([string]::IsNullOrWhiteSpace($LineageReport)) {
    $LineageReport = Join-Path $repoRoot "local-output\opendp_aim\privacy_report.json"
}

$oldPythonInstallDir = $env:UV_PYTHON_INSTALL_DIR
$oldCacheDir = $env:UV_CACHE_DIR
$oldVirtualEnv = $env:VIRTUAL_ENV

try {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot ".tools\python"
    $env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
    $env:VIRTUAL_ENV = $null
    Push-Location $repoRoot
    try {
        & $uv run `
            --project (Join-Path $repoRoot "starter") `
            --frozen `
            python (Join-Path $repoRoot "opendp_aim\scripts\toy_attack.py") `
            --c $CPath `
            --dist (Join-Path $repoRoot "participant_data") `
            --output-dir $OutputDir `
            --target-name $TargetName `
            --seed $Seed `
            --permutations $Permutations `
            --defense-config $DefenseConfig `
            --privacy-report $LineageReport
        if ($LASTEXITCODE -ne 0) {
            throw "toy attack failed (exit code $LASTEXITCODE)"
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:UV_PYTHON_INSTALL_DIR = $oldPythonInstallDir
    $env:UV_CACHE_DIR = $oldCacheDir
    $env:VIRTUAL_ENV = $oldVirtualEnv
}
