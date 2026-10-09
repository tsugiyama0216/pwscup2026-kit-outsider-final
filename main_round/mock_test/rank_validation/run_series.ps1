param([string]$PanelRoot, [string]$Output, [string]$Python, [switch]$ReportOnly)
$ErrorActionPreference = 'Stop'
$mockRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path $mockRoot -Parent) -Parent
if (-not $PanelRoot) { $PanelRoot = Join-Path $mockRoot 'local-output/ranking-v5-20261009/panels' }
$PanelRoot = [IO.Path]::GetFullPath($PanelRoot)
$panels = @('confirm-2026100961','confirm-2026100962') | ForEach-Object { Join-Path $PanelRoot $_ }
if (-not $Output) { $Output = Join-Path $mockRoot ('local-output/rv6-series-comparison/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
$Output = [IO.Path]::GetFullPath($Output)
if (-not $Python) {
    $Python = Join-Path $repoRoot 'main_round/attack/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $commonGit = & git -C $repoRoot rev-parse --path-format=absolute --git-common-dir
        if ($LASTEXITCODE -ne 0) { throw 'Pass -Python with the existing attack environment.' }
        $Python = Join-Path (Split-Path $commonGit -Parent) 'main_round/attack/.venv/Scripts/python.exe'
    }
}
$Python = (Resolve-Path -LiteralPath $Python).Path
if (-not $ReportOnly) {
    foreach ($panel in $panels) {
        & (Join-Path $PSScriptRoot 'run_joint.ps1') -Action Series -Panel $panel -Python $Python
    }
}
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = '1'
Push-Location $mockRoot
try {
    & $Python -B -m rank_validation.series_report --panels @panels --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Series report failed; existing results are preserved.' }
} finally {
    Pop-Location
    $env:PYTHONUTF8 = $oldUtf8
}
