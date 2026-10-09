param([string]$PanelRoot, [string]$Output, [string]$Python, [switch]$ReportOnly)
$ErrorActionPreference = 'Stop'
$mockRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path $mockRoot -Parent) -Parent
if (-not $PanelRoot) { $PanelRoot = Join-Path $mockRoot 'local-output/ranking-v5-20261009/panels' }
$PanelRoot = [IO.Path]::GetFullPath($PanelRoot)
$panels = @('confirm-2026100961','confirm-2026100962') | ForEach-Object { Join-Path $PanelRoot $_ }
if (-not $Output) { $Output = Join-Path $mockRoot ('local-output/additional-comparison/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
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
$oldUtf8 = $env:PYTHONUTF8
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $repoRoot 'main_round/attack/official-kit/starter/src'
if ($oldPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $oldPythonPath }
Push-Location $mockRoot
try {
    if (-not $ReportOnly) {
        foreach ($panel in $panels) {
            & $Python -B -m rank_validation.additional_panel --panel $panel
            if ($LASTEXITCODE -ne 0) { throw 'Additional prediction/scoring failed; existing outputs are preserved.' }
        }
    }
    & $Python -B -m rank_validation.additional_report --panels @panels --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Additional report failed; existing outputs are preserved.' }
} finally {
    Pop-Location
    $env:PYTHONUTF8 = $oldUtf8
    $env:PYTHONPATH = $oldPythonPath
}
