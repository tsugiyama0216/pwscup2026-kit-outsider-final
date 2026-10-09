param(
    [Parameter(Mandatory=$true)][string]$Plugin,
    [Parameter(Mandatory=$true)][string]$Name,
    [ValidateSet('Rv1','Rv2','pl1','Tv105')][string]$Reference = 'Rv2',
    [string]$PanelRoot,
    [string]$Output,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$mockRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path $mockRoot -Parent) -Parent
$Plugin = (Resolve-Path -LiteralPath $Plugin).Path
$pluginDigest = (Get-FileHash -LiteralPath $Plugin -Algorithm SHA256).Hash
if (-not $PanelRoot) { $PanelRoot = Join-Path $mockRoot 'local-output/ranking-v5-20261009/panels' }
$PanelRoot = [IO.Path]::GetFullPath($PanelRoot)
$panels = @('confirm-2026100961','confirm-2026100962') | ForEach-Object { Join-Path $PanelRoot $_ }
foreach ($panel in $panels) {
    if (-not (Test-Path -LiteralPath (Join-Path $panel 'panel.json'))) { throw "Saved reference panel not found: $panel" }
    foreach ($baseline in @('Rv1','Rv2','pl1','Tv105')) {
        if (-not (Test-Path -LiteralPath (Join-Path $panel "scores/$baseline/result.json"))) { throw "Reference model missing: $panel / $baseline" }
    }
}
if (-not $Output) { $Output = Join-Path $mockRoot ('local-output/candidate-comparison/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')) }
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
foreach ($panel in $panels) {
    if ((Get-FileHash -LiteralPath $Plugin -Algorithm SHA256).Hash -ne $pluginDigest) { throw 'Candidate plugin changed between scenarios.' }
    & (Join-Path $PSScriptRoot 'run_joint.ps1') -Action Predict -Panel $panel -Model $Name -Plugin $Plugin -Python $Python
}
if ((Get-FileHash -LiteralPath $Plugin -Algorithm SHA256).Hash -ne $pluginDigest) { throw 'Candidate plugin changed during evaluation.' }
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = '1'
Push-Location $mockRoot
try {
    & $Python -B -m rank_validation.candidate_report --candidate $Name --reference $Reference --output $Output --panels @panels
    if ($LASTEXITCODE -ne 0) { throw 'Candidate report failed; all saved predictions are preserved.' }
} finally {
    Pop-Location
    $env:PYTHONUTF8 = $oldUtf8
}
