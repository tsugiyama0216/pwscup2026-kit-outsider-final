#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [Parameter(Mandatory)] [string]$C,
    [Parameter(Mandatory)] [string]$Score,
    [string]$Data,
    [string]$Python,
    [string]$KitRoot,
    [string]$OutDir,
    [Nullable[double]]$Phi,
    [string]$ResistanceReport
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$evaluation = $PSScriptRoot
$mDir = Split-Path -Parent $evaluation
$team = Split-Path -Parent $mDir
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $team))
if (-not $Data) { $Data = Join-Path $team 'participant_data' }
if (-not $KitRoot) { $KitRoot = Join-Path $root 'prelim\workspace' }
if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
if (-not $OutDir) { $OutDir = Join-Path $evaluation ("results\$Model") }
if ($null -eq $Phi) {
    $phiPath = Join-Path $evaluation 'phi_team10.json'
    if (Test-Path -LiteralPath $phiPath -PathType Leaf) {
        $Phi = [double](Get-Content -LiteralPath $phiPath -Raw | ConvertFrom-Json).phi_i
    }
}
if (-not (Test-Path -LiteralPath $C -PathType Leaf)) { throw "C.csv not found: $C" }
if (-not (Test-Path -LiteralPath $Score -PathType Leaf)) { throw "score JSON not found: $Score" }
if (-not (Test-Path -LiteralPath $Data -PathType Container)) { throw "data directory not found: $Data" }
if (-not (Test-Path -LiteralPath $KitRoot -PathType Container)) { throw "kit root not found: $KitRoot" }
$jsonOut = Join-Path $OutDir 'defense_score.json'
$markdownOut = Join-Path $OutDir 'DEFENSE_SCORE.md'
$arguments = @(
    (Join-Path $evaluation 'scripts\score_defense.py'),
    '--model', $Model, '--c', $C, '--score', $Score, '--data', $Data,
    '--kit-root', $KitRoot, '--out', $jsonOut, '--markdown-out', $markdownOut
)
if ($ResistanceReport) {
    $arguments += @('--resistance-report', $ResistanceReport)
}
if ($null -ne $Phi) {
    $arguments += @('--phi', [string]$Phi)
}
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Defense score evaluation failed: $Model" }
Write-Host "JSON: $jsonOut"
Write-Host "Markdown: $markdownOut"
