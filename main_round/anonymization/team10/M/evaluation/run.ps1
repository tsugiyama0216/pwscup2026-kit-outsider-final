#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [Parameter(Mandatory)] [string]$C,
    [Parameter(Mandatory)] [string]$Score,
    [string]$Python,
    [string]$KitRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$evaluation = $PSScriptRoot
$mDir = Split-Path -Parent $evaluation
$team = Split-Path -Parent $mDir
if (-not $Python) { $Python = Join-Path $mDir '.venv-score\Scripts\python.exe' }
if (-not $KitRoot) {
    $KitRoot = Join-Path $mDir 'M1\local-output\official-kit\pwscup2026-kit-3dd71bd3436f2abfdbb64da97e860f22729ea054'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $KitRoot -PathType Container)) { throw "Official kit not found: $KitRoot" }
$result = Join-Path $evaluation "results\$Model\resistance_report.json"
& $Python (Join-Path $evaluation 'scripts\evaluate_resistance.py') `
    --model $Model --c $C --score $Score --data (Join-Path $team 'participant_data') `
    --kit-root $KitRoot --out $result
if ($LASTEXITCODE -ne 0) { throw "Resistance evaluation failed: $Model" }
& $Python (Join-Path $evaluation 'scripts\build_comparison.py') `
    --results (Join-Path $evaluation 'results') `
    --json-out (Join-Path $evaluation 'model_comparison.json') `
    --markdown-out (Join-Path $evaluation 'MODEL_COMPARISON.md') `
    --diagnostics-out (Join-Path $evaluation 'MODEL_DIAGNOSTICS.md')
if ($LASTEXITCODE -ne 0) { throw 'Comparison build failed.' }
