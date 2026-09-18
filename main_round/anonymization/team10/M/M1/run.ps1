#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('Generate', 'Score', 'Validate')]
    [string]$Mode = 'Validate',
    [string]$Python,
    [string]$KitRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$model = $PSScriptRoot
$team = [IO.Path]::GetFullPath((Join-Path $model '..\..'))
$data = Join-Path $team 'participant_data'
if (-not $Python) { $Python = Join-Path $team 'M\.venv-score\Scripts\python.exe' }
if (-not $KitRoot) {
    $KitRoot = Join-Path $model 'local-output\official-kit\pwscup2026-kit-3dd71bd3436f2abfdbb64da97e860f22729ea054'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$env:PYTHONPATH = Join-Path $KitRoot 'starter\src'

if ($Mode -eq 'Generate') {
    & $Python (Join-Path $model 'scripts\generate_m1.py') `
        --s1 (Join-Path $team 'S\S1\C.csv') `
        --background (Join-Path $data 'A_bg.csv') `
        --schema (Join-Path $data 'schema.json') `
        --config (Join-Path $team 'S\S1\config\aim_main_team10_S1.json') `
        --output (Join-Path $model 'C.csv') `
        --report (Join-Path $model 'generation_report.json') `
        --strength 1.0 --seed 20260919 --death-weight 0.31 `
        --outcome-method nearest-risk
    if ($LASTEXITCODE -ne 0) { throw 'M1 generation failed.' }
}
if ($Mode -in @('Generate', 'Score')) {
    $dist = Join-Path $model 'local-output\selfscore-dist'
    & $Python -m pwscup2026.kit.selfscore (Join-Path $model 'C.csv') --dist $dist --json
    if ($LASTEXITCODE -ne 0) { throw 'M1 scoring failed.' }
}
if ($Mode -in @('Generate', 'Validate')) {
    & $Python (Join-Path $model 'scripts\validate_m1.py')
    if ($LASTEXITCODE -ne 0) { throw 'M1 validation failed.' }
}
