[CmdletBinding()]
param(
    [ValidateSet('Run', 'Generate', 'Evaluate')][string]$Action = 'Run',
    [long]$Seed = 2026100701,
    [ValidateSet('mixed', 'stress')][string]$Profile = 'mixed',
    [ValidateSet('dev', 'holdout')][string]$Split = 'dev',
    [string]$CaseDirectory,
    [string]$Plugin,
    [string]$Output,
    [ValidateRange(2, 10000)][int]$FloorRepetitions = 1000,
    [long]$FloorSeed
)
$ErrorActionPreference = 'Stop'
$sugiyamaPython = Join-Path $PSScriptRoot '../attack/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $sugiyamaPython)) { throw 'Run ../attack/setup.ps1 first.' }
if ($Action -eq 'Evaluate') {
    if (-not $CaseDirectory) { throw 'Evaluate requires CaseDirectory.' }
    foreach ($sugiyamaOption in @('Seed', 'Profile', 'Split')) {
        if ($PSBoundParameters.ContainsKey($sugiyamaOption)) { throw "$sugiyamaOption belongs to Generate/Run. Evaluate uses the saved case." }
    }
} elseif ($CaseDirectory) { throw 'CaseDirectory is only valid for Evaluate.' }
if (-not $Output) {
    $sugiyamaStamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $Output = Join-Path $PSScriptRoot "local-output/sugiyama/$sugiyamaStamp-$($Action.ToLowerInvariant())"
}
$sugiyamaArgs = @('-B', '-m', 'sugiyama_v2', $Action.ToLowerInvariant(), '--output', [IO.Path]::GetFullPath($Output),
    '--floor-repetitions', "$FloorRepetitions")
if ($Action -ne 'Evaluate') { $sugiyamaArgs += @('--seed', "$Seed", '--profile', $Profile, '--split', $Split) }
if ($CaseDirectory) { $sugiyamaArgs += @('--case', (Resolve-Path -LiteralPath $CaseDirectory).Path) }
if ($Plugin) { $sugiyamaArgs += @('--plugin', (Resolve-Path -LiteralPath $Plugin).Path) }
if ($PSBoundParameters.ContainsKey('FloorSeed')) { $sugiyamaArgs += @('--floor-seed', "$FloorSeed") }
if ($Action -ne 'Generate') { & (Join-Path $PSScriptRoot '../attack/prepare_attack_m.ps1') }
$sugiyamaOldUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    Push-Location $PSScriptRoot
    try {
        & $sugiyamaPython @sugiyamaArgs
        if ($LASTEXITCODE -ne 0) { throw "Sugiyama evaluation failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
} finally { $env:PYTHONUTF8 = $sugiyamaOldUtf8 }
