[CmdletBinding()]
param(
    [ValidateSet('Run','Generate','Export','Attack','Score','Benchmark')][string]$Action = 'Run',
    [long]$Seed = 2026100701,
    [ValidateRange(1,30)][int]$Count = 3,
    [long]$SelectionSeed = 20261007,
    [ValidateRange(1,12)][int]$RCount = 10,
    [string]$SuiteDirectory,
    [string]$CaseId,
    [string]$Model,
    [string[]]$Models,
    [string[]]$Cases,
    [string]$BundleDirectory,
    [string]$PredictionDirectory,
    [string]$Plugin,
    [ValidateSet('official','background_only','constant')][string]$Method = 'official',
    [ValidateRange(2,10000)][int]$FloorRepetitions = 1000,
    [ValidateRange(1,86400)][int]$Timeout = 600,
    [string]$Output
)
$ErrorActionPreference = 'Stop'
$mockTsPython = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../attack/.venv/Scripts/python.exe'))
if (-not (Test-Path -LiteralPath $mockTsPython)) { throw 'Run main_round/attack/setup.ps1 first.' }
$mockTsArgs = @('-B', '-m', 'mock_TS', $Action.ToLowerInvariant())
if ($Output) { $mockTsArgs += @('--output', [IO.Path]::GetFullPath($Output)) }
if ($Action -in @('Run','Generate')) {
    $mockTsArgs += @('--seed', "$Seed", '--count', "$Count", '--selection-seed', "$SelectionSeed", '--r-count', "$RCount")
} elseif (@('Seed','Count','SelectionSeed','RCount') | Where-Object { $PSBoundParameters.ContainsKey($_) }) {
    throw 'Generation options only apply to Run/Generate; saved suites keep their conditions.'
}
if ($Action -in @('Export','Score','Benchmark')) {
    if (-not $SuiteDirectory) { throw "$Action requires SuiteDirectory." }
    $mockTsArgs += @('--suite', (Resolve-Path -LiteralPath $SuiteDirectory).Path)
}
if ($Action -in @('Export','Score')) {
    if (-not $CaseId -or -not $Model) { throw "$Action requires CaseId and Model." }
    $mockTsArgs += @('--case', $CaseId, '--model', $Model)
}
if ($Action -eq 'Attack') {
    if (-not $BundleDirectory) { throw 'Attack requires BundleDirectory.' }
    $mockTsArgs += @('--bundle', (Resolve-Path -LiteralPath $BundleDirectory).Path)
}
if ($Action -eq 'Score') {
    if (-not $PredictionDirectory) { throw 'Score requires PredictionDirectory.' }
    $mockTsArgs += @('--predictions', (Resolve-Path -LiteralPath $PredictionDirectory).Path)
}
if ($Action -in @('Run','Benchmark','Attack')) {
    if ($Plugin -and $PSBoundParameters.ContainsKey('Method')) { throw 'Choose Plugin or Method.' }
    if ($Plugin) { $mockTsArgs += @('--plugin', (Resolve-Path -LiteralPath $Plugin).Path) }
    else { $mockTsArgs += @('--method', $Method) }
    $mockTsArgs += @('--timeout', "$Timeout")
}
if ($Action -in @('Run','Benchmark','Score')) { $mockTsArgs += @('--floor-repetitions', "$FloorRepetitions") }
if ($Models) { $mockTsArgs += @('--models') + $Models }
if ($Cases) { $mockTsArgs += @('--cases') + $Cases }
$mockTsPreviousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    Push-Location (Split-Path -Parent $PSScriptRoot)
    try {
        & $mockTsPython @mockTsArgs
        if ($LASTEXITCODE -ne 0) { throw "mock_TS failed (exit $LASTEXITCODE)." }
    } finally { Pop-Location }
} finally { $env:PYTHONUTF8 = $mockTsPreviousUtf8 }
