param(
    [Parameter(Mandatory=$true)][ValidateSet('Generate','Prepare','Predict','Batch','Series','Score','Report')][string]$Action,
    [string]$Case,
    [string]$Panel,
    [string[]]$Panels,
    [string]$Model,
    [string]$Plugin,
    [long]$Seed,
    [ValidateSet('v3','v4','v5')][string]$Generation = 'v5',
    [string]$Output,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$mockRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path $mockRoot -Parent) -Parent
if (-not $Python) {
    $Python = Join-Path $repoRoot 'main_round/attack/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $commonGit = & git -C $repoRoot rev-parse --path-format=absolute --git-common-dir
        if ($LASTEXITCODE -ne 0) { throw 'Cannot locate existing Python; pass -Python.' }
        $Python = Join-Path (Split-Path $commonGit -Parent) 'main_round/attack/.venv/Scripts/python.exe'
    }
}
if (-not (Test-Path -LiteralPath $Python)) { throw 'Existing attack Python not found; pass -Python.' }
$Python = (Resolve-Path -LiteralPath $Python).Path
foreach ($entry in @(@('Case',$Case), @('Panel',$Panel), @('Plugin',$Plugin))) {
    if ($entry[1] -and -not [IO.Path]::IsPathRooted($entry[1])) {
        Set-Variable -Name $entry[0] -Value ([IO.Path]::GetFullPath($entry[1]))
    }
}
if ($Panels) { $Panels = @($Panels | ForEach-Object { [IO.Path]::GetFullPath($_) }) }
if ($Output) { $Output = [IO.Path]::GetFullPath($Output) }
switch ($Action) {
    'Generate' {
        if (-not $Seed -or -not $Output) { throw 'Generate requires -Seed and a new -Output.' }
        $arguments = @('-B','-m',"rank_validation.generation_$Generation",'--seed',"$Seed",'--output',$Output)
    }
    'Prepare' {
        if (-not $Case -or -not $Output) { throw 'Prepare requires -Case and a new -Output.' }
        $arguments = @('-B','-m','rank_validation.panel','prepare','--case',$Case,'--output',$Output)
    }
    { $_ -in @('Batch','Series') } {
        if (-not $Panel) { throw 'Batch requires -Panel.' }
        $arguments = @('-B','-m','rank_validation.panel',$Action.ToLowerInvariant(),'--panel',$Panel)
    }
    { $_ -in @('Predict','Score') } {
        if (-not $Panel -or -not $Model) { throw 'Predict/Score requires -Panel and -Model.' }
        $arguments = @('-B','-m','rank_validation.panel',$Action.ToLowerInvariant(),'--panel',$Panel,'--model',$Model)
        if ($Plugin) { $arguments += @('--plugin',$Plugin) }
    }
    'Report' {
        if (-not $Panels -or -not $Output) { throw 'Report requires -Panels and a new -Output.' }
        $protocol = Join-Path $PSScriptRoot "generation_${Generation}_protocol.json"
        $reportModule = if ($Generation -eq 'v5') { 'rank_validation.stable_report' } else { 'rank_validation.joint_report' }
        $arguments = @('-B','-m',$reportModule,'--output',$Output,'--protocol',$protocol,'--panels') + $Panels
    }
}
$oldPythonPath = $env:PYTHONPATH
$oldUtf8 = $env:PYTHONUTF8
$oldThreads = $env:OMP_NUM_THREADS
$env:PYTHONPATH = (Join-Path $repoRoot 'main_round/attack/official-kit/starter/src')
if ($oldPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $oldPythonPath }
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
Push-Location $mockRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Joint evaluation $Action failed. Existing outputs are preserved." }
} finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONUTF8 = $oldUtf8
    $env:OMP_NUM_THREADS = $oldThreads
}
