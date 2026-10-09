param([Parameter(Mandatory=$true)][string]$OutputDirectory,[string]$Python)
$ErrorActionPreference='Stop'
$repoRoot=Split-Path (Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent) -Parent
if (-not $Python) {
    $Python=Join-Path $repoRoot 'main_round/attack/.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $rvGit=& git -C $repoRoot rev-parse --path-format=absolute --git-common-dir
        if ($LASTEXITCODE -ne 0) { throw 'Pass -Python with the existing attack environment.' }
        $Python=Join-Path (Split-Path $rvGit -Parent) 'main_round/attack/.venv/Scripts/python.exe'
    }
}
$Python=(Resolve-Path -LiteralPath $Python).Path
& (Join-Path $repoRoot 'main_round/attack/prepare_attack_m.ps1')
$oldUtf8=$env:PYTHONUTF8
$env:PYTHONUTF8='1'
try {
    & $Python -B (Join-Path $PSScriptRoot 'generate.py') --output ([IO.Path]::GetFullPath($OutputDirectory))
    if ($LASTEXITCODE -ne 0) { throw 'Rv12 generation failed; existing artifacts retained.' }
} finally { $env:PYTHONUTF8=$oldUtf8 }
