[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('M-v1-diagnose', 'M-v1-evaluate', 'M-v1-predict', 'M-v1-test',
        'v1-diagnose', 'v1-evaluate', 'v1-predict', 'v1-test',
        'M-v2-aia-evaluate', 'M-v2-mia-evaluate', 'M-v2-predict', 'M-v2-test',
        'M-v2-freeze', 'M-v2-complete')]
    [string]$Action = 'M-v2-test',
    [ValidateSet('dev', 'holdout')][string]$Stage = 'dev',
    [string]$Selected = 'ensemble_nearest',
    [string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$shared = Split-Path $PSScriptRoot -Parent
& (Join-Path $shared 'prepare_attack_m.ps1')
$forward = @{ Action = $Action; Stage = $Stage; Selected = $Selected }
if ($OutputDirectory) { $forward.OutputDirectory = $OutputDirectory }
& (Join-Path $shared 'attack.ps1') @forward
