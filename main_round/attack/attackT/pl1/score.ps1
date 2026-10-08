param([ValidateSet('dev','holdout')][string]$Split = 'dev', [string]$Output)
$ErrorActionPreference = 'Stop'
$pl1Scorer = Join-Path $PSScriptRoot '../../../mock_test/sugiyama.ps1'
$pl1Options = @{ Plugin = (Join-Path $PSScriptRoot 'model.py'); FloorRepetitions = 1000 }
if ($Split -eq 'holdout') { $pl1Options.Split='holdout'; $pl1Options.Profile='stress'; $pl1Options.Seed=2026100702 }
else { $pl1Options.Split='dev'; $pl1Options.Profile='mixed'; $pl1Options.Seed=2026100701 }
if ($Output) { $pl1Options.Output=$Output }
& $pl1Scorer @pl1Options
