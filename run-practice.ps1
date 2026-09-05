#requires -Version 5.1

[CmdletBinding()]
param(
    [ValidateSet('copula_survival', 'copula_synth')]
    [string]$DefenseMethod = 'copula_survival',

    [ValidateRange(0, 2147483647)]
    [int]$Seed = 0,

    # 省略時は練習用サンプルトークンを使う。本番提出時だけ自分のトークンを指定する。
    [string]$Token,

    [switch]$SkipAttack
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$starterDir = Join-Path $repoRoot 'starter'
$distDir = Join-Path $repoRoot 'participant_data'
$outputDir = Join-Path $repoRoot 'local-output'
$uvExe = Join-Path $repoRoot '.tools\uv\uv.exe'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $repoRoot '.tools\python'
$env:UV_CACHE_DIR = Join-Path $repoRoot '.uv-cache'

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(Mandatory)] [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "コマンドが終了コード $LASTEXITCODE で失敗しました: $FilePath $($Arguments -join ' ')"
    }
}

function New-FlatZip {
    param(
        [Parameter(Mandatory)] [string[]]$Files,
        [Parameter(Mandatory)] [string]$Destination
    )

    # 提出物の ZIP 直下に、指定したファイルだけを格納する。
    Compress-Archive `
        -LiteralPath $Files `
        -DestinationPath $Destination `
        -CompressionLevel Optimal `
        -Force
}

if (-not (Test-Path -LiteralPath $uvExe)) {
    throw 'ローカル環境がありません。先に .\setup-local.ps1 を実行してください。'
}
if (-not (Test-Path -LiteralPath (Join-Path $starterDir '.venv'))) {
    throw 'Python 仮想環境がありません。先に .\setup-local.ps1 を実行してください。'
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$cPath = Join-Path $outputDir 'C.csv'
$miaPath = Join-Path $outputDir 'F_mia.csv'
$aiaPath = Join-Path $outputDir 'F_aia.csv'
$tokenPath = Join-Path $outputDir 'token.txt'
$defenseZip = Join-Path $outputDir 'defense_submission.zip'
$attackZip = Join-Path $outputDir 'attack_submission.zip'

Write-Host "[1/5] 防御データを作成します（method=$DefenseMethod / seed=$Seed）..."
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $starterDir, '--frozen',
    'python', (Join-Path $repoRoot 'my_solution\run_defense.py'),
    '--dist', $distDir,
    '--out', $cPath,
    '--method', $DefenseMethod,
    '--seed', $Seed.ToString()
)

Write-Host ''
Write-Host '[2/5] 防御データの有用性を自己採点します...'
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $starterDir, '--frozen',
    'python', '-m', 'pwscup2026.kit.selfscore',
    $cPath, '--dist', $distDir
)

if ($PSBoundParameters.ContainsKey('Token')) {
    $tokenValue = $Token.Trim()
} else {
    $sampleTokenPath = Join-Path $distDir 'sample_submissions\token.txt'
    $tokenValue = (Get-Content -LiteralPath $sampleTokenPath -Raw).Trim()
    Write-Warning '練習用サンプルトークンを使います。本番提出時は -Token に自分のトークンを指定してください。'
}
if ($tokenValue -notmatch '^[A-Za-z0-9]{16,}$') {
    throw 'Token は英数字だけで16文字以上にしてください。'
}
[System.IO.File]::WriteAllText(
    $tokenPath,
    $tokenValue + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host ''
Write-Host '[3/5] 防御提出 ZIP を作成して検証します...'
New-FlatZip -Files @($cPath, $tokenPath) -Destination $defenseZip
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $starterDir, '--frozen',
    'python', '-m', 'pwscup2026.kit.validate',
    $defenseZip, '--kind', 'defense', '--dist', $distDir
)

if (-not $SkipAttack) {
    Write-Host ''
    Write-Host '[4/5] 参照 MIA / AIA を実行します...'
    Invoke-NativeCommand -FilePath $uvExe -Arguments @(
        'run', '--project', $starterDir, '--frozen',
        'python', (Join-Path $starterDir 'reference\run_mia.py'),
        '--dist', $distDir, '--out', $miaPath
    )
    Invoke-NativeCommand -FilePath $uvExe -Arguments @(
        'run', '--project', $starterDir, '--frozen',
        'python', (Join-Path $starterDir 'reference\run_aia.py'),
        '--dist', $distDir, '--out', $aiaPath
    )

    Write-Host ''
    Write-Host '[5/5] 攻撃提出 ZIP を作成して検証します...'
    New-FlatZip -Files @($miaPath, $aiaPath, $tokenPath) -Destination $attackZip
    Invoke-NativeCommand -FilePath $uvExe -Arguments @(
        'run', '--project', $starterDir, '--frozen',
        'python', '-m', 'pwscup2026.kit.validate',
        $attackZip, '--kind', 'attack', '--dist', $distDir
    )
} else {
    Write-Host ''
    Write-Host '[4/5, 5/5] -SkipAttack のため攻撃処理を省略しました。'
}

Write-Host ''
Write-Host '完了しました。出力先:'
Write-Host "  $outputDir"
Write-Host "  防御 ZIP: $defenseZip"
if (-not $SkipAttack) {
    Write-Host "  攻撃 ZIP: $attackZip"
}
