#requires -Version 5.1

[CmdletBinding()]
param(
    [switch]$Notebook
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = $PSScriptRoot
$starterDir = Join-Path $repoRoot 'starter'
$uvVersion = '0.11.32'
$uvDir = Join-Path $repoRoot '.tools\uv'
$uvExe = Join-Path $uvDir 'uv.exe'

# Python 本体とダウンロードキャッシュもリポジトリ内に閉じ込める。
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

if (-not (Test-Path -LiteralPath $uvExe)) {
    Write-Host "uv $uvVersion をリポジトリ内へ導入します..."
    $env:UV_UNMANAGED_INSTALL = $uvDir
    $env:UV_NO_MODIFY_PATH = '1'

    # Astral 公式のバージョン固定インストーラー。PATH や PowerShell プロファイルは変更しない。
    $installerUrl = "https://astral.sh/uv/$uvVersion/install.ps1"
    $installer = Invoke-RestMethod -Uri $installerUrl
    Invoke-Expression $installer
}

if (-not (Test-Path -LiteralPath $uvExe)) {
    throw "uv の導入を確認できませんでした: $uvExe"
}

$syncArgs = @(
    'sync',
    '--project', $starterDir,
    '--python', '3.12',
    '--frozen'
)
if ($Notebook) {
    $syncArgs += @('--extra', 'notebook')
}

Write-Host 'Python 3.12 とロック済み依存関係を同期します...'
Invoke-NativeCommand -FilePath $uvExe -Arguments $syncArgs

Write-Host ''
Write-Host '環境構築が完了しました。'
Invoke-NativeCommand -FilePath $uvExe -Arguments @(
    'run', '--project', $starterDir, '--frozen', 'python', '--version'
)
Write-Host '次のコマンドで練習一式を実行できます:'
Write-Host '  powershell -ExecutionPolicy Bypass -File .\run-practice.ps1'
