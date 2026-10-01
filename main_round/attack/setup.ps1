[CmdletBinding()]
param(
    [string]$Python,
    [string]$Uv,
    [switch]$Offline
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'prepare_attack_m.ps1')
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$legacyUv = Join-Path $repoRoot 'prelim/workspace/.tools/uv/uv.exe'
$legacyPython = Join-Path $repoRoot 'prelim/workspace/.tools/python/cpython-3.12.13-windows-x86_64-none/python.exe'
$attackPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$starter = Join-Path $PSScriptRoot 'official-kit/starter'
$requirements = Join-Path $starter 'requirements-kit.txt'
foreach ($requiredPath in @($requirements, (Join-Path $starter 'pyproject.toml'))) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Required local file missing: $requiredPath" }
}

function Resolve-Executable([string]$Value) {
    if (Test-Path -LiteralPath $Value -PathType Leaf) {
        return (Resolve-Path -LiteralPath $Value).Path
    }
    $command = Get-Command -Name $Value -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    return $null
}

function Find-CompatiblePython([string]$Executable, [string[]]$Arguments = @()) {
    if (-not $Executable) { return $null }
    # Avoid launching the Microsoft Store through a Windows execution alias.
    if ($Executable -match '\\WindowsApps\\python(?:3)?\.exe$') { return $null }
    try {
        $pythonResult = & $Executable @Arguments -I -c 'import sys; print(sys.executable); sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>$null
        if ($LASTEXITCODE -eq 0 -and $pythonResult -and (Test-Path -LiteralPath $pythonResult -PathType Leaf)) {
            return [string]$pythonResult
        }
    } catch {
        # Continue automatic discovery when a candidate cannot run.
    }
    return $null
}

$attackUv = $null
if ($Uv) {
    $attackUv = Resolve-Executable $Uv
    if (-not $attackUv) { throw "uv executable not found: $Uv" }
} elseif (Test-Path -LiteralPath $legacyUv -PathType Leaf) {
    $attackUv = $legacyUv
} else {
    $attackUv = Resolve-Executable 'uv'
}

$basePython = $null
if ($Python) {
    $basePython = Find-CompatiblePython (Resolve-Executable $Python)
    if (-not $basePython) { throw '-Python must point to a working Python 3.11 or newer executable.' }
} else {
    $basePython = Find-CompatiblePython (Resolve-Executable $legacyPython)
    foreach ($candidate in @('python', 'python3')) {
        if (-not $basePython) { $basePython = Find-CompatiblePython (Resolve-Executable $candidate) }
    }
    if (-not $basePython) { $basePython = Find-CompatiblePython (Resolve-Executable 'py') @('-3') }
}
if (Test-Path -LiteralPath $attackPython -PathType Leaf) {
    if (-not (Find-CompatiblePython $attackPython)) {
        throw 'The existing .venv is unusable or older than Python 3.11. Rename it and rerun setup.ps1 with a compatible interpreter.'
    }
} elseif (-not $attackUv -and -not $basePython) {
    throw 'Install Python 3.11+ or uv first, or supply -Python / -Uv with an existing executable path.'
}
if ($Offline -and -not $attackUv) {
    throw '-Offline requires uv and a populated local uv cache. Omit -Offline for standard venv/pip setup.'
}

$oldUvCache = $env:UV_CACHE_DIR
$oldUvPythonInstall = $env:UV_PYTHON_INSTALL_DIR
$oldPipCache = $env:PIP_CACHE_DIR
try {
    $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache/uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.tools/python'
    $env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache/pip'
    if ($attackUv) {
        $uvOptions = @()
        if ($Offline) { $uvOptions += '--offline' }
        if (-not (Test-Path -LiteralPath $attackPython)) {
            # With uv alone, request Python 3.12; any managed download stays here.
            $pythonRequest = if ($basePython) { $basePython } else { '3.12' }
            & $attackUv @uvOptions venv --python $pythonRequest (Join-Path $PSScriptRoot '.venv')
            if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
        }
        & $attackUv @uvOptions pip install --python $attackPython -r $requirements -e $starter
    } else {
        if (-not (Test-Path -LiteralPath $attackPython)) {
            & $basePython -m venv (Join-Path $PSScriptRoot '.venv')
            if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
        }
        & $attackPython -m pip install --disable-pip-version-check -r $requirements -e $starter
    }
    if ($LASTEXITCODE -ne 0) { throw 'Official dependency installation failed.' }
} finally {
    $env:UV_CACHE_DIR = $oldUvCache
    $env:UV_PYTHON_INSTALL_DIR = $oldUvPythonInstall
    $env:PIP_CACHE_DIR = $oldPipCache
}
& (Join-Path $PSScriptRoot 'attack.ps1') check
