[CmdletBinding()]
param()

# Preserve M's frozen module names, source bytes, and historical hash keys.
# Canonical files live in Attack_M; M is an ignored local runtime view.
$ErrorActionPreference = 'Stop'
$canonical = Join-Path $PSScriptRoot 'Attack_M'
$compat = Join-Path $PSScriptRoot 'M'
$marker = Join-Path $compat '.attack_m_compat'
if (-not (Test-Path -LiteralPath (Join-Path $canonical 'v2/selection.json'))) {
    throw 'Attack_M frozen sources are missing.'
}
if (Test-Path -LiteralPath $compat) {
    $item = Get-Item -LiteralPath $compat -Force
    # OneDrive also sets ReparsePoint on ordinary cloud-backed directories.
    # Reject directory links, while allowing the managed local OneDrive folder.
    if ($item.LinkType -in @('Junction', 'SymbolicLink') -or
        -not $item.PSIsContainer -or -not (Test-Path -LiteralPath $marker)) {
        throw 'M already exists and is not a managed compatibility directory. Preserve it and resolve the collision first.'
    }
} else {
    New-Item -ItemType Directory -Path $compat | Out-Null
    [IO.File]::WriteAllText($marker, 'Local compatibility view of ../Attack_M; do not edit here.')
}

$sources = @(Get-Item -LiteralPath (Join-Path $canonical '__init__.py'))
foreach ($folder in @('v1', 'v2', 'strategy')) {
    $sources += @(Get-ChildItem -LiteralPath (Join-Path $canonical $folder) -File -Recurse |
        Where-Object { $_.FullName -notmatch '[\\/]__pycache__[\\/]' -and $_.Extension -ne '.pyc' })
}
foreach ($source in $sources) {
    $relative = $source.FullName.Substring($canonical.Length + 1)
    $destination = Join-Path $compat $relative
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash -ne
            (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
            throw "Compatibility copy differs: $relative. No existing file was overwritten."
        }
    } else {
        New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
        Copy-Item -LiteralPath $source.FullName -Destination $destination
    }
}

# Outputs have one physical location, so new and old entry points see the same runs.
$canonicalOutput = Join-Path $canonical 'local-output'
$compatOutput = Join-Path $compat 'local-output'
if (-not (Test-Path -LiteralPath $canonicalOutput)) {
    New-Item -ItemType Directory -Path $canonicalOutput | Out-Null
}
$outputItem = Get-Item -LiteralPath $compatOutput -Force -ErrorAction SilentlyContinue
if ($outputItem) {
    if ($outputItem.LinkType -ne 'Junction' -or
        [IO.Path]::GetFullPath([string]$outputItem.Target) -ne [IO.Path]::GetFullPath($canonicalOutput)) {
        throw 'M/local-output is not the expected junction to Attack_M/local-output.'
    }
} else {
    New-Item -ItemType Junction -Path $compatOutput -Target $canonicalOutput | Out-Null
}
Write-Output "Attack_M compatibility ready: $($sources.Count) unchanged source files; outputs remain in Attack_M/local-output."
