<#
.SYNOPSIS
    Build the TRELLIS.2 engine (trellis-server.exe and friends) from a pinned
    trellis.cpp commit, and package it as the one zip models.py downloads.

.DESCRIPTION
    Why this exists: v0.8.1, the last upstream release, has no --faces. The face
    target (QEM simplification *before* the engine's own UV and PBR bake, a
    per-request /generate field) landed on upstream's default branch afterwards and
    is in no release, so the engine is built from a commit we choose and publish
    ourselves. The pin is -Commit; the default below is the head that was read on
    2026-10-15 (src/trellis-server.cpp there accepts "faces", "model" and "fov" as
    multipart fields, and CMakeLists.txt defines the post-replay target).

    Follows native/build.ps1's idiom: comment help, $ErrorActionPreference = 'Stop',
    the vcvars lookup, a generated batch file for anything that needs the developer
    environment. Everything is written under build/ (gitignored). Nothing is
    uploaded: the last thing printed is the `gh release create` line to run by hand,
    because a published release is the one step here that cannot be taken back.

    What it checks rather than assumes:
      - the checked-out head equals -Commit, and the source tree is clean;
      - the CUDA toolkit is major version 13 (the zip ships cudart64_13 and friends,
        and models.py's probe list names them) -- the machine default may well be 12;
      - the built trellis-server.exe lists --faces and --model in its own --help
        (the capability the whole slice exists for).

    CMAKE_CUDA_ARCHITECTURES is upstream's own CI matrix for the CUDA 13 leg
    (.github/workflows/release.yml): 75;80;86;89;90;120, SASS-only for Blackwell.
    Upstream marks its Windows CUDA leg `experimental`, so a failure to build here is
    a finding to report, not a bug in this script.

.PARAMETER Commit
    The trellis.cpp commit to build (full SHA recommended; any unambiguous prefix
    is resolved and the full SHA is what gets recorded).

.PARAMETER CudaPath
    A CUDA 13.x toolkit root. Default: the highest CUDA_PATH_V13_* in the environment.

.PARAMETER Retopo
    Also build the optional native quad retopology (-DTRELLIS_RETOPO_MATCHING=ON
    -DTRELLIS_RETOPO_COLLISION=ON) into a separate -retopo zip. Off the critical path:
    CUDA together with retopo has not been shown to build on Windows. Retopology
    links CGAL (GPL), so this also writes the matching sources beside the zip; if
    that zip is ever published, publish the sources with it.

.PARAMETER Repo
    The upstream clone URL.

.PARAMETER Clean
    Delete the build directory (not the source clone) before configuring.
#>
[CmdletBinding()]
param(
    [string]$Commit = '610c90e9aca282ac2fb489b5d62b7ed5d1c177c9',
    [string]$CudaPath,
    [switch]$Retopo,
    [string]$Repo = 'https://github.com/pwilkin/trellis.cpp',
    [string]$CudaArchs = '75;80;86;89;90;120',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$work = Join-Path $root 'build'
$src = Join-Path $work 'trellis-src'

function Invoke-Native {
    # A native command's failure does not trip $ErrorActionPreference; say so loudly.
    param([Parameter(Mandatory)][string]$What, [Parameter(Mandatory)][scriptblock]$Run)
    & $Run
    if ($LASTEXITCODE -ne 0) { throw "$What failed ($LASTEXITCODE)" }
}

function Find-VcVars {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $install = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath 2>$null
    if (-not $install) { return $null }
    $vcvars = Join-Path $install 'VC\Auxiliary\Build\vcvars64.bat'
    return (Test-Path $vcvars) ? $vcvars : $null
}

function Find-Cuda13 {
    if ($CudaPath) { return $CudaPath }
    $found = Get-ChildItem Env: | Where-Object { $_.Name -match '^CUDA_PATH_V13_(\d+)$' } |
        Sort-Object { [int]($_.Name -replace '^CUDA_PATH_V13_', '') } -Descending |
        Select-Object -First 1
    if (-not $found) { throw 'no CUDA 13 toolkit found: pass -CudaPath (CUDA_PATH_V13_* is unset)' }
    return $found.Value
}

# --- toolchain ---------------------------------------------------------------
$vcvars = Find-VcVars
if (-not $vcvars) { throw 'MSVC (vcvars64.bat) not found; install the VC build tools' }
foreach ($tool in 'git', 'cmake', 'ninja') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is not on PATH" }
}
$cuda = Find-Cuda13
$nvcc = Join-Path $cuda 'bin\nvcc.exe'
if (-not (Test-Path $nvcc)) { throw "no nvcc at $nvcc" }
$nvccText = (& $nvcc --version) -join "`n"
if ($nvccText -notmatch 'release (\d+)\.(\d+)') { throw "cannot read the nvcc version from: $nvccText" }
if ([int]$Matches[1] -ne 13) {
    throw "CUDA $($Matches[1]).$($Matches[2]) at ${cuda}: the engine zip ships the CUDA 13 runtime (cudart64_13), build with a 13.x toolkit"
}
$nvccVersion = "$($Matches[1]).$($Matches[2])"
Write-Host "CUDA: $cuda ($nvccVersion)"
Write-Host "MSVC: $vcvars"

# --- source, pinned ----------------------------------------------------------
if (-not (Test-Path (Join-Path $src '.git'))) {
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    Invoke-Native 'git clone' { git clone $Repo $src }
} else {
    Invoke-Native 'git fetch' { git -C $src fetch --tags origin }
}
Invoke-Native 'git checkout' { git -C $src checkout --detach $Commit }
Invoke-Native 'git submodule update' { git -C $src submodule update --init --recursive }
$head = (git -C $src rev-parse HEAD).Trim()
$want = (git -C $src rev-parse "$Commit^{commit}").Trim()
if ($head -ne $want) { throw "head is $head, expected $want" }
if (git -C $src status --porcelain) { throw "$src is not clean; refusing to build a tree that is not the pinned commit" }
$short = $head.Substring(0, 8)
Write-Host "trellis.cpp: $head"

# --- configure and build -----------------------------------------------------
$suffix = $Retopo ? '-retopo' : ''
$buildDir = Join-Path $work "trellis-build$suffix"
$dist = Join-Path $work "trellis-dist$suffix"
$outDir = Join-Path $work "trellis-out\$short"
if ($Clean -and (Test-Path $buildDir)) { Remove-Item -Recurse -Force $buildDir }
foreach ($d in $dist, $outDir) {
    if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# GGML_OPENMP=OFF is what upstream's Windows legs pass: ggml's own thread pool,
# no libomp DLL to ship.
$flags = @(
    '-DCMAKE_BUILD_TYPE=Release', '-DGGML_CUDA=ON', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs",
    '-DGGML_OPENMP=OFF', "-DCMAKE_CUDA_COMPILER=$($nvcc -replace '\\', '/')"
)
$targets = @('trellis-server', 'trellis-cli', 'post-replay')
if ($Retopo) {
    $flags += '-DTRELLIS_RETOPO_MATCHING=ON', '-DTRELLIS_RETOPO_COLLISION=ON'
    $targets += 'trellis-retopo-atlas'
}
$flagArgs = ($flags | ForEach-Object { '"' + $_ + '"' }) -join ' '
$msvcFile = Join-Path $work 'msvc-version.txt'
$bat = Join-Path $work "_build-trellis$suffix.bat"
@(
    '@echo off'
    "call `"$vcvars`" >nul || exit /b 1"
    "echo %VCToolsVersion%> `"$msvcFile`""
    "set `"CUDA_PATH=$cuda`""
    "set `"PATH=$cuda\bin;$cuda\bin\x64;%PATH%`""
    # Every flag quoted: the nvcc path has spaces, and cmd treats ';' (the arch list) as a separator.
    "cmake -S `"$src`" -B `"$buildDir`" -G Ninja $flagArgs || exit /b 1"
    "cmake --build `"$buildDir`" --target $($targets -join ' ') -j || exit /b 1"
    'exit /b 0'
) | Set-Content -Path $bat -Encoding ascii
Invoke-Native 'trellis.cpp configure/build' { & cmd.exe /c "`"$bat`"" }
$msvcVersion = (Get-Content $msvcFile -ErrorAction SilentlyContinue | Select-Object -First 1)

# --- assemble the flat zip ---------------------------------------------------
foreach ($exe in $targets) {
    $path = Join-Path $buildDir "$exe.exe"
    if (-not (Test-Path $path)) { throw "build produced no $exe.exe" }
    Copy-Item $path $dist
}
# ggml's DLLs land beside the executables or in bin/, depending on its CMake.
Get-ChildItem -Path $buildDir, (Join-Path $buildDir 'bin') -Filter '*.dll' -File -ErrorAction SilentlyContinue |
    Copy-Item -Destination $dist -Force
# CUDA 13 moved the Windows runtime DLLs into bin\x64; look wherever they live and
# fail rather than ship a zip with no CUDA runtime.
foreach ($dll in 'cudart64_13', 'cublas64_13', 'cublasLt64_13') {
    $hit = Get-ChildItem -Path $cuda -Recurse -Filter "$dll.dll" -File | Select-Object -First 1
    if (-not $hit) { throw "$dll.dll not found under $cuda" }
    Copy-Item $hit.FullName $dist -Force
}

# The capability the slice exists for, read from the binary rather than assumed.
$help = (& (Join-Path $dist 'trellis-server.exe') --help 2>&1) -join "`n"
$caps = [ordered]@{
    faces       = [bool]($help -match '--faces')
    model       = [bool]($help -match '--model')
    dump_post   = [bool]($help -match '--dump-post')
    no_texture  = [bool]($help -match '--no-texture')
    post_replay = [bool](Test-Path (Join-Path $dist 'post-replay.exe'))
}
if (-not $caps.faces) { throw "trellis-server.exe built from $short does not list --faces in --help" }

[ordered]@{
    repo        = $Repo
    commit      = $head
    flags       = $flags
    targets     = $targets
    cuda        = $nvccVersion
    cuda_archs  = $CudaArchs
    msvc        = $msvcVersion
    retopo      = [bool]$Retopo
    capabilities = $caps
    built_utc   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
} | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $dist 'trellis-build.json') -Encoding utf8

$asset = "trellis-cuda-windows-x64$suffix.zip"
$zip = Join-Path $outDir $asset
Compress-Archive -Path (Join-Path $dist '*') -DestinationPath $zip -CompressionLevel Optimal

if ($Retopo) {
    # CGAL is GPL: whoever receives the retopo binaries is owed the matching source.
    $srcZip = Join-Path $outDir "trellis-cpp-$short-sources.zip"
    Invoke-Native 'git archive' { git -C $src archive --format=zip -o $srcZip $head }
    git -C $src submodule status --recursive | Set-Content -Path (Join-Path $outDir "submodules-$short.txt")
}

# --- report ------------------------------------------------------------------
$sha = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
$bytes = (Get-Item $zip).Length
$unpacked = (Get-ChildItem $dist -File | Measure-Object Length -Sum).Sum
$files = Get-ChildItem $dist -File | Sort-Object Name
Write-Host ''
Write-Host "zip:       $zip"
Write-Host "sha256:    $sha"
Write-Host ("size:      {0:N0} B ({1:N2} GiB); unpacked {2:N0} B ({3:N2} GiB)" -f $bytes, ($bytes / 1GB), $unpacked, ($unpacked / 1GB))
Write-Host "caps:      $($caps | ConvertTo-Json -Compress)"
Write-Host ''
Write-Host '# --- paste into src/realmspinner/models.py (TRELLIS_RUNTIME_*) ---'
Write-Host "TRELLIS_RUNTIME_VERSION = `"engine-trellis-$short`""
Write-Host "TRELLIS_RUNTIME_SHA256 = (`n    `"$sha`"`n)"
Write-Host 'TRELLIS_RUNTIME_FILES = ('
$files | ForEach-Object { Write-Host "    `"$($_.Name)`"," }
Write-Host ')'
Write-Host 'TRELLIS_RUNTIME_DIGESTS: tuple[tuple[str, str], ...] = ('
$files | ForEach-Object {
    Write-Host "    (`"$($_.Name)`", `"$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())`"),"
}
Write-Host ')'
Write-Host ("# Fetch(size_gib={0:N2}, unpack_gib={1:N2})" -f ($bytes / 1GB), ($unpacked / 1GB))
Write-Host ''
Write-Host '# --- publish by hand (this script does not): ---'
Write-Host "gh release create engine-trellis-$short `"$zip`" --repo <owner>/<repo> --title `"TRELLIS.2 engine $short`" --notes `"trellis.cpp $head, CUDA $nvccVersion`""
