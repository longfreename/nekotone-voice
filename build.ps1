# Build Voicekit in one go: tests, release binaries, the app, the docs
# site, and the installer (via Forgeset).
#
#   .\build.ps1            everything -> dist\Voicekit-<version>-Setup.exe and a portable zip
#   .\build.ps1 -Quick     skip tests and the app (reuses the last app build,
#                          so the previous build must have used -KeepBuildCache)
#   .\build.ps1 -Install   also install/upgrade it on this PC
#   .\build.ps1 -KeepBuildCache   keep the compiler caches (several GB on C:);
#                          by default they are deleted after a successful build
#
# `G:` is a slow CIFS share: build output always goes to %LOCALAPPDATA%, never
# under the repo itself (see HANDOFF.md section 2).

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$SkipTests,
    [switch]$SkipApp,
    [switch]$Install,
    [switch]$KeepBuildCache,
    [string]$Target = "$env:LOCALAPPDATA\nekotone-voice-target\main"
)
$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot
$started = Get-Date
if ($Quick) { $SkipTests = $true; $SkipApp = $true }
function Say($m, $c = "Cyan") { Write-Host "`n==> $m" -ForegroundColor $c }
function Fail($m) { Write-Host "`nBUILD FAILED: $m" -ForegroundColor Red; exit 1 }
function Have($exe) { $null -ne (Get-Command $exe -ErrorAction SilentlyContinue) }

Say "Checking tools"
if (-not (Have cargo)) { Fail "cargo is not on PATH (install Rust from https://rustup.rs)." }
$forgesetExe = (Get-Command forgeset -ErrorAction SilentlyContinue).Source
if (-not $forgesetExe -and (Test-Path "$env:LOCALAPPDATA\Programs\Forgeset\bin\forgeset.exe")) { $forgesetExe = "$env:LOCALAPPDATA\Programs\Forgeset\bin\forgeset.exe" }
if (-not $forgesetExe) { Fail "forgeset was not found on PATH or in %LOCALAPPDATA%\Programs\Forgeset\bin (install Forgeset)." }
if (-not $SkipApp -and -not (Have npm)) { Fail "npm is not on PATH (needed for the app). Install Node.js LTS or use -SkipApp." }
$ver = (Select-String -Path "$repo\Cargo.toml" -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
Write-Host "  Voicekit $ver"

$env:CARGO_TARGET_DIR = $Target
if (-not $SkipTests) {
    Say "Tests"
    cargo test --workspace --all-features --manifest-path "$repo\Cargo.toml"
    if ($LASTEXITCODE) { Fail "a test failed." }
}

Say "Release binaries"
cargo build --release -p nekotone-voice-cli --manifest-path "$repo\Cargo.toml"
if ($LASTEXITCODE) { Fail "voicekit.exe did not compile." }

if (-not $SkipApp) {
    Say "Voicekit app"
    Push-Location "$repo\app"
    try {
        if (-not (Test-Path node_modules)) { npm ci } else { npm ci --prefer-offline --no-audit --no-fund }
        if ($LASTEXITCODE) { Fail "npm ci failed." }
        $env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\nekotone-voice-target\app"
        npx tauri build --no-bundle
        if ($LASTEXITCODE) { Fail "the app did not build." }
    } finally { Pop-Location; $env:CARGO_TARGET_DIR = $Target }
}
$studio = "$env:LOCALAPPDATA\nekotone-voice-target\app\release\voicekit-studio.exe"
if (-not (Test-Path $studio)) { Fail "voicekit-studio.exe was not built (run without -SkipApp once)." }

Say "Staging"
$stage = "$repo\packaging\stage"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$stage\bin" | Out-Null
Copy-Item "$Target\release\voicekit.exe" "$stage\bin\"
Copy-Item $studio "$stage\bin\"
# ONNX Runtime's DLL sits next to the executables that use it (when it is dynamic).
Get-ChildItem "$Target\release" -Filter "onnxruntime*.dll" | Copy-Item -Destination "$stage\bin\"
Get-ChildItem (Split-Path $studio) -Filter "onnxruntime*.dll" | Copy-Item -Destination "$stage\bin\" -ErrorAction SilentlyContinue

# App-local Visual C++ runtime (same reasoning as Nekotone's build.ps1: ONNX
# Runtime needs msvcp140.dll 14.40+, and a file in the program's own folder
# is found before an older system copy).
$crt = @("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll")
foreach ($dll in $crt) {
    $src = Join-Path $env:WINDIR "System32\$dll"
    if (-not (Test-Path $src)) { Fail "$dll is missing from System32; install the Visual C++ 2015-2022 Redistributable (x64) on the build PC." }
    $v = (Get-Item $src).VersionInfo
    if ($dll -like "msvcp140*" -and ($v.FileMinorPart -lt 40)) { Fail "$dll on the build PC is $($v.FileVersion); 14.40 or newer is needed (update the Visual C++ Redistributable)." }
    Copy-Item $src "$stage\bin\"
}
Write-Host "  Visual C++ runtime $((Get-Item "$stage\bin\msvcp140.dll").VersionInfo.FileVersion) (app-local)"

# DirectML (GPU for the models, feature `gpu`, part of the default feature
# set since nekotone-voice-core always carries CPU + DirectML + TensorRT-RTX
# and picks at runtime). Without a DirectX 12 GPU the models fall back to
# the CPU; the DLL itself loads everywhere.
$dml = Get-ChildItem "$env:LOCALAPPDATA\ort.pyke.io\dfbin\x86_64-pc-windows-msvc" -Recurse -Filter "DirectML.dll" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $dml) { Fail "DirectML.dll was not found in the ONNX Runtime download cache (%LOCALAPPDATA%\ort.pyke.io); build once (the gpu feature is on by default)." }
Copy-Item $dml.FullName "$stage\bin\"
Write-Host "  DirectML $($dml.VersionInfo.FileVersion) (app-local)"

# Recommended models, bundled as installer components. gpu-nvidia is not a
# model: NVIDIA's TensorRT for RTX runtime, unpacked by `models get`.
$models = Join-Path $env:LOCALAPPDATA "NekotoneVoice\models"
foreach ($m in @("whisper-base", "tts-kokoro", "tts-chatterbox", "gpu-nvidia")) {
    & "$Target\release\voicekit.exe" models get $m | Out-Null   # no-op when present and verified
    if ($LASTEXITCODE) { Fail "could not get the $m model (network?)." }
    New-Item -ItemType Directory -Force "$stage\models\$m" | Out-Null
    Get-ChildItem "$models\$m" -File | Where-Object { $_.Extension -ne ".part" } | Copy-Item -Destination "$stage\models\$m\"
    Write-Host ("  model {0,-14} {1,7:N1} MB" -f $m, ((Get-ChildItem "$stage\models\$m" | Measure-Object Length -Sum).Sum / 1MB))
}

Say "Documentation site"
if (Test-Path "$repo\docs\nav.toml") {
    & $forgesetExe docs build "$repo\docs" --strict
    if ($LASTEXITCODE) { Fail "docs build failed." }
} else {
    New-Item -ItemType Directory -Force "$repo\docs\site" | Out-Null
    Copy-Item "$repo\README.md" "$repo\docs\site\" -Force
}

Say "Installer (Forgeset)"
& $forgesetExe build "$repo\packaging\forgeset.toml"
if ($LASTEXITCODE) { Fail "forgeset build failed." }

$took = [math]::Round(((Get-Date) - $started).TotalMinutes, 1)
Say "Built Voicekit $ver in $took min" "Green"
Get-ChildItem "$repo\dist" | ForEach-Object { Write-Host ("  {0,-44} {1,7:N1} MB" -f $_.Name, ($_.Length / 1MB)) }

if ($Install) {
    Say "Installing"
    $setup = Get-ChildItem "$repo\dist\Voicekit-$ver-Setup.exe" | Select-Object -First 1
    $tmp = Join-Path $env:TEMP $setup.Name
    Copy-Item $setup.FullName $tmp -Force
    $p = Start-Process $tmp -ArgumentList "/VERYSILENT", "/TASKS=path,desktop" -Wait -PassThru
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { Fail "the installer returned exit code $($p.ExitCode)" }
    if ($p.ExitCode -eq 3010) { Write-Host "  installed; one file is replaced when Windows restarts" }
    Write-Host "  installed; open a new terminal for the updated PATH"
}

# Clean up after ourselves: the compiler caches are many GB each and
# everything needed is now in dist\. -KeepBuildCache keeps them for a
# faster rebuild.
if (-not $KeepBuildCache) {
    Say "Cleaning up build caches"
    foreach ($dir in @($Target, "$env:LOCALAPPDATA\nekotone-voice-target\app", $stage)) {
        if (Test-Path $dir) {
            $gb = [math]::Round(((Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum) / 1GB, 1)
            Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  removed $dir ($gb GB)"
        }
    }
}
