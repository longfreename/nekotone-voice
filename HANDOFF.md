# Voicekit: handoff for the next engineer (or model)

This file is the continuation point for Voicekit. It says what Voicekit is,
how it is built, what state every part is in, and how the fork from
Nekotone was done. Keep it current: update the *State* table and the
*Change log* whenever you land something. User-facing changes also go in
`CHANGELOG.md`; module layout and data flow in `ARCHITECTURE.md`.

## 1. What it is

Voicekit changes your voice live (Voice Studio: presets, pads, a virtual
microphone), clones your voice from a short sample, and uses the clone two
ways: **Speak for me** (typed or spoken text, instantly, in your voice) and
**Change my voice** (re-reads your own speech in a chosen voice). A
compute-server mode lets a PC with a graphics card do the voice-clone work
for another PC on the network (`voicekit serve`, or a Windows
service/systemd unit).

It is a **fork of [Nekotone](https://github.com/longfreename/nekotone)**,
an audio indexer and player: Voicekit keeps only the voice features (Voice
Studio, voice cloning, Speak for me) and the Windows/Linux compute-server
mechanism, dropped everything else (indexing, search, the player, dub,
MIDI, the notebook), and reskinned the app with a sleek, professional
look — no cat mascot, no window-skin picker.

## 2. Layout and build

```
G:\nekotone-voice
├── Cargo.toml                       workspace: crates/*  (app/ is excluded: its own Cargo project)
├── crates/nekotone-voice-core       the engine, trimmed from nekotone-core (see ARCHITECTURE.md for the module table)
├── crates/nekotone-voice-cli        `voicekit` command: models, serve, service (thin over the core; no library commands)
├── app                              the Voicekit app: Tauri 2 + Vite + TypeScript, reskinned
├── packaging/server                 Dockerfile/README/compose for the compute server on Linux
├── packaging/forgeset.toml          the installer (Forgeset, fluent template, dark, Express install)
├── packaging/gen_icon.py            regenerates the app icon set (python packaging/gen_icon.py; needs Pillow)
├── docs/                            the manual (nav.toml + guides; `forgeset docs build docs --strict`)
├── CHANGELOG.md
└── HANDOFF.md                       this file
```

Build commands (PowerShell, from `G:\nekotone-voice`; `G:` is a slow network
share, so build output is always redirected off it):

```powershell
$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\nekotone-voice-target\main"
cargo test --workspace
cargo clippy --workspace --all-targets
cargo build --release -p nekotone-voice-cli    # → voicekit.exe

cd app; npm ci
npx tsc --noEmit
$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\nekotone-voice-target\app"; npx tauri build --no-bundle   # → the Voicekit app
```

**`G:` is a CIFS share: cargo reads every source file over the network on
each incremental build, which made a from-scratch `cargo build -p
nekotone-voice-core --all-features` take over an hour during the fork (the
same build takes well under a minute from a local disk).** If a build on
`G:` is taking far longer than its size would suggest, don't wait it out:
`robocopy` the repo (excluding `node_modules`/`target`) to a local path,
redirect `CARGO_TARGET_DIR` there too, iterate locally, then `robocopy`
the fixed source back to `G:` (or do the git commit/push from the local
copy and mirror it back). This is how the core crate was actually finished.

## 3. How the fork was done

Source: Nekotone at commit `e6956f4` (origin/main, 2026-10-04), after a
correctness pass (see its own HANDOFF.md). Decisions made with the user
before forking:

- Scope: Voice Studio + voice cloning + Speak for me + the compute server.
  Excluded: Dub-to-English, the notebook/recorder, library/search/indexing,
  MIDI, the full player.
- Code sharing: a **new, trimmed core crate** (`nekotone-voice-core`), not
  a feature-flagged reuse of `nekotone-core` — the two engines can now
  diverge independently.
- Location: a brand-new, separate GitHub repository
  (`longfreename/nekotone-voice`), not a branch or folder of Nekotone.
- A matching slim CLI ships (`voicekit`): `models`, `serve`, `service` only
  — mirroring Nekotone's own split, where the live voice changer and Speak
  for me are interactive and only live in the app, never the CLI.
- Reskin: sleek and professional, explicitly **no Neko/cat skin option
  anywhere** (user's instruction, verbatim).

Module-by-module inclusion/exclusion reasoning is in `ARCHITECTURE.md`.

## 4. State

| Part | State |
|---|---|
| `nekotone-voice-cli` (`models`, `serve`, `service`) | Done. Builds and its 5 end-to-end tests pass against the real core crate. |
| `nekotone-voice-core` | Done. Extracted from `nekotone-core` per the module table in ARCHITECTURE.md. `cargo test -p nekotone-voice-core --all-features`: 195 passed, 2 flaky, 40 ignored (need downloaded models). `cargo clippy -p nekotone-voice-core --all-features --all-targets`: 4 warnings, all explained in §5 — no errors. |
| `app` (Tauri, reskinned) | Done for this pass. New chrome (sidebar nav, dark/light theme, no mascot/skin picker), Voice Studio / Settings / Help views. `npx tsc --noEmit`, `npm run build`, and `cargo check` on `src-tauri` (against the finished core crate) all pass. GUI binary renamed `voicekit-studio` (was colliding with the CLI's `voicekit` bin name at install-staging time). New icon: a plain waveform mark on the app's own dark/indigo gradient (`packaging/gen_icon.py`), replacing Nekotone's cat mascot art the scaffold had copied in by default. |
| `packaging/server` | Ported from Nekotone's `packaging/server`, renamed (`voicekit` binary, `nekotone-voice-cli`). Not yet built/tested (needs the `cuda` feature and a Linux target to validate for real). |
| `packaging/forgeset.toml` (installer) | **Built end to end.** `build.ps1 -SkipTests` from a local (non-`G:`) copy ran the full pipeline: release CLI + app, all 4 model downloads (whisper-base, tts-kokoro, tts-chatterbox, gpu-nvidia), docs site, then `forgeset build` — in 11 min total. Output: `Voicekit-0.1.0-Setup.exe` (1.5 GB, 74 files staged, checksums OK) and `Voicekit-0.1.0-portable.zip` (34 MB, core files only). Not code-signed yet (SmartScreen will warn — expected for now). `packaging/forgeset.lock` is now committed per Forgeset's own advice, to protect future upgrades. Not yet tested: `forgeset test` (sandboxed install/uninstall) and `forgeset diff` against a prior release (no prior release exists yet). |
| Docs (`docs/guides/*`, `CLI.md`, `FAQ.md`, `ARCHITECTURE.md`) | Written, scoped to the voice-only feature set; `voice-studio.md` is Nekotone's guide with only the product name substituted (its content did not reference any dropped feature). |
| CI / first `git push` | This commit is the first one; pushed to `origin/main`. |

## 5. Known open items

- Whether `denoise.rs`/`cache.rs` are needed in the trimmed core: resolved
  by grep before extraction — neither is referenced by `voice`/`tts`/
  `record::{align,capture,vad}` outside test modules, so both are dropped.
- Two `voice::speak::tests` (`change_my_voice_keeps_my_pauses`,
  `change_my_voice_catches_up_in_pauses`) are timing-sensitive and can fail
  under parallel `cargo test` load (CPU contention skews the real-time
  pacing they measure); both pass reliably run alone or with
  `--test-threads=1`. Same category as the one flaky test already
  documented in Nekotone's own HANDOFF.md — not a logic bug.
- `cargo clippy -p nekotone-voice-core` reports 4 warnings on the plain
  `lib` target: `VadConfig::captions()` and `Utterance::{start_secs,
  end_secs}` are only called from that file's own `#[cfg(test)]` module
  (invisible to dead-code analysis in a non-test build), plus 2 pre-existing
  style nits (`type_complexity`, `collapsible_match`) ported as-is from
  Nekotone. Building the core crate as a plain dependency (from the CLI or
  the app) additionally flags a handful of `nvidia.rs` helpers used only
  under `cfg(windows, feature = "gpu")` or only by its own tests — expected,
  not a regression.
- App icon/branding art: replaced — `packaging/gen_icon.py` draws a plain
  waveform mark on the app's own dark/indigo gradient (no mascot), used by
  both the app window and the installer. Re-run it after any palette change
  in `styles.css` to keep them in sync.
- GPU support (DirectML, TensorRT for RTX) must stay in `nekotone-voice-core`'s
  **default** features (`default = ["ml", "gpu", "player"]`), matching
  Nekotone: consumers (`app`, `nekotone-voice-cli`) only add `features =
  ["server"]`/`["cuda"]` on top without `default-features = false`, so
  dropping `"gpu"` from the defaults (as a first extraction pass did)
  silently ships a binary with no GPU code paths at all. One binary always
  carries CPU + DirectML + TensorRT-RTX; `Accelerator::Auto` (the default)
  decides at runtime which one actually runs, falling back on any failure
  — never gate GPU support behind a separate build.
- The CLI and the app's Tauri package both defaulted to a `[[bin]]` named
  `voicekit`; harmless for `cargo build` (different, excluded workspaces)
  but would collide when both land in the installer's `{app}/bin`. The
  app's binary is now `voicekit-studio`; the `voicekit` command stays the
  CLI/compute-server entry point.
- `packaging/forgeset.toml`/`build.ps1` have now been run end to end from
  a local (non-`G:`) copy, producing a real `Voicekit-0.1.0-Setup.exe`
  (1.5 GB) and `Voicekit-0.1.0-portable.zip` (34 MB); `forgeset inspect`
  confirms all streams/checksums OK. Not signed (SmartScreen will warn —
  expected pre-signing-cert). Still open: a sandboxed install/uninstall
  smoke test (`forgeset test`) to confirm nothing is left behind, and
  `forgeset diff` once a second release exists.
- `voicekit.exe` release build smoke-tested directly (command help surface,
  `models list`, `service status`, `serve`): GPU auto-detection genuinely
  found this machine's NVIDIA RTX 4070 Ti through DirectML (confirms the
  "one binary, runtime decides" requirement for real, not just by reading
  the code); `service status` correctly reports not installed; `serve`
  fails with a clear, actionable message and no hang/crash/leftover process
  when the voice-clone model isn't downloaded yet. Did not install a real
  Windows service or download the 1.3 GB voice-clone model in this pass —
  both are real side effects disproportionate to a smoke test; `service
  install`/`uninstall` is covered by `nekotone-voice-cli`'s unit tests.
- The scripted UI test harness (`app/tests/ui/*.js`, driven by
  `--capture`/`--script`) has not been ported yet — it does not exist in
  this repo. Follow-up; the app's correctness so far rests on `tsc`,
  `npm run build`, and `cargo check`, not an automated UI walkthrough.

## Change log

- **(unreleased)** First fork from Nekotone: trimmed `nekotone-voice-core`
  (engine), `nekotone-voice-cli` (`voicekit`: `models`/`serve`/`service`),
  the reskinned Tauri `app` (sleek/professional, no cat mascot or skin
  picker), `packaging/server`, and docs. Workspace builds, tests and
  clippy all pass (see §4/§5 for the handful of known, explained
  warnings/flaky tests). First commit and push to `origin/main`.
- **(unreleased)** Installer and branding follow-up: wrote
  `packaging/forgeset.toml` (fluent template, dark, the app's own accent
  colour, no file-type associations/library page) and a matching
  `build.ps1`; replaced the cat-mascot icon the scaffold had copied in
  with a plain waveform mark (`packaging/gen_icon.py`) used by both the
  app and the installer; renamed the app's binary to `voicekit-studio` to
  stop it colliding with the CLI's `voicekit` at install-staging time.
- **(unreleased)** Built the real distributable: `build.ps1 -SkipTests`
  from a local copy ran release compiles, model downloads, docs, and
  `forgeset build` end to end (11 min), producing `Voicekit-0.1.0-Setup.exe`
  (1.5 GB) and `Voicekit-0.1.0-portable.zip` (34 MB); `forgeset inspect`
  confirmed checksums OK. Committed `packaging/forgeset.lock` (protects
  future upgrade paths, per Forgeset's own guidance).
