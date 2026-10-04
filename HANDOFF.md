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
├── packaging/forgeset.toml          the installer (not yet adapted — follow-up)
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
| `app` (Tauri, reskinned) | Done for this pass. New chrome (sidebar nav, dark/light theme, no mascot/skin picker), Voice Studio / Settings / Help views. `npx tsc --noEmit`, `npm run build`, and `cargo check` on `src-tauri` (against the finished core crate) all pass. Icons are still Nekotone's placeholders. |
| `packaging/server` | Ported from Nekotone's `packaging/server`, renamed (`voicekit` binary, `nekotone-voice-cli`). Not yet built/tested (needs the `cuda` feature and a Linux target to validate for real). |
| `packaging/forgeset.toml` (installer) | Not started — follow-up. |
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
- App icon/branding art: still Nekotone's placeholders; a follow-up before
  the installer is built.
- GPU support (DirectML, TensorRT for RTX) must stay in `nekotone-voice-core`'s
  **default** features (`default = ["ml", "gpu", "player"]`), matching
  Nekotone: consumers (`app`, `nekotone-voice-cli`) only add `features =
  ["server"]`/`["cuda"]` on top without `default-features = false`, so
  dropping `"gpu"` from the defaults (as a first extraction pass did)
  silently ships a binary with no GPU code paths at all. One binary always
  carries CPU + DirectML + TensorRT-RTX; `Accelerator::Auto` (the default)
  decides at runtime which one actually runs, falling back on any failure
  — never gate GPU support behind a separate build.
- The Forgeset installer config and `build.ps1`/`build-linux.sh` equivalents
  for this repo are not started.

## Change log

- **(unreleased)** First fork from Nekotone: trimmed `nekotone-voice-core`
  (engine), `nekotone-voice-cli` (`voicekit`: `models`/`serve`/`service`),
  the reskinned Tauri `app` (sleek/professional, no cat mascot or skin
  picker), `packaging/server`, and docs. Workspace builds, tests and
  clippy all pass (see §4/§5 for the handful of known, explained
  warnings/flaky tests). First commit and push to `origin/main`.
