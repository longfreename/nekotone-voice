# Voicekit architecture

Voicekit changes your voice live, clones it, and uses the clone to read text
or to re-read what you say — nothing else. It is a trimmed, reskinned fork
of [Nekotone](https://github.com/longfreename/nekotone); where this document
is silent, Nekotone's `ARCHITECTURE.md` describes the shared mechanism
(models/acceleration, the async-Tauri rule, the morph rule).

```
G:\nekotone-voice
├── crates/nekotone-voice-core   the engine: decode, the voice changer, voice cloning, speak-for-me,
│                                the compute-server protocol
├── crates/nekotone-voice-cli    `voicekit` command: compute server (serve/service), model management
├── app                          the Voicekit app: Tauri 2 + TypeScript, a sleek/professional chrome
│                                (no mascot, no window-skin picker)
├── packaging/forgeset.toml      the installer
└── docs                         guides (rendered by `forgeset docs build`)
```

## Modules (nekotone-voice-core)

Ported near-verbatim from Nekotone's `nekotone-core`, keeping only what the
voice features touch:

| Module | What it does | Models |
|---|---|---|
| `audio` | probe, decode to mono/stereo f32, resample (aligned sinc), streaming `Source`, WAV writer | – |
| `record::{align, capture, vad}` | resampling alignment, capture sinks/sources (mic, loopback), voice-activity detection — used by `voice::speak`'s listen path | – |
| `stt` | Whisper through ONNX Runtime — used by Speak-for-me's listen path | `whisper-*` |
| `voice` | real-time voice changer (LP-PSOLA, preset chains, pads, virtual mic), `voice::identity` (speaker embedding for cloning), `voice::speak` (Speak for me: VAD → Whisper → Kokoro/clone → effect → output) | `tts-kokoro`, `whisper-*` |
| `tts` | text to speech: English normalisation, misaki-lexicon G2P, Kokoro-82M (24 kHz, sentence streaming), `tts::chatterbox` (the voice-clone model) | `tts-kokoro` (CPU only), `tts-chatterbox` |
| `remote` | the compute-server protocol: client and server halves, used by both `voicekit serve` and the app's **Share this PC** | – |
| `models` | catalogue (pinned URLs, sizes, SHA-256) trimmed to the models this product uses, resumable cancellable downloads, `onnx_session` | – |
| `nvidia` | TensorRT for RTX detection and benchmarking (see Nekotone's HANDOFF §5 for why some graphs stay on DirectML/CPU) | – |
| `threads`, `error` | worker counts/priorities; user-facing error type | – |

Dropped entirely (library/search/player/notebook-only, not reachable from
the voice features): `audioset`, `cache`, `classify`, `diarize`, `dub`,
`embed`, `fingerprint`, `index`, `insight`, `library`, `llm`, `midi`,
`midi_tune`, `player`, `similar`, `stems`, `clap.rs` (the CLAP sound model),
and `record::{live, notes, writer}` (Notebook-only).

## Data flow

```text
voice changer: mic ─ voice::Processor (preset chain: pitch/formant, pads) ─▶ virtual mic / monitor
speak:         mic (record::CaptureSource) ─ listen thread: mono → 16 kHz → record::vad ─▶ stt thread: Whisper
               ─▶ queue ◀─ say(text) ─▶ tts thread: tts::Tts (Kokoro or chatterbox clone, sentence by sentence)
               → sinc 24 kHz→device → voice::Processor (effect, limiter)
               ─▶ SPSC rings ─▶ OutputTap in the output/monitor callbacks; watch thread: Speaking/Done events + latency
compute server: app (Settings → Share this PC) or `voicekit serve`/`service` ─▶ remote::ChatterboxEngine
               (loads tts::chatterbox once) ─▶ HTTP ─▶ another Voicekit's remote client, used in place of the
               local clone when it is configured and answers faster than the local CPU would
```

Rules kept from Nekotone: nothing blocks the UI thread (every Tauri command
touching disk, a device or ML is async + `spawn_blocking`); a view fills in
every missing settings key itself, because `settings.json` persists across
upgrades; views re-render only on visible change and morph the DOM in place
(the morph rule).

## Models and acceleration

Same mechanism as Nekotone: `models::onnx_session*` tries TensorRT for RTX,
then DirectML, then the CPU, falling back on any failure. Kokoro and the
speaker/voice-clone embedding stay on the CPU (DirectML fails Kokoro's
ConvTranspose; see Nekotone HANDOFF §5/§5f for the measurements behind this).

## Data on disk

- Models: `%LOCALAPPDATA%\NekotoneVoice\models\<id>\…`.
- Settings: `%LOCALAPPDATA%\NekotoneVoice\settings.json` (app).
- Uploaded voice prints for the compute server: `<data dir>\server-prints`.
- No index, no cache, no library: nothing is written next to your audio.

## The app

Tauri 2 + TypeScript, no framework — adapted from Nekotone's app shell
(`dom.ts`'s `h()`/`dom.morph`, `store.ts`, `api.ts`). Views: **Voice Studio**
(device picker, presets, pads, virtual-mic toggle), **Speak for me** (type,
clone management), **Settings** (devices, compute server / Share this PC,
voice/tts model management), a small **Help**. No Library, Search, Tools,
Notebook, or player views; no cat mascot; no window-skin picker.

## Quality bar

Same bar as Nekotone: `cargo test --workspace` and
`cargo clippy --workspace --all-targets` clean; tests synthesise audio
(tones, chords, noise, silence) and anything needing a model or the network
is `#[ignore]`; the CLI has end-to-end tests that run the built binary with
an empty `--models-dir`; errors are sentences a person can act on; nothing
blocks the UI thread.
