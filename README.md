# Voicekit

**A real-time voice changer, your own cloned voice, and a professional, simplified cockpit for using them.**

Voicekit changes your voice as you speak (pitch, formants, a set of character
presets, pads you can trigger live) and sends the result to a virtual
microphone any app can use — a game, a call, a stream. It can also clone your
voice from a short sample and use it two ways: **Speak for me** turns typed
text into speech in your voice, instantly, and **Change my voice** re-reads
your own recorded or live speech in a chosen voice.

Everything runs on your PC: no accounts, no uploads. The models (an LP-PSOLA
voice changer, Kokoro for text to speech, a voice-clone model) run through
ONNX Runtime, on the GPU through DirectML when there is one (NVIDIA
TensorRT for RTX where it measures faster). Heavier voice-clone work can
also run on another PC on your network — see **Compute server**, below.

Voicekit is a focused fork of [Nekotone](https://github.com/longfreename/nekotone),
an audio indexer and player; it keeps only the voice features, reskinned
and simplified, with no library, search, or player.

- [CHANGELOG.md](CHANGELOG.md): what changed in each version.
- [ARCHITECTURE.md](ARCHITECTURE.md): how it is built. [HANDOFF.md](HANDOFF.md): the state of the work and how to continue it.
- [packaging/server/README.md](packaging/server/README.md): run the compute server on another PC (Docker, Windows service, or a systemd unit).

## Building

```powershell
$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\nekotone-voice-target\main"
cargo test --workspace
cargo build --release -p nekotone-voice-cli    # voicekit.exe (the compute server and model management)

cd app; npm ci
npx tsc --noEmit
$env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\nekotone-voice-target\app"; npx tauri build --no-bundle   # the Voicekit app
```

## Compute server

`voicekit serve` (or `voicekit service install` for a Windows service / systemd
unit) runs the voice-clone model for another PC's Voicekit app, so a laptop
without a GPU can still use **Speak for me** and **Change my voice** at full
speed. See [packaging/server/README.md](packaging/server/README.md) for Docker,
Windows service, and Linux instructions, and measured round-trip times.

## Credits

Voicekit is licensed under the [Apache License 2.0](LICENSE-APACHE). The
models it downloads keep their own licences; see
[docs/guides/models-and-privacy.md](docs/guides/models-and-privacy.md).
