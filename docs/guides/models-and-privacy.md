# Models and privacy

This guide is for anyone deciding which models to download, and anyone who
needs to know what Voicekit does with their voice. Short version: everything
runs on your PC (or, if you choose, another PC you control — see
[The compute server](./compute-server.md)). Your voice, recordings and
cloned-voice prints never leave your network; the only traffic is
downloading models you ask for, and an update check when you press the
button for it.

**Time:** 5 minutes.

## The models

| Model | Size | For | Licence and source |
|---|---|---|---|
| `whisper-base` | 79 MB | listening for what you say, in **Speak for me** | MIT, OpenAI Whisper (ONNX export by onnx-community) |
| `whisper-small` / `whisper-medium` / `whisper-large-v3-turbo` | 252 MB–1.1 GB | the same, more accurate (optional, slower without a GPU) | MIT, OpenAI Whisper |
| `tts-kokoro` | 353 MB | the built-in voices (Speakers, Transform, Creatures, Spaces & FX) in Voice Studio and Speak for me | Apache-2.0, Kokoro-82M by hexgrad (ONNX export by onnx-community), with the misaki pronunciation dictionaries (Apache-2.0, hexgrad) |
| `tts-chatterbox` | 1.33 GB | **Speak for me** in your own voice (cloned from 5–15 s of you) and **Change my voice**; your recording and voice print stay on your PC (or the compute server you chose) | MIT, Chatterbox-Turbo by Resemble AI (ONNX export `ResembleAI/chatterbox-turbo-ONNX`) |
| `gpu-nvidia` | 82 MB download, 204 MB installed | not a model: NVIDIA TensorRT for RTX, the fastest engine on an NVIDIA RTX card | NVIDIA's own licence (NVIDIA-LICENSE.txt, installed with it), downloaded from NVIDIA's package on PyPI |

Exact sizes and a tick for the installed ones: `voicekit models list`, or
**Settings → Models** in the app. All Whisper models understand 99
languages; `base` leans English.

## Where they live

`%LOCALAPPDATA%\NekotoneVoice\models\<model>\`. Remove one with
`voicekit models remove <model>` or the Remove button in Settings. On
another PC, install or download again.

## How a download works

1. Each file is fetched over HTTPS from a location pinned to one exact
   version (the Hugging Face model hub or the project's GitHub).
2. It is written to a `.part` file. An interrupted or cancelled download
   (`Ctrl+C` on the command line) keeps the part and resumes from there
   next time.
3. Its SHA-256 is compared with the value built into Voicekit. A file that
   does not match is deleted and reported; nothing unverified is ever
   loaded.
4. The model is used only when every one of its files is present.

A proxy from `HTTPS_PROXY` is honoured.

## What runs where

Every model runs through ONNX Runtime, on the GPU (NVIDIA TensorRT for RTX
or DirectML) or the CPU; `tts-kokoro` and the voice-clone's speaker
embedding always use the CPU (DirectML rejects a layer they need). Nothing
is sent anywhere to be processed unless you set up a
[compute server](./compute-server.md) yourself.

## What leaves your PC

| Traffic | When |
|---|---|
| Model downloads | only when you ask (`models get`, Settings → Models, or a feature that offers to download its model) |
| Compute-server requests | only if you configure **Settings → Compute server**, and only to the address you typed |
| Update check | only when you press **Check for updates** under **Settings → About → Updates**, and only in an installed copy set up for updates; there is no automatic check |

That is all. There is no telemetry and no account.

## What is written to disk

| Where | What |
|---|---|
| `%LOCALAPPDATA%\NekotoneVoice\settings.json` | presets, pads, device choices, compute-server address |
| `%LOCALAPPDATA%\NekotoneVoice\models\` | the models |
| `%LOCALAPPDATA%\NekotoneVoice\voices\` | your recorded samples and cloned-voice prints |

## Next steps

- [Voice Studio](./voice-studio.md): the live changer, presets and pads.
- [The compute server](./compute-server.md): run the voice clone on another PC.
