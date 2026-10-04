# Voicekit compute server

`voicekit serve` runs the voice clone for another PC's Voicekit. Voicekit
uses it when **Settings → Compute server** names it, does the work itself
when the server does not answer, and sets a busy server aside for 30 s (an
answer slower than the speech it made).

Any machine that runs Voicekit can serve. Pick one way:

## Linux with an NVIDIA GPU (Docker)

Needs Docker and the NVIDIA Container Toolkit. Works on x86-64 and arm64
(a DGX Spark / GB10 included). Everything is compiled inside the image.

```bash
# from the repository root
docker build -f packaging/server/Dockerfile -t voicekit-server .
docker run -d --name voicekit-server --hostname "$(hostname)" --gpus all \
  -p 8199:8199 -v voicekit-data:/data --restart unless-stopped voicekit-server
docker logs -f voicekit-server     # first start downloads the voice-clone model (~1.3 GB)
curl http://localhost:8199/v1/info
```

or with Compose: `NEKOTONE_HOSTNAME=$(hostname) docker compose -f packaging/server/compose.yaml up -d --build`.

The image is NVIDIA's CUDA 13 runtime (with cuDNN) plus ONNX Runtime GPU
1.30 for the machine's architecture, about 3.4 GB. Older drivers: build
with `--build-arg CUDA_IMAGE=nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`
and an ONNX Runtime built for CUDA 12 (`--build-arg ORT_VERSION=…`).

## Windows: the Voicekit app, or a service

The easiest: on the PC with the graphics card, open Voicekit, **Settings →
Share this PC**, switch it on. It serves while Voicekit is open and shows
the address other computers type into their **Settings → Compute server**.

To serve without anyone logged in, install it as a Windows service from an
administrator terminal (the `voicekit` command comes with Voicekit):

```powershell
voicekit service install            # same options as serve: --port, --token, --name, --threads
voicekit service status
voicekit service stop | start
voicekit service uninstall
```

It starts with Windows, uses the models folder of the user who installed it
(nothing downloads again), keeps uploaded voices in that user's
`server-prints` folder, and opens its port in Windows Firewall. It uses the
graphics card the way the app does (DirectML, or NVIDIA TensorRT for RTX).

## Linux without Docker

Build with `cargo build --release -p nekotone-voice-cli` (add `--features cuda`
and point `ORT_DYLIB_PATH` at an onnxruntime-gpu `libonnxruntime.so` for
NVIDIA), then run `voicekit serve`, or `voicekit service install` for a
systemd user unit (`loginctl enable-linger $USER` keeps it running when you
are logged out).

## Options

| Option | Environment | Default |
|---|---|---|
| `--port` | `NEKOTONE_SERVER_PORT` | 8199 |
| `--host` | `NEKOTONE_SERVER_HOST` | 0.0.0.0 |
| `--token` | `NEKOTONE_SERVER_TOKEN` | none (clients then need it) |
| `--name` | `NEKOTONE_SERVER_NAME` | the host name |
| `--threads` | `NEKOTONE_SERVER_THREADS` | 2 |
| `--prints` | `NEKOTONE_SERVER_PRINTS` | `<data dir>/server-prints` |
| `--no-download` | | download the model if missing |

The server speaks plain HTTP for a home network or a VPN; do not expose it
to the internet. The protocol is described in `crates/nekotone-voice-core/src/remote.rs`.

Voicekit is a trimmed fork of [Nekotone](https://github.com/longfreename/nekotone);
the server protocol and measured performance are unchanged from there.
