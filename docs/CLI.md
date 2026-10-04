# Command line

`voicekit` runs the compute server and manages its models — the headless
half of Voicekit. The live voice changer and Speak for me are interactive
and only live in the Voicekit app.

```sh
voicekit --help
voicekit <command> --help
```

## Options for every command

| Option | Effect |
|---|---|
| `--models-dir <folder>` | Look for (and download) models here instead of `%LOCALAPPDATA%\NekotoneVoice\models`. |
| `--gpu auto` | Default. Models that benefit from it run on the GPU (NVIDIA TensorRT for RTX, then DirectML), with the CPU as the fallback. |
| `--gpu cpu` | Every model runs on the CPU. |
| `--version`, `--help` | |

## Commands

### `models`

```sh
voicekit models list [--json]           # every model, size, licence, installed?
voicekit models get <name>               # download (resumes an interrupted download)
voicekit models remove <name>            # delete a downloaded model
```

Model names: `whisper-tiny`, `whisper-base`, `whisper-small`,
`whisper-medium`, `whisper-large-v3-turbo`, `tts-kokoro`, `tts-chatterbox`,
`gpu-nvidia`. See [Models and privacy](guides/models-and-privacy.md).

### `serve`

```sh
voicekit serve [--host 0.0.0.0] [--port 8199] [--token …] [--name …] [--threads 2] [--prints <folder>] [--no-download]
```

Runs the voice-clone compute server in the foreground; `Ctrl+C` stops it.
Downloads `tts-chatterbox` the first time unless `--no-download`. Every
option also comes from the environment (`NEKOTONE_SERVER_HOST`,
`NEKOTONE_SERVER_PORT`, `NEKOTONE_SERVER_TOKEN`, `NEKOTONE_SERVER_NAME`,
`NEKOTONE_SERVER_THREADS`, `NEKOTONE_SERVER_PRINTS`), for containers and
services. See [The compute server](guides/compute-server.md).

### `service`

```sh
voicekit service install [the same options as serve]   # Windows: an administrator terminal
voicekit service status
voicekit service start | stop
voicekit service uninstall
```

Runs `serve` in the background, starting with the machine: a real Windows
service on Windows (runs with nobody logged in), a systemd user unit on
Linux. Installs with the models folder and `--prints` folder of the
installing user, and opens the port in Windows Firewall.

## Next steps

- [The compute server](guides/compute-server.md): when and how to use `serve`/`service`.
- [Models and privacy](guides/models-and-privacy.md): every model, its size and licence.
