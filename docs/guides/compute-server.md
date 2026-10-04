# The compute server

This guide is for running the voice-clone model on another machine, so a
laptop without a graphics card still gets full-speed **Speak for me** and
**Change my voice**.

**Time:** 5 minutes.

## Why

Voicekit's voice clone can run on another PC on your network instead of (or
alongside) this one. That machine can be a Linux box with an NVIDIA card
(or a DGX Spark), or a Windows PC set aside for it. That machine runs
`voicekit serve`:

- **Windows**: install Voicekit there, then run `voicekit serve` in a
  terminal (it uses that PC's graphics card through DirectML or NVIDIA
  TensorRT for RTX, as the app would), or turn on **Settings → Share this
  PC** in its own Voicekit app.
- **Linux with an NVIDIA GPU**: build the Docker image in
  `packaging/server` (instructions in its README) and start it; it
  downloads the voice-clone model the first time.
- **Any computer**: `voicekit serve` also runs on the CPU alone, slower.

`voicekit serve` listens on port 8199 by default. `--port`, `--host`,
`--token`, `--name`, `--threads` and `--prints` change that, and each has an
environment variable (`NEKOTONE_SERVER_PORT` and so on) for services and
containers.

## Use a server from this PC

In **Settings → Compute server**, type the machine's name or address (with
`:port` if it is not 8199) and press **Test**:

```
gpubox: Voicekit 0.1.0 on CUDA (convert, synth)
```

From then on the voice clone's work is shared between the server and this
PC. **Share the work** sets how:

- **Auto** (the default) starts each phrase on both and uses whichever
  answer comes first. When one side wins six phrases in a row it works
  alone (the server still with this PC taking over when it is late), and
  every 8–12 phrases one phrase runs on both again to check; a late or
  failed answer goes straight back to both.
- **Both always** runs every phrase on both, first answer wins: the
  smoothest, at the cost of this PC doing the work every time.
- **Server first** uses the server, and this PC only when an answer is
  late (about half the phrase's length) or fails.

The Speak for me status shows how many answers came from each side and
which is in charge. Making a voice still happens on this PC; each voice is
sent to the server the first time it is used.

A token is needed only if the server was started with `--token`. The server
speaks plain HTTP, for a home network or a VPN; do not expose it to the
internet.

## Share this PC

Any PC with Voicekit can be the compute server for others: **Settings →
Share this PC** switches it on while the app is open (Windows asks once
whether to let it through the firewall), and the note shows the address
other computers type into their **Settings → Compute server**, for example
`192.168.1.50:8199`. A token (the second box) makes the others type it too.

To keep serving with nobody logged in, install it as a Windows service from
an administrator terminal: `voicekit service install` (then
`voicekit service status`, `stop`, `start`, `uninstall`). It starts with
Windows and uses the models already on the PC. On Linux the same command
installs a systemd unit; `packaging/server/README.md` has the Docker image
too.

## Next steps

- [Voice Studio](./voice-studio.md): the live changer, presets and pads.
- [Models and privacy](./models-and-privacy.md): every model, its size and licence.
