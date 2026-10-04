# FAQ

**Does my real voice ever reach the other side?**
No. Once you are **on air** only the processed voice (or silence, when muted) goes to the virtual microphone; there is no path for the unprocessed microphone signal to reach the output.

**Does anything leave my PC?**
Only the model downloads you request, requests to a compute server you configure yourself, and an update check when you press **Check for updates**. See [Models and privacy](guides/models-and-privacy.md).

**Can I clone my own voice?**
Yes: **My voices** in Voice Studio clones from about 15 seconds of you talking (Chatterbox-Turbo, running on your PC). Only clone your own voice, or a voice whose owner agreed. See [Voice Studio](guides/voice-studio.md#speak-in-your-own-voice).

**What is the difference between Speak for me and Change my voice?**
**Speak for me** replaces your voice: you type or speak, and a synthetic or cloned voice says it, with no path for your real voice to the output. **Change my voice** keeps your own words, timing and delivery, and changes only the voice, about a second after you stop talking. Both need the voice-clone model for a cloned voice; built-in voices work without it.

**Can I use my GPU?**
Yes, automatically, through DirectML on any DirectX 12 card (NVIDIA TensorRT for RTX too, on an RTX card). Kokoro's voices and the voice-clone's speaker embedding always run on the CPU; everything else prefers the GPU. Turn the GPU off with `NEKOTONE_GPU=0` or `--gpu cpu`. See [Models and privacy](guides/models-and-privacy.md).

**Can another PC do the heavy work for me?**
Yes: a PC with a graphics card (or a Linux box, or a DGX Spark) can run `voicekit serve` or install it as a Windows service, and this PC uses it from **Settings → Compute server**. See [The compute server](guides/compute-server.md).

**Why does the voice changer delay my speech?**
The processing itself adds about 26 ms; the rest of the delay you see comes from your audio devices' buffers. For the lowest delay, use a smaller buffer in the device's own driver settings, with the microphone and output on the same device.

**The other program hears nothing.**
It is probably listening to the wrong device. Choose **CABLE Output (VB-Audio Virtual Cable)** (or whichever virtual cable you installed) as its microphone. See [Getting started](guides/getting-started.md).

**Where are the logs?**
`%LOCALAPPDATA%\NekotoneVoice\logs`. **Settings → About** has **Copy diagnostics** for bug reports.

**How do I remove everything?**
Uninstall from Apps & Features and tick "Also delete settings and downloaded models". Nothing else is written to disk.
