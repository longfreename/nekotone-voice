# Getting started

This guide is for anyone who wants to change their voice live for a game,
call or stream, or have their typed (or spoken) words said aloud in a voice
of their choosing. You install Voicekit, set up a virtual microphone once,
and go live.

**Time:** about 10 minutes, including the one-time virtual-cable setup.

## What you will have at the end

- Your voice, changed live, reaching Discord, Teams, OBS or a game as a
  microphone.
- A cloned version of your own voice, usable from typed text (**Speak for
  me**) or to re-read what you say (**Change my voice**).

## Before you start

- Windows 10 or 11, 64-bit (Linux is supported for the compute server; see
  [The compute server](./compute-server.md)).
- A microphone, and headphones (speakers make the microphone hear the
  changed voice and howl).
- About 0.5 GB of free disk space for Voicekit and its built-in voices;
  more if you clone your own voice (about 1.3 GB) or add larger Whisper
  models.

## Step 1. Install

Run `Voicekit-<version>-Setup.exe` and press **Install now**. It installs
for your user only (no administrator password) into
`%LOCALAPPDATA%\Programs\Voicekit`, with the built-in voices included, and
adds Voicekit to the Start menu and the Desktop, and the `voicekit` command
to your PATH (open a new terminal afterwards so it sees it).

## Step 2. Give other programs a way to hear it

Windows cannot create a new microphone without a signed driver, so Voicekit
sends its voice into a free *virtual audio cable*, and your other program
listens to the cable.

1. Install **VB-CABLE** (free) from
   [vb-audio.com/Cable](https://vb-audio.com/Cable/) and restart the PC if
   it asks.
2. Open **Setup**. Under **Send to other apps as a microphone** Voicekit
   shows the cable it found and the exact name to choose in the other
   program, usually **CABLE Output (VB-Audio Virtual Cable)**.
3. In Discord, Teams, OBS or your game, choose that name as the microphone.

VoiceMeeter and Virtual Audio Cable work too; Voicekit detects them by name.
It never installs drivers itself.

## Step 3. Pick a voice and go live

1. In **Setup**, choose your microphone.
2. Open **Voices**. Press **Preview** on any voice to hear a sample, then
   click a voice and **Go live**. The bar at the top shows **ON AIR**, the
   input and output levels, and the delay.
3. **Mute** (or **Ctrl+Alt+M**) sends silence at once, from anywhere.

See [Voice Studio](./voice-studio.md) for the full voice list, tuning knobs,
pads, and **Speak for me**/**Change my voice** (your own cloned voice).

## Check it worked

- The other program's microphone meter moves while you talk.
- `voicekit models list` shows the built-in voices as installed (✓).
- **Hear yourself** in Setup plays the changed voice to your headphones.

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| The other program hears nothing | It is listening to the wrong device | Choose **CABLE Output (VB-Audio Virtual Cable)** as its microphone |
| No cable is found | VB-CABLE is not installed, or the PC was not restarted | Install it, restart, press **Check again** |
| Howling | **Hear yourself** is on without headphones | Turn it off or use headphones |

## Next steps

- [Voice Studio](./voice-studio.md): the full guide — voices, tuning, pads, Speak for me, Change my voice.
- [The compute server](./compute-server.md): run the voice clone on another PC for full speed without a local GPU.
- [Models and privacy](./models-and-privacy.md): every model, its size and licence, and what never leaves your PC.
