# Changelog

All notable changes to Voicekit are documented here.

## Unreleased

- First fork of Voicekit from [Nekotone](https://github.com/longfreename/nekotone):
  Voice Studio (real-time voice changer, presets, pads, virtual microphone),
  voice cloning ("Speak for me" text to speech, "Change my voice"), and the
  Windows/Linux compute server, reskinned with a sleek, professional look
  (no cat mascot, no window-skin picker).
- New app icon: a plain waveform mark, replacing placeholder cat-mascot art.
- Installer (`packaging/forgeset.toml`): Windows 11-styled wizard, dark
  theme, no file-type associations (Voicekit has no player).
- Built the first real distributable: `Voicekit-0.1.0-Setup.exe` (1.5 GB)
  and a portable `Voicekit-0.1.0-portable.zip` (34 MB), via `build.ps1`.
  Not code-signed yet, so SmartScreen will warn on first run.
