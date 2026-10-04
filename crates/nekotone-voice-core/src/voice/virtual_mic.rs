//! Getting the processed voice into other programs as a microphone.
//!
//! * **Windows** cannot create a microphone without a signed kernel driver,
//!   and Nekotone never installs drivers. Instead it sends the voice to a
//!   virtual cable the user installed (VB-Audio CABLE, VoiceMeeter, Virtual
//!   Audio Cable); the other end of the cable appears as a microphone that
//!   Discord/Teams/games select. [`virtual_mic_status`] detects them by name
//!   and says which device to pick.
//! * **Linux** (PipeWire or PulseAudio): Nekotone creates a real virtual
//!   microphone with `pactl` (a null sink plus a remapped source called
//!   "Nekotone_Voice"), plays into the sink, and unloads both modules on stop,
//!   on drop, and — after a crash — on the next start ([`linux`]).
//! * **macOS**: a loopback driver such as BlackHole is detected like a cable.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum VirtualMicKind {
    /// VB-Audio Virtual Cable (and its A/B/C/D variants).
    VbCable,
    /// VB-Audio VoiceMeeter (Banana/Potato) virtual inputs.
    VoiceMeeter,
    /// Virtual Audio Cable (Muzychenko), "Line N".
    VirtualAudioCable,
    /// BlackHole / Loopback on macOS.
    Loopback,
    /// PipeWire/PulseAudio null sink created by Nekotone (Linux).
    PulseNullSink,
    /// Nothing usable found.
    None,
}

/// One detected cable: send the voice to `output_device`; other programs select `mic_name`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VirtualCable {
    pub kind: VirtualMicKind,
    /// Playback device Nekotone writes to (e.g. "CABLE Input (VB-Audio Virtual Cable)").
    pub output_device: String,
    /// Recording device other apps choose (e.g. "CABLE Output (VB-Audio Virtual Cable)").
    pub mic_name: String,
    /// True when the recording side was found among the input devices.
    pub mic_present: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct VirtualMicStatus {
    /// "windows", "linux", "macos", …
    pub platform: &'static str,
    /// A virtual microphone can be used right now.
    pub available: bool,
    pub kind: VirtualMicKind,
    /// Where the engine sends the voice when `VoiceConfig::virtual_mic` is on.
    pub output_device: Option<String>,
    /// What other programs should select as their microphone.
    pub mic_name: Option<String>,
    /// A sentence or two for the Settings page.
    pub instructions: String,
    /// Every cable detected (the first is the one used).
    pub candidates: Vec<VirtualCable>,
}

/// Name of the Linux null sink / source Nekotone creates.
pub const LINUX_SINK: &str = "nekotone_voice";
pub const LINUX_SOURCE: &str = "nekotone_mic";
pub const LINUX_MIC_DESCRIPTION: &str = "Nekotone_Voice";

/// Find virtual cables among device names (pure; unit-tested).
pub fn detect_cables(outputs: &[String], inputs: &[String]) -> Vec<VirtualCable> {
    let mut out = Vec::new();
    let find_input = |want: &str| -> Option<String> {
        inputs.iter().find(|i| i.eq_ignore_ascii_case(want)).cloned().or_else(|| {
            // tolerate a changed suffix: match on the part before " ("
            let head = want.split(" (").next().unwrap_or(want).to_ascii_lowercase();
            inputs.iter().find(|i| i.to_ascii_lowercase().starts_with(&head)).cloned()
        })
    };
    for o in outputs {
        let l = o.to_ascii_lowercase();
        let (kind, mic) = if l.contains("cable") && l.contains("input") && (l.contains("vb-audio") || l.starts_with("cable")) {
            (VirtualMicKind::VbCable, replace_word(o, "Input", "Output"))
        } else if l.contains("voicemeeter") && l.contains("input") {
            (VirtualMicKind::VoiceMeeter, replace_word(o, "Input", "Output"))
        } else if l.contains("virtual audio cable") || (l.starts_with("line ") && l.contains("virtual")) {
            (VirtualMicKind::VirtualAudioCable, o.clone())
        } else if l.contains("blackhole") || l.contains("loopback audio") {
            (VirtualMicKind::Loopback, o.clone())
        } else {
            continue;
        };
        let found = find_input(&mic);
        out.push(VirtualCable { kind, output_device: o.clone(), mic_present: found.is_some(), mic_name: found.unwrap_or(mic) });
    }
    // prefer a plain VB-CABLE, then VoiceMeeter's main input, then others; complete pairs first
    out.sort_by_key(|c| {
        let k = match c.kind {
            VirtualMicKind::VbCable => 0,
            VirtualMicKind::VirtualAudioCable => 1,
            VirtualMicKind::VoiceMeeter => 2,
            _ => 3,
        };
        (!c.mic_present, k, c.output_device.to_ascii_lowercase().contains("aux") as u8)
    });
    out
}

fn replace_word(s: &str, from: &str, to: &str) -> String {
    // case-insensitive single replacement of the first occurrence
    let l = s.to_ascii_lowercase();
    match l.find(&from.to_ascii_lowercase()) {
        Some(i) => format!("{}{}{}", &s[..i], to, &s[i + from.len()..]),
        None => s.to_string(),
    }
}

/// Status for the current machine (lists devices; cheap, call it when the
/// Settings page opens).
pub fn virtual_mic_status() -> VirtualMicStatus {
    #[cfg(target_os = "linux")]
    {
        linux::status()
    }
    #[cfg(not(target_os = "linux"))]
    {
        #[cfg(feature = "player")]
        let (outs, ins): (Vec<String>, Vec<String>) = (
            super::devices::output_devices().into_iter().map(|d| d.name).collect(),
            super::devices::input_devices().into_iter().map(|d| d.name).collect(),
        );
        #[cfg(not(feature = "player"))]
        let (outs, ins): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
        status_from_cables(detect_cables(&outs, &ins))
    }
}

/// Build the status text from detected cables (Windows/macOS wording).
pub fn status_from_cables(candidates: Vec<VirtualCable>) -> VirtualMicStatus {
    let platform = std::env::consts::OS;
    match candidates.first() {
        Some(c) => {
            let mut instructions = format!(
                "Nekotone sends your changed voice to \"{}\". In Discord, Teams, OBS or your game, choose \"{}\" as the microphone (input device).",
                c.output_device, c.mic_name
            );
            if c.kind == VirtualMicKind::VoiceMeeter {
                instructions.push_str(" In VoiceMeeter, route that virtual input strip to bus B1 so it reaches the VoiceMeeter Output microphone.");
            }
            if !c.mic_present {
                instructions.push_str(" (The recording side of this cable was not found; check that the cable is enabled in Sound settings → Recording.)");
            }
            instructions.push_str(" Keep your real microphone out of those apps so only the changed voice is heard.");
            VirtualMicStatus {
                platform,
                available: true,
                kind: c.kind,
                output_device: Some(c.output_device.clone()),
                mic_name: Some(c.mic_name.clone()),
                instructions,
                candidates,
            }
        }
        None => VirtualMicStatus {
            platform,
            available: false,
            kind: VirtualMicKind::None,
            output_device: None,
            mic_name: None,
            instructions: if platform == "macos" {
                "No virtual microphone found. Install a free loopback driver such as BlackHole (existential.audio/blackhole), then choose \"BlackHole 2ch\" as the microphone in other apps.".into()
            } else {
                "No virtual cable found. Windows needs a virtual audio cable to offer a new microphone: install the free VB-Audio Virtual Cable (vb-audio.com/Cable), restart Nekotone, then choose \"CABLE Output\" as the microphone in Discord, Teams or your game. Nekotone never installs drivers itself.".into()
            },
            candidates,
        },
    }
}

/// PipeWire / PulseAudio virtual microphone through `pactl`.
#[cfg(target_os = "linux")]
pub mod linux {
    use super::*;
    use crate::{Error, Result};
    use std::process::Command;

    fn pactl(args: &[&str]) -> Result<String> {
        let out = Command::new("pactl").args(args).output().map_err(|e| {
            Error::Output(format!(
                "pactl is not available ({e}); install it (Debian/Ubuntu: `sudo apt install pulseaudio-utils`, Fedora: `sudo dnf install pulseaudio-utils`) — it works with PipeWire too"
            ))
        })?;
        if !out.status.success() {
            return Err(Error::Output(format!("pactl {} failed: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim())));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    pub fn pactl_available() -> bool {
        Command::new("pactl").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
    }

    /// Module ids (from `pactl list short modules`) that Nekotone created.
    pub fn parse_our_modules(list: &str) -> Vec<u32> {
        list.lines()
            .filter(|l| l.contains(&format!("sink_name={LINUX_SINK}")) || l.contains(&format!("source_name={LINUX_SOURCE}")))
            .filter_map(|l| l.split_whitespace().next()?.parse().ok())
            .collect()
    }

    /// Unload modules left behind by a crashed run.
    pub fn cleanup_stale() {
        if let Ok(list) = pactl(&["list", "short", "modules"]) {
            // unload the source (remap) first, then the sink
            let mut ids = parse_our_modules(&list);
            ids.sort_unstable_by(|a, b| b.cmp(a));
            for id in ids {
                let _ = pactl(&["unload-module", &id.to_string()]);
            }
        }
    }

    /// The null sink + remapped source. Unloaded on drop.
    pub struct VirtualMic {
        sink_module: u32,
        source_module: u32,
    }

    impl VirtualMic {
        pub fn create() -> Result<VirtualMic> {
            cleanup_stale();
            let sink = pactl(&[
                "load-module",
                "module-null-sink",
                &format!("sink_name={LINUX_SINK}"),
                "sink_properties=device.description=NekotoneVoiceSink",
            ])?;
            let sink_module: u32 = sink.trim().parse().map_err(|_| Error::Output(format!("unexpected pactl output: {sink}")))?;
            let src = pactl(&[
                "load-module",
                "module-remap-source",
                &format!("master={LINUX_SINK}.monitor"),
                &format!("source_name={LINUX_SOURCE}"),
                &format!("source_properties=device.description={LINUX_MIC_DESCRIPTION}"),
            ]);
            let source_module = match src.and_then(|s| s.trim().parse().map_err(|_| Error::Output(format!("unexpected pactl output: {s}")))) {
                Ok(v) => v,
                Err(e) => {
                    let _ = pactl(&["unload-module", &sink_module.to_string()]);
                    return Err(e);
                }
            };
            Ok(VirtualMic { sink_module, source_module })
        }

        /// Our process's sink-inputs (playback streams) right now.
        pub fn our_sink_inputs() -> Vec<u32> {
            let pid = std::process::id().to_string();
            let Ok(txt) = pactl(&["list", "sink-inputs"]) else { return Vec::new() };
            let mut out = Vec::new();
            let mut cur: Option<u32> = None;
            for line in txt.lines() {
                let t = line.trim();
                if let Some(rest) = t.strip_prefix("Sink Input #") {
                    cur = rest.trim().parse().ok();
                } else if t.starts_with("application.process.id") && t.contains(&format!("\"{pid}\"")) {
                    if let Some(c) = cur {
                        out.push(c);
                    }
                }
            }
            out
        }

        /// Move playback streams that are not in `before` into the null sink.
        pub fn capture_new_streams(before: &[u32]) -> usize {
            let mut moved = 0;
            for id in Self::our_sink_inputs() {
                if !before.contains(&id) && pactl(&["move-sink-input", &id.to_string(), LINUX_SINK]).is_ok() {
                    moved += 1;
                }
            }
            moved
        }
    }

    impl Drop for VirtualMic {
        fn drop(&mut self) {
            let _ = pactl(&["unload-module", &self.source_module.to_string()]);
            let _ = pactl(&["unload-module", &self.sink_module.to_string()]);
        }
    }

    pub fn status() -> VirtualMicStatus {
        if !pactl_available() {
            return VirtualMicStatus {
                platform: "linux",
                available: false,
                kind: VirtualMicKind::None,
                output_device: None,
                mic_name: None,
                instructions: "Nekotone creates its virtual microphone with `pactl`, which was not found. Install it (Debian/Ubuntu: `sudo apt install pulseaudio-utils`; Fedora: `sudo dnf install pulseaudio-utils`); it works with both PipeWire and PulseAudio.".into(),
                candidates: Vec::new(),
            };
        }
        VirtualMicStatus {
            platform: "linux",
            available: true,
            kind: VirtualMicKind::PulseNullSink,
            output_device: Some("NekotoneVoiceSink".into()),
            mic_name: Some(LINUX_MIC_DESCRIPTION.into()),
            instructions: format!(
                "While the voice changer runs, Nekotone adds a microphone called \"{LINUX_MIC_DESCRIPTION}\". Choose it as the input device in Discord, Teams, OBS or your game. It disappears again when you stop."
            ),
            candidates: Vec::new(),
        }
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn parses_module_list() {
            let l = "22\tmodule-null-sink\tsink_name=nekotone_voice sink_properties=x\n23\tmodule-remap-source\tmaster=nekotone_voice.monitor source_name=nekotone_mic\n5\tmodule-alsa-card\tdevice_id=0";
            assert_eq!(super::parse_our_modules(l), vec![22, 23]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn detects_vb_cable_and_names_the_mic() {
        let outs = s(&["Speakers (Realtek(R) Audio)", "CABLE Input (VB-Audio Virtual Cable)", "VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)"]);
        let ins = s(&["Microphone (USB)", "CABLE Output (VB-Audio Virtual Cable)", "VoiceMeeter Aux Output (VB-Audio VoiceMeeter AUX VAIO)"]);
        let c = detect_cables(&outs, &ins);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].kind, VirtualMicKind::VbCable);
        assert_eq!(c[0].mic_name, "CABLE Output (VB-Audio Virtual Cable)");
        assert!(c[0].mic_present);
        let st = status_from_cables(c);
        assert!(st.available);
        assert!(st.instructions.contains("CABLE Output"));
    }

    #[test]
    fn detects_vac_and_voicemeeter_and_explains_absence() {
        let c = detect_cables(&s(&["Line 1 (Virtual Audio Cable)", "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)"]), &s(&["Line 1 (Virtual Audio Cable)"]));
        assert_eq!(c[0].kind, VirtualMicKind::VirtualAudioCable);
        assert_eq!(c[0].mic_name, "Line 1 (Virtual Audio Cable)");
        assert_eq!(c[1].mic_name, "VoiceMeeter Output (VB-Audio VoiceMeeter VAIO)");
        assert!(!c[1].mic_present);
        let none = status_from_cables(detect_cables(&s(&["Speakers"]), &s(&["Mic"])));
        assert!(!none.available);
        assert!(none.instructions.contains("never installs drivers") || none.instructions.contains("BlackHole"));
    }
}
