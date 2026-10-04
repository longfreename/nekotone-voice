//! `%LOCALAPPDATA%\NekotoneVoice\settings.json`: the small set of preferences
//! Voicekit keeps between runs. Unknown keys survive upgrades.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const FILE: &str = "settings.json";
pub const DEFAULT_WIDTH: u32 = 1240;
pub const DEFAULT_HEIGHT: u32 = 860;
pub const LAYOUT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SpeakSettings {
    pub voice: String,
    pub speed: f32,
    pub effect_id: Option<String>,
    pub accent: String,
    pub listen: bool,
    pub translate: bool,
    pub convert: bool,
    pub keep_pauses: bool,
    pub clone_consent: bool,
    pub language: Option<String>,
    pub trim_fillers: bool,
    pub mask_profanity: bool,
    pub half_duplex: bool,
    pub hangover_ms: u32,
    pub auto_pause: bool,
    pub output_gain_db: f32,
    pub draft: String,
    pub show_all: bool,
}

impl Default for SpeakSettings {
    fn default() -> Self {
        SpeakSettings {
            voice: "af_heart".into(),
            speed: 1.0,
            effect_id: None,
            accent: "voice".into(),
            listen: true,
            translate: false,
            convert: false,
            keep_pauses: true,
            clone_consent: false,
            language: None,
            trim_fillers: true,
            mask_profanity: false,
            half_duplex: false,
            hangover_ms: 275,
            auto_pause: true,
            output_gain_db: 0.0,
            draft: String::new(),
            show_all: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct VoiceSettings {
    pub tab: String,
    pub preset_id: String,
    pub input: Option<String>,
    pub target: String,
    pub output: Option<String>,
    pub monitor: bool,
    pub monitor_device: Option<String>,
    pub monitor_gain_db: f32,
    pub output_gain_db: f32,
    pub ceiling_db: f32,
    pub gate_auto: bool,
    pub gate_db: f32,
    pub hpf_hz: f32,
    pub deesser: bool,
    pub use_recorded_preview: bool,
    pub sample_seconds: u32,
    pub preview_text: String,
    pub clone_name: String,
    pub speak: SpeakSettings,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        VoiceSettings {
            tab: "changer".into(),
            preset_id: "female".into(),
            input: None,
            target: "virtual".into(),
            output: None,
            monitor: true,
            monitor_device: None,
            monitor_gain_db: -6.0,
            output_gain_db: 0.0,
            ceiling_db: -1.0,
            gate_auto: true,
            gate_db: -48.0,
            hpf_hz: 70.0,
            deesser: true,
            use_recorded_preview: true,
            sample_seconds: 8,
            preview_text: "Testing Voicekit. This preview uses the current voice settings.".into(),
            clone_name: "My voice".into(),
            speak: SpeakSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    pub theme: String,
    pub accelerator: String,
    pub voice_learn: bool,
    pub whisper_model: String,
    pub compute_server: String,
    pub compute_token: String,
    pub compute_mode: String,
    pub serve_enabled: bool,
    pub serve_port: u16,
    pub serve_token: String,
    pub window_width: u32,
    pub window_height: u32,
    pub window_x: Option<i32>,
    pub window_y: Option<i32>,
    pub window_maximized: bool,
    pub window_layout: u32,
    pub voice: VoiceSettings,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            theme: "dark".into(),
            accelerator: "auto".into(),
            voice_learn: false,
            whisper_model: "whisper-base".into(),
            compute_server: String::new(),
            compute_token: String::new(),
            compute_mode: "auto".into(),
            serve_enabled: false,
            serve_port: 8199,
            serve_token: String::new(),
            window_width: DEFAULT_WIDTH,
            window_height: DEFAULT_HEIGHT,
            window_x: None,
            window_y: None,
            window_maximized: false,
            window_layout: 0,
            voice: VoiceSettings::default(),
            extra: BTreeMap::new(),
        }
    }
}

pub fn path(dir: &Path) -> PathBuf {
    dir.join(FILE)
}

pub fn nudge_window(s: &mut Settings) {
    if s.window_layout < LAYOUT_VERSION {
        if s.window_width < 920 {
            s.window_width = DEFAULT_WIDTH;
        }
        if s.window_height < 620 {
            s.window_height = DEFAULT_HEIGHT;
        }
        s.window_layout = LAYOUT_VERSION;
    }
}

pub fn load(dir: &Path) -> Settings {
    let p = path(dir);
    match std::fs::read_to_string(&p) {
        Ok(text) => match serde_json::from_str::<Settings>(&text) {
            Ok(mut s) => {
                if s.voice.preset_id.trim().is_empty() {
                    s.voice.preset_id = "female".into();
                }
                if s.voice.speak.voice.trim().is_empty() {
                    s.voice.speak.voice = "af_heart".into();
                }
                nudge_window(&mut s);
                s
            }
            Err(e) => {
                crate::log_file(&format!("settings: {} is not valid ({e}); starting fresh", p.display()));
                let _ = std::fs::rename(&p, p.with_extension("json.bad"));
                Settings::default()
            }
        },
        Err(_) => Settings::default(),
    }
}

pub fn store(dir: &Path, s: &Settings) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let p = path(dir);
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(s)?)?;
    std::fs::rename(&tmp, &p)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_unknown_keys() {
        let dir = crate::data_dir().join("settings-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(path(&dir), r#"{"theme":"light","futureKey":{"a":1},"voice":{"cloneName":"Test"}}"#).unwrap();
        let s = load(&dir);
        assert_eq!(s.theme, "light");
        assert_eq!(s.voice.clone_name, "Test");
        store(&dir, &s).unwrap();
        let again = std::fs::read_to_string(path(&dir)).unwrap();
        assert!(again.contains("futureKey"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
