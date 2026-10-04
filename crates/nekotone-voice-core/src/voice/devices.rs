//! Audio device enumeration and opening helpers for the voice engine (cpal).

use crate::{Error, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AudioDevice {
    pub name: String,
    /// The system default for this direction.
    pub default: bool,
    /// Looks like a virtual cable (VB-CABLE, VoiceMeeter, VAC, BlackHole…).
    pub virtual_cable: bool,
    pub channels: u16,
    pub sample_rate: u32,
}

fn is_virtual(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    ["vb-audio", "voicemeeter", "virtual audio cable", "blackhole", "cable input", "cable output", "nekotone"].iter().any(|k| l.contains(k))
}

fn list(input: bool) -> Vec<AudioDevice> {
    let host = cpal::default_host();
    let default = if input { host.default_input_device() } else { host.default_output_device() }
        .and_then(|d| d.description().ok())
        .map(|d| d.name().to_string());
    let devices = if input { host.input_devices().map(|d| d.collect::<Vec<_>>()) } else { host.output_devices().map(|d| d.collect::<Vec<_>>()) };
    let mut out: Vec<AudioDevice> = Vec::new();
    for d in devices.unwrap_or_default() {
        let Ok(desc) = d.description() else { continue };
        let name = desc.name().to_string();
        if out.iter().any(|o| o.name == name) {
            continue;
        }
        let cfg = if input { d.default_input_config() } else { d.default_output_config() };
        let (channels, sample_rate) = cfg.map(|c| (c.channels(), c.sample_rate())).unwrap_or((0, 0));
        out.push(AudioDevice { default: default.as_deref() == Some(name.as_str()), virtual_cable: is_virtual(&name), name, channels, sample_rate });
    }
    out
}

/// Microphones and other capture devices.
pub fn input_devices() -> Vec<AudioDevice> {
    list(true)
}

/// Speakers, headphones and virtual cables.
pub fn output_devices() -> Vec<AudioDevice> {
    list(false)
}

/// Find a device by exact name (or the default when `name` is None or "").
pub(crate) fn find(input: bool, name: Option<&str>) -> Result<(cpal::Device, String)> {
    let host = cpal::default_host();
    let what = if input { "input" } else { "output" };
    match name.filter(|n| !n.is_empty()) {
        Some(want) => {
            let devices = if input { host.input_devices().map(|d| d.collect::<Vec<_>>()) } else { host.output_devices().map(|d| d.collect::<Vec<_>>()) };
            for d in devices.unwrap_or_default() {
                if let Ok(desc) = d.description() {
                    if desc.name() == want {
                        return Ok((d, want.to_string()));
                    }
                }
            }
            Err(Error::Output(format!("no {what} device called \"{want}\"; it may be unplugged — pick another in Voice → Setup")))
        }
        None => {
            let d = if input { host.default_input_device() } else { host.default_output_device() }
                .ok_or_else(|| Error::Output(format!("no default {what} device is available")))?;
            let n = d.description().map(|x| x.name().to_string()).unwrap_or_else(|_| "Default".into());
            Ok((d, n))
        }
    }
}

/// Choose a stream config: the device's default format and channels, at
/// `want_rate` when the device supports it, with a fixed buffer of
/// `block` frames when supported.
pub(crate) fn choose_config(device: &cpal::Device, input: bool, want_rate: Option<u32>, block: u32) -> Result<(cpal::SampleFormat, cpal::StreamConfig)> {
    let def = if input { device.default_input_config() } else { device.default_output_config() }
        .map_err(|e| Error::Output(format!("the device has no usable format ({e})")))?;
    let fmt = def.sample_format();
    let mut chosen = def;
    if let Some(r) = want_rate {
        if r != def.sample_rate() {
            let ranges: Vec<cpal::SupportedStreamConfigRange> = if input {
                device.supported_input_configs().map(|i| i.collect()).unwrap_or_default()
            } else {
                device.supported_output_configs().map(|i| i.collect()).unwrap_or_default()
            };
            if let Some(c) = ranges
                .into_iter()
                .filter(|c| c.sample_format() == fmt && c.channels() == def.channels())
                .find_map(|c| c.try_with_sample_rate(r))
            {
                chosen = c;
            }
        }
    }
    let mut cfg = chosen.config();
    if let cpal::SupportedBufferSize::Range { min, max } = chosen.buffer_size() {
        if block >= *min && block <= *max {
            cfg.buffer_size = cpal::BufferSize::Fixed(block);
        }
    }
    Ok((fmt, cfg))
}

