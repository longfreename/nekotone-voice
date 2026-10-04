//! Capture/time-alignment helpers used by the voice features.

pub(crate) mod align;
pub(crate) mod capture;
pub(crate) mod vad;

pub use capture::{input_devices, loopback_devices, CaptureFormat, CaptureSink, CaptureSource, Clock, DeviceInfo, FakeCapture, ManualClock, RealClock, LINUX_DEFAULT_MONITOR};
#[cfg(feature = "player")]
pub use capture::CpalCapture;

/// A source's part in the capture pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceRole {
    Mic,
    System,
}

impl SourceRole {
    pub fn name(self) -> &'static str {
        match self {
            SourceRole::Mic => "mic",
            SourceRole::System => "system",
        }
    }
}
