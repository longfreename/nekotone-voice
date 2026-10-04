//! Helpers shared by the commands: the models folder, Ctrl+C, number formatting.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

/// Where the command runs: the models folder chosen on the command line
/// (or the default, `%LOCALAPPDATA%\NekotoneVoice\models`).
pub struct Ctx {
    pub models: nekotone_voice_core::models::ModelManager,
}

/// The flag Ctrl+C sets. Installed once; long jobs pass it as their cancel
/// flag and stop soon after it is set (files already written stay).
pub fn cancel_flag() -> &'static AtomicBool {
    static FLAG: OnceLock<&'static AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| {
        let flag: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
        let _ = ctrlc::set_handler(move || {
            if flag.swap(true, Ordering::SeqCst) {
                // Second Ctrl+C: stop now.
                std::process::exit(130);
            }
            eprintln!("\nstopping after the current step (Ctrl+C again to quit at once)…");
        });
        flag
    })
}

pub fn megabytes(bytes: u64) -> String {
    let mb = bytes as f64 / 1e6;
    if mb < 10.0 {
        format!("{mb:.1} MB")
    } else {
        format!("{mb:.0} MB")
    }
}

/// An absolute version of `p` without Windows' `\\?\` prefix.
#[allow(dead_code)]
pub fn absolute(p: &std::path::Path) -> PathBuf {
    let a = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf());
    let s = a.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => a,
    }
}
