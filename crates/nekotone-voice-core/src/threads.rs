//! Thread budget and priorities, so background analysis never makes the
//! player or the window feel slow.
//!
//! - Analysis (indexing, classification, transcription) runs on
//!   `background_threads()` workers at *below normal* priority.
//! - The playback engine thread runs *above normal*; the audio callback is
//!   already real-time on the driver's thread.
//! - ONNX Runtime sessions get `ml_threads()` intra-op threads (half the
//!   cores by default, or `NEKOTONE_THREADS`).

/// Worker threads for analysis: all cores but two, at least one; or the
/// requested count when it is not zero.
pub fn background_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    cores.saturating_sub(2).max(1)
}

/// Intra-op threads for one ONNX Runtime session.
pub fn ml_threads() -> usize {
    if let Some(n) = std::env::var("NEKOTONE_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|n| *n > 0) {
        return n;
    }
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    (cores / 2).max(1)
}

/// Lower the current thread's priority (Windows: BELOW_NORMAL).
pub fn lower_current_thread_priority() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

/// Raise the current thread's priority (Windows: ABOVE_NORMAL), for the
/// playback engine.
pub fn raise_current_thread_priority() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn budgets_are_sane() {
        assert!(super::background_threads(0) >= 1);
        assert_eq!(super::background_threads(3), 3);
        assert!(super::ml_threads() >= 1);
    }
}
