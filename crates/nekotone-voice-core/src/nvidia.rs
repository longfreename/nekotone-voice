//! NVIDIA TensorRT for RTX: the fastest way to run the ONNX models on an
//! NVIDIA RTX graphics card (Windows).
//!
//! ONNX Runtime (the `nvrtx` build) carries a small provider DLL,
//! `onnxruntime_providers_nv_tensorrt_rtx.dll`, next to the program. That
//! provider needs NVIDIA's runtime, `tensorrt_rtx_1_4.dll` and
//! `tensorrt_onnxparser_rtx_1_4.dll` (≈ 204 MB), and the driver's
//! `nvml.dll`. The runtime is optional: an installer component (ticked when
//! an RTX card is found) or a download in Settings → Models
//! (`ModelId::GpuNvidia`, NVIDIA's own wheel from PyPI, unpacked here). It
//! lives in `models/gpu-nvidia/` and is loaded from there by full path
//! before the first session, so the provider's imports resolve to it.
//!
//! Nothing here is required: without the runtime, without an RTX card, or
//! when TensorRT-RTX fails once, [`crate::models::onnx_session`] uses
//! DirectML and then the CPU, and the reason is kept for Settings.

// TensorRT for RTX is Windows-only; elsewhere its helpers are unused
#![cfg_attr(not(windows), allow(dead_code))]

use crate::{Error, Result};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The two runtime DLLs and NVIDIA's licence, as unpacked into the model
/// folder: (name inside NVIDIA's wheel, file name here, size, SHA-256).
pub(crate) const UNPACKED: [(&str, &str, u64, &str); 3] = [
    ("tensorrt_rtx_libs/tensorrt_rtx_1_4.dll", "tensorrt_rtx_1_4.dll", 200_527_360, "5ad0c4de71682017047c60d8f22e1af23e80f57e79c91fc591bc84019b6da62b"),
    (
        "tensorrt_rtx_libs/tensorrt_onnxparser_rtx_1_4.dll",
        "tensorrt_onnxparser_rtx_1_4.dll",
        3_271_168,
        "6a608b1005ffff6c2d967e3f66349ca71670096a8e528ebbf50a332f5b937b90",
    ),
    ("tensorrt_rtx_cu13_libs-1.4.0.76.dist-info/LICENSE.txt", "NVIDIA-LICENSE.txt", 47_141, "c86915fd95bbefbdda3135aace3e6ad6b4612846dcbdbc5d2f5a38b139dd88b4"),
];

/// The runtime files are all present (right sizes) in `dir`.
pub fn installed_in(dir: &Path) -> bool {
    UNPACKED.iter().all(|(_, name, size, _)| std::fs::metadata(dir.join(name)).map(|m| m.is_file() && m.len() == *size).unwrap_or(false))
}

/// Unpack the runtime from NVIDIA's wheel (a zip) into `dir`, verify every
/// file's SHA-256, then delete the wheel.
#[cfg(feature = "ml")]
pub(crate) fn unpack(wheel: &Path, dir: &Path) -> Result<()> {
    let bytes = std::fs::read(wheel)?;
    let bad = |what: &str| Error::Model(format!("the NVIDIA runtime download is damaged ({what}); remove it in Settings → Models and download it again"));
    let entries = zip_entries(&bytes).ok_or_else(|| bad("not a zip file"))?;
    for (member, name, size, sha) in UNPACKED {
        let e = entries.iter().find(|e| e.name == member).ok_or_else(|| bad(&format!("{member} is missing")))?;
        let data = e.read(&bytes).ok_or_else(|| bad(&format!("{member} does not unpack")))?;
        if data.len() as u64 != size || sha256_hex(&data) != sha {
            return Err(bad(&format!("{member} has the wrong checksum")));
        }
        let part = dir.join(format!("{name}.part"));
        std::fs::write(&part, &data)?;
        std::fs::rename(&part, dir.join(name))?;
    }
    let _ = std::fs::remove_file(wheel);
    Ok(())
}

#[cfg(feature = "ml")]
fn sha256_hex(data: &[u8]) -> String {
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(data))
}

/// One member of a zip file (from the central directory).
struct ZipEntry {
    name: String,
    method: u16,
    comp_size: u64,
    size: u64,
    local_offset: u64,
}

impl ZipEntry {
    #[cfg(all(feature = "ml", windows))]
    fn read(&self, zip: &[u8]) -> Option<Vec<u8>> {
        let lo = self.local_offset as usize;
        if zip.get(lo..lo + 4)? != b"PK\x03\x04" {
            return None;
        }
        let n = u16::from_le_bytes(zip.get(lo + 26..lo + 28)?.try_into().ok()?) as usize;
        let m = u16::from_le_bytes(zip.get(lo + 28..lo + 30)?.try_into().ok()?) as usize;
        let start = lo + 30 + n + m;
        let raw = zip.get(start..start + self.comp_size as usize)?;
        let out = match self.method {
            0 => raw.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(raw, self.size as usize).ok()?,
            _ => return None,
        };
        (out.len() as u64 == self.size).then_some(out)
    }
    #[cfg(all(feature = "ml", not(windows)))]
    fn read(&self, _zip: &[u8]) -> Option<Vec<u8>> {
        None // the runtime is Windows-only
    }
}

/// The central directory of a (non-zip64) zip file.
fn zip_entries(zip: &[u8]) -> Option<Vec<ZipEntry>> {
    let u16_at = |i: usize| zip.get(i..i + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |i: usize| zip.get(i..i + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    // end of central directory: the last "PK\x05\x06" (comment up to 64 KB)
    let from = zip.len().saturating_sub(22 + 65_535);
    let eocd = (from..zip.len().saturating_sub(21)).rev().find(|&i| &zip[i..i + 4] == b"PK\x05\x06")?;
    let count = u16_at(eocd + 10)? as usize;
    let mut p = u32_at(eocd + 16)? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        if u32_at(p)? != 0x0201_4b50 {
            return None;
        }
        let (n, m, k) = (u16_at(p + 28)? as usize, u16_at(p + 30)? as usize, u16_at(p + 32)? as usize);
        out.push(ZipEntry {
            method: u16_at(p + 10)?,
            comp_size: u32_at(p + 20)? as u64,
            size: u32_at(p + 24)? as u64,
            local_offset: u32_at(p + 42)? as u64,
            name: String::from_utf8_lossy(zip.get(p + 46..p + 46 + n)?).into_owned(),
        });
        p += 46 + n + m + k;
    }
    Some(out)
}

// ───────────────────────────── the card ─────────────────────────────

/// The NVIDIA RTX graphics card's name ("NVIDIA GeForce RTX 4070 Ti"), from
/// the display adapters Windows lists; `None` without one (or off Windows).
/// TensorRT for RTX needs an RTX card (GeForce RTX 20 series or newer, RTX
/// professional cards).
pub fn rtx_gpu() -> Option<String> {
    static GPU: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    GPU.get_or_init(|| display_adapters().into_iter().find(|n| is_rtx(n))).clone()
}

pub(crate) fn is_rtx(name: &str) -> bool {
    let l = name.to_ascii_lowercase();
    l.contains("nvidia") && l.contains("rtx")
}

#[cfg(windows)]
fn display_adapters() -> Vec<String> {
    use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};
    let wide = |s: &str| s.encode_utf16().chain(std::iter::once(0)).collect::<Vec<u16>>();
    let value = wide("DriverDesc");
    let mut out = Vec::new();
    for i in 0..32 {
        let key = wide(&format!(r"SYSTEM\CurrentControlSet\Control\Class\{{4d36e968-e325-11ce-bfc1-08002be10318}}\{i:04}"));
        let mut buf = [0u16; 256];
        let mut len = (buf.len() * 2) as u32;
        // SAFETY: valid NUL-terminated strings and a buffer of `len` bytes.
        let rc = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, key.as_ptr(), value.as_ptr(), RRF_RT_REG_SZ, std::ptr::null_mut(), buf.as_mut_ptr().cast(), &mut len) };
        if rc == 0 {
            let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
            out.push(String::from_utf16_lossy(&buf[..n]));
        }
    }
    out
}

#[cfg(not(windows))]
fn display_adapters() -> Vec<String> {
    Vec::new()
}

// ───────────────────────────── loading ─────────────────────────────

/// Model folders searched for the runtime: the default one, plus any a
/// ModelManager with another root has used (`--models-dir`).
static ROOTS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

/// Also look for the runtime under `root/gpu-nvidia` (ModelManager::new).
pub(crate) fn add_models_root(root: &Path) {
    let mut r = ROOTS.lock().unwrap_or_else(|e| e.into_inner());
    if !r.iter().any(|x| x == root) {
        r.push(root.to_path_buf());
    }
}

/// The folder holding a complete runtime, if any.
pub fn runtime_dir() -> Option<PathBuf> {
    let mut roots = ROOTS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    roots.push(crate::data_dir().join("models"));
    roots.into_iter().map(|r| r.join(crate::models::ModelId::GpuNvidia.name())).find(|d| installed_in(d))
}

/// Whether TensorRT-RTX is usable in this process: `Ok` once the runtime
/// is loaded; otherwise the reason (no RTX card, no runtime, a load or
/// session failure). A failure sticks for the rest of the process so every
/// model does not pay for it again.
static STATE: Mutex<Option<std::result::Result<(), String>>> = Mutex::new(None);

pub(crate) fn ready() -> std::result::Result<(), String> {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = st.as_ref() {
        return s.clone();
    }
    let r = load();
    // "not installed" is re-checked next time (a download may finish meanwhile)
    if !matches!(&r, Err(e) if e == NOT_INSTALLED) {
        *st = Some(r.clone());
    }
    r
}

const NOT_INSTALLED: &str = "the NVIDIA TensorRT for RTX runtime is not installed";

/// Stop using TensorRT-RTX in this process (a session failed on it).
pub(crate) fn disable(reason: String) {
    log::warn!("TensorRT-RTX disabled for this session: {reason}");
    *STATE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Err(reason));
}

fn load() -> std::result::Result<(), String> {
    if rtx_gpu().is_none() {
        return Err("no NVIDIA RTX graphics card was found".into());
    }
    let dir = runtime_dir().ok_or_else(|| NOT_INSTALLED.to_string())?;
    preload(&dir.join(UNPACKED[0].1))?;
    preload(&dir.join(UNPACKED[1].1))?;
    log::info!("TensorRT-RTX runtime loaded from {}", dir.display());
    Ok(())
}

/// Load a DLL by full path and keep it loaded, so later imports of the
/// same module name resolve to it.
#[cfg(windows)]
fn preload(path: &Path) -> std::result::Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH};
    let w: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    // SAFETY: a NUL-terminated path; the handle is deliberately never freed.
    let h = unsafe { LoadLibraryExW(w.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
    if h.is_null() {
        return Err(format!("could not load {} ({})", path.display(), std::io::Error::last_os_error()));
    }
    Ok(())
}

#[cfg(not(windows))]
fn preload(path: &Path) -> std::result::Result<(), String> {
    Err(format!("{}: TensorRT for RTX is used on Windows only", path.display()))
}

// ───────────────────────── which models TensorRT-RTX takes ─────────────────────────

/// Operators TensorRT for RTX cannot build (the int8 "dynamic quantisation"
/// of the Whisper, text-embedding and Kokoro exports). Trying such a model
/// cost ~90 s of failed engine building before DirectML took over
/// (Whisper small, measured), so these models go straight to DirectML.
const UNSUPPORTED_OPS: [&[u8]; 3] = [b"DynamicQuantizeLinear", b"MatMulInteger", b"ConvInteger"];

/// Why TensorRT-RTX should not be tried for this model file, if it
/// shouldn't: an operator it cannot build, or a failure remembered from an
/// earlier try (kept in `cache/trt-rtx/failed.txt`, keyed by file size and
/// date, so a changed model is tried again). Scans are cached per process.
/// `fixed_shapes`: the caller declares the input sizes, so an earlier
/// "variable size" refusal of the same file does not apply.
pub(crate) fn model_skip_reason(path: &Path, fixed_shapes: bool) -> Option<String> {
    static SEEN: Mutex<Vec<(PathBuf, bool, Option<String>)>> = Mutex::new(Vec::new());
    if let Some((_, _, r)) = SEEN.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(p, f, _)| p == path && *f == fixed_shapes) {
        return r.clone();
    }
    let r = remembered_failure(path)
        .filter(|why| !(fixed_shapes && why.contains("variable size")))
        .or_else(|| unsupported_op(path).map(|op| format!("it uses {op}, which TensorRT for RTX cannot run")));
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).push((path.to_path_buf(), fixed_shapes, r.clone()));
    r
}

/// The first unsupported operator name found in the model file (the ONNX
/// protobuf stores operator types as plain strings).
fn unsupported_op(path: &Path) -> Option<&'static str> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let keep = UNSUPPORTED_OPS.iter().map(|o| o.len()).max().unwrap_or(0);
    let mut buf = vec![0u8; 8 << 20];
    let mut carry = 0usize;
    loop {
        let n = f.read(&mut buf[carry..]).ok()?;
        if n == 0 {
            return None;
        }
        let hay = &buf[..carry + n];
        for op in UNSUPPORTED_OPS {
            if hay.windows(op.len()).any(|w| w == op) {
                return Some(std::str::from_utf8(op).unwrap_or("?"));
            }
        }
        // keep the tail so a name split across reads is still found
        let tail = hay.len().min(keep);
        let start = hay.len() - tail;
        buf.copy_within(start..start + tail, 0);
        carry = tail;
    }
}

fn failure_log() -> PathBuf {
    crate::data_dir().join("cache").join("trt-rtx").join("failed.txt")
}

fn model_key(path: &Path) -> Option<String> {
    let m = std::fs::metadata(path).ok()?;
    let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs();
    Some(format!("{}\t{}\t{}", path.display(), m.len(), t))
}

fn remembered_failure(path: &Path) -> Option<String> {
    let key = model_key(path)?;
    let text = std::fs::read_to_string(failure_log()).ok()?;
    text.lines().find(|l| l.starts_with(&key)).map(|l| format!("it failed before: {}", l[key.len()..].trim()))
}

/// Remember that TensorRT-RTX could not build this model file.
pub(crate) fn remember_model_failure(path: &Path, error: &str) {
    let Some(key) = model_key(path) else { return };
    let log = failure_log();
    if let Some(d) = log.parent() {
        let _ = std::fs::create_dir_all(d);
    }
    let first_line: String = error.lines().next().unwrap_or("").chars().take(200).collect();
    let line = format!("{key}\t{first_line}\n");
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// What Settings shows about the NVIDIA path.
#[derive(Debug, Clone, Serialize)]
pub struct NvidiaStatus {
    /// The RTX card, when there is one.
    pub gpu: Option<String>,
    /// The runtime files are installed.
    pub runtime: bool,
    /// Why TensorRT-RTX is not in use (after a try), when it is not.
    pub error: Option<String>,
}

pub fn status() -> NvidiaStatus {
    let tried = STATE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    NvidiaStatus { gpu: rtx_gpu(), runtime: runtime_dir().is_some(), error: tried.and_then(|r| r.err()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtx_names() {
        assert!(is_rtx("NVIDIA GeForce RTX 4070 Ti"));
        assert!(is_rtx("NVIDIA RTX A4000"));
        assert!(!is_rtx("NVIDIA GeForce GTX 1080"));
        assert!(!is_rtx("AMD Radeon RX 7900 XTX"));
        assert!(!is_rtx("Intel(R) UHD Graphics 630"));
    }

    /// One engine on real models and real audio (ignored: needs the
    /// installed models; run once per engine, each in its own process so no
    /// session is shared: `NEKOTONE_GPU=0`, `NEKOTONE_GPU=directml`, unset).
    /// The CPU run saves its outputs to `%TEMP%\nekotone-bench\`; the other
    /// runs compare with them (same top AudioSet label, scores within 0.05,
    /// ≥ 90 % of transcript words, vocals and cleaned speech within 20 dB of
    /// the CPU's) and print cold/warm times with the engine each model got.
    /// Audio: `NEKOTONE_BENCH_MUSIC` (default: a DDO music file, read in
    /// One engine on real models and real audio (ignored: needs the
    /// installed models; run once per engine, each in its own process so no
    /// session is shared: `NEKOTONE_GPU=0`, `NEKOTONE_GPU=directml`, unset).
    #[test]
    #[ignore]
    #[cfg(all(feature = "ml", feature = "gpu", windows))]
    fn bench_engine_on_real_models() {
        use crate::models::{accelerator, last_backend, Accelerator, Backend, ModelId, ModelManager};
        use crate::stt::{Transcriber, TranscribeOptions, WhisperOnnx};
        use std::time::Instant;

        let speech = std::env::var("NEKOTONE_BENCH_SPEECH")
            .unwrap_or_else(|_| std::env::temp_dir().join("nekotone-notes-speech.wav").display().to_string());
        let dir = std::env::temp_dir().join("nekotone-bench");
        std::fs::create_dir_all(&dir).unwrap();
        let cpu = accelerator() == Accelerator::Cpu;
        let mm = ModelManager::default();
        let speech = crate::audio::decode(Path::new(&speech)).expect("speech decodes");
        let engine = |cpu: bool| if cpu { Backend::Cpu } else { last_backend().unwrap_or(Backend::Cpu) };
        let mut rows: Vec<String> = Vec::new();

        for id in [ModelId::WhisperSmall, ModelId::WhisperLargeTurbo] {
            if mm.path(id).is_none() {
                continue;
            }
            let opts = TranscribeOptions { language: Some("en".into()), ..Default::default() };
            let t = Instant::now();
            let w = WhisperOnnx::load(&mm, id).expect("Whisper loads");
            let tr = w.transcribe(&speech, &opts, &mut |_| {}).expect("transcribes");
            let cold = t.elapsed().as_secs_f32();
            let who = engine(cpu);
            let t = Instant::now();
            let _ = w.transcribe(&speech, &opts, &mut |_| {}).unwrap();
            rows.push(format!("{:<17} {:<24} cold {cold:6.2} s  warm {:6.2} s  ({:.1} s of speech)", id.name(), who.label(), t.elapsed().as_secs_f32(), speech.samples.len() as f32 / speech.sample_rate as f32));
            let text: String = tr.segments.iter().map(|s| s.text.trim()).collect::<Vec<_>>().join(" ");
            let file = format!("{}.txt", id.name());
            if cpu {
                std::fs::write(dir.join(&file), &text).unwrap();
            } else if let Ok(r) = std::fs::read_to_string(dir.join(&file)) {
                let norm = |s: &str| s.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()).collect::<Vec<_>>();
                let (a, b) = (norm(&r), norm(&text));
                let same = a.iter().zip(&b).filter(|(x, y)| x == y).count();
                let agree = same as f32 / a.len().max(b.len()).max(1) as f32;
                rows.push(format!("                  transcript agreement with the CPU {:.0} %", agree * 100.0));
                assert!(agree > 0.9, "{}: transcript differs
  cpu: {r}
  now: {text}", id.name());
            }
        }
        eprintln!("
== engine setting: {:?}", accelerator());
        for r in rows {
            eprintln!("{r}");
        }
    }

    /// A tiny stored + deflated zip, written by hand, reads back.
    #[test]
    #[cfg(all(feature = "ml", windows))]
    fn zip_reader_reads_stored_and_deflated() {
        let a = b"hello hello hello hello".to_vec();
        let b = miniz_oxide::deflate::compress_to_vec(&a, 6);
        let mut zip = Vec::new();
        let mut central = Vec::new();
        for (name, method, data) in [("a.txt", 0u16, &a), ("b.txt", 8u16, &b)] {
            let off = zip.len() as u32;
            let mut h = b"PK\x03\x04".to_vec();
            h.extend([20, 0, 0, 0]);
            h.extend(method.to_le_bytes());
            h.extend([0u8; 8]); // time, date, crc (unchecked)
            h.extend((data.len() as u32).to_le_bytes());
            h.extend((a.len() as u32).to_le_bytes());
            h.extend((name.len() as u16).to_le_bytes());
            h.extend(0u16.to_le_bytes());
            h.extend(name.as_bytes());
            zip.extend(h);
            zip.extend(data.iter());
            let mut c = b"PK\x01\x02".to_vec();
            c.extend([20, 0, 20, 0, 0, 0]);
            c.extend(method.to_le_bytes());
            c.extend([0u8; 8]);
            c.extend((data.len() as u32).to_le_bytes());
            c.extend((a.len() as u32).to_le_bytes());
            c.extend((name.len() as u16).to_le_bytes());
            c.extend([0u8; 12]);
            c.extend(off.to_le_bytes());
            c.extend(name.as_bytes());
            central.extend(c);
        }
        let cd_off = zip.len() as u32;
        zip.extend(&central);
        zip.extend(b"PK\x05\x06");
        zip.extend([0u8; 4]);
        zip.extend(2u16.to_le_bytes());
        zip.extend(2u16.to_le_bytes());
        zip.extend((central.len() as u32).to_le_bytes());
        zip.extend(cd_off.to_le_bytes());
        zip.extend(0u16.to_le_bytes());
        let e = zip_entries(&zip).expect("entries");
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].read(&zip).unwrap(), a);
        assert_eq!(e[1].read(&zip).unwrap(), a);
    }
}
