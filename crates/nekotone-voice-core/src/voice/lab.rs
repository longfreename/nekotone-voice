//! Voice lab: offline renders of presets (built-in or JSON) over real speech
//! files, for listening tests and objective measurements. Not run by default.
//!
//! ```text
//! $env:NEKOTONE_LAB_JOB = "C:\...\job.json"
//! cargo test --release -p nekotone-core --lib voice::lab -- --ignored --nocapture
//! ```
//!
//! Job file:
//! ```json
//! { "inputs": {"male": "tts-male.wav"}, "out_dir": "renders", "tail_secs": 1.5,
//!   "dump_builtins": "builtins.json",
//!   "jobs": [ {"name": "dragon-A", "builtin": "dragon"},
//!             {"name": "dragon-B", "preset": { ...Preset JSON... }} ] }
//! ```
//! Writes `<out_dir>/<name>__<input>.wav` (48 kHz mono 16-bit) and
//! `<out_dir>/results.json` (CPU fraction and latency per render).

use super::*;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Job {
    name: String,
    #[serde(default)]
    builtin: Option<String>,
    #[serde(default)]
    preset: Option<Preset>,
    /// Engine tuning for experiments (see `dsp::shifter::Tuning`).
    #[serde(default)]
    tune: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct LabFile {
    inputs: BTreeMap<String, String>,
    out_dir: String,
    #[serde(default)]
    tail_secs: f32,
    #[serde(default)]
    dump_builtins: Option<String>,
    #[serde(default)]
    jobs: Vec<Job>,
}

#[test]
#[ignore]
fn lab_render() {
    let path = std::env::var("NEKOTONE_LAB_JOB").expect("set NEKOTONE_LAB_JOB");
    let lab: LabFile = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let out = PathBuf::from(&lab.out_dir);
    std::fs::create_dir_all(&out).unwrap();
    if let Some(d) = &lab.dump_builtins {
        std::fs::write(d, serde_json::to_string_pretty(&builtin_presets()).unwrap()).unwrap();
    }
    let rate = 48000u32;
    let mut inputs = Vec::new();
    for (k, f) in &lab.inputs {
        let clip = crate::audio::decode(std::path::Path::new(f)).unwrap();
        let clip = if clip.sample_rate != rate { crate::audio::resample(&clip, rate).unwrap() } else { clip };
        inputs.push((k.clone(), clip.samples));
    }
    let mut results = Vec::new();
    for job in &lab.jobs {
        let mut p = match (&job.builtin, &job.preset) {
            (Some(id), _) => preset_by_id(id).unwrap_or_else(|| panic!("no builtin {id}")),
            (None, Some(p)) => p.clone(),
            _ => panic!("job {} has neither builtin nor preset", job.name),
        };
        p.sanitize();
        *crate::voice::dsp::shifter::TUNE.lock().unwrap() =
            job.tune.as_ref().map(|t| serde_json::from_value(t.clone()).expect("bad tune"));
        for (k, x) in &inputs {
            let o = RenderOptions { tail_secs: lab.tail_secs, ..Default::default() };
            let t0 = std::time::Instant::now();
            let y = render_with(x, rate, &p, &o);
            let el = t0.elapsed().as_secs_f32();
            let secs = y.len() as f32 / rate as f32;
            let f = out.join(format!("{}__{}.wav", job.name, k));
            crate::audio::write_wav(&f, &y, rate, 1).unwrap();
            let lat = Processor::new(rate as f32, 128, o.front_end, &p.blocks, 0.0, -1.0).latency();
            results.push(serde_json::json!({
                "name": job.name, "input": k, "file": f.display().to_string(),
                "cpu": el / secs, "latency_ms": lat as f32 * 1000.0 / rate as f32,
            }));
            println!("{:<24} {:<8} cpu {:5.2} %", job.name, k, el / secs * 100.0);
        }
    }
    std::fs::write(out.join("results.json"), serde_json::to_string_pretty(&results).unwrap()).unwrap();
}
