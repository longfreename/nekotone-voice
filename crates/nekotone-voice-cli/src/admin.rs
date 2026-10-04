//! `voicekit models`: which voice/TTS models exist, download, remove.

use crate::util::Ctx;
use anyhow::Context;
use anyhow::Result;
use clap::Subcommand;
use nekotone_voice_core::models::{self, ModelId};
use nekotone_voice_core::Progress;

#[derive(Subcommand)]
pub enum ModelsCmd {
    /// Which models exist, their sizes and licences, and which are downloaded
    List {
        #[arg(long)]
        json: bool,
    },
    /// Download a model (resumes an interrupted download; Ctrl+C stops and keeps the part)
    Get { name: String },
    /// Delete a downloaded model
    Remove { name: String },
}

fn model_id(name: &str) -> Result<ModelId> {
    let n = name.trim().to_ascii_lowercase();
    let n = match n.as_str() {
        "tiny" | "base" | "small" | "medium" => format!("whisper-{n}"),
        _ => n,
    };
    ModelId::from_name(&n).with_context(|| {
        let all: Vec<&str> = ModelId::ALL.iter().map(|m| m.name()).collect();
        format!("unknown model `{name}`; the models are {}", all.join(", "))
    })
}

fn where_it_runs() -> String {
    if !models::gpu_supported() {
        return "models run on the CPU (this build has no GPU support)".into();
    }
    match models::accelerator() {
        models::Accelerator::Cpu if std::env::var("NEKOTONE_GPU").map(|v| v.trim() == "0").unwrap_or(false) => {
            "models run on the CPU (NEKOTONE_GPU=0)".into()
        }
        models::Accelerator::Cpu => "models run on the CPU (--gpu cpu)".into(),
        models::Accelerator::DirectMl => "models use the GPU through DirectML (NEKOTONE_GPU=directml), else the CPU".into(),
        models::Accelerator::Auto => {
            let nv = nekotone_voice_core::nvidia::status();
            let how = match (&nv.gpu, nv.runtime) {
                (Some(g), true) => format!("the {g} through DirectML, else the CPU"),
                (Some(g), false) => format!("the {g} through DirectML, else the CPU (`voicekit models get gpu-nvidia` speeds up the voice clone, where it measured faster)"),
                (None, _) => "the GPU through DirectML when there is a DirectX 12 card, else the CPU".into(),
            };
            format!("models use {how}")
        }
    }
}

pub fn models_cmd(ctx: &Ctx, what: ModelsCmd) -> Result<()> {
    let mm = &ctx.models;
    match what {
        ModelsCmd::List { json } => {
            let all = mm.status();
            if json {
                println!("{}", serde_json::to_string_pretty(&all)?);
                return Ok(());
            }
            for s in &all {
                let mb = s.info.total_bytes() as f64 / 1e6;
                let size = if mb < 10.0 { format!("{mb:>6.1}") } else { format!("{mb:>6.0}") };
                println!(
                    "{} {:<21} {size} MB  {}  ({})",
                    if s.installed { "✓" } else { " " },
                    s.info.id.name(),
                    s.info.purpose,
                    s.info.note
                );
                println!("  {:<21}           licence: {}", "", s.info.license);
            }
            println!("stored in {}", mm.root().display());
            println!("{}", where_it_runs());
        }
        ModelsCmd::Get { name } => {
            let id = model_id(&name)?;
            let info = models::info(id);
            if mm.path(id).is_none() {
                eprintln!("{}: {} ({}; licence {})", id.name(), crate::util::megabytes(info.total_bytes()), info.purpose, info.license);
            }
            let cancel = crate::util::cancel_flag();
            let r = mm.ensure_cancellable(
                id,
                &mut |p: Progress| {
                    if let Some(f) = p.fraction {
                        eprint!("\r{} {:>3}%   ", p.message, (f * 100.0) as u32);
                    }
                },
                cancel,
            );
            eprintln!();
            match r {
                Ok(p) => println!("ready: {}", p.display()),
                Err(nekotone_voice_core::Error::Cancelled) => {
                    anyhow::bail!("download stopped; run the same command again to resume where it stopped")
                }
                Err(e) => return Err(e.into()),
            }
        }
        ModelsCmd::Remove { name } => {
            let id = model_id(&name)?;
            let had = mm.path(id).is_some() || mm.dir_of(id).is_dir();
            mm.remove(id)?;
            println!("{} {}", if had { "removed" } else { "was not downloaded:" }, id.name());
        }
    }
    Ok(())
}
