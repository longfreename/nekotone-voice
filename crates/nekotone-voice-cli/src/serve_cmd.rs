//! `voicekit serve`: the compute server (nekotone_voice_core::remote). Another
//! PC's app sends it the voice-clone work (your voice reading typed text,
//! Change my voice) and falls back to itself when the server is away.
//!
//! Runs wherever voicekit runs: Windows (DirectML, or TensorRT for RTX), Linux
//! with an NVIDIA GPU (a `--features cuda` build; packaging/server has a
//! Docker image), or any CPU. Every option can also come from the
//! environment (NEKOTONE_SERVER_*), for containers and services.

use crate::util::Ctx;
use anyhow::{Context, Result};
use nekotone_voice_core::models::ModelId;
use nekotone_voice_core::remote;
use nekotone_voice_core::Progress;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(clap::Args)]
pub struct ServeArgs {
    /// Address to listen on [env: NEKOTONE_SERVER_HOST]
    #[arg(long, env = "NEKOTONE_SERVER_HOST", default_value = "0.0.0.0")]
    pub host: String,
    /// Port [env: NEKOTONE_SERVER_PORT]
    #[arg(long, env = "NEKOTONE_SERVER_PORT", default_value_t = remote::DEFAULT_PORT)]
    pub port: u16,
    /// Require this token from clients (Authorization: ****** [env: NEKOTONE_SERVER_TOKEN]
    #[arg(long, env = "NEKOTONE_SERVER_TOKEN", hide_env_values = true)]
    pub token: Option<String>,
    /// The name clients show for this server (default: the host name) [env: NEKOTONE_SERVER_NAME]
    #[arg(long, env = "NEKOTONE_SERVER_NAME")]
    pub name: Option<String>,
    /// Where uploaded voice prints are kept (default: <data dir>/server-prints) [env: NEKOTONE_SERVER_PRINTS]
    #[arg(long, env = "NEKOTONE_SERVER_PRINTS")]
    pub prints: Option<PathBuf>,
    /// Requests handled at once [env: NEKOTONE_SERVER_THREADS]
    #[arg(long, env = "NEKOTONE_SERVER_THREADS", default_value_t = 2)]
    pub threads: usize,
    /// Do not download the voice-clone model when it is missing
    #[arg(long)]
    pub no_download: bool,
}

pub fn serve(ctx: &Ctx, a: ServeArgs) -> Result<()> {
    let id = ModelId::TtsChatterbox;
    let dir = match ctx.models.path(id) {
        Some(d) => d,
        None if a.no_download => anyhow::bail!("the voice-clone model is not downloaded: run `voicekit models get tts-chatterbox`"),
        None => {
            eprintln!("downloading the voice-clone model (once, about {} MB)…", nekotone_voice_core::models::info(id).total_bytes() / 1_000_000);
            let mut last = 0u32;
            ctx.models
                .ensure(id, &mut |p: Progress| {
                    let pct = p.fraction.map(|f| (f * 100.0) as u32).unwrap_or(0);
                    if pct >= last + 10 {
                        last = pct;
                        eprintln!("  {pct}%");
                    }
                })
                .context("could not download the voice-clone model")?
        }
    };
    let token = a.token.filter(|t| !t.trim().is_empty());
    let t0 = std::time::Instant::now();
    let cb = Arc::new(nekotone_voice_core::tts::chatterbox::Chatterbox::load_dir(&dir, nekotone_voice_core::models::accelerator())?);
    eprintln!("voice clone loaded in {:.1} s on {} ({:?})", t0.elapsed().as_secs_f32(), cb.device(), cb.engines());
    let prints = a.prints.unwrap_or_else(|| nekotone_voice_core::data_dir().join("server-prints"));
    let listener = remote::bind(&format!("{}:{}", a.host, a.port))?;
    eprintln!("voicekit server listening on {}:{}{}", a.host, listener.port(), if token.is_some() { " (token required)" } else { "" });
    listener.run(Arc::new(remote::ChatterboxEngine::new(cb)), &prints, token, a.threads, a.name)?;
    Ok(())
}
