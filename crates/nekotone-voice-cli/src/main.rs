//! `voicekit`: a slim command line for the compute server, trimmed from
//! Nekotone. Every command is a thin call into `nekotone-voice-core`, so the
//! command line and the Voicekit app do exactly the same thing.
//!
//! The live voice changer and Speak-for-me are interactive and only live in
//! the Voicekit app; this command line runs the headless half: the compute
//! server another PC's Voicekit sends voice-clone work to, and its model
//! management.

mod admin;
mod serve_cmd;
mod service_cmd;
mod util;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use nekotone_voice_core::models::{self, Accelerator, ModelManager};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "voicekit", version, about = "Run Voicekit's voice-clone compute server, and manage its models")]
struct Cli {
    /// Models folder (default: %LOCALAPPDATA%\NekotoneVoice\models)
    #[arg(long, global = true)]
    models_dir: Option<PathBuf>,
    /// Where models run: auto (GPU through DirectML when possible, else CPU) or cpu
    #[arg(long, global = true, value_enum, default_value = "auto")]
    gpu: GpuArg,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum GpuArg {
    Auto,
    Cpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Manage the downloadable models
    Models {
        #[command(subcommand)]
        what: admin::ModelsCmd,
    },
    /// Run a compute server: another PC's Voicekit sends it the voice-clone work
    Serve(serve_cmd::ServeArgs),
    /// Run the compute server in the background, starting with this machine (a Windows service; a systemd unit on Linux)
    Service {
        #[command(subcommand)]
        what: service_cmd::ServiceCmd,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.gpu, GpuArg::Cpu) {
        models::set_accelerator(Accelerator::Cpu);
    }
    let ctx = util::Ctx {
        models: match cli.models_dir {
            Some(d) => ModelManager::new(d),
            None => ModelManager::default(),
        },
    };
    match cli.cmd {
        Cmd::Models { what } => admin::models_cmd(&ctx, what),
        Cmd::Serve(a) => serve_cmd::serve(&ctx, a),
        Cmd::Service { what } => service_cmd::service(&ctx, what),
    }
}
