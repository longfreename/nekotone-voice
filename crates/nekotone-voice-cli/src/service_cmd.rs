//! `voicekit service …`: run the compute server in the background, starting
//! with the machine. Windows: a real Windows service (runs with nobody
//! logged in). Linux: a systemd user unit (or use the Docker image in
//! packaging/server).

use crate::serve_cmd::ServeArgs;
use crate::util::Ctx;
use anyhow::{Context, Result};

#[derive(clap::Subcommand)]
pub enum ServiceCmd {
    /// Install and start it (Windows: run from an administrator terminal)
    Install(ServeArgs),
    /// Stop and remove it
    Uninstall,
    /// Start it (installed, stopped)
    Start,
    /// Stop it (it starts again with the machine)
    Stop,
    /// Whether it is installed and running, and where it listens
    Status,
    /// Run as the service (Windows calls this; not for use by hand)
    #[command(hide = true)]
    Run(ServeArgs),
}

/// The name it is registered under.
pub const NAME: &str = "NekotoneVoiceServer";

pub fn service(ctx: &Ctx, cmd: ServiceCmd) -> Result<()> {
    imp::service(ctx, cmd)
}

/// The arguments the service runs with: the models folder and prints of the
/// installing user (a service runs as another account, whose own folders
/// would be empty), then the serve options.
fn run_args(ctx: &Ctx, a: &ServeArgs) -> Vec<String> {
    let mut v = vec!["--models-dir".into(), ctx.models.root().display().to_string(), "service".into(), "run".into()];
    v.extend(["--host".into(), a.host.clone(), "--port".into(), a.port.to_string(), "--threads".into(), a.threads.to_string()]);
    let prints = a.prints.clone().unwrap_or_else(|| nekotone_voice_core::data_dir().join("server-prints"));
    v.extend(["--prints".into(), prints.display().to_string()]);
    if let Some(t) = a.token.as_ref().filter(|t| !t.trim().is_empty()) {
        v.extend(["--token".into(), t.clone()]);
    }
    let name = a.name.clone().or_else(|| std::env::var("COMPUTERNAME").ok()).or_else(|| std::fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()));
    if let Some(n) = name.filter(|n| !n.is_empty()) {
        v.extend(["--name".into(), n]);
    }
    v
}

#[cfg(windows)]
mod imp {
    use super::*;
    use std::ffi::OsString;
    use std::time::Duration;
    use windows_service::service::{ServiceAccess, ServiceControl, ServiceControlAccept, ServiceErrorControl, ServiceExitCode, ServiceInfo, ServiceStartType, ServiceState, ServiceStatus, ServiceType};
    use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
    use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};
    use windows_service::{define_windows_service, service_dispatcher};

    fn manager(create: bool) -> Result<ServiceManager> {
        let access = if create { ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE } else { ServiceManagerAccess::CONNECT };
        ServiceManager::local_computer(None::<&str>, access).context("could not reach the Windows service manager (installing needs an administrator terminal)")
    }

    pub fn service(ctx: &Ctx, cmd: ServiceCmd) -> Result<()> {
        match cmd {
            ServiceCmd::Install(a) => {
                let exe = std::env::current_exe()?;
                let args = run_args(ctx, &a);
                let m = manager(true)?;
                let info = ServiceInfo {
                    name: OsString::from(NAME),
                    display_name: OsString::from("Voicekit compute server"),
                    service_type: ServiceType::OWN_PROCESS,
                    start_type: ServiceStartType::AutoStart,
                    error_control: ServiceErrorControl::Normal,
                    executable_path: exe,
                    launch_arguments: args.iter().map(OsString::from).collect(),
                    dependencies: vec![],
                    account_name: None, // LocalSystem
                    account_password: None,
                };
                let s = m.create_service(&info, ServiceAccess::CHANGE_CONFIG | ServiceAccess::START | ServiceAccess::QUERY_STATUS).context("could not create the service (run this from an administrator terminal; `voicekit service uninstall` first if it exists)")?;
                let _ = s.set_description("Runs Voicekit's voice clone for other computers (voicekit serve).");
                // let other computers in (best effort; the rule is removed on uninstall)
                let rule = format!("name=Voicekit server ({})", a.port);
                let fw = std::process::Command::new("netsh").args(["advfirewall", "firewall", "add", "rule", &rule, "dir=in", "action=allow", "protocol=TCP", &format!("localport={}", a.port)]).output();
                if !matches!(fw, Ok(ref o) if o.status.success()) {
                    eprintln!("note: could not add a firewall rule for port {}; other computers may be blocked", a.port);
                }
                s.start::<&str>(&[]).context("installed, but could not start it")?;
                println!("installed and started {NAME}: other computers connect to {}:{}", nekotone_voice_core::remote::lan_address().map(|i| i.to_string()).unwrap_or_else(|| "this-pc".into()), a.port);
                println!("models: {}", ctx.models.root().display());
                Ok(())
            }
            ServiceCmd::Uninstall => {
                let m = manager(false)?;
                let s = m.open_service(NAME, ServiceAccess::STOP | ServiceAccess::DELETE | ServiceAccess::QUERY_STATUS | ServiceAccess::QUERY_CONFIG).context("the service is not installed")?;
                // the port it was installed with, for its firewall rule
                let port = s
                    .query_config()
                    .ok()
                    .and_then(|c| {
                        let cmd = c.executable_path.display().to_string();
                        let mut it = cmd.split_whitespace().map(|w| w.trim_matches('"'));
                        while let Some(w) = it.next() {
                            if w == "--port" {
                                return it.next().and_then(|p| p.parse::<u16>().ok());
                            }
                        }
                        None
                    })
                    .unwrap_or(nekotone_voice_core::remote::DEFAULT_PORT);
                if s.query_status()?.current_state != ServiceState::Stopped {
                    let _ = s.stop();
                    for _ in 0..30 {
                        if s.query_status()?.current_state == ServiceState::Stopped {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(200));
                    }
                }
                s.delete().context("could not remove the service (administrator terminal?)")?;
                let _ = std::process::Command::new("netsh").args(["advfirewall", "firewall", "delete", "rule", &format!("name=Voicekit server ({port})")]).output();
                println!("removed {NAME}");
                Ok(())
            }
            ServiceCmd::Start => {
                let s = manager(false)?.open_service(NAME, ServiceAccess::START).context("the service is not installed: voicekit service install")?;
                s.start::<&str>(&[])?;
                println!("started");
                Ok(())
            }
            ServiceCmd::Stop => {
                let s = manager(false)?.open_service(NAME, ServiceAccess::STOP).context("the service is not installed")?;
                s.stop()?;
                println!("stopped");
                Ok(())
            }
            ServiceCmd::Status => {
                match manager(false)?.open_service(NAME, ServiceAccess::QUERY_STATUS | ServiceAccess::QUERY_CONFIG) {
                    Err(_) => println!("not installed (voicekit service install, from an administrator terminal)"),
                    Ok(s) => {
                        let st = s.query_status()?;
                        println!("{NAME}: {:?}", st.current_state);
                        if let Ok(c) = s.query_config() {
                            println!("runs: {}", c.executable_path.display());
                        }
                    }
                }
                Ok(())
            }
            ServiceCmd::Run(a) => run_as_service(ctx, a),
        }
    }

    // The service manager calls ffi_service_main on its own thread; the
    // arguments come from the command line we registered.
    static ARGS: std::sync::Mutex<Option<(Option<std::path::PathBuf>, ServeArgsOwned)>> = std::sync::Mutex::new(None);

    #[derive(Clone)]
    struct ServeArgsOwned {
        host: String,
        port: u16,
        token: Option<String>,
        name: Option<String>,
        prints: Option<std::path::PathBuf>,
        threads: usize,
    }

    define_windows_service!(ffi_service_main, service_main);

    fn run_as_service(ctx: &Ctx, a: ServeArgs) -> Result<()> {
        *ARGS.lock().unwrap() = Some((Some(ctx.models.root().to_path_buf()), ServeArgsOwned { host: a.host, port: a.port, token: a.token, name: a.name, prints: a.prints, threads: a.threads }));
        service_dispatcher::start(NAME, ffi_service_main).context("not started by the Windows service manager (use `voicekit serve` to run it by hand)")?;
        Ok(())
    }

    fn service_main(_args: Vec<OsString>) {
        let (models, a) = ARGS.lock().unwrap().clone().expect("arguments");
        let listener = match nekotone_voice_core::remote::bind(&format!("{}:{}", a.host, a.port)) {
            Ok(l) => l,
            Err(_) => return,
        };
        let stop = listener.stopper();
        let handler = move |c| match c {
            ServiceControl::Stop | ServiceControl::Shutdown => {
                stop.stop();
                ServiceControlHandlerResult::NoError
            }
            ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
            _ => ServiceControlHandlerResult::NotImplemented,
        };
        let Ok(h) = service_control_handler::register(NAME, handler) else { return };
        let status = |state, accept| ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: state,
            controls_accepted: accept,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::from_secs(30),
            process_id: None,
        };
        let _ = h.set_service_status(status(ServiceState::StartPending, ServiceControlAccept::empty()));
        let mm = match models {
            Some(d) => nekotone_voice_core::models::ModelManager::new(d),
            None => nekotone_voice_core::models::ModelManager::default(),
        };
        let engine = mm
            .path(nekotone_voice_core::models::ModelId::TtsChatterbox)
            .ok_or_else(|| anyhow::anyhow!("no voice-clone model"))
            .and_then(|d| Ok(std::sync::Arc::new(nekotone_voice_core::tts::chatterbox::Chatterbox::load_dir(&d, nekotone_voice_core::models::accelerator())?)));
        let Ok(cb) = engine else {
            let _ = h.set_service_status(ServiceStatus { exit_code: ServiceExitCode::ServiceSpecific(1), ..status(ServiceState::Stopped, ServiceControlAccept::empty()) });
            return;
        };
        let _ = h.set_service_status(status(ServiceState::Running, ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN));
        let prints = a.prints.unwrap_or_else(|| nekotone_voice_core::data_dir().join("server-prints"));
        let _ = listener.run(std::sync::Arc::new(nekotone_voice_core::remote::ChatterboxEngine::new(cb)), &prints, a.token, a.threads, a.name);
        let _ = h.set_service_status(status(ServiceState::Stopped, ServiceControlAccept::empty()));
    }
}

#[cfg(not(windows))]
mod imp {
    use super::*;

    fn unit_path() -> Result<std::path::PathBuf> {
        let home = std::env::var("HOME").context("HOME is not set")?;
        Ok(std::path::PathBuf::from(home).join(".config/systemd/user/nekotone-voice-server.service"))
    }

    pub fn service(ctx: &Ctx, cmd: ServiceCmd) -> Result<()> {
        let systemctl = |args: &[&str]| -> Result<()> {
            let st = std::process::Command::new("systemctl").arg("--user").args(args).status().context("systemctl is not available (use the Docker image in packaging/server instead)")?;
            anyhow::ensure!(st.success(), "systemctl --user {} failed", args.join(" "));
            Ok(())
        };
        match cmd {
            ServiceCmd::Install(a) => {
                let exe = std::env::current_exe()?;
                let args = run_args(ctx, &a);
                // systemd runs `serve`, not the Windows `service run`
                let serve: Vec<String> = args.iter().map(|s| if s == "service" { "serve".to_string() } else { s.clone() }).filter(|s| s != "run").collect();
                let quoted: Vec<String> = serve.iter().map(|s| format!("\"{}\"", s.replace('"', "\\\""))).collect();
                let unit = format!(
                    "[Unit]\nDescription=Voicekit compute server\nAfter=network-online.target\n\n[Service]\nExecStart=\"{}\" {}\nRestart=on-failure\n\n[Install]\nWantedBy=default.target\n",
                    exe.display(),
                    quoted.join(" ")
                );
                let p = unit_path()?;
                std::fs::create_dir_all(p.parent().unwrap())?;
                std::fs::write(&p, unit)?;
                systemctl(&["daemon-reload"])?;
                systemctl(&["enable", "--now", "nekotone-voice-server.service"])?;
                println!("installed {} and started it; to keep it running when you are logged out: loginctl enable-linger $USER", p.display());
                Ok(())
            }
            ServiceCmd::Uninstall => {
                let _ = systemctl(&["disable", "--now", "nekotone-voice-server.service"]);
                let _ = std::fs::remove_file(unit_path()?);
                let _ = systemctl(&["daemon-reload"]);
                println!("removed");
                Ok(())
            }
            ServiceCmd::Start => systemctl(&["start", "nekotone-voice-server.service"]),
            ServiceCmd::Stop => systemctl(&["stop", "nekotone-voice-server.service"]),
            ServiceCmd::Status => systemctl(&["status", "--no-pager", "nekotone-voice-server.service"]),
            ServiceCmd::Run(a) => crate::serve_cmd::serve(ctx, a),
        }
    }
}
