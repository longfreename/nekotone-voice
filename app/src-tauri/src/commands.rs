use crate::{settings, AppState};
use nekotone_voice_core::audio;
use nekotone_voice_core::models::{ModelId, ModelManager, ModelStatus};
use nekotone_voice_core::{Error, Progress};
use serde::Serialize;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, State};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CmdError {
    pub kind: &'static str,
    pub message: String,
}

pub type CmdResult<T> = Result<T, CmdError>;

pub fn err(e: Error) -> CmdError {
    let kind = match &e {
        Error::NotImplemented(_) => "not-implemented",
        Error::ModelMissing { .. } => "model-missing",
        Error::NotFound(_) => "not-found",
        Error::Cancelled => "cancelled",
        _ => "error",
    };
    CmdError { kind, message: e.to_string() }
}

pub fn any(e: impl std::fmt::Display) -> CmdError {
    CmdError { kind: "error", message: e.to_string() }
}

pub async fn blocking<T: Send + 'static>(f: impl FnOnce() -> CmdResult<T> + Send + 'static) -> CmdResult<T> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(any)?
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Meta {
    pub version: String,
    pub data_dir: String,
    pub models_dir: String,
    pub log_path: String,
    pub supported_extensions: Vec<String>,
}

#[tauri::command]
pub fn meta(state: State<'_, AppState>) -> Meta {
    Meta {
        version: env!("CARGO_PKG_VERSION").into(),
        data_dir: state.data_dir.display().to_string(),
        models_dir: state.models.root().display().to_string(),
        log_path: crate::log_path().display().to_string(),
        supported_extensions: audio::SUPPORTED_EXTENSIONS.iter().map(|s| s.to_string()).collect(),
    }
}

#[tauri::command]
pub fn get_settings(state: State<'_, AppState>) -> settings::Settings {
    state.settings.lock().clone()
}

#[tauri::command]
pub async fn set_settings(app: AppHandle, value: settings::Settings) -> CmdResult<()> {
    let dir = {
        let state = app.state::<AppState>();
        let mut current = state.settings.lock();
        let mut next = value;
        next.window_width = current.window_width;
        next.window_height = current.window_height;
        next.window_x = current.window_x;
        next.window_y = current.window_y;
        next.window_maximized = current.window_maximized;
        next.window_layout = current.window_layout;
        nekotone_voice_core::models::set_accelerator(accel(&next.accelerator));
        *current = next;
        state.data_dir.clone()
    };
    blocking(move || {
        let state = app.state::<AppState>();
        let _write = state.settings_write.lock();
        let latest = state.settings.lock().clone();
        settings::store(&dir, &latest).map_err(any)
    })
    .await
}

fn accel(name: &str) -> nekotone_voice_core::models::Accelerator {
    match name {
        "cpu" => nekotone_voice_core::models::Accelerator::Cpu,
        "directml" => nekotone_voice_core::models::Accelerator::DirectMl,
        _ => nekotone_voice_core::models::Accelerator::Auto,
    }
}

fn voice_model_ids() -> &'static [ModelId] {
    &[
        ModelId::WhisperTiny,
        ModelId::WhisperBase,
        ModelId::WhisperSmall,
        ModelId::WhisperMedium,
        ModelId::WhisperLargeTurbo,
        ModelId::TtsKokoro,
        ModelId::TtsChatterbox,
        ModelId::GpuNvidia,
    ]
}

fn voice_models_only(models: &ModelManager) -> Vec<ModelStatus> {
    let ids = voice_model_ids();
    models.status().into_iter().filter(|item| ids.contains(&item.info.id)).collect()
}

#[tauri::command]
pub async fn voice_models(app: AppHandle) -> CmdResult<Vec<ModelStatus>> {
    let models = app.state::<AppState>().models.clone();
    blocking(move || Ok(voice_models_only(&models))).await
}

#[tauri::command]
pub async fn voice_model_download(app: AppHandle, id: String, on_progress: Channel<Progress>) -> CmdResult<String> {
    let mid = ModelId::from_name(&id).ok_or_else(|| any(format!("unknown model {id}")))?;
    let models = app.state::<AppState>().models.clone();
    blocking(move || {
        let mut on = |progress: Progress| {
            let _ = on_progress.send(progress);
        };
        models.ensure(mid, &mut on).map(|path| path.display().to_string()).map_err(err)
    })
    .await
}

#[tauri::command]
pub async fn voice_model_remove(app: AppHandle, id: String) -> CmdResult<()> {
    let mid = ModelId::from_name(&id).ok_or_else(|| any(format!("unknown model {id}")))?;
    let models = app.state::<AppState>().models.clone();
    blocking(move || models.remove(mid).map_err(err)).await
}

#[tauri::command]
pub async fn open_path(path: String) -> CmdResult<()> {
    blocking(move || open_target(&path)).await
}

fn open_target(path: &str) -> CmdResult<()> {
    #[cfg(target_os = "windows")]
    {
        let status = std::process::Command::new("cmd").args(["/C", "start", "", path]).status().map_err(any)?;
        if status.success() {
            Ok(())
        } else {
            Err(any(format!("could not open {path}")))
        }
    }
    #[cfg(target_os = "macos")]
    {
        let status = std::process::Command::new("open").arg(path).status().map_err(any)?;
        if status.success() {
            Ok(())
        } else {
            Err(any(format!("could not open {path}")))
        }
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        let status = std::process::Command::new("xdg-open").arg(path).status().map_err(any)?;
        if status.success() {
            Ok(())
        } else {
            Err(any(format!("could not open {path}")))
        }
    }
}

#[tauri::command]
pub fn log_line(line: String) {
    eprintln!("[ui] {line}");
    crate::log_file(&format!("[ui] {line}"));
}

#[tauri::command]
pub fn exit_app(app: AppHandle, code: i32) {
    let state = app.state::<AppState>();
    let _ = settings::store(&state.data_dir, &state.settings.lock());
    app.exit(code);
}
