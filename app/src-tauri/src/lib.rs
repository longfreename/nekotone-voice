pub mod cmd_voice;
pub mod commands;
mod settings;

use nekotone_voice_core::models::ModelManager;
use parking_lot::Mutex;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use tauri::Manager;

pub struct AppState {
    pub settings: Mutex<settings::Settings>,
    pub settings_write: Mutex<()>,
    pub data_dir: PathBuf,
    pub models: ModelManager,
}

pub fn data_dir() -> PathBuf {
    nekotone_voice_core::data_dir()
}

pub fn log_path() -> PathBuf {
    data_dir().join("voicekit.log")
}

pub fn log_file(line: &str) {
    let dir = data_dir();
    let _ = std::fs::create_dir_all(&dir);
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(log_path()) {
        let _ = writeln!(file, "{line}");
    }
}

struct EngineLog;

impl log::Log for EngineLog {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn && metadata.target().starts_with("nekotone")
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            log_file(&format!("{}: {}", record.level().as_str().to_lowercase(), record.args()));
        }
    }
    fn flush(&self) {}
}

fn make_state() -> AppState {
    let data_dir = data_dir();
    let _ = std::fs::create_dir_all(&data_dir);
    let mut settings = settings::load(&data_dir);
    settings::nudge_window(&mut settings);
    let models = ModelManager::new(data_dir.join("models"));
    AppState { settings: Mutex::new(settings), settings_write: Mutex::new(()), data_dir, models }
}

fn special_window(window: &tauri::Window) -> bool {
    window.is_maximized().unwrap_or(false) || window.is_fullscreen().unwrap_or(false) || window.is_minimized().unwrap_or(false)
}

fn place_window(win: &tauri::WebviewWindow, state: &AppState) {
    let (mut width, mut height, pos, maximized) = {
        let s = state.settings.lock();
        (s.window_width.max(920) as f64, s.window_height.max(620) as f64, s.window_x.zip(s.window_y), s.window_maximized)
    };
    if let Ok(Some(monitor)) = win.current_monitor().or_else(|_| win.primary_monitor()) {
        let scale = monitor.scale_factor();
        let area = monitor.work_area();
        width = width.min(area.size.width as f64 / scale);
        height = height.min(area.size.height as f64 / scale);
    }
    let _ = win.set_size(tauri::Size::Logical(tauri::LogicalSize::new(width, height)));
    if let Some((x, y)) = pos {
        let _ = win.set_position(tauri::PhysicalPosition::new(x, y));
    } else {
        let _ = win.center();
    }
    if maximized {
        let _ = win.maximize();
    }
}

pub fn run() {
    static ENGINE_LOG: EngineLog = EngineLog;
    if log::set_logger(&ENGINE_LOG).is_ok() {
        log::set_max_level(log::LevelFilter::Warn);
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let state = make_state();
            let accel = state.settings.lock().accelerator.clone();
            nekotone_voice_core::models::set_accelerator(match accel.as_str() {
                "cpu" => nekotone_voice_core::models::Accelerator::Cpu,
                "directml" => nekotone_voice_core::models::Accelerator::DirectMl,
                _ => nekotone_voice_core::models::Accelerator::Auto,
            });
            let serve_enabled = state.settings.lock().serve_enabled;
            app.manage(state);
            if let Some(window) = app.get_webview_window("main") {
                place_window(&window, &app.state::<AppState>());
                let _ = window.show();
            }
            if serve_enabled {
                let handle = app.handle().clone();
                std::thread::spawn(move || {
                    let _ = cmd_voice::start_serving(&handle);
                });
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            match event {
                tauri::WindowEvent::Resized(size) => {
                    let state = window.state::<AppState>();
                    if !special_window(window) {
                        if let Ok(scale) = window.scale_factor() {
                            let logical = size.to_logical::<u32>(scale);
                            if logical.width >= 920 && logical.height >= 620 {
                                let mut s = state.settings.lock();
                                s.window_width = logical.width;
                                s.window_height = logical.height;
                            }
                        }
                    }
                    state.settings.lock().window_maximized = window.is_maximized().unwrap_or(false);
                }
                tauri::WindowEvent::Moved(pos) => {
                    if !special_window(window) && pos.x > -16000 && pos.y > -16000 {
                        let state = window.state::<AppState>();
                        let mut s = state.settings.lock();
                        s.window_x = Some(pos.x);
                        s.window_y = Some(pos.y);
                    }
                }
                tauri::WindowEvent::Destroyed => {
                    cmd_voice::shutdown();
                    let state = window.state::<AppState>();
                    let _ = settings::store(&state.data_dir, &state.settings.lock());
                }
                _ => {}
            }
        })
        .invoke_handler(tauri::generate_handler![
            commands::meta,
            commands::get_settings,
            commands::set_settings,
            commands::voice_models,
            commands::voice_model_download,
            commands::voice_model_remove,
            commands::open_path,
            commands::log_line,
            commands::exit_app,
            cmd_voice::voice_devices,
            cmd_voice::voice_presets,
            cmd_voice::voice_block_types,
            cmd_voice::voice_start,
            cmd_voice::voice_stop,
            cmd_voice::voice_status,
            cmd_voice::voice_set_preset,
            cmd_voice::voice_set_mute,
            cmd_voice::voice_set_output_gain,
            cmd_voice::voice_set_monitor_gain,
            cmd_voice::voice_set_front_end,
            cmd_voice::voice_set_ceiling,
            cmd_voice::voice_sample,
            cmd_voice::voice_import_sample,
            cmd_voice::voice_store_sample,
            cmd_voice::voice_delete_sample,
            cmd_voice::voice_preview,
            cmd_voice::voice_profile,
            cmd_voice::voice_calibrate,
            cmd_voice::voice_forget_profile,
            cmd_voice::compute_serve,
            cmd_voice::compute_serve_status,
            cmd_voice::compute_server_test,
            cmd_voice::speak_info,
            cmd_voice::speak_start,
            cmd_voice::speak_stop,
            cmd_voice::speak_status,
            cmd_voice::speak_say,
            cmd_voice::speak_skip,
            cmd_voice::speak_clear,
            cmd_voice::speak_set_mute,
            cmd_voice::speak_set_listening,
            cmd_voice::speak_set_voice,
            cmd_voice::speak_set_effect,
            cmd_voice::speak_set_accent,
            cmd_voice::speak_set_translate,
            cmd_voice::speak_set_options,
            cmd_voice::speak_set_gains,
            cmd_voice::speak_preview,
            cmd_voice::speak_clone_info,
            cmd_voice::speak_clone_make,
            cmd_voice::speak_clone_delete,
            cmd_voice::speak_clone_rename
        ])
        .run(tauri::generate_context!())
        .expect("error while running Voicekit");
}
