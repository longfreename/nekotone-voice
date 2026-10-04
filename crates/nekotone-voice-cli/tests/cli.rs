//! End-to-end tests of the `voicekit` binary: an empty models folder
//! (`--models-dir`), so nothing touches the user's real models and no
//! network or GPU is needed (an empty folder reports every model as not
//! installed).

use std::path::PathBuf;
use std::process::{Command, Output};

struct Cli {
    _dir: tempfile::TempDir,
    models: PathBuf,
}

impl Cli {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        Cli { _dir: dir, models }
    }

    /// `voicekit --models-dir <empty> args…`
    fn run(&self, args: &[&str]) -> Output {
        let mut c = Command::new(env!("CARGO_BIN_EXE_voicekit"));
        c.arg("--models-dir").arg(&self.models).args(args);
        c.output().expect("run voicekit")
    }

    fn ok(&self, args: &[&str]) -> String {
        let o = self.run(args);
        assert!(o.status.success(), "voicekit {args:?} failed: {}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8(o.stdout).unwrap()
    }
}

#[test]
fn models_list_with_nothing_downloaded() {
    let cli = Cli::new();
    let out = cli.ok(&["models", "list"]);
    // every model line starts with a space (not installed) or a checkmark
    assert!(out.contains("stored in"), "{out}");
    assert!(!out.contains('✓'), "nothing should be installed yet:\n{out}");
}

#[test]
fn models_list_json_is_parseable() {
    let cli = Cli::new();
    let out = cli.ok(&["models", "list", "--json"]);
    let v: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    assert!(v.as_array().is_some_and(|a| !a.is_empty()), "{out}");
}

#[test]
fn models_get_unknown_name_is_a_clear_error() {
    let cli = Cli::new();
    let o = cli.run(&["models", "get", "not-a-real-model"]);
    assert!(!o.status.success());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("unknown model"), "{err}");
}

#[test]
fn models_remove_when_not_downloaded_says_so() {
    let cli = Cli::new();
    let out = cli.ok(&["models", "remove", "tts-chatterbox"]);
    assert!(out.contains("was not downloaded"), "{out}");
}

#[test]
fn service_status_when_not_installed() {
    let cli = Cli::new();
    let out = cli.ok(&["service", "status"]);
    assert!(out.to_lowercase().contains("not installed"), "{out}");
}
