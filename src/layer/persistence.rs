use crate::layer::settings::{self, Settings};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use toml_edit::{DocumentMut, Item, Table, Value};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// A single user-driven setting change to persist into the configuration file.
pub enum Change<'a> {
    Mode {
        mode: &'a str,
        algorithm: Option<&'a str>,
        manual_fps: Option<f32>,
    },
    ManualFps(f32),
    AutoValue {
        algorithm: &'a str,
        order: &'a [String],
        field: &'a str,
        value: f32,
    },
    OverlayMode(u8),
    Renderer(&'a str),
}

/// Returns `$HOME/.config/framecap.toml`, or `NotFound` if `HOME` is unset or empty.
pub fn config_path() -> io::Result<PathBuf> {
    env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(".config/framecap.toml"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set"))
}

/// Creates the default configuration file at the standard path if none exists.
pub fn ensure_default() -> io::Result<()> {
    ensure_default_at(&config_path()?)
}

/// Takes an exclusive `flock` on a lock file beside the configuration; it is released when the returned file is dropped.
fn lock(path: &Path) -> io::Result<File> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("configuration has no parent"))?;
    fs::create_dir_all(parent)?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(parent.join("framecap.toml.lock"))?;
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(file)
}

/// Atomically writes validated default settings to `path` via a temp file and hard link, leaving any existing file untouched.
pub(super) fn ensure_default_at(path: &Path) -> io::Result<()> {
    let _lock = lock(path)?;
    match fs::symlink_metadata(path) {
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error),
    }
    let defaults = toml::to_string_pretty(&Settings::default()).map_err(io::Error::other)?;
    let mut document = defaults.parse::<DocumentMut>().map_err(io::Error::other)?;
    set(
        &mut document,
        &["manual-mode", "startup_mode"],
        Value::from("manual"),
    )?;
    set(
        &mut document,
        &["diagnostics", "append_log"],
        Value::from(0),
    )?;
    let content = document.to_string();
    settings::validate_configuration(&content).map_err(io::Error::other)?;
    let temp = write_temp(path, content.as_bytes(), None)?;
    let result = fs::hard_link(&temp, path);
    let cleanup = fs::remove_file(&temp);
    match result {
        Ok(()) => {
            cleanup?;
            sync_parent(path)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => cleanup,
        Err(error) => {
            if let Err(cleanup_error) = cleanup {
                eprintln!("Framecap: unable to remove temporary configuration: {cleanup_error}");
            }
            Err(error)
        }
    }
}

/// Applies `change` to the configuration at `path`, logging instead of returning any failure.
pub fn save(path: &Path, change: Change<'_>) {
    if let Err(error) = save_at(path, change) {
        crate::layer::diagnostics::error(format!("Unable to save user configuration: {error}"));
    }
}

/// Under the lock, edits the existing file in place (preserving comments and formatting), validates the result, and atomically replaces it.
fn save_at(path: &Path, change: Change<'_>) -> io::Result<()> {
    let _lock = lock(path)?;
    // Resolve links before replacing the target so an installed configuration link survives.
    let target = fs::canonicalize(path)?;
    OpenOptions::new().write(true).open(&target)?;
    let content = fs::read_to_string(&target)?;
    settings::validate_configuration(&content).map_err(io::Error::other)?;
    let mut document = content.parse::<DocumentMut>().map_err(io::Error::other)?;
    match change {
        Change::Mode {
            mode,
            algorithm,
            manual_fps,
        } => {
            set(
                &mut document,
                &["manual-mode", "startup_mode"],
                Value::from(mode),
            )?;
            if let Some(algorithm) = algorithm {
                set(
                    &mut document,
                    &["manual-mode", "startup_algorithm"],
                    Value::from(algorithm),
                )?;
            }
            if let Some(fps) = manual_fps {
                set(
                    &mut document,
                    &["manual-mode", "startup_fps"],
                    Value::from(f64::from(fps)),
                )?;
            }
        }
        Change::ManualFps(fps) => {
            set(
                &mut document,
                &["manual-mode", "startup_fps"],
                Value::from(f64::from(fps)),
            )?;
        }
        Change::AutoValue {
            algorithm,
            order,
            field,
            value,
        } => {
            if document
                .get("auto-mode")
                .and_then(|item| item.get("algorithms"))
                .and_then(|item| item.get(algorithm))
                .is_none()
            {
                let defaults = toml::to_string(&Settings::default()).map_err(io::Error::other)?;
                let defaults = defaults.parse::<DocumentMut>().map_err(io::Error::other)?;
                if document.get("auto-mode").is_none() {
                    document["auto-mode"] = Item::Table(Table::new());
                }
                let table = document["auto-mode"]
                    .as_table_like_mut()
                    .ok_or_else(|| io::Error::other("auto-mode is not a table"))?;
                if !table.contains_key("algorithms") {
                    table.insert("algorithms", Item::Table(Table::new()));
                }
                let algorithms = table
                    .get_mut("algorithms")
                    .and_then(Item::as_table_like_mut)
                    .ok_or_else(|| io::Error::other("algorithms is not a table"))?;
                let default_algorithms = defaults["auto-mode"]["algorithms"]
                    .as_table_like()
                    .ok_or_else(|| io::Error::other("default algorithms are not a table"))?;
                for name in order {
                    if !algorithms.contains_key(name) {
                        let default = default_algorithms
                            .get(name)
                            .ok_or_else(|| io::Error::other(format!("unknown algorithm {name}")))?;
                        algorithms.insert(name, default.clone());
                    }
                }
                let mut array = toml_edit::Array::new();
                for name in order {
                    array.push(name.as_str());
                }
                set(
                    &mut document,
                    &["auto-mode", "algorithm_order"],
                    Value::Array(array),
                )?;
            }
            set(
                &mut document,
                &["auto-mode", "algorithms", algorithm, field],
                Value::from(f64::from(value)),
            )?;
        }
        Change::OverlayMode(mode) => {
            set(
                &mut document,
                &["overlay", "startup"],
                Value::from(i64::from(mode)),
            )?;
        }
        Change::Renderer(renderer) => {
            set(
                &mut document,
                &["overlay", "renderer"],
                Value::from(renderer),
            )?;
        }
    }
    let updated = document.to_string();
    settings::validate_configuration(&updated).map_err(io::Error::other)?;
    let permissions = fs::metadata(&target)?.permissions();
    let temp = write_temp(&target, updated.as_bytes(), Some(permissions))?;
    if let Err(error) = fs::rename(&temp, &target) {
        if let Err(cleanup_error) = fs::remove_file(&temp) {
            eprintln!("Framecap: unable to remove temporary configuration: {cleanup_error}");
        }
        return Err(error);
    }
    sync_parent(&target)
}

/// Sets the value at a nested key path, creating missing tables and keeping the old value's comments and spacing.
fn set(document: &mut DocumentMut, path: &[&str], mut value: Value) -> io::Result<()> {
    let (key, parents) = path
        .split_last()
        .ok_or_else(|| io::Error::other("empty configuration path"))?;
    let mut table = document.as_table_mut() as &mut dyn toml_edit::TableLike;
    for parent in parents {
        if !table.contains_key(parent) {
            table.insert(parent, Item::Table(Table::new()));
        }
        table = table
            .get_mut(parent)
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| io::Error::other(format!("{parent} is not a table")))?;
    }
    if let Some(old) = table.get(key).and_then(Item::as_value) {
        *value.decor_mut() = old.decor().clone();
    }
    table.insert(key, Item::Value(value));
    Ok(())
}

/// Writes `content` to a new uniquely named temp file next to `path`, optionally applying `permissions`, and syncs it to disk.
fn write_temp(
    path: &Path,
    content: &[u8],
    permissions: Option<fs::Permissions>,
) -> io::Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("configuration has no parent"))?;
    loop {
        let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let temp = parent.join(format!(".framecap-{}-{id}.tmp", std::process::id()));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        let result = (|| {
            if let Some(permissions) = permissions {
                file.set_permissions(permissions)?;
            }
            file.write_all(content)?;
            file.sync_all()
        })();
        if let Err(error) = result {
            if let Err(cleanup_error) = fs::remove_file(&temp) {
                eprintln!("Framecap: unable to remove temporary configuration: {cleanup_error}");
            }
            return Err(error);
        }
        return Ok(temp);
    }
}

/// Syncs the parent directory so a preceding rename or link is durable.
fn sync_parent(path: &Path) -> io::Result<()> {
    File::open(
        path.parent()
            .ok_or_else(|| io::Error::other("configuration has no parent"))?,
    )?
    .sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::config::{Config, StartupMode};
    use std::os::unix::fs::symlink;

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "framecap-persistence-test-{}-{}",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn config(&self) -> PathBuf {
            self.0.join("framecap.toml")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn load(path: &Path) -> Config {
        Config::from_settings(Settings::load_from_path(path.to_owned()))
    }

    #[test]
    fn creates_builtin_defaults_and_restores_all_shortcut_settings() {
        let fixture = Fixture::new();
        let path = fixture.config();
        ensure_default_at(&path).unwrap();
        let config = load(&path);
        assert_eq!(config.target_fps, Some(60.0));
        assert_eq!(config.startup_mode, StartupMode::Manual);
        assert!(fs::read_to_string(&path)
            .unwrap()
            .contains("startup_mode = \"manual\""));
        assert_eq!(config.auto_algorithms.len(), 3);

        save_at(&path, Change::ManualFps(75.0)).unwrap();
        let order = ["gpu_threshold".into(), "vrm_temp".into(), "fan_rpm".into()];
        save_at(
            &path,
            Change::AutoValue {
                algorithm: "gpu_threshold",
                order: &order,
                field: "hysteresis",
                value: 7.0,
            },
        )
        .unwrap();
        save_at(
            &path,
            Change::AutoValue {
                algorithm: "vrm_temp",
                order: &order,
                field: "target",
                value: 105.0,
            },
        )
        .unwrap();
        save_at(
            &path,
            Change::AutoValue {
                algorithm: "fan_rpm",
                order: &order,
                field: "target",
                value: 2050.0,
            },
        )
        .unwrap();
        save_at(&path, Change::OverlayMode(1)).unwrap();
        save_at(&path, Change::Renderer("helix")).unwrap();
        save_at(
            &path,
            Change::Mode {
                mode: "auto",
                algorithm: Some("fan_rpm"),
                manual_fps: None,
            },
        )
        .unwrap();
        let config = load(&path);
        assert_eq!(config.target_fps, Some(75.0));
        assert_eq!(config.startup_mode, StartupMode::Automatic(2));
        assert_eq!(config.auto_algorithms[0].hysteresis, 7.0);
        assert_eq!(
            config.auto_algorithms[1].target,
            crate::layer::config::AutoTarget::Value(105.0)
        );
        assert_eq!(
            config.auto_algorithms[2].target,
            crate::layer::config::AutoTarget::Value(2050.0)
        );
        assert_eq!(config.startup, 1);
        assert_eq!(config.overlay_renderer, settings::OverlayRenderer::Helix);

        save_at(
            &path,
            Change::Mode {
                mode: "off",
                algorithm: None,
                manual_fps: None,
            },
        )
        .unwrap();
        let config = load(&path);
        assert_eq!(config.startup_mode, StartupMode::Off);
        assert_eq!(config.target_fps, Some(75.0));
        save_at(
            &path,
            Change::Mode {
                mode: "manual",
                algorithm: None,
                manual_fps: Some(75.0),
            },
        )
        .unwrap();
        assert_eq!(load(&path).startup_mode, StartupMode::Manual);
    }

    #[test]
    fn preserves_links_comments_and_implicit_algorithms() {
        let fixture = Fixture::new();
        let target = fixture.0.join("original.toml");
        fs::write(&target, "[manual-mode]\nstartup_fps = 60 # keep this\n\n[auto-mode]\nalgorithm_order = [\"fan_rpm\", \"gpu_threshold\"]\n").unwrap();
        symlink(&target, fixture.config()).unwrap();
        ensure_default_at(&fixture.config()).unwrap();
        save_at(&fixture.config(), Change::ManualFps(65.0)).unwrap();
        save_at(
            &fixture.config(),
            Change::AutoValue {
                algorithm: "fan_rpm",
                order: &["fan_rpm".into(), "gpu_threshold".into()],
                field: "target",
                value: 2050.0,
            },
        )
        .unwrap();
        assert!(fs::symlink_metadata(fixture.config())
            .unwrap()
            .file_type()
            .is_symlink());
        assert!(fs::read_to_string(&target).unwrap().contains("# keep this"));
        let config = load(&fixture.config());
        assert_eq!(config.target_fps, Some(65.0));
        assert_eq!(
            config
                .auto_algorithms
                .iter()
                .map(|a| a.name.as_str())
                .collect::<Vec<_>>(),
            ["fan_rpm", "gpu_threshold"]
        );
        assert_eq!(
            config.auto_algorithms[0].target,
            crate::layer::config::AutoTarget::Value(2050.0)
        );
    }

    #[test]
    fn invalid_file_is_never_overwritten() {
        let fixture = Fixture::new();
        let path = fixture.config();
        fs::write(&path, "[manual-mode]\nunknown = 1\n").unwrap();
        ensure_default_at(&path).unwrap();
        assert!(save_at(&path, Change::ManualFps(75.0)).is_err());
        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "[manual-mode]\nunknown = 1\n"
        );
    }

    #[test]
    fn dangling_configuration_link_is_not_replaced() {
        let fixture = Fixture::new();
        let path = fixture.config();
        symlink(fixture.0.join("missing.toml"), &path).unwrap();
        ensure_default_at(&path).unwrap();
        assert!(save_at(&path, Change::ManualFps(75.0)).is_err());
        assert!(fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
    }

    #[test]
    fn saves_into_inline_tables_and_legacy_off_configuration() {
        let fixture = Fixture::new();
        let path = fixture.config();
        fs::write(
            &path,
            "auto-mode = { algorithm_order = [\"gpu_threshold\", \"vrm_temp\", \"fan_rpm\"] }\nmanual-mode = { startup_fps = 0 }\n",
        )
        .unwrap();
        assert_eq!(load(&path).startup_mode, StartupMode::Off);
        save_at(
            &path,
            Change::Mode {
                mode: "manual",
                algorithm: None,
                manual_fps: Some(60.0),
            },
        )
        .unwrap();
        save_at(
            &path,
            Change::AutoValue {
                algorithm: "fan_rpm",
                order: &["gpu_threshold".into(), "vrm_temp".into(), "fan_rpm".into()],
                field: "target",
                value: 2050.0,
            },
        )
        .unwrap();
        let config = load(&path);
        assert_eq!(config.startup_mode, StartupMode::Manual);
        assert_eq!(config.target_fps, Some(60.0));
        assert_eq!(
            config.auto_algorithms[2].target,
            crate::layer::config::AutoTarget::Value(2050.0)
        );
    }

    #[test]
    fn saved_auto_mode_follows_algorithm_name_when_order_changes() {
        let fixture = Fixture::new();
        let path = fixture.config();
        ensure_default_at(&path).unwrap();
        save_at(
            &path,
            Change::Mode {
                mode: "auto",
                algorithm: Some("fan_rpm"),
                manual_fps: None,
            },
        )
        .unwrap();
        let mut document = fs::read_to_string(&path)
            .unwrap()
            .parse::<DocumentMut>()
            .unwrap();
        let mut order = toml_edit::Array::new();
        for name in ["fan_rpm", "gpu_threshold", "vrm_temp"] {
            order.push(name);
        }
        set(
            &mut document,
            &["auto-mode", "algorithm_order"],
            Value::Array(order),
        )
        .unwrap();
        fs::write(&path, document.to_string()).unwrap();
        assert_eq!(load(&path).startup_mode, StartupMode::Automatic(0));
    }

    #[test]
    fn concurrent_writes_preserve_distinct_fields() {
        let fixture = Fixture::new();
        let path = fixture.config();
        ensure_default_at(&path).unwrap();
        let a = path.clone();
        let b = path.clone();
        let first = std::thread::spawn(move || save_at(&a, Change::ManualFps(75.0)).unwrap());
        let second = std::thread::spawn(move || save_at(&b, Change::Renderer("modern")).unwrap());
        first.join().unwrap();
        second.join().unwrap();
        let config = load(&path);
        assert_eq!(config.target_fps, Some(75.0));
        assert_eq!(config.overlay_renderer, settings::OverlayRenderer::Modern);
    }
}
