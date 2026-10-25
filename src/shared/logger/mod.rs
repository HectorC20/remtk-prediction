use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::EnvFilter;

/// Configuración de salida y persistencia para el sistema de observabilidad (`Logger`).
#[derive(Debug, Clone)]
pub struct LoggerConfig {
    /// Directorio donde se persisten los archivos de log rotativos diarios (default: `logs`).
    pub logs_dir: PathBuf,
    /// Filtro de niveles por módulo (default: `info,remtk_prediction_server=debug,profiler=info,prediction_trace=info`).
    pub env_filter: String,
    /// Habilita escritura dual en archivo rotativo diario (`logs/prediction-YYYY-MM-DD.log`).
    pub file_logging_enabled: bool,
    /// Formato JSON estructurado si `LOG_FORMAT=json`.
    pub json_format: bool,
}

impl LoggerConfig {
    pub fn from_env() -> Self {
        let logs_dir = env::var("LOGS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("logs"));

        let env_filter = env::var("RUST_LOG").unwrap_or_else(|_| {
            "info,remtk_prediction_server=debug,profiler=info,prediction_trace=info,http_trace=info"
                .to_string()
        });

        let file_logging_enabled = env::var("LOG_TO_FILE")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        let json_format = env::var("LOG_FORMAT")
            .map(|v| v.to_lowercase() == "json")
            .unwrap_or(false);

        Self {
            logs_dir,
            env_filter,
            file_logging_enabled,
            json_format,
        }
    }
}

/// Estado interno del archivo rotativo diario.
struct RotatingFileState {
    logs_dir: PathBuf,
    current_date: String,
    file: Option<File>,
}

impl RotatingFileState {
    fn new(logs_dir: PathBuf) -> Self {
        let _ = fs::create_dir_all(&logs_dir);
        let date = current_utc_date_string();
        let file = open_daily_log_file(&logs_dir, &date);
        Self {
            logs_dir,
            current_date: date,
            file,
        }
    }

    fn write_all(&mut self, buf: &[u8]) {
        let today = current_utc_date_string();
        if today != self.current_date {
            self.current_date = today.clone();
            self.file = open_daily_log_file(&self.logs_dir, &today);
        }
        if let Some(ref mut f) = self.file {
            let _ = f.write_all(buf);
        }
    }

    fn flush(&mut self) {
        if let Some(ref mut f) = self.file {
            let _ = f.flush();
        }
    }
}

fn open_daily_log_file(logs_dir: &Path, date_str: &str) -> Option<File> {
    let filename = format!("prediction-{}.log", date_str);
    let path = logs_dir.join(filename);
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()
}

/// Calcula la fecha UTC actual en formato `YYYY-MM-DD` sin dependencias externas usando el algoritmo civil de Howard Hinnant.
pub fn current_utc_date_string() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", year, m, d)
}

/// Escritor dual (`DualWriter`) que emite trazas simultáneamente en `stdout` y en el archivo rotativo diario.
#[derive(Clone)]
pub struct DualMakeWriter {
    file_state: Option<Arc<Mutex<RotatingFileState>>>,
}

impl DualMakeWriter {
    pub fn new(logs_dir: PathBuf, file_enabled: bool) -> Self {
        let file_state = if file_enabled {
            Some(Arc::new(Mutex::new(RotatingFileState::new(logs_dir))))
        } else {
            None
        };
        Self { file_state }
    }
}

pub struct DualWriter {
    stdout: io::Stdout,
    file_state: Option<Arc<Mutex<RotatingFileState>>>,
}

impl Write for DualWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.stdout.write(buf)?;
        if let Some(ref state_mutex) = self.file_state {
            if let Ok(mut guard) = state_mutex.lock() {
                guard.write_all(&buf[..written]);
            }
        }
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stdout.flush()?;
        if let Some(ref state_mutex) = self.file_state {
            if let Ok(mut guard) = state_mutex.lock() {
                guard.flush();
            }
        }
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for DualMakeWriter {
    type Writer = DualWriter;

    fn make_writer(&'a self) -> Self::Writer {
        DualWriter {
            stdout: io::stdout(),
            file_state: self.file_state.clone(),
        }
    }
}

/// Inicializa el suscriptor global de `tracing` con salida dual (consola + archivo rotativo diario).
pub fn init_logger() -> LoggerConfig {
    let cfg = LoggerConfig::from_env();
    let filter = EnvFilter::try_new(&cfg.env_filter)
        .unwrap_or_else(|_| EnvFilter::new("info,remtk_prediction_server=debug"));

    let writer = DualMakeWriter::new(cfg.logs_dir.clone(), cfg.file_logging_enabled);

    if cfg.json_format {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .json()
            .with_writer(writer)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(false)
            .with_writer(writer)
            .try_init();
    }

    cfg
}

/// Recorta de forma segura en bordes UTF-8 una cadena para previsualización en logs.
pub fn preview_text(text: &str, max_chars: usize) -> String {
    let clean = text.replace('\r', " ").replace('\n', " ↵ ");
    let trimmed = clean.trim();
    if trimmed.chars().count() <= max_chars {
        trimmed.to_string()
    } else {
        let head: String = trimmed.chars().take(max_chars).collect();
        format!("{}…(+{} chars)", head, trimmed.chars().count() - max_chars)
    }
}

/// Formatea una lista de herramientas puntuadas `(nombre, score)` para trazabilidad compacta.
pub fn format_scored_slice(items: &[(String, f32)], limit: usize) -> String {
    if items.is_empty() {
        return "[]".to_string();
    }
    let formatted: Vec<String> = items
        .iter()
        .take(limit)
        .map(|(name, score)| format!("{}({:.4})", name, score))
        .collect();
    if items.len() > limit {
        format!("[{}, …+{}]", formatted.join(", "), items.len() - limit)
    } else {
        format!("[{}]", formatted.join(", "))
    }
}
