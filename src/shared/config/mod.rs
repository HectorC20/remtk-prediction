use std::env;
use std::path::{Path, PathBuf};

/// Configuración tipada del monolito modular remtk-prediction.
#[derive(Debug, Clone)]
pub struct AppConfig {
    /// Puerto HTTP del servicio de predicción de herramientas y juicio MoE (default: 6776).
    pub port_predict: u16,
    /// Puerto HTTP del servicio de embeddings vectoriales (default: 6777).
    pub port_embed: u16,
    /// Puerto HTTP del servicio de recall BM25 y sincronización Qdrant (default: 6775).
    pub port_qdrant: u16,

    /// Ruta al directorio que contiene los modelos ONNX.
    pub models_path: PathBuf,
    /// Tamaño del modelo ONNX por defecto ("small", "large", "gte").
    pub model_size: String,

    /// Hilos intra-op de ONNX Runtime para aceleración SIMD AVX2.
    pub intra_op_threads: usize,
    /// Hilos inter-op de ONNX Runtime.
    pub inter_op_threads: usize,

    /// Umbral mínimo de score para considerar una herramienta como ganadora (default: 0.80).
    pub adaptive_min_score: f32,
    /// Umbral de brecha (gap) entre Top-1 y candidatos sucesivos (default: 0.03).
    pub adaptive_gap_threshold: f32,
    /// Umbral de probabilidad de abstención para retorno nulo (default: 0.52).
    pub abstencion_threshold: f32,

    /// Refuerzo por coincidencia de palabras clave internas (default: 0.05).
    pub keyword_boost: f32,
    /// Refuerzo por afinidad en el nombre de la herramienta (default: 0.08).
    pub name_affinity_boost: f32,
    /// Penalización por familia / namespace ajeno (default: 0.15).
    pub family_gate_penalty: f32,

    /// Límite máximo de candidatas recuperadas en recall (default: 20).
    pub recall_limit: usize,
    /// Tope máximo de herramientas devueltas en la respuesta final (default: 8).
    pub max_output_tools: usize,
    /// Tope máximo de categorías activas evaluadas en el juicio (default: 6).
    pub max_categories: usize,
}

impl AppConfig {
    /// Carga la configuración a partir de variables de entorno con fallbacks robustos.
    pub fn from_env() -> Self {
        let port_predict = env_u16("PORT_PREDICT", 6776);
        let port_embed = env_u16("PORT_EMBED", 6777);
        let port_qdrant = env_u16("PORT_QDRANT", 6775);

        let models_path_raw = env::var("MODELS_PATH")
            .or_else(|_| env::var("ONNX_MODELS_PATH"))
            .unwrap_or_else(|_| {
                if Path::new("models").exists() {
                    "models".to_string()
                } else if Path::new("../models").exists() {
                    "../models".to_string()
                } else {
                    "models".to_string()
                }
            });
        let models_path = PathBuf::from(models_path_raw);

        let model_size = env::var("ONNX_MODEL_SIZE").unwrap_or_else(|_| "small".to_string());

        let intra_op_threads = env_usize("ONNX_INTRA_OP_THREADS", 4);
        let inter_op_threads = env_usize("ONNX_INTER_OP_THREADS", 1);

        let adaptive_min_score = env_f32("ONNX_ADAPTIVE_MIN_SCORE", 0.84);
        let adaptive_gap_threshold = env_f32("ONNX_ADAPTIVE_GAP_THRESHOLD", 0.03);
        let abstencion_threshold = env_f32("ABSTENCION_THRESHOLD", 0.52);

        let keyword_boost = env_f32("KEYWORD_BOOST", 0.05);
        let name_affinity_boost = env_f32("NAME_AFFINITY_BOOST", 0.08);
        let family_gate_penalty = env_f32("FAMILY_GATE_PENALTY", 0.15);

        let recall_limit = env_usize("RECALL_LIMIT", 20);
        let max_output_tools = env_usize("MAX_OUTPUT_TOOLS", 8);
        let max_categories = env_usize("MAX_CATEGORIES", 6);

        Self {
            port_predict,
            port_embed,
            port_qdrant,
            models_path,
            model_size,
            intra_op_threads,
            inter_op_threads,
            adaptive_min_score,
            adaptive_gap_threshold,
            abstencion_threshold,
            keyword_boost,
            name_affinity_boost,
            family_gate_penalty,
            recall_limit,
            max_output_tools,
            max_categories,
        }
    }
}

fn env_u16(key: &str, default: u16) -> u16 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    env::var(key)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}
