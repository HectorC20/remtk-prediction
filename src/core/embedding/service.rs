use std::sync::Arc;
use crate::core::inference::engine::{InferenceEngine, EMBED_DIM};
use crate::core::inference::tokenizer::DEFAULT_MAX_QUERY_LEN;
use crate::shared::dto::{EmbedResponse, ModelInfoResponse};
use crate::shared::error::AppError;
use crate::shared::tensor::l2_normalize;

/// Servicio de dominio para cálculo, pooling y normalización de embeddings densos.
pub struct EmbeddingService {
    engine: Arc<InferenceEngine>,
}

impl EmbeddingService {
    pub fn new(engine: Arc<InferenceEngine>) -> Self {
        Self { engine }
    }

    /// Genera el embedding denso de 384 dimensiones para un texto aplicando Mean Pooling y L2.
    pub fn embed(&self, text: &str, size: &str) -> Result<EmbedResponse, AppError> {
        let t_start = std::time::Instant::now();
        let pooled = self.engine.embed_vector(text)?;
        let total_elapsed = t_start.elapsed();

        tracing::info!(
            target: "profiler",
            "[profile:embed] text_len={} total={:.2}ms dim=384 model={}",
            text.len(),
            total_elapsed.as_secs_f64() * 1000.0,
            size
        );

        let model_path = self
            .engine
            .models_path()
            .join("multilingual-e5-small-onnx")
            .to_string_lossy()
            .to_string();

        Ok(EmbedResponse {
            embedding: pooled,
            model: size.to_string(),
            dim: EMBED_DIM,
            model_path,
        })
    }

    /// Obtiene información de metadatos del modelo activo.
    pub fn get_model_info(&self) -> ModelInfoResponse {
        let model_path = self
            .engine
            .models_path()
            .join("multilingual-e5-small-onnx")
            .to_string_lossy()
            .to_string();

        ModelInfoResponse {
            model: "multilingual-e5-small".to_string(),
            dim: EMBED_DIM,
            model_path,
        }
    }
}
