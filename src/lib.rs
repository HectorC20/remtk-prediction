//! remtk-prediction-server — Monolito Modular en Rust Nativo
//!
//! Inspirado directamente en la arquitectura de dominio de `key-remtk/src`:
//! - `core/`: Motores e inferencia matemática pura desacoplada (ONNX Runtime, MoE, Embeddings, Predicción).
//! - `support/`: Servicios de apoyo e infraestructura en memoria (Caché FP16 en DashMap, Spatial, Qdrant).
//! - `shared/`: Configuración, errores tipados, tensores zero-copy y DTOs de serialización Serde.
//! - `api/`: Capa de transporte y entrega HTTP concurrente sobre Tokio y Axum (Puertos 6776, 6777, 6775).

pub mod api;
pub mod core;
pub mod shared;
pub mod support;

// Re-exports ergonómicos de primer nivel
pub use api::state::AppState;
pub use core::embedding::EmbeddingService;
pub use core::inference::{InferenceEngine, MoeDecisionOutput, E5Tokenizer, EMBED_DIM};
pub use core::prediction::PredictionService;
pub use shared::config::AppConfig;
pub use shared::error::AppError;
pub use support::qdrant::QdrantService;
pub use support::registry::{CachedTool, GlobalToolCache, TenantCache, ToolRegistry};
pub use support::spatial::SpatialService;
