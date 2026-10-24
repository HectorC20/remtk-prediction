use std::sync::Arc;
use crate::core::embedding::EmbeddingService;
use crate::core::prediction::PredictionService;
use crate::shared::config::AppConfig;
use crate::support::qdrant::QdrantService;
use crate::support::registry::TenantCache;
use crate::support::spatial::SpatialService;

/// Estado global de inyección de dependencias para los routers de la capa API.
#[derive(Clone)]
pub struct AppState {
    pub prediction: Arc<PredictionService>,
    pub embedding: Arc<EmbeddingService>,
    pub spatial: Arc<SpatialService>,
    pub qdrant: Arc<QdrantService>,
    pub cache: Arc<TenantCache>,
    pub config: Arc<AppConfig>,
}

impl AppState {
    pub fn new(
        prediction: Arc<PredictionService>,
        embedding: Arc<EmbeddingService>,
        spatial: Arc<SpatialService>,
        qdrant: Arc<QdrantService>,
        cache: Arc<TenantCache>,
        config: Arc<AppConfig>,
    ) -> Self {
        Self {
            prediction,
            embedding,
            spatial,
            qdrant,
            cache,
            config,
        }
    }
}
