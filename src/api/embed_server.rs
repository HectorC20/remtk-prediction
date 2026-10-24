use std::collections::HashMap;
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};

use crate::api::state::AppState;
use crate::shared::dto::*;
use crate::shared::error::AppError;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/models", get(models_handler))
        .route("/embed", post(embed_handler))
        .with_state(state)
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// GET /models: Metadatos del modelo activo.
async fn models_handler(
    State(state): State<AppState>,
) -> Json<HashMap<String, ModelInfoResponse>> {
    let mut map = HashMap::new();
    let size = state.config.model_size.clone();
    let info = state.embedding.get_model_info();
    map.insert(size, info);
    Json(map)
}

/// POST /embed: Generación de embeddings densos con e5-small (384 dims, normalizados L2).
async fn embed_handler(
    State(state): State<AppState>,
    Json(payload): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, AppError> {
    if payload.text.trim().is_empty() {
        return Err(AppError::BadRequest("El campo 'text' es requerido".into()));
    }

    let embedding_service = state.embedding.clone();
    let text = payload.text;
    let size = payload.size;

    let response = tokio::task::spawn_blocking(move || {
        embedding_service.embed(&text, &size)
    })
    .await??;

    Ok(Json(response))
}
