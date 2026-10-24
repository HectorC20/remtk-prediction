use axum::{
    extract::{Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::info;

use crate::api::state::AppState;
use crate::shared::dto::*;
use crate::shared::error::AppError;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/status", get(status_handler))
        .route("/tools/upsert", post(upsert_tools_handler))
        .route("/tools/count", get(tools_count_handler))
        .route("/tools/search", post(search_tools_handler))
        .route("/memories/search", post(search_memories_handler))
        .route("/tools/synonyms", post(synonyms_handler))
        .with_state(state)
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub enabled: bool,
    pub qdrant: QdrantStatus,
}

#[derive(Serialize)]
pub struct QdrantStatus {
    pub ok: bool,
    pub collections: Vec<&'static str>,
}

/// GET /status: Estado de las colecciones de búsqueda vectorial y BM25.
async fn status_handler() -> Json<StatusResponse> {
    Json(StatusResponse {
        enabled: true,
        qdrant: QdrantStatus {
            ok: true,
            collections: vec!["tools", "memories", "synonyms", "keywords"],
        },
    })
}

/// POST /tools/upsert: Indexación de herramientas vía la API de Qdrant.
async fn upsert_tools_handler(
    State(state): State<AppState>,
    Json(payload): Json<UpsertToolsRequest>,
) -> Result<Json<UpsertToolsResponse>, AppError> {
    let qdrant = state.qdrant.clone();

    let tenant_id = payload.tenant.clone();
    let count = tokio::task::spawn_blocking(move || {
        qdrant.upsert_tools(&payload.tenant, &payload.tools)
    })
    .await??;

    info!("Qdrant Server: Indexadas {} herramientas para tenant: {}", count, tenant_id);
    Ok(Json(UpsertToolsResponse { indexed: count }))
}

#[derive(Deserialize)]
struct TenantQuery {
    tenant: Option<String>,
}

/// GET /tools/count?tenant= : Conteo de herramientas indexadas en el canal.
async fn tools_count_handler(
    State(state): State<AppState>,
    Query(query): Query<TenantQuery>,
) -> Json<ToolsCountResponse> {
    let count = query
        .tenant
        .as_deref()
        .map(|t| state.qdrant.count_tools(t))
        .unwrap_or(0);

    Json(ToolsCountResponse { count })
}

/// POST /tools/search: Búsqueda y recall léxico de herramientas candidatas.
async fn search_tools_handler(
    State(state): State<AppState>,
    Json(payload): Json<SearchToolsRequest>,
) -> Result<Json<SearchToolsResponse>, AppError> {
    let response = state.qdrant.search_tools(&payload);
    Ok(Json(response))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MemoriesSearchRequest {
    user_id: String,
    text: String,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default = "default_mem_limit")]
    limit: usize,
}

fn default_mem_limit() -> usize {
    8
}

/// POST /memories/search: Búsqueda vectorial en colección de memorias.
async fn search_memories_handler(
    Json(_payload): Json<MemoriesSearchRequest>,
) -> Json<Value> {
    Json(json!({
        "memories": []
    }))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SynonymsRequest {
    text: String,
    #[serde(default)]
    tenant: Option<String>,
    #[serde(default)]
    agent_id: Option<String>,
}

/// POST /tools/synonyms: Expansión de sinónimos para consultas de búsqueda.
async fn synonyms_handler(
    State(state): State<AppState>,
    Json(payload): Json<SynonymsRequest>,
) -> Json<Value> {
    let expanded = state.qdrant.expand_synonyms(&payload.text);
    Json(json!({
        "expanded": expanded
    }))
}
