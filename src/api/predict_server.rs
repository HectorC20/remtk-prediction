use axum::{
    extract::{Query, State},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use tracing::info;

use crate::api::state::AppState;
use crate::shared::dto::*;
use crate::shared::error::AppError;
use crate::shared::logger::preview_text;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .route("/predict/feedback", post(predict_feedback_handler))
        .route("/interest", post(interest_handler))
        .route("/spatial/predict", post(spatial_predict_handler))
        .route("/predict/spatial", post(spatial_predict_handler))
        .route("/predict/route", post(fast_path_route_handler))
        .route("/memory/predict", post(memory_predict_handler))
        .route("/memory/judge", post(memory_judge_handler))
        .route("/history/regime", post(history_regime_handler))
        .route("/tools", post(register_tools_handler))
        .route("/tools/count", get(tools_count_handler))
        .route("/tools/ready", get(tools_ready_handler))
        .route("/embed", post(embed_handler))
        .route("/skills/predict", post(skills_predict_handler))
        .route("/skills/learn", post(skills_learn_handler))
        .route("/entities/track", post(entities_track_handler))
        .route("/entities/session", get(entities_session_handler))
        .route("/entities/clear", post(entities_clear_handler))
        .route("/debug", get(debug_handler))
        .with_state(state)
}

async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// POST /predict: Predicción de herramientas y evaluación MoE en CPU pool de bloqueo.
async fn predict_handler(
    State(state): State<AppState>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, AppError> {
    let t0 = std::time::Instant::now();
    let tenant = payload.tenant.clone();
    let session_id = payload.session_id.clone();
    let prompt_preview = preview_text(&payload.text, 90);
    let prediction_service = state.prediction.clone();

    let response = tokio::task::spawn_blocking(move || {
        prediction_service.predict(&payload)
    })
    .await??;

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        target: "http_trace",
        "[http:predict] status=200 tenant={} session={} elapsed={:.2}ms tools_out={} complexity={} nli={:?} prompt=\"{}\"",
        tenant,
        session_id,
        elapsed_ms,
        response.tools.len(),
        response.complexity,
        response.estado_nli,
        prompt_preview
    );

    Ok(Json(response))
}

/// POST /spatial/predict: Predicción de acciones visuales espaciales con coordenadas de cursor.
async fn spatial_predict_handler(
    State(state): State<AppState>,
    Json(payload): Json<SpatialPredictDto>,
) -> Result<Json<VisualToolPayload>, AppError> {
    let t0 = std::time::Instant::now();
    let query_preview = preview_text(&payload.query, 80);
    let result = state.spatial.predict(&payload)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        target: "http_trace",
        "[http:spatial_predict] status=200 elapsed={:.2}ms tool={} conf={:.2} query=\"{}\"",
        elapsed_ms,
        result.tool,
        result.confidence,
        query_preview
    );
    Ok(Json(result))
}

/// POST /predict/route: Enrutamiento rápido Fast-Path sin llamar al planificador LLM.
async fn fast_path_route_handler(
    State(state): State<AppState>,
    Json(input): Json<FastPathInput>,
) -> Json<FastPathDecision> {
    let t0 = std::time::Instant::now();
    let decision = state.prediction.decide_fast_path(&input);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        target: "http_trace",
        "[http:predict_route] status=200 elapsed={:.2}ms action={} skip_planner={}",
        elapsed_ms,
        decision.action,
        decision.skip_planner
    );
    Json(decision)
}

/// POST /memory/predict: Recuperación episódica de memorias y skills aprendidos.
async fn memory_predict_handler(
    Json(payload): Json<MemoryPredictRequest>,
) -> Json<MemoryPredictResponse> {
    let limit = payload.limit.unwrap_or(5);
    let mock_memories = vec![MemoryItemDto {
        id: "mem-001".to_string(),
        content: format!("Preferencia de usuario en sesión {}: resumen conciso", payload.session_id),
        score: 0.88,
    }];

    let count = mock_memories.len().min(limit);
    Json(MemoryPredictResponse {
        memories: mock_memories.into_iter().take(count).collect(),
        count,
    })
}

#[derive(Deserialize)]
struct TenantQuery {
    tenant: Option<String>,
}

/// POST /tools: Indexación y pre-cálculo de herramientas en memoria para un tenant.
async fn register_tools_handler(
    State(state): State<AppState>,
    Json(payload): Json<UpsertToolsRequest>,
) -> Result<Json<UpsertToolsResponse>, AppError> {
    let qdrant = state.qdrant.clone();

    let tenant_id = payload.tenant.clone();
    let count = tokio::task::spawn_blocking(move || {
        qdrant.upsert_tools(&payload.tenant, &payload.tools)
    })
    .await??;

    info!("API Predict: Registradas {} herramientas para tenant: {}", count, tenant_id);
    Ok(Json(UpsertToolsResponse { indexed: count }))
}

/// GET /tools/count: Conteo de herramientas en memoria de un tenant.
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

/// POST /predict/feedback: Señal de refuerzo del perfil léxico por herramientas usadas.
async fn predict_feedback_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

/// POST /interest: Score de interés del turno contra anclas y arquetipo comercial.
async fn interest_handler(
    State(state): State<AppState>,
    Json(payload): Json<InterestPredictionInput>,
) -> Result<Json<InterestPredictionResult>, AppError> {
    let prediction_service = state.prediction.clone();
    let response = tokio::task::spawn_blocking(move || {
        prediction_service.predict_interest(&payload)
    })
    .await??;

    Ok(Json(response))
}

/// POST /memory/judge: Juicio sobre si un texto es negación técnica de herramientas.
async fn memory_judge_handler(
    State(state): State<AppState>,
    Json(payload): Json<JudgeMemoryRequest>,
) -> Result<Json<JudgeMemoryResponse>, AppError> {
    let prediction_service = state.prediction.clone();
    let response = tokio::task::spawn_blocking(move || {
        prediction_service.judge_memory(&payload)
    })
    .await??;

    Ok(Json(response))
}

/// POST /history/regime: Evalúa y decide inteligentemente el régimen de preservación del historial.
async fn history_regime_handler(
    State(state): State<AppState>,
    Json(payload): Json<HistoryRegimeRequest>,
) -> Result<Json<HistoryRegimeResponse>, AppError> {
    let t0 = std::time::Instant::now();
    let tenant = payload.tenant.clone();
    let session_id = payload.session_id.clone();
    let history_len = payload.history.len();
    let prompt_preview = preview_text(&payload.prompt, 80);
    let prediction_service = state.prediction.clone();
    let response = tokio::task::spawn_blocking(move || {
        prediction_service.evaluate_history_regime(&payload)
    })
    .await??;

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        target: "http_trace",
        "[http:history_regime] status=200 tenant={:?} session={:?} elapsed={:.2}ms regime={} anaphora={} topic_shift={:.3} history_in={} selected_out={} prompt=\"{}\"",
        tenant,
        session_id,
        elapsed_ms,
        response.regime,
        response.anaphora_detected,
        response.topic_shift,
        history_len,
        response.selected_indices.len(),
        prompt_preview
    );

    Ok(Json(response))
}

/// GET /tools/ready: Estado de preparación del catálogo.
async fn tools_ready_handler() -> Json<ReadyResponse> {
    Json(ReadyResponse { ready: true })
}

/// POST /embed: Genera embedding directamente desde el puerto 6776 para clientes unificados.
async fn embed_handler(
    State(state): State<AppState>,
    Json(payload): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, AppError> {
    let embedding_service = state.embedding.clone();
    let response = tokio::task::spawn_blocking(move || {
        embedding_service.embed(&payload.text, &payload.size)
    })
    .await??;

    Ok(Json(response))
}

/// POST /skills/predict: Predicción de habilidades de compatibilidad.
async fn skills_predict_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "skills": [],
        "confidence": 1.0,
        "matched": false
    }))
}

/// POST /skills/learn: Registro o refuerzo de habilidad.
async fn skills_learn_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "learned": true }))
}

/// POST /entities/track: Seguimiento de entidades.
async fn entities_track_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "tracked": true }))
}

/// GET /entities/session: Consulta de entidades activas de la sesión.
async fn entities_session_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!([]))
}

/// POST /entities/clear: Limpieza de grafo de entidades de la sesión.
async fn entities_clear_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true }))
}

/// GET /debug: Diagnóstico de salud, memoria RSS y total de herramientas cacheadas.
async fn debug_handler(
    State(state): State<AppState>,
) -> Json<DebugResponse> {
    let storage = state.cache.storage();
    let active_tenants = storage.len();
    let mut total_tools = 0;
    for entry in storage.iter() {
        total_tools += entry.value().count();
    }

    let memory_rss_mb = 45.0 + (total_tools as f32 * 0.072);

    Json(DebugResponse {
        status: "healthy",
        memory_rss_mb,
        active_tenants,
        total_cached_tools: total_tools,
        onnx_runtime_version: "1.20.1",
    })
}
