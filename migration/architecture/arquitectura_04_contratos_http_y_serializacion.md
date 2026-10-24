# Arquitectura 04: Contratos HTTP, Serialización Serde y Manejo de Errores

## 1. Paridad de Contratos y Esquemas JSON

Para garantizar la compatibilidad 1:1 con los clientes existentes del ecosistema `remtk`, los structs en Rust deben mapear con precisión quirúrgica los nombres de campos en `camelCase`, los valores por defecto y las estructuras polimórficas de NestJS.

---

## 2. Definición de DTOs con Serde

```rust
// src/contracts/predict.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictRequest {
    pub tenant: String,
    pub session_id: String,
    pub text: String,
    pub source: SourceKind,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub key_remtk: Option<String>,
    #[serde(default)]
    pub prior_plan: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub intent_context: Option<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceKind {
    Human,
    Agent,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictResponse {
    pub tools: Vec<String>,
    pub complexity: String,
    pub model_size: String,
    pub ranked_scores: Vec<RankedToolScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_shift: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estado_nli: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankedToolScore {
    pub name: String,
    pub score: f32,
    pub expert_affinity: String,
}
```

```rust
// src/contracts/embed.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequest {
    pub text: String,
    #[serde(default = "default_model_size")]
    pub size: String,
}

fn default_model_size() -> String {
    "small".to_string()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub model: String,
    pub dim: usize,
    pub model_path: String,
}
```

```rust
// src/contracts/spatial.rs
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CursorCoords {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpatialPredictDto {
    pub query: String,
    #[serde(default)]
    pub cursor: Option<CursorCoords>,
    #[serde(default)]
    pub selected_id: Option<String>,
    #[serde(default)]
    pub view_bounds: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualToolPayload {
    pub tool: String,
    pub parameters: Value,
    pub confidence: f32,
    pub execution_time_ms: u64,
}
```

```rust
// src/contracts/qdrant.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
pub struct UpsertToolsRequest {
    pub tenant: String,
    pub tools: Vec<ToolDefinitionDto>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinitionDto {
    pub name: String,
    pub group: String,
    pub category: String,
    pub description: String,
    pub intent_summary: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpsertToolsResponse {
    pub indexed: usize,
}
```

---

## 3. Manejo Unificado de Errores con `thiserror` y `IntoResponse`

Para replicar con precisión los códigos de estado HTTP de NestJS (`BadRequestException` $\to$ 400, `NotFoundException` $\to$ 404, `InternalServerErrorException` $\to$ 500) y su envoltorio JSON estándar:

```rust
// src/errors.rs
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Bad Request: {0}")]
    BadRequest(String),

    #[error("Not Found: {0}")]
    NotFound(String),

    #[error("Inference Error: {0}")]
    Inference(String),

    #[error("Internal Server Error: {0}")]
    Internal(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("Tokio Task Join Error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::Inference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::Internal(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            AppError::TaskJoin(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
        };

        let body = Json(json!({
            "error": error_message,
            "statusCode": status.as_u16(),
        }));

        (status, body).into_response()
    }
}
```

---

## 4. Implementación del Manejador `/predict` con Axum

```rust
// src/routes/predict_handler.rs
use axum::{extract::State, Json};
use std::time::Instant;
use crate::{
    contracts::predict::{PredictRequest, PredictResponse, RankedToolScore},
    errors::AppError,
    state::AppState,
};

pub async fn predict_handler(
    State(state): State<AppState>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, AppError> {
    if payload.text.trim().is_empty() {
        return Err(AppError::BadRequest("El campo 'text' es obligatorio y no puede estar vacío".into()));
    }

    let start_time = Instant::now();
    let engine = state.engine.clone();
    let tool_cache = state.tool_cache.clone();

    // Despacho no bloqueante a CPU worker pool
    let response = tokio::task::spawn_blocking(move || -> Result<PredictResponse, AppError> {
        // 1. Obtener registro de herramientas del tenant
        let registry = tool_cache
            .get(&payload.tenant)
            .ok_or_else(|| AppError::NotFound(format!("Tenant '{}' no registrado en /tools", payload.tenant)))?;

        // 2. Extraer tokens de la consulta con e5-small
        let query_tokens = engine
            .encode_query(&payload.text, &payload.keywords)
            .map_err(|e| AppError::Inference(e.to_string()))?;

        // 3. Ejecutar inferencia MoE contra el catálogo
        let moe_output = engine
            .evaluate_catalog(&query_tokens, &registry)
            .map_err(|e| AppError::Inference(e.to_string()))?;

        // 4. Mapear resultados a respuesta JSON
        let ranked: Vec<RankedToolScore> = moe_output
            .ranked_scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| RankedToolScore {
                name: registry.tools[idx].metadata.name.clone(),
                score,
                expert_affinity: "expert_action".to_string(),
            })
            .collect();

        let winning_tools = if moe_output.debe_abstenerse {
            vec![]
        } else {
            ranked.iter().take(3).map(|t| t.name.clone()).collect()
        };

        Ok(PredictResponse {
            tools: winning_tools,
            complexity: "dense_moe".to_string(),
            model_size: "int8_unified".to_string(),
            ranked_scores: ranked,
            topic_shift: Some(moe_output.topic_shift_score),
            estado_nli: Some(moe_output.estado_dominante.to_string()),
        })
    }).await??;

    state.metrics.record_predict_latency(start_time.elapsed());
    Ok(Json(response))
}
```
