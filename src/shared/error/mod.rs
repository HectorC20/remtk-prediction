use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Jerarquía unificada de errores del sistema remtk-prediction en Rust.
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Bad Request: {0}")]
    BadRequest(String),

    #[error("Not Found: {0}")]
    NotFound(String),

    #[error("Inference Error: {0}")]
    Inference(String),

    #[error("Model Loading Error: {0}")]
    ModelLoad(String),

    #[error("Tokenizer Error: {0}")]
    Tokenizer(String),

    #[error("Concurrency Task Error: {0}")]
    TaskJoin(#[from] tokio::task::JoinError),

    #[error("Internal Server Error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AppError::Inference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::ModelLoad(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::Tokenizer(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::TaskJoin(err) => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        // Formato canónico estándar de NestJS: { error: string, statusCode: number }
        let body = Json(json!({
            "error": message,
            "statusCode": status.as_u16(),
        }));

        (status, body).into_response()
    }
}
