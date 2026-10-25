use std::time::Instant;
use serde_json::json;
use crate::shared::dto::{SpatialPredictDto, VisualToolPayload};
use crate::shared::error::AppError;
use crate::shared::logger::preview_text;

/// Servicio de soporte para predicción sobre lienzo espacial interactivo.
pub struct SpatialService;

impl SpatialService {
    pub fn new() -> Self {
        Self
    }

    /// Evalúa la consulta del usuario con su contexto de cursor espacial y anáforas.
    pub fn predict(&self, payload: &SpatialPredictDto) -> Result<VisualToolPayload, AppError> {
        if payload.query.trim().is_empty() {
            return Err(AppError::BadRequest("El campo query es requerido".into()));
        }

        let start = Instant::now();
        let q_lower = payload.query.to_lowercase();

        let (tool, params, conf) = if q_lower.contains("zoom") || q_lower.contains("acercar") {
            ("spatial_zoom_in", json!({"factor": 1.25}), 0.95f32)
        } else if q_lower.contains("alejar") {
            ("spatial_zoom_out", json!({"factor": 0.80}), 0.94f32)
        } else if q_lower.contains("mover") || q_lower.contains("pan") {
            let cursor = payload.cursor.as_ref().map(|c| (c.x, c.y)).unwrap_or((0.5, 0.5));
            ("spatial_pan", json!({"to": {"x": cursor.0, "y": cursor.1}}), 0.92f32)
        } else if q_lower.contains("seleccionar") || q_lower.contains("click") {
            ("spatial_select", json!({"targetId": payload.selected_id}), 0.90f32)
        } else {
            ("spatial_inspect_canvas", json!({"query": payload.query}), 0.85f32)
        };

        let elapsed_ms = start.elapsed().as_millis() as u64;
        tracing::info!(
            target: "prediction_trace",
            "[trace:spatial] tool={} confidence={:.2} elapsed={}ms cursor={:?} selected_id={:?} query=\"{}\"",
            tool,
            conf,
            elapsed_ms,
            payload.cursor,
            payload.selected_id,
            preview_text(&payload.query, 90)
        );

        Ok(VisualToolPayload {
            tool: tool.to_string(),
            parameters: params,
            confidence: conf,
            execution_time_ms: elapsed_ms,
        })
    }
}
