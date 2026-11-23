use std::sync::Arc;
use std::time::Instant;
use axum::{
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::Json,
    routing::{get, post},
    Router,
};
use candle_core::{Device, Tensor};
use serde_json::json;
use crate::e5::E5Embedder;
use crate::model::RemtkModel;
use crate::registry::TenantRegistry;
use crate::types::*;

pub struct AppState {
    pub model: Arc<RemtkModel>,
    pub embedder: Arc<E5Embedder>,
    pub registry: Arc<TenantRegistry>,
    pub dev: Device,
}

#[derive(serde::Deserialize)]
pub struct ScopeQuery {
    pub tenant: Option<String>,
    #[serde(rename = "agentId")]
    pub agent_id: Option<String>,
}

fn resolve_scope(headers: &HeaderMap, query: &ScopeQuery, body_tenant: Option<&str>, body_agent: Option<&str>) -> String {
    let header_key = headers
        .get("key-remtk")
        .or_else(|| headers.get("x-remtk-key"))
        .and_then(|h| h.to_str().ok());

    let tenant = body_tenant
        .or_else(|| query.tenant.as_deref())
        .or(header_key);

    let agent_id = body_agent
        .or_else(|| query.agent_id.as_deref());

    TenantRegistry::extract_scope_key(tenant, agent_id, header_key)
}

pub async fn health_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ScopeQuery>,
) -> Json<HealthResponse> {
    let scope_key = resolve_scope(&headers, &query, None, None);
    let count = state.registry.count_tools(&scope_key);
    let is_isolated = state.registry.get_tenant(&scope_key).is_some();

    Json(HealthResponse {
        status: "ok".to_string(),
        model: "remtk-juicio-jev-moe-3.82b-candle-rust".to_string(),
        tools_count: count,
        global_catalog_size: state.registry.global_tool_ids.len(),
        tenant_isolated: is_isolated,
        runtime: "candle-rust-native".to_string(),
    })
}

pub async fn tools_count_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ScopeQuery>,
) -> Json<ToolsCountResponse> {
    let scope_key = resolve_scope(&headers, &query, None, None);
    let count = state.registry.count_tools(&scope_key);
    Json(ToolsCountResponse { count })
}

pub async fn tools_ready_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ScopeQuery>,
) -> Json<ToolsReadyResponse> {
    let scope_key = resolve_scope(&headers, &query, None, None);
    let count = state.registry.count_tools(&scope_key);
    let ready = state.registry.get_tenant(&scope_key).map(|_| true).unwrap_or(true);
    Json(ToolsReadyResponse { ready, count })
}

pub async fn register_tools_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ScopeQuery>,
    Json(payload): Json<RegisterToolsPayload>,
) -> Result<Json<RegisterToolsResponse>, (StatusCode, String)> {
    let scope_key = resolve_scope(&headers, &query, payload.tenant.as_deref(), payload.agent_id.as_deref());
    let count = state.registry.register(&scope_key, payload.tools, &state.dev)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to register tools: {}", e)))?;

    Ok(Json(RegisterToolsResponse { indexed: count }))
}

fn select_candidates(
    search_pooled: &Tensor,
    pool_pooled: &Tensor,
    pool_ids: &[String],
    k: usize,
    forced: &[String],
) -> Result<(Vec<String>, Vec<u32>), candle_core::Error> {
    let total = pool_ids.len();
    if total == 0 {
        return Ok((vec![], vec![]));
    }
    let k_actual = k.min(total);

    let sims = search_pooled.matmul(&pool_pooled.t()?)?.squeeze(0)?;
    let sims_vec = sims.to_vec1::<f32>()?;

    let mut indexed: Vec<(usize, f32)> = sims_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut top_indices: Vec<usize> = indexed.iter().take(k_actual).map(|(idx, _)| *idx).collect();
    let mut cands: Vec<String> = top_indices.iter().map(|&idx| pool_ids[idx].clone()).collect();

    // Handle forced candidates
    for fc in forced.iter().rev() {
        if let Some(pos) = pool_ids.iter().position(|id| id == fc) {
            if !cands.contains(fc) {
                if !cands.is_empty() {
                    cands.pop();
                    top_indices.pop();
                }
                cands.insert(0, fc.clone());
                top_indices.insert(0, pos);
            }
        }
    }

    let u32_indices = top_indices.into_iter().map(|i| i as u32).collect();
    Ok((cands, u32_indices))
}

pub async fn predict_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(query): Query<ScopeQuery>,
    Json(payload): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, String)> {
    let t0 = Instant::now();
    let query_text = payload.text.trim();
    if query_text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "The field 'text' is required".to_string()));
    }

    let scope_key = resolve_scope(&headers, &query, payload.tenant.as_deref(), payload.agent_id.as_deref());
    let tenant_cat = state.registry.get_tenant(&scope_key);

    let (pool_ids, pool_tensors, pool_pooled, pool_dict) = if let Some(ref cat) = tenant_cat {
        if !cat.tool_ids.is_empty() {
            (
                cat.tool_ids.clone(),
                cat.tool_tensors.clone(),
                cat.tool_pooled.clone(),
                Some(cat.tools_dict.clone()),
            )
        } else {
            (
                state.registry.global_tool_ids.clone(),
                state.registry.global_tensors.clone(),
                state.registry.global_pooled.clone(),
                None,
            )
        }
    } else {
        (
            state.registry.global_tool_ids.clone(),
            state.registry.global_tensors.clone(),
            state.registry.global_pooled.clone(),
            None,
        )
    };

    // 1. Encode query
    let (q_tokens, q_pooled) = state.embedder.encode_single(query_text, "query: ", 24)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to encode query: {}", e)))?;

    // Assistant offer
    let mut assistant_offer = String::new();
    for msg in payload.history.iter().rev() {
        if msg.role.as_deref() == Some("assistant") {
            if let Some(ref content) = msg.content {
                assistant_offer = content.trim().to_string();
                break;
            }
        }
    }
    if assistant_offer.is_empty() {
        if let Some(ref ctx) = payload.context {
            assistant_offer = ctx.trim().to_string();
        }
    }

    let search_pooled = if !assistant_offer.is_empty() {
        if let Ok((_, a_pooled)) = state.embedder.encode_single(&assistant_offer, "passage: ", 24) {
            let q_part = q_pooled.mul(&Tensor::new(&[0.40f32], &state.dev).unwrap())
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?;
            let a_part = a_pooled.mul(&Tensor::new(&[0.60f32], &state.dev).unwrap())
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?;
            let blended = q_part.add(&a_part)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?;
            let eps_s = Tensor::new(&[1e-6f32], &state.dev).unwrap();
            let b_norm = blended.sqr()
                .and_then(|x| x.sum_keepdim(1))
                .and_then(|x| x.sqrt())
                .and_then(|x| x.broadcast_maximum(&eps_s))
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?;
            blended.broadcast_div(&b_norm)
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?
        } else {
            q_pooled.clone()
        }
    } else {
        q_pooled.clone()
    };

    let (cands, cand_indices) = select_candidates(
        &search_pooled,
        &pool_pooled,
        &pool_ids,
        32,
        &payload.keywords,
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed candidate selection: {}", e)))?;

    let k_actual = cand_indices.len();
    if k_actual == 0 {
        return Err((StatusCode::BAD_REQUEST, "No candidate tools available in catalog".to_string()));
    }

    let idx_tensor = Tensor::from_slice(&cand_indices, (k_actual,), &state.dev)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?;
    let t_tokens = pool_tensors.index_select(&idx_tensor, 0)
        .and_then(|t| t.narrow(1, 0, 32))
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?; // [1, K, 32, 384]
    let t_pooled = pool_pooled.index_select(&idx_tensor, 0)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Tensor error: {}", e)))?; // [1, K, 384]

    // 5. Inferencia en RemtkModel
    let out = state.model.forward(
        &q_tokens,
        &q_pooled,
        &t_tokens,
        &t_pooled,
    ).map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Model inference error: {}", e)))?;

    let lat_ms = (t0.elapsed().as_secs_f64() * 1000.0) as f32;

    let gated_scores = out.gated_scores;
    let master_gate = out.master_gate;
    let p_sink = out.p_sink;
    let p_noul_cov = out.p_noul_coverage;
    let p_noul_intent = out.p_noul_intent;
    let red_prob = out.redundancy_prob;

    let mut ranked: Vec<(usize, f32)> = gated_scores.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_cands: Vec<String> = ranked.iter().take(3).map(|(i, _)| cands[*i].clone()).collect();
    let top_scores: Vec<f32> = ranked.iter().take(3).map(|(_, s)| *s).collect();

    let top_score_0 = top_scores.get(0).copied().unwrap_or(0.0);
    let is_abstain = (master_gate < 0.20) || (p_sink > 0.70) || (p_noul_cov < 0.15) || (top_score_0 < 0.25);
    let decision = if is_abstain { "ABSTAIN" } else { "EXECUTE" };

    let mut predicted_tools: Vec<PredictedToolOut> = Vec::new();
    let mut ranked_scores_out: Vec<f32> = Vec::new();
    let mut calibrated_scores_out: Vec<f32> = Vec::new();

    if !is_abstain {
        for (idx, tid) in top_cands.iter().enumerate() {
            if payload.exclude.contains(tid) {
                continue;
            }
            let score = top_scores[idx];
            if score >= 0.20 {
                let tool_meta = if let Some(ref d) = pool_dict {
                    d.get(tid).cloned()
                } else {
                    None
                };

                let global_meta = state.registry.global_catalog.get(tid);

                let name = tool_meta.as_ref().map(|t| t.name.clone())
                    .or_else(|| global_meta.and_then(|g| g["name"].as_str().map(String::from)))
                    .unwrap_or_else(|| tid.clone());

                let group = tool_meta.as_ref().and_then(|t| t.group.clone())
                    .or_else(|| global_meta.and_then(|g| g["group"].as_str().or_else(|| g["domain_name"].as_str()).map(String::from)))
                    .unwrap_or_else(|| "general".to_string());

                let category = tool_meta.as_ref().and_then(|t| t.category.clone())
                    .or_else(|| global_meta.and_then(|g| g["category"].as_str().or_else(|| g["affordance"].as_str()).map(String::from)))
                    .unwrap_or_else(|| "QUERY".to_string());

                let description = tool_meta.as_ref().and_then(|t| t.description.clone())
                    .or_else(|| global_meta.and_then(|g| g["desc_es"].as_str().or_else(|| g["description"].as_str()).map(String::from)))
                    .unwrap_or_default();

                let tags = tool_meta.as_ref().map(|t| t.tags.clone())
                    .or_else(|| global_meta.and_then(|g| g["tags"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())))
                    .unwrap_or_default();

                let intent_summary = tool_meta.as_ref().and_then(|t| t.intent_summary.clone())
                    .or_else(|| global_meta.and_then(|g| g["intentSummary"].as_str().or_else(|| g["desc_es"].as_str()).map(String::from)))
                    .unwrap_or_default();

                let input_schema = tool_meta.as_ref().and_then(|t| t.input_schema.clone())
                    .or_else(|| global_meta.and_then(|g| g.get("inputSchema").cloned()))
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}));

                let prerequisites = tool_meta.as_ref().map(|t| t.prerequisites.clone())
                    .or_else(|| global_meta.and_then(|g| g["prerequisites"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())))
                    .unwrap_or_default();

                let conflicts = tool_meta.as_ref().map(|t| t.conflicts.clone())
                    .or_else(|| global_meta.and_then(|g| g["conflicts"].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())))
                    .unwrap_or_default();

                predicted_tools.push(PredictedToolOut {
                    id: tid.clone(),
                    name,
                    group,
                    category,
                    description,
                    tags,
                    intent_summary,
                    input_schema,
                    prerequisites,
                    conflicts,
                });
                ranked_scores_out.push((score * 10000.0).round() / 10000.0);
                calibrated_scores_out.push(((0.30f32).max(score * master_gate) * 10000.0).round() / 10000.0);
            }
        }
    }

    let tool_names: Vec<String> = predicted_tools.iter().map(|t| t.name.clone()).collect();
    let graph = GraphDto {
        nodes: tool_names.clone(),
        edges: vec![],
        execution_order: tool_names.clone(),
    };

    let is_file_tool = tool_names.iter().any(|t| {
        let l = t.to_lowercase();
        l.contains("file") || l.contains("workspace")
    });

    let file_command_mode = FileCommandModeDto {
        enabled: is_file_tool,
        confidence: if is_file_tool { master_gate } else { 0.0 },
        reason: if is_file_tool { "predicted_tool".to_string() } else { "none".to_string() },
    };

    let debug = PredictDebugDto {
        decision: decision.to_string(),
        top_tool: top_cands.get(0).cloned().unwrap_or_else(|| "none".to_string()),
        top_3_tools: top_cands,
        master_gate: (master_gate * 10000.0).round() / 10000.0,
        p_sink: (p_sink * 10000.0).round() / 10000.0,
        p_noul_intent: (p_noul_intent * 10000.0).round() / 10000.0,
        p_noul_coverage: (p_noul_cov * 10000.0).round() / 10000.0,
        redundancy_prob: (red_prob * 10000.0).round() / 10000.0,
        is_abstain,
        latency_ms: (lat_ms * 10.0).round() / 10.0,
    };

    let complexity = if predicted_tools.len() <= 1 { "simple" } else { "moderate" }.to_string();

    println!(
        "[/predict] 200 OK ({:.1}ms) | decision='{}' | tools={:?} | prompt='{}'",
        lat_ms,
        decision,
        tool_names,
        if query_text.len() > 45 { &query_text[..45] } else { query_text }
    );

    Ok(Json(PredictResponse {
        tools: predicted_tools,
        complexity,
        model_size: "remtk-juicio-jev-moe-3.82b-candle-rust".to_string(),
        ranked_scores: ranked_scores_out,
        calibrated_scores: calibrated_scores_out,
        graph,
        file_command_mode,
        debug,
    }))
}

// Auxiliary stub handlers
pub async fn predict_feedback_handler() -> Json<serde_json::Value> {
    Json(json!({"ok": true, "events": 1}))
}

pub async fn memory_predict_handler() -> Json<serde_json::Value> {
    Json(json!({
        "memories": [],
        "topicShift": false,
        "topicScore": 0.0,
        "modelSize": "3.82b-candle-rust",
        "rankedScores": []
    }))
}

pub async fn memory_judge_handler() -> Json<serde_json::Value> {
    Json(json!({"isNoise": false, "noiseScore": 0.0}))
}

pub async fn history_regime_handler(Json(payload): Json<serde_json::Value>) -> Json<serde_json::Value> {
    let history = payload.get("history").and_then(|h| h.as_array());
    let n = history.map(|h| h.len()).unwrap_or(0);
    let selected_indices: Vec<usize> = (0..n).collect();
    Json(json!({
        "regime": if n <= 4 { "PRESERVE_ALL" } else { "PRESERVE_RECENT" },
        "topicShift": 0.0,
        "anaphoraDetected": false,
        "dependencyScore": 0.5,
        "selectedIndices": selected_indices,
        "reason": "candle_3_82b_history_preservation"
    }))
}

pub async fn skills_predict_handler() -> Json<serde_json::Value> {
    Json(json!({"skills": []}))
}

pub async fn skills_learn_handler() -> Json<serde_json::Value> {
    Json(json!({"learned": true}))
}

pub async fn entities_track_handler() -> Json<serde_json::Value> {
    Json(json!({"ok": true}))
}

pub async fn spatial_predict_handler() -> Json<serde_json::Value> {
    Json(json!({"tool": "none", "confidence": 0.0, "x": 0.0, "y": 0.0}))
}

pub async fn fast_path_route_handler() -> Json<serde_json::Value> {
    Json(json!({"action": "plan", "skip_planner": false}))
}

pub async fn interest_handler() -> Json<serde_json::Value> {
    Json(json!({"hasInterest": false, "score": 0.0, "matches": []}))
}

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .route("/predict/feedback", post(predict_feedback_handler))
        .route("/tools/count", get(tools_count_handler))
        .route("/tools/ready", get(tools_ready_handler))
        .route("/tools", post(register_tools_handler))
        .route("/memory/predict", post(memory_predict_handler))
        .route("/memory/judge", post(memory_judge_handler))
        .route("/history/regime", post(history_regime_handler))
        .route("/skills/predict", post(skills_predict_handler))
        .route("/skills/learn", post(skills_learn_handler))
        .route("/entities/track", post(entities_track_handler))
        .route("/spatial/predict", post(spatial_predict_handler))
        .route("/predict/spatial", post(spatial_predict_handler))
        .route("/predict/route", post(fast_path_route_handler))
        .route("/interest", post(interest_handler))
        .with_state(state)
}
