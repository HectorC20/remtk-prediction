use std::sync::Arc;
use serde_json::json;

use crate::core::inference::engine::InferenceEngine;
use crate::shared::config::AppConfig;
use crate::shared::dto::{
    FastPathDecision, FastPathInput, HistoryMessageDto, HistoryRegimeRequest, HistoryRegimeResponse,
    InterestMatch, InterestPredictionInput, InterestPredictionResult, JudgeMemoryRequest,
    JudgeMemoryResponse, PredictRequest, PredictResponse, RankedToolScore,
};
use crate::shared::error::AppError;
use crate::shared::logger::{format_scored_slice, preview_text};
use crate::shared::tensor::cosine_similarity_normalized;
use crate::support::registry::TenantCache;

const CANONICAL_INTEREST_PROBE: &str = "interés comercial en adquirir, comprar, cotizar o contratar un producto o servicio";
const CANONICAL_DENIAL_PROBES: [&str; 3] = [
    "afirmaciones del agente negando herramientas o capacidades técnicas para la tarea",
    "el asistente afirma que no tiene herramientas ni funciones disponibles para realizar la acción",
    "no tengo ninguna herramienta ni función disponible para realizar esta tarea",
];
const INTEREST_TOPIC_WEIGHT: f32 = 0.6;
const INTEREST_INTENT_WEIGHT: f32 = 0.4;
const INTEREST_MAX_ANCHORS: usize = 32;
const INTEREST_TOP_MATCHES: usize = 5;

/// Servicio principal de decisión y predicción de herramientas (`PredictionService`).
/// Orquesta la interacción entre la extracción de tokens, la caché en memoria y el cabezal MoE.
pub struct PredictionService {
    engine: Arc<InferenceEngine>,
    cache: Arc<TenantCache>,
    config: Arc<AppConfig>,
}

impl PredictionService {
    pub fn new(engine: Arc<InferenceEngine>, cache: Arc<TenantCache>, config: Arc<AppConfig>) -> Self {
        Self { engine, cache, config }
    }

    /// Ejecuta el pipeline completo de predicción y juicio de herramientas para un turno.
    pub fn predict(&self, payload: &PredictRequest) -> Result<PredictResponse, AppError> {
        if payload.text.trim().is_empty() {
            return Err(AppError::BadRequest("El campo 'text' es requerido".into()));
        }

        let storage = self.cache.storage();
        let registry_guard = storage.get(&payload.tenant);

        let registry = match registry_guard {
            Some(ref r) => r.clone(),
            None => {
                tracing::warn!(
                    target: "prediction_trace",
                    "[trace:predict] tenant={} session={} status=COLD_CATALOG prompt=\"{}\"",
                    payload.tenant,
                    payload.session_id,
                    preview_text(&payload.text, 100)
                );
                // Catálogo frío / no registrado
                return Ok(PredictResponse {
                    tools: vec![],
                    complexity: "cold_catalog".to_string(),
                    model_size: "int8_unified".to_string(),
                    ranked_scores: vec![],
                    topic_shift: Some(0.0),
                    estado_nli: Some("TRANSICION".to_string()),
                });
            }
        };

        if registry.count() == 0 {
            tracing::warn!(
                target: "prediction_trace",
                "[trace:predict] tenant={} session={} status=EMPTY_CATALOG prompt=\"{}\"",
                payload.tenant,
                payload.session_id,
                preview_text(&payload.text, 100)
            );
            return Ok(PredictResponse {
                tools: vec![],
                complexity: "empty_catalog".to_string(),
                model_size: "int8_unified".to_string(),
                ranked_scores: vec![],
                topic_shift: Some(0.0),
                estado_nli: Some("TRANSICION".to_string()),
            });
        }

        let t_start = std::time::Instant::now();

        // 1. Extracción de embedding semántico de la consulta (Bi-Encoder E5)
        let t0 = std::time::Instant::now();
        let query_emb = self.engine.embed_vector(&format!("query: {}", payload.text))?;
        let d_query_emb = t0.elapsed();

        // 2. Fase de Recall Híbrida: Semántica Densa (Bi-Encoder E5) + Anclaje Léxico/Dominio
        let t0 = std::time::Instant::now();
        let exclude_set: std::collections::HashSet<String> = payload
            .exclude
            .iter()
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();

        let mut candidate_indices: Vec<usize> = registry
            .retrieve_candidates_semantic(&query_emb, self.config.recall_limit + exclude_set.len())
            .into_iter()
            .filter(|&idx| {
                idx < registry.tools.len()
                    && !exclude_set.contains(&registry.tools[idx].metadata.name.to_lowercase())
            })
            .take(self.config.recall_limit)
            .collect();

        // Inyectar candidatos con coincidencia léxica o de dominio explícita que no hayan entrado en el top-K semántico
        let query_and_kw = format!("{} {}", payload.text, payload.keywords.join(" "));
        let mut domain_boosted: Vec<(usize, f32)> = registry
            .tools
            .iter()
            .enumerate()
            .filter(|(idx, tool)| {
                !exclude_set.contains(&tool.metadata.name.to_lowercase())
                    && !candidate_indices.contains(idx)
            })
            .map(|(idx, tool)| {
                let boost = Self::compute_lexical_and_domain_affinity(
                    &payload.text,
                    &payload.keywords,
                    &tool.metadata.name,
                    &tool.metadata.group,
                    &tool.metadata.intent_summary,
                    &tool.metadata.description,
                    &tool.metadata.tags,
                );
                (idx, boost)
            })
            .filter(|(_, boost)| *boost >= 0.06)
            .collect();

        domain_boosted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (idx, _) in domain_boosted.into_iter().take(4) {
            if candidate_indices.len() >= self.config.recall_limit && !candidate_indices.is_empty() {
                candidate_indices.pop();
            }
            candidate_indices.push(idx);
        }
        let d_recall = t0.elapsed();

        if candidate_indices.is_empty() {
            return Ok(PredictResponse {
                tools: vec![],
                complexity: "simple".to_string(),
                model_size: "int8_unified".to_string(),
                ranked_scores: vec![],
                topic_shift: Some(0.0),
                estado_nli: Some("TRANSICION".to_string()),
            });
        }

        // 3. Extracción de tokens de consulta (1, q_len, 384) normalizados L2 por token
        let t0 = std::time::Instant::now();
        let query_text = if payload.keywords.is_empty() {
            payload.text.clone()
        } else {
            query_and_kw
        };
        let query_tokens = self.engine.extract_query_tokens(&query_text, 256)?;
        let d_query_tokens = t0.elapsed();

        // 4. Extracción de tokens de contexto histórico o intentContext (1, ctx_len, 384)
        let t0 = std::time::Instant::now();
        let ctx_text = payload
            .intent_context
            .as_deref()
            .or(payload.prior_plan.as_deref())
            .unwrap_or("contextual_turn");
        let context_tokens = self.engine.extract_context_tokens(ctx_text, 256)?;
        let d_ctx_tokens = t0.elapsed();

        // 5. Ensamblado Zero-Copy del tensor de herramientas candidatas (K, 96, 384)
        let t0 = std::time::Instant::now();
        let tools_tensor = registry.assemble_candidates_tensor(&candidate_indices);
        let d_assemble = t0.elapsed();

        // 6. Inferencia MoE de 3 etapas en una sola pasada (Arquitectura Cognitiva JEV de 3 Capas)
        let t0 = std::time::Instant::now();
        let moe_output = self.engine.evaluate_moe(
            query_tokens.view(),
            context_tokens.view(),
            tools_tensor.view(),
        )?;
        let d_moe = t0.elapsed();

        // 7. Enrutador Cognitivo de Sistema 1 (JEV 3 Capas: Choice + MaxSim + Noul/Score)
        let max_aff = moe_output
            .ranked_scores
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let uniform_p = 1.0 / (candidate_indices.len().max(1) as f32);

        let mut max_lex_boost = 0.0f32;
        let mut candidate_boosts = vec![0.0f32; candidate_indices.len()];
        for (i, &tool_idx) in candidate_indices.iter().enumerate() {
            if tool_idx < registry.tools.len() {
                let tool = &registry.tools[tool_idx];
                let lb = Self::compute_lexical_and_domain_affinity(
                    &payload.text,
                    &payload.keywords,
                    &tool.metadata.name,
                    &tool.metadata.group,
                    &tool.metadata.intent_summary,
                    &tool.metadata.description,
                    &tool.metadata.tags,
                );
                candidate_boosts[i] = lb;
                if lb > max_lex_boost {
                    max_lex_boost = lb;
                }
            }
        }

        let is_conceptual = Self::is_pure_conceptual_query(&payload.text)
            && payload.keywords.is_empty()
            && max_lex_boost < 0.08;

        // Compuerta de abstención de Capa 2 (INDEPENDENT) y Capa 3 (decision_valid_noul < 0.50):
        // Si el verificador JEV dictamina abstención y no hay un comando léxico explícito inequívoco, abstenerse.
        let abstain_gate = is_conceptual || (moe_output.debe_abstenerse && max_lex_boost < 0.12);

        let mut ranked = Vec::new();
        for (i, &tool_idx) in candidate_indices.iter().enumerate() {
            if tool_idx < registry.tools.len() {
                let tool = &registry.tools[tool_idx];
                let sem_sim = cosine_similarity_normalized(&query_emb, &tool.embedding);
                let lex_boost = candidate_boosts[i];

                let raw_score = if moe_output.is_jev {
                    let choice_prob = moe_output.tool_choice_probs.get(i).copied().unwrap_or(uniform_p);
                    let jev_rel = ((choice_prob - uniform_p) / uniform_p.max(1e-6)).clamp(-0.5, 2.5);
                    let tool_aff = moe_output.ranked_scores.get(i).copied().unwrap_or(sem_sim);

                    let name_lower = tool.metadata.name.to_lowercase();
                    let route_bonus = match moe_output.ctx_route.as_str() {
                        "MEMORY" if name_lower.contains("memory") || name_lower.contains("history") => 0.08,
                        "TOOL" if lex_boost > 0.0 => 0.03,
                        _ => 0.0,
                    };

                    if max_aff >= 0.35 {
                        // Cabezal de afinidad individual desacoplado
                        0.50 * sem_sim + 0.40 * tool_aff + 0.03 * jev_rel + lex_boost + route_bonus
                    } else {
                        // Cabezal de afinidad comprimido: usar calibración E5 + preferencia relativa JEV Choice + léxico
                        sem_sim + 0.035 * jev_rel + lex_boost + route_bonus
                    }
                } else {
                    let moe_score = moe_output.ranked_scores.get(i).copied().unwrap_or(sem_sim);
                    0.5 * sem_sim + 0.5 * moe_score + lex_boost
                };

                let final_score = if abstain_gate {
                    (raw_score * 0.35).clamp(0.0, 0.45)
                } else {
                    raw_score.clamp(0.0, 0.99)
                };

                ranked.push(RankedToolScore {
                    name: tool.metadata.name.clone(),
                    score: final_score,
                    expert_affinity: if moe_output.is_jev {
                        format!("jev_{}", moe_output.ctx_route.to_lowercase())
                    } else {
                        "expert_action".to_string()
                    },
                });
            }
        }

        // Ordenar candidatos por score descendente
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // 8. Filtrado por umbral adaptativo calibrado para JEV 3-Capas
        let effective_min_score = if abstain_gate {
            0.95
        } else if moe_output.ctx_route == "PREV_RESPONSE" && max_lex_boost >= 0.06 {
            0.79
        } else {
            0.81
        };

        let mut winning_tools = if abstain_gate {
            Vec::new()
        } else {
            Self::adaptive_threshold(
                &ranked,
                1,
                self.config.max_output_tools,
                self.config.adaptive_gap_threshold,
                effective_min_score,
            )
        };

        // Si ninguna herramienta superó el umbral adaptativo (turno sin herramientas),
        // asegurar que ranked_scores quede por debajo de 0.48 para que el consumidor no rescate ruido.
        if winning_tools.is_empty() {
            for item in &mut ranked {
                if item.score >= 0.48 {
                    item.score = (item.score * 0.45).min(0.45);
                }
            }
        } else {
            // Acotar a máximo 5 herramientas cuando no hay anclas múltiples explícitas
            if winning_tools.len() > 5 && payload.keywords.len() <= 2 {
                winning_tools.truncate(5);
            }
        }

        let total_elapsed = t_start.elapsed();

        let top_candidates: Vec<(String, f32)> = ranked
            .iter()
            .take(8)
            .map(|r| (r.name.clone(), r.score))
            .collect();

        let winning_set: std::collections::HashSet<&str> =
            winning_tools.iter().map(|s| s.as_str()).collect();
        let filtered_out: Vec<(String, f32)> = ranked
            .iter()
            .filter(|r| !winning_set.contains(r.name.as_str()))
            .take(8)
            .map(|r| (r.name.clone(), r.score))
            .collect();

        let complexity = if winning_tools.is_empty() {
            "simple".to_string()
        } else {
            "dense_moe".to_string()
        };

        tracing::info!(
            target: "profiler",
            "[profile:predict] tenant={} session={} total={:.2}ms (recall={:.2}ms q_emb={:.2}ms query_tokens={:.2}ms ctx_tokens={:.2}ms assemble={:.2}ms moe={:.2}ms) catalog={} candidates={} winning={} route={} nli={} valid_noul={:.3} topic_shift={:.3} abstain={}",
            payload.tenant,
            payload.session_id,
            total_elapsed.as_secs_f64() * 1000.0,
            d_recall.as_secs_f64() * 1000.0,
            d_query_emb.as_secs_f64() * 1000.0,
            d_query_tokens.as_secs_f64() * 1000.0,
            d_ctx_tokens.as_secs_f64() * 1000.0,
            d_assemble.as_secs_f64() * 1000.0,
            d_moe.as_secs_f64() * 1000.0,
            registry.count(),
            candidate_indices.len(),
            winning_tools.len(),
            moe_output.ctx_route,
            moe_output.estado_dominante,
            moe_output.valid_noul,
            moe_output.topic_shift_score,
            abstain_gate
        );

        tracing::info!(
            target: "prediction_trace",
            "[trace:predict] session={} source={} prompt=\"{}\" keywords={:?} min_score={:.2} gap_thresh={:.2} selected={:?} top_candidates={} filtered_out={}",
            payload.session_id,
            payload.source,
            preview_text(&payload.text, 120),
            payload.keywords,
            effective_min_score,
            self.config.adaptive_gap_threshold,
            winning_tools,
            format_scored_slice(&top_candidates, 8),
            format_scored_slice(&filtered_out, 8)
        );

        Ok(PredictResponse {
            tools: winning_tools,
            complexity,
            model_size: "int8_unified".to_string(),
            ranked_scores: ranked,
            topic_shift: Some(moe_output.topic_shift_score),
            estado_nli: Some(moe_output.estado_dominante),
        })
    }

    /// Detecta si la consulta es puramente conceptual/definicional sin verbos operativos de búsqueda o herramienta.
    fn is_pure_conceptual_query(text: &str) -> bool {
        let lower = text.trim().to_lowercase();
        let operational_cues = [
            "busca", "buscar", "internet", "web", "google", "crea", "crear", "crees",
            "lista", "listar", "programa", "programar", "alerta", "alertas", "cron",
            "recordatorio", "archivo", "lee", "leer", "guarda", "guardar", "registra",
            "registrar", "actualiza", "elimina", "borra", "ejecuta", "herramienta",
            "herramientas", "herraamienta", "herraamientas", "heramienta", "herramientad",
            "mcp", "hora", "fecha", "clima", "pronostico", "pronóstico",
            "memoria", "historial", "publica", "publicar", "envia", "enviar", "revisa",
        ];
        if operational_cues.iter().any(|cue| lower.contains(cue)) {
            return false;
        }
        let conceptual_prefixes = [
            "dime qué es", "dime que es", "¿qué es ", "que es ", "qué es ",
            "explícame qué es", "explicame que es", "define ", "¿cómo funciona ",
        ];
        conceptual_prefixes.iter().any(|p| lower.starts_with(p))
    }

    /// Calcula la afinidad léxica y de puentes de dominio entre la consulta y una herramienta candidata.
    fn compute_lexical_and_domain_affinity(
        query: &str,
        keywords: &[String],
        tool_name: &str,
        tool_group: &str,
        intent_summary: &str,
        description: &str,
        tags: &[String],
    ) -> f32 {
        let query_lower = query.trim().to_lowercase();
        let name_lower = tool_name.trim().to_lowercase();
        let group_lower = tool_group.trim().to_lowercase();
        let summary_lower = intent_summary.to_lowercase();
        let desc_lower = description.to_lowercase();

        let mut boost = 0.0f32;

        // 1. Anclaje directo por keywords del planificador
        for kw in keywords {
            let kw_lower = kw.trim().to_lowercase();
            if kw_lower.is_empty() {
                continue;
            }
            if kw_lower == name_lower {
                boost += 0.14;
            } else if name_lower.contains(&kw_lower) || kw_lower.contains(&name_lower) {
                boost += 0.08;
            }
        }

        // 2. Puentes semántico-léxicos de dominio (Cron / Alertas / Recordatorios / Tareas programadas)
        let has_schedule_domain = query_lower.contains("cron")
            || query_lower.contains("alerta")
            || query_lower.contains("recordatorio")
            || query_lower.contains("alarma")
            || query_lower.contains("programada")
            || query_lower.contains("cada 10 min")
            || query_lower.contains("cada 5 min")
            || (query_lower.contains("cada ") && query_lower.contains(" min") && (query_lower.contains("cre") || query_lower.contains("program")));

        if has_schedule_domain {
            let is_create = query_lower.contains("crea")
                || query_lower.contains("crees")
                || query_lower.contains("crear")
                || query_lower.contains("programa")
                || query_lower.contains("configura")
                || query_lower.contains("agrega")
                || query_lower.contains("pon");
            let is_delete = query_lower.contains("elimina")
                || query_lower.contains("borra")
                || query_lower.contains("cancela")
                || query_lower.contains("desactiva");

            if is_delete && (name_lower == "schedule_delete" || name_lower == "schedule_cancel") {
                boost += 0.16;
            } else if is_create && name_lower == "schedule_task" {
                boost += 0.16;
            } else if !is_create && !is_delete && name_lower == "schedule_list" {
                boost += 0.16;
            } else if name_lower.starts_with("schedule_") {
                boost += 0.06;
            }
        }

        // 3. Puente de introspección de herramientas / MCP
        let has_tool_introspection = query_lower.contains("herramienta")
            || query_lower.contains("herraamienta")
            || query_lower.contains("heramienta")
            || query_lower.contains("herramientad")
            || query_lower.contains("capacidades")
            || query_lower.contains("habilidades")
            || query_lower.contains("mcp")
            || query_lower.contains("tools");
        if has_tool_introspection {
            if name_lower == "list_tools" || name_lower == "list_own_tools" {
                boost += 0.22;
            } else if name_lower == "search_knowledge_base"
                || name_lower == "register_tool"
                || name_lower == "search_memory"
            {
                boost += 0.09;
            }
        }

        // 4. Puente de búsqueda web explícita
        let has_web_intent = (query_lower.contains("internet") || query_lower.contains("web") || query_lower.contains("google"))
            && (query_lower.contains("busca") || query_lower.contains("investiga") || query_lower.contains("consulta") || query_lower.contains("sobre"));
        if has_web_intent && (name_lower == "web_search" || name_lower == "web_fetch") {
            boost += 0.12;
        }

        // 5. Solapamiento léxico directo con nombre, tags e intent_summary
        let stopwords = [
            "para", "como", "sobre", "donde", "cuando", "este", "esta", "estos", "estas",
            "porque", "hacer", "quiero", "puedes", "decir", "favor", "dame", "mostrar",
            "tiene", "tienen", "desde", "hasta", "solo", "eso", "probar", "actualmente",
            "realmente", "existe", "existen", "cuales", "cual", "tengo", "tengp", "mis",
        ];

        let query_tokens: Vec<&str> = query_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4 && !stopwords.contains(w))
            .collect();

        let mut token_hits = 0usize;
        for tok in &query_tokens {
            let in_name = name_lower.contains(tok);
            let in_tags = tags.iter().any(|t| t.to_lowercase().contains(tok));
            let in_group = group_lower.contains(tok);
            if in_name || in_tags || in_group {
                token_hits += 2;
            } else if summary_lower.contains(tok) || desc_lower.contains(tok) {
                token_hits += 1;
            }
        }

        if token_hits > 0 {
            boost += (token_hits as f32 * 0.025).min(0.10);
        }

        boost.min(0.22)
    }

    /// Implementación de umbral adaptativo con corte por gap natural e intervalo de relevancia.
    pub fn adaptive_threshold(
        scored: &[RankedToolScore],
        min_tools: usize,
        max_tools: usize,
        gap_threshold: f32,
        min_score: f32,
    ) -> Vec<String> {
        if scored.is_empty() {
            return Vec::new();
        }

        let top = scored[0].score;
        // Piso de relevancia: si la mejor herramienta no supera el umbral, abstenerse (0 tools)
        if top < min_score {
            return Vec::new();
        }

        let min = min_tools.min(scored.len());
        let max = min.max(max_tools);
        let band = gap_threshold * 2.0;

        // Corte por gap natural después del mínimo: gap > gap_threshold && top - score > band
        for i in min..scored.len().min(max) {
            let gap = scored[i - 1].score - scored[i].score;
            if scored[i].score < min_score || (gap > gap_threshold && (top - scored[i].score) > band) {
                return scored[0..i].iter().map(|t| t.name.clone()).collect();
            }
        }

        // Sin gap natural (ranking denso): agrupa el cluster superior dentro de la banda del top
        let mut cluster = scored.len();
        for i in 0..scored.len() {
            if scored[i].score < min_score || (top - scored[i].score) > band {
                cluster = i;
                break;
            }
        }

        let count = min.max(cluster.min(max));
        scored[0..count].iter().map(|t| t.name.clone()).collect()
    }

    /// Evalúa la decisión de Fast-Path para omitir llamadas redundantes al planificador LLM.
    pub fn decide_fast_path(&self, input: &FastPathInput) -> FastPathDecision {
        let has_plan = input.has_pending_plan.unwrap_or(false);
        let tokens_saved = input.average_planner_tokens.unwrap_or(2500);

        let decision = if input.tools.is_empty() && !has_plan {
            FastPathDecision {
                action: "DIRECT_CHAT".to_string(),
                skip_planner: true,
                reason: "Sin herramientas requeridas y sin plan pendiente de confirmación".to_string(),
                estimated_tokens_saved: tokens_saved,
                direct_tool: None,
                suggested_chat_context: Some(json!({
                    "situation": "DIRECT_CHAT",
                    "toolCount": 0,
                    "promptRole": "conversational_assistant"
                })),
            }
        } else if input.tools.len() == 1 && !has_plan {
            let tool_name = input.tools[0].get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
            FastPathDecision {
                action: "DIRECT_EXECUTE".to_string(),
                skip_planner: true,
                reason: "Herramienta única determinante identificada para ejecución directa".to_string(),
                estimated_tokens_saved: tokens_saved,
                direct_tool: Some(tool_name.to_string()),
                suggested_chat_context: None,
            }
        } else {
            FastPathDecision {
                action: "FULL_PLANNER".to_string(),
                skip_planner: false,
                reason: "Se requiere orquestación multi-herramienta o resolución de dependencias".to_string(),
                estimated_tokens_saved: 0,
                direct_tool: None,
                suggested_chat_context: None,
            }
        };

        tracing::info!(
            target: "prediction_trace",
            "[trace:route] action={} skip_planner={} tools_count={} direct_tool={:?} tokens_saved={} prompt=\"{}\"",
            decision.action,
            decision.skip_planner,
            input.tools.len(),
            decision.direct_tool,
            decision.estimated_tokens_saved,
            preview_text(&input.text, 100)
        );

        decision
    }

    /// POST /interest: Score semántico de interés del turno contra anclas y arquetipo comercial.
    pub fn predict_interest(
        &self,
        input: &InterestPredictionInput,
    ) -> Result<InterestPredictionResult, AppError> {
        let text = input.text.trim();
        if text.is_empty() {
            return Ok(InterestPredictionResult {
                interest_score: 0.0,
                topic: None,
                topic_score: 0.0,
                intent_score: 0.0,
                matches: vec![],
                method: "empty".to_string(),
            });
        }

        let t_start = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let prompt_emb = self.engine.embed_vector(text)?;
        let d_prompt = t0.elapsed();

        let t0 = std::time::Instant::now();
        let probe_emb = self.engine.embed_vector(CANONICAL_INTEREST_PROBE)?;
        let intent_score = cosine_similarity_normalized(&prompt_emb, &probe_emb).max(0.0);
        let d_probe = t0.elapsed();

        let limit = input.limit.unwrap_or(INTEREST_MAX_ANCHORS).min(INTEREST_MAX_ANCHORS);
        let mut matches = Vec::new();

        let t0 = std::time::Instant::now();
        if let Some(ref anchors) = input.anchors {
            for anchor in anchors.iter().take(limit) {
                let trimmed = anchor.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(anchor_emb) = self.engine.embed_vector(trimmed) {
                    let score = cosine_similarity_normalized(&prompt_emb, &anchor_emb);
                    matches.push(InterestMatch {
                        anchor: trimmed.to_string(),
                        score,
                    });
                }
            }
            matches.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }
        let d_anchors = t0.elapsed();

        let topic_score = matches.first().map(|m| m.score).unwrap_or(0.0);
        let topic = matches.first().map(|m| m.anchor.clone());

        let interest_score = if matches.is_empty() {
            intent_score.clamp(0.0, 1.0)
        } else {
            (topic_score * INTEREST_TOPIC_WEIGHT + intent_score * INTEREST_INTENT_WEIGHT).clamp(0.0, 1.0)
        };

        let top_matches = matches.into_iter().take(INTEREST_TOP_MATCHES).collect();
        let total_elapsed = t_start.elapsed();

        tracing::info!(
            target: "profiler",
            "[profile:interest] session={:?} total={:.2}ms (prompt_embed={:.2}ms probe_embed={:.2}ms anchors_embed={:.2}ms) score={:.3} intent={:.3} topic={:.3}",
            input.session_id,
            total_elapsed.as_secs_f64() * 1000.0,
            d_prompt.as_secs_f64() * 1000.0,
            d_probe.as_secs_f64() * 1000.0,
            d_anchors.as_secs_f64() * 1000.0,
            interest_score,
            intent_score,
            topic_score
        );

        Ok(InterestPredictionResult {
            interest_score,
            topic,
            topic_score,
            intent_score,
            matches: top_matches,
            method: "multilingual-e5-small".to_string(),
        })
    }

    /// POST /memory/judge: Evalúa si un texto es ruido de negación técnica o incapacidad.
    pub fn judge_memory(&self, req: &JudgeMemoryRequest) -> Result<JudgeMemoryResponse, AppError> {
        let text = req.text.trim();
        if text.is_empty()
            || text.starts_with("Prompt del usuario:")
            || text.starts_with("[HISTORIAL RECIENTE]")
            || text.starts_with("user:")
        {
            return Ok(JudgeMemoryResponse {
                is_noise: false,
                noise_score: 0.0,
            });
        }

        // Limpiar viñetas, scores o prefijos típicos de la memoria ("- (0.85) ...")
        let mut clean_text = text.trim_start_matches(|c: char| c == '-' || c == '*' || c == '•' || c.is_whitespace());
        if clean_text.starts_with('(') {
            if let Some(idx) = clean_text.find(')') {
                clean_text = clean_text[idx + 1..].trim();
            }
        }

        let prompt_emb = self.engine.embed_vector(clean_text)?;
        let mut max_score = 0.0f32;

        for probe in CANONICAL_DENIAL_PROBES {
            if let Ok(probe_emb) = self.engine.embed_vector(probe) {
                let sim = cosine_similarity_normalized(&prompt_emb, &probe_emb);
                if sim > max_score {
                    max_score = sim;
                }
            }
        }

        let is_noise = max_score >= 0.82;
        tracing::debug!(
            target: "prediction_trace",
            "[trace:memory_judge] is_noise={} noise_score={:.4} text=\"{}\"",
            is_noise,
            max_score,
            preview_text(clean_text, 80)
        );
        Ok(JudgeMemoryResponse {
            is_noise,
            noise_score: max_score,
        })
    }

    /// POST /history/regime: Evalúa y decide inteligentemente el régimen de preservación del historial.
    pub fn evaluate_history_regime(
        &self,
        req: &HistoryRegimeRequest,
    ) -> Result<HistoryRegimeResponse, AppError> {
        let prompt_trimmed = req.prompt.trim();
        let prompt_lower = prompt_trimmed.to_lowercase();
        let history_len = req.history.len();

        if history_len == 0 || prompt_trimmed.is_empty() {
            return Ok(HistoryRegimeResponse {
                regime: "PRESERVE_ALL".to_string(),
                topic_shift: 0.0,
                anaphora_detected: false,
                dependency_score: 0.0,
                selected_indices: (0..history_len).collect(),
                reason: "Historial vacío o prompt no analizable; régimen neutro".to_string(),
            });
        }

        // 1. Detección de marcadores anafóricos, deícticos, conectores de continuación o elipsis
        let continuation_prefixes = [
            "y ", "¿y ", "e ", "¿e ", "o ", "¿o ", "pero ", "¿pero ",
            "sobre ", "¿sobre ", "del ", "¿del ", "de la ", "¿de la ", "de los ", "¿de los ",
            "de las ", "¿de las ", "de lo ", "¿de lo ", "en cuanto a ", "¿en cuanto a ",
            "con respecto a ", "¿con respecto a ", "respecto a ", "¿respecto a ", "respecto de ",
            "qué hay de ", "que hay de ", "¿qué hay de ", "¿que hay de ", "qué tal ", "que tal ",
            "¿qué tal ", "¿que tal ", "muéstrame ", "muestrame ", "dame ", "pásame ", "pasame ",
            "enséñame ", "enseñame ", "cómo va ", "como va ", "¿cómo va ", "¿como va ",
            "cuéntame ", "cuentame ", "explica ", "explícame ", "explicame ", "continúa", "continua",
        ];

        let anaphora_cues = [
            "ambos", "ambas", "los dos", "las dos", "el anterior", "la anterior", "lo anterior",
            "los anteriores", "las anteriores", "anterior", "anteriores", "eso", "esto", "aquello",
            "esos", "esas", "estos", "estas", "compara", "comparar", "comparacion", "comparación",
            "diferencia", "cuál de", "cual de", "respecto a", "respecto de", "de ellos", "de ellas",
            "del primero", "del segundo", "el primero", "el segundo", "el tercero", "el último",
            "el ultimo", "la primera", "la segunda", "la última", "la ultima", " y en ", " ¿y en ",
            " y del ", " ¿y del ", " y de la ", " ¿y de la ", " y de los ", " ¿y de los ", " y de las ",
            " ¿y de las ", " y de ", " ¿y de ", " y el ", " ¿y el ", " y la ", " ¿y la ", " y los ",
            " ¿y los ", " y las ", " ¿y las ", " pero en ", "menciónalo", "mencionalo", "dímelo",
            "dimelo", "detállalo", "detallalo", "explícalo", "explicalo", "muéstralo", "muestralo",
            "cuál era", "cual era", "de los cuales", "de las cuales", "qué opinas de eso",
            "que opinas de eso", "de qué hemos estado hablando", "de que hemos estado hablando",
            "también", "tambien", "tampoco", "además", "ademas", "otra vez", "de nuevo", "mismo", "misma",
        ];

        let mut anaphora_detected = false;
        let mut matched_cue = "";

        for prefix in continuation_prefixes {
            if prompt_lower.starts_with(prefix) {
                anaphora_detected = true;
                matched_cue = prefix;
                break;
            }
        }

        if !anaphora_detected {
            for cue in anaphora_cues {
                if prompt_lower.contains(cue) {
                    anaphora_detected = true;
                    matched_cue = cue;
                    break;
                }
            }
        }

        // Heurística de seguimiento breve: preguntas elípticas cortas (<= 5 palabras y <= 35 caracteres)
        let word_count = prompt_trimmed.split_whitespace().count();
        let is_brief_followup = word_count <= 5 && prompt_trimmed.len() <= 35;
        if !anaphora_detected && is_brief_followup {
            anaphora_detected = true;
            matched_cue = "consulta breve de seguimiento";
        }

        // 2. Co-referencia y solapamiento léxico con turnos recientes (últimos 4 mensajes)
        let recent_slice_start = history_len.saturating_sub(4);
        let recent_context_lower: String = req.history[recent_slice_start..]
            .iter()
            .map(|m| m.content.to_lowercase())
            .collect::<Vec<_>>()
            .join(" ");

        let stopwords = [
            "para", "como", "sobre", "donde", "cuando", "este", "esta", "estos", "estas",
            "porque", "hacer", "quiero", "puedes", "decir", "favor", "busca", "dame", "mostrar",
            "diagrama", "tiene", "tienen", "desde", "hasta",
        ];

        let prompt_keywords: Vec<&str> = prompt_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4 && !stopwords.contains(w))
            .collect();

        let lexical_overlap = !prompt_keywords.is_empty()
            && prompt_keywords.iter().any(|kw| recent_context_lower.contains(kw));

        if anaphora_detected || lexical_overlap {
            let reason = if anaphora_detected {
                format!("Dependencia referencial activa ('{}'); se preserva el historial completo.", matched_cue)
            } else {
                format!("Co-referencia léxica con diálogo reciente ({:?}); se preserva el diálogo.", prompt_keywords)
            };
            tracing::info!(
                target: "profiler",
                "[profile:history_regime] Continuidad contextual detectada -> PRESERVE_ALL (total_msgs={})",
                history_len
            );
            return Ok(HistoryRegimeResponse {
                regime: "PRESERVE_ALL".to_string(),
                topic_shift: 0.0,
                anaphora_detected: true,
                dependency_score: 1.0,
                selected_indices: (0..history_len).collect(),
                reason,
            });
        }

        // 3. Inferencia semántica con e5-small contra el último turno completo (user y assistant)
        let last_user_msg = req
            .history
            .iter()
            .rev()
            .find(|m| m.role == "user" && !m.content.trim().is_empty())
            .map(|m| m.content.trim())
            .unwrap_or("");

        let last_assistant_msg = req
            .history
            .iter()
            .rev()
            .find(|m| m.role == "assistant" && !m.content.trim().is_empty())
            .map(|m| {
                // Acotar a los primeros 600 caracteres para el embedding de contexto
                let content = m.content.trim();
                if content.len() > 600 {
                    &content[..600]
                } else {
                    content
                }
            })
            .unwrap_or("");

        if last_user_msg.is_empty() && last_assistant_msg.is_empty() {
            return Ok(HistoryRegimeResponse {
                regime: "PRESERVE_ALL".to_string(),
                topic_shift: 0.0,
                anaphora_detected: false,
                dependency_score: 0.5,
                selected_indices: (0..history_len).collect(),
                reason: "Sin mensajes previos con contenido evaluable".to_string(),
            });
        }

        let prompt_emb = self.engine.embed_vector(prompt_trimmed)?;
        let mut max_similarity = 0.0f32;

        if !last_user_msg.is_empty() {
            if let Ok(user_emb) = self.engine.embed_vector(last_user_msg) {
                let sim = cosine_similarity_normalized(&prompt_emb, &user_emb);
                if sim > max_similarity {
                    max_similarity = sim;
                }
            }
        }

        if !last_assistant_msg.is_empty() {
            if let Ok(assistant_emb) = self.engine.embed_vector(last_assistant_msg) {
                let sim = cosine_similarity_normalized(&prompt_emb, &assistant_emb);
                if sim > max_similarity {
                    max_similarity = sim;
                }
            }
        }

        let similarity = max_similarity;
        let topic_shift = (1.0 - similarity).clamp(0.0, 1.0);
        let dependency_score = similarity.max(0.0);

        // Umbral calibrado de afinidad temática E5 (0.70 para evitar falsos positivos de ruptura)
        const TOPIC_SHIFT_THRESHOLD: f32 = 0.70;

        // 4. Decisión del régimen según similitud de intención
        if similarity < TOPIC_SHIFT_THRESHOLD {
            tracing::info!(
                target: "profiler",
                "[profile:history_regime] Ruptura temática real (sim={:.3} < {:.2}, topicShift={:.3}) -> ISOLATE",
                similarity,
                TOPIC_SHIFT_THRESHOLD,
                topic_shift
            );
            Ok(HistoryRegimeResponse {
                regime: "ISOLATE".to_string(),
                topic_shift,
                anaphora_detected: false,
                dependency_score,
                selected_indices: vec![],
                reason: format!("Ruptura temática detectada (sim={:.2} < {:.2}); se aísla la consulta para evitar arrastrar contexto irrelevante.", similarity, TOPIC_SHIFT_THRESHOLD),
            })
        } else {
            let start_idx = history_len.saturating_sub(6);
            let selected: Vec<usize> = (start_idx..history_len).collect();
            tracing::info!(
                target: "profiler",
                "[profile:history_regime] Continuidad temática (sim={:.3} >= {:.2}, topicShift={:.3}) -> PRESERVE_RECENT (selected={})",
                similarity,
                TOPIC_SHIFT_THRESHOLD,
                topic_shift,
                selected.len()
            );
            Ok(HistoryRegimeResponse {
                regime: "PRESERVE_RECENT".to_string(),
                topic_shift,
                anaphora_detected: false,
                dependency_score,
                selected_indices: selected,
                reason: format!("Continuidad temática (sim={:.2}); se preserva el diálogo reciente.", similarity),
            })
        }
    }
}
