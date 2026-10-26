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

        // 2. Fase de Recall Semántico Densa (Vectorial en memoria sobre catálogo de embeddings L2)
        let t0 = std::time::Instant::now();
        let candidate_indices = registry.retrieve_candidates_semantic(&query_emb, self.config.recall_limit);
        let d_recall = t0.elapsed();

        if candidate_indices.is_empty() {
            return Ok(PredictResponse {
                tools: vec![],
                complexity: "empty_recall".to_string(),
                model_size: "int8_unified".to_string(),
                ranked_scores: vec![],
                topic_shift: Some(0.0),
                estado_nli: Some("TRANSICION".to_string()),
            });
        }

        // 3. Extracción de tokens de consulta (1, q_len, 384)
        let t0 = std::time::Instant::now();
        let query_text = format!("{} {}", payload.text, payload.keywords.join(" "));
        let query_tokens = self.engine.extract_query_tokens(&query_text, 256)?;
        let d_query_tokens = t0.elapsed();

        // 4. Extracción de tokens de contexto histórico o intentContext (1, ctx_len, 384)
        let t0 = std::time::Instant::now();
        let ctx_text = payload.intent_context.as_deref().unwrap_or("query: contextual_turn");
        let context_tokens = self.engine.extract_context_tokens(ctx_text, 256)?;
        let d_ctx_tokens = t0.elapsed();

        // 5. Ensamblado Zero-Copy del tensor de herramientas candidatas (K, 96, 384)
        let t0 = std::time::Instant::now();
        let tools_tensor = registry.assemble_candidates_tensor(&candidate_indices);
        let d_assemble = t0.elapsed();

        // 6. Inferencia MoE de 3 etapas en una sola pasada
        let t0 = std::time::Instant::now();
        let moe_output = self.engine.evaluate_moe(
            query_tokens.view(),
            context_tokens.view(),
            tools_tensor.view(),
        )?;
        let d_moe = t0.elapsed();

        // 7. Fusión semántica E5 (Bi-Encoder) + Juicio Neuronal MoE (Reranker)
        let mut ranked = Vec::new();
        for (i, &tool_idx) in candidate_indices.iter().enumerate() {
            if tool_idx < registry.tools.len() {
                let tool = &registry.tools[tool_idx];
                let sem_sim = cosine_similarity_normalized(&query_emb, &tool.embedding);
                let moe_score = if i < moe_output.ranked_scores.len() {
                    moe_output.ranked_scores[i]
                } else {
                    sem_sim
                };

                // Fusión ponderada pura: 50% similitud semántica global + 50% juicio fino del modelo
                let final_score = (0.5 * sem_sim + 0.5 * moe_score).clamp(0.0, 1.0);
                ranked.push(RankedToolScore {
                    name: tool.metadata.name.clone(),
                    score: final_score,
                    expert_affinity: "expert_action".to_string(),
                });
            }
        }

        // Ordenar candidatos por score descendente
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // 8. Filtrado por umbral adaptativo (abstención eleva umbral dinámico en vez de silenciar a ciegas)
        let effective_min_score = if moe_output.debe_abstenerse {
            (self.config.adaptive_min_score + 0.08).min(0.88)
        } else {
            0.72
        };
        let winning_tools = Self::adaptive_threshold(
            &ranked,
            1,
            self.config.max_output_tools,
            self.config.adaptive_gap_threshold,
            effective_min_score,
        );

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

        tracing::info!(
            target: "profiler",
            "[profile:predict] tenant={} session={} total={:.2}ms (recall={:.2}ms q_emb={:.2}ms query_tokens={:.2}ms ctx_tokens={:.2}ms assemble={:.2}ms moe={:.2}ms) catalog={} candidates={} winning={} nli={} topic_shift={:.3} abstain={}",
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
            moe_output.estado_dominante,
            moe_output.topic_shift_score,
            moe_output.debe_abstenerse
        );

        tracing::info!(
            target: "prediction_trace",
            "[trace:predict] session={} source={} prompt=\"{}\" keywords={:?} min_score={:.2} gap_thresh={:.2} selected={:?} top_candidates={} filtered_out={}",
            payload.session_id,
            payload.source,
            preview_text(&payload.text, 120),
            payload.keywords,
            self.config.adaptive_min_score,
            self.config.adaptive_gap_threshold,
            winning_tools,
            format_scored_slice(&top_candidates, 8),
            format_scored_slice(&filtered_out, 8)
        );

        Ok(PredictResponse {
            tools: winning_tools,
            complexity: "dense_moe".to_string(),
            model_size: "int8_unified".to_string(),
            ranked_scores: ranked,
            topic_shift: Some(moe_output.topic_shift_score),
            estado_nli: Some(moe_output.estado_dominante),
        })
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
