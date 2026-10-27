use std::path::Path;
use std::sync::{Arc, Mutex};
use ndarray::ArrayView3;
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use crate::shared::error::AppError;

/// Salida procesada y calibrada del cabezal de decisión MoE de 3 etapas (Legacy + JEV 3-Capas).
#[derive(Debug, Clone)]
pub struct MoeDecisionOutput {
    /// Puntuaciones de afinidad individual [0.0, 1.0] para cada una de las K herramientas candidatas.
    pub ranked_scores: Vec<f32>,
    /// Distribución de probabilidad JEV Choice [0.0, 1.0] sobre las K herramientas (suma = 1.0).
    pub tool_choice_probs: Vec<f32>,
    /// Capa 1 (JEV Choice): Probabilidades de idioma [ES, EN].
    pub lang_probs: [f32; 2],
    /// Capa 2 (JEV Choice): Ruta cognitiva dominante ("TOOL", "MEMORY", "PREV_RESPONSE", "INDEPENDENT").
    pub ctx_route: String,
    /// Probabilidades de estado conversacional / ruta:
    /// En JEV: [TOOL, MEMORY, PREV_RESPONSE, INDEPENDENT].
    /// En Legacy: [TRANSICION, CONTINUACION, CORRECCION, ABANDONO].
    pub estado_probs: [f32; 4],
    /// Estado NLI dominante del turno (compatible con contrato HTTP).
    pub estado_dominante: String,
    /// Capa 3 (JEV Noul): Probabilidad [0.0, 1.0] de validez de la decisión del Sistema 1.
    pub valid_noul: f32,
    /// Capa 3 (JEV Score): Calidad esperada de la decisión [0.0, 3.0].
    pub quality_score: f32,
    /// Probabilidad [0.0, 1.0] de abstención para retorno nulo.
    pub abstencion_prob: f32,
    /// Booleano de decisión de abstención.
    pub debe_abstenerse: bool,
    /// Puntuación de cambio de tópico.
    pub topic_shift_score: f32,
    /// Pesos de activación de los 6 expertos de interacción.
    pub active_experts: Vec<f32>,
    /// Indica si la inferencia proviene de la arquitectura JEV de 3 capas.
    pub is_jev: bool,
}

/// Ejecutor de bajo nivel del modelo neuronal remtk-juicio-moe-int8.onnx.
pub struct MoeStagesExecutor {
    session: Mutex<Session>,
    abstencion_threshold: f32,
}

impl MoeStagesExecutor {
    pub fn new<P: AsRef<Path>>(
        env: &Arc<Environment>,
        model_path: P,
        intra_threads: usize,
        inter_threads: usize,
        abstencion_threshold: f32,
    ) -> Result<Self, AppError> {
        let session = Session::builder(env)
            .map_err(|e| AppError::ModelLoad(format!("SessionBuilder MoE error: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .with_intra_threads(intra_threads)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .with_inter_threads(inter_threads)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| AppError::ModelLoad(format!("Fallo al cargar modelo MoE ONNX: {}", e)))?;

        Ok(Self {
            session: Mutex::new(session),
            abstencion_threshold,
        })
    }

    /// Ejecuta el grafo MoE recibiendo tensores 3D mediante vistas prestadas Zero-Copy.
    pub fn evaluate(
        &self,
        query_tokens: ArrayView3<f32>,
        context_tokens: ArrayView3<f32>,
        tools_tokens: ArrayView3<f32>,
    ) -> Result<MoeDecisionOutput, AppError> {
        let q_shape = [query_tokens.shape()[0], query_tokens.shape()[1], query_tokens.shape()[2]];
        let q_vec: Vec<f32> = query_tokens.iter().copied().collect();
        let q_tensor = Tensor::from_array((q_shape, q_vec))
            .map_err(|e| AppError::Inference(format!("Error en tensor query_tokens: {}", e)))?;

        let c_shape = [context_tokens.shape()[0], context_tokens.shape()[1], context_tokens.shape()[2]];
        let c_vec: Vec<f32> = context_tokens.iter().copied().collect();
        let c_tensor = Tensor::from_array((c_shape, c_vec))
            .map_err(|e| AppError::Inference(format!("Error en tensor context_tokens: {}", e)))?;

        let t_shape = [tools_tokens.shape()[0], tools_tokens.shape()[1], tools_tokens.shape()[2]];
        let t_vec: Vec<f32> = tools_tokens.iter().copied().collect();
        let t_tensor = Tensor::from_array((t_shape, t_vec))
            .map_err(|e| AppError::Inference(format!("Error en tensor tools_tokens: {}", e)))?;

        let mut session = self.session.lock().map_err(|_| {
            AppError::Internal("Fallo al adquirir cerrojo de sesión ONNX MoE".into())
        })?;

        let results = session
            .run(ort::inputs![
                "query_tokens" => q_tensor,
                "context_tokens" => c_tensor,
                "tools_tokens" => t_tensor,
            ])
            .map_err(|e| AppError::Inference(format!("Error ejecutando grafo MoE: {}", e)))?;

        let is_jev_arch = results.len() >= 7;

        let ranked_scores: Vec<f32>;
        let mut tool_choice_probs: Vec<f32> = Vec::new();
        let mut lang_probs = [1.0f32, 0.0f32];
        let mut estado_probs = [0.0f32; 4];
        let mut valid_noul = 1.0f32;
        let mut quality_score = 3.0f32;
        let mut abstencion_prob = 0.0f32;
        let mut topic_shift_score = 0.0f32;
        let mut active_experts = Vec::new();

        if is_jev_arch {
            // Arquitectura JEV MoE (7 salidas):
            // 0: lang_probs [1, 2] -> [ES, EN]
            // 1: context_choice_probs [1, 4] -> [TOOL, MEMORY, PREV_RESPONSE, INDEPENDENT]
            // 2: tool_choice_probs [num_tools]
            // 3: tool_affinity_scores [num_tools]
            // 4: decision_valid_noul [1, 1]
            // 5: decision_quality_score [1, 1]
            // 6: sparse_gates [1, 6]
            if let Ok((_s, raw)) = results[0].try_extract_tensor::<f32>() {
                for (i, val) in raw.iter().enumerate().take(2) {
                    lang_probs[i] = *val;
                }
            }

            if let Ok((_s, raw)) = results[1].try_extract_tensor::<f32>() {
                for (i, val) in raw.iter().enumerate().take(4) {
                    estado_probs[i] = *val;
                }
            }

            let (_shape, choice_slice) = results[2]
                .try_extract_tensor::<f32>()
                .map_err(|e| AppError::Inference(format!("Error al extraer tool_choice_probs: {}", e)))?;
            tool_choice_probs = choice_slice.to_vec();

            if let Ok((_s, aff_slice)) = results[3].try_extract_tensor::<f32>() {
                ranked_scores = aff_slice.to_vec();
            } else {
                ranked_scores = tool_choice_probs.clone();
            }

            if let Ok((_s, raw)) = results[4].try_extract_tensor::<f32>() {
                valid_noul = raw.first().copied().unwrap_or(1.0);
                abstencion_prob = (1.0 - valid_noul).max(estado_probs[3]);
            } else {
                abstencion_prob = estado_probs[3];
            }

            if let Ok((_s, raw)) = results[5].try_extract_tensor::<f32>() {
                quality_score = raw.first().copied().unwrap_or(0.0);
                let quality_norm = (quality_score / 3.0).clamp(0.0, 1.0);
                topic_shift_score = (1.0 - quality_norm).clamp(0.0, 1.0);
            }

            if let Ok((_s, raw)) = results[6].try_extract_tensor::<f32>() {
                active_experts = raw.to_vec();
            }
        } else {
            // Arquitectura Legacy MoE (5 salidas):
            // 0: ranked_scores [num_tools]
            // 1: estado_probs [1, 4]
            // 2: abstencion_prob [1, 1]
            // 3: topic_shift_score [1, 1]
            // 4: sparse_gates [1, 6]
            let (_shape, scores_slice) = results[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| AppError::Inference(format!("Error al extraer ranked_scores: {}", e)))?;
            ranked_scores = scores_slice.to_vec();
            tool_choice_probs = ranked_scores.clone();

            if results.len() > 1 {
                if let Ok((_s, raw)) = results[1].try_extract_tensor::<f32>() {
                    for (i, val) in raw.iter().enumerate().take(4) {
                        estado_probs[i] = *val;
                    }
                }
            }

            if results.len() > 2 {
                if let Ok((_s, raw)) = results[2].try_extract_tensor::<f32>() {
                    abstencion_prob = raw.first().copied().unwrap_or(0.0);
                    valid_noul = (1.0 - abstencion_prob).clamp(0.0, 1.0);
                }
            }

            if results.len() > 3 {
                if let Ok((_s, raw)) = results[3].try_extract_tensor::<f32>() {
                    topic_shift_score = raw.first().copied().unwrap_or(0.0);
                }
            }

            if results.len() > 4 {
                if let Ok((_s, raw)) = results[4].try_extract_tensor::<f32>() {
                    active_experts = raw.to_vec();
                }
            }
        }

        let max_idx = estado_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let ctx_route = if is_jev_arch {
            match max_idx {
                0 => "TOOL",
                1 => "MEMORY",
                2 => "PREV_RESPONSE",
                _ => "INDEPENDENT",
            }
        } else {
            match max_idx {
                0 => "TOOL",
                1 => "PREV_RESPONSE",
                2 => "PREV_RESPONSE",
                _ => "INDEPENDENT",
            }
        }
        .to_string();

        let estado_dominante = if is_jev_arch {
            match max_idx {
                0 => "TRANSICION",   // TOOL
                1 => "CONTINUACION", // MEMORY
                2 => "CONTINUACION", // PREV_RESPONSE
                _ => "ABANDONO",     // INDEPENDENT
            }
        } else {
            match max_idx {
                0 => "TRANSICION",
                1 => "CONTINUACION",
                2 => "CORRECCION",
                _ => "ABANDONO",
            }
        }
        .to_string();

        let debe_abstenerse = if is_jev_arch {
            ctx_route == "INDEPENDENT" || valid_noul < 0.50 || abstencion_prob >= self.abstencion_threshold
        } else {
            abstencion_prob >= self.abstencion_threshold
        };

        if active_experts.is_empty() {
            active_experts = vec![0.0; 6];
        }

        tracing::info!(
            target: "prediction_trace",
            "[trace:moe] arch={} tools_evaluated={} ctx_route={} nli_dominant={} route_probs=[TOOL={:.3}, MEM={:.3}, PREV={:.3}, INDEP={:.3}] valid_noul={:.4} quality={:.2}/3.0 abstencion_prob={:.4} (threshold={:.2}, abstain={}) lang=[ES={:.2}, EN={:.2}] experts={:?}",
            if is_jev_arch { "JEV_3LAYER" } else { "LEGACY_5OUT" },
            ranked_scores.len(),
            ctx_route,
            estado_dominante,
            estado_probs[0],
            estado_probs[1],
            estado_probs[2],
            estado_probs[3],
            valid_noul,
            quality_score,
            abstencion_prob,
            self.abstencion_threshold,
            debe_abstenerse,
            lang_probs[0],
            lang_probs[1],
            active_experts
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<f32>>()
        );

        Ok(MoeDecisionOutput {
            ranked_scores,
            tool_choice_probs,
            lang_probs,
            ctx_route,
            estado_probs,
            estado_dominante,
            valid_noul,
            quality_score,
            abstencion_prob,
            debe_abstenerse,
            topic_shift_score,
            active_experts,
            is_jev: is_jev_arch,
        })
    }
}
