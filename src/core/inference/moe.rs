use std::path::Path;
use std::sync::{Arc, Mutex};
use ndarray::ArrayView3;
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use crate::shared::error::AppError;

/// Salida procesada y calibrada del cabezal de decisión MoE de 3 etapas.
#[derive(Debug, Clone)]
pub struct MoeDecisionOutput {
    /// Puntuaciones calibradas [0.0, 1.0] para cada una de las K herramientas candidatas.
    pub ranked_scores: Vec<f32>,
    /// Probabilidades de estado conversacional NLI: [TRANSICION, CONTINUACION, CORRECCION, ABANDONO].
    pub estado_probs: [f32; 4],
    /// Estado NLI dominante del turno.
    pub estado_dominante: String,
    /// Probabilidad [0.0, 1.0] de abstención para retorno nulo.
    pub abstencion_prob: f32,
    /// Booleano de decisión de abstención.
    pub debe_abstenerse: bool,
    /// Puntuación de cambio de tópico.
    pub topic_shift_score: f32,
    /// Pesos de activación de los 6 expertos de interacción.
    pub active_experts: Vec<f32>,
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

        // 1. ranked_scores (dimensión: [num_tools])
        let (_shape, scores_slice) = results[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::Inference(format!("Error al extraer ranked_scores: {}", e)))?;
        let ranked_scores: Vec<f32> = scores_slice.to_vec();

        // 2. probabilidades de estado NLI (dimensión: [1, 4])
        let mut estado_probs = [0.0f32; 4];
        if results.len() > 1 {
            if let Ok((_s, raw)) = results[1].try_extract_tensor::<f32>() {
                for (i, val) in raw.iter().enumerate().take(4) {
                    estado_probs[i] = *val;
                }
            }
        }

        let max_idx = estado_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let estado_dominante = match max_idx {
            0 => "TRANSICION",
            1 => "CONTINUACION",
            2 => "CORRECCION",
            _ => "ABANDONO",
        }
        .to_string();

        // 3. abstencion_prob (dimensión: [1, 1])
        let mut abstencion_prob = 0.0f32;
        if results.len() > 2 {
            if let Ok((_s, raw)) = results[2].try_extract_tensor::<f32>() {
                abstencion_prob = raw.first().copied().unwrap_or(0.0);
            }
        }
        let debe_abstenerse = abstencion_prob >= self.abstencion_threshold;

        // 4. topic_shift_score (dimensión: [1, 1])
        let mut topic_shift_score = 0.0f32;
        if results.len() > 3 {
            if let Ok((_s, raw)) = results[3].try_extract_tensor::<f32>() {
                topic_shift_score = raw.first().copied().unwrap_or(0.0);
            }
        }

        // 5. active_experts (dimensión: [1, 6])
        let mut active_experts = Vec::new();
        if results.len() > 4 {
            if let Ok((_s, raw)) = results[4].try_extract_tensor::<f32>() {
                active_experts = raw.to_vec();
            }
        }
        if active_experts.is_empty() {
            active_experts = vec![0.0; 6];
        }

        tracing::info!(
            target: "prediction_trace",
            "[trace:moe] tools_evaluated={} nli_dominant={} nli_probs=[TRANS={:.3}, CONT={:.3}, CORR={:.3}, ABAN={:.3}] abstencion_prob={:.4} (threshold={:.2}, abstain={}) topic_shift={:.4} experts={:?}",
            ranked_scores.len(),
            estado_dominante,
            estado_probs[0],
            estado_probs[1],
            estado_probs[2],
            estado_probs[3],
            abstencion_prob,
            self.abstencion_threshold,
            debe_abstenerse,
            topic_shift_score,
            active_experts
                .iter()
                .map(|v| (v * 1000.0).round() / 1000.0)
                .collect::<Vec<f32>>()
        );

        Ok(MoeDecisionOutput {
            ranked_scores,
            estado_probs,
            estado_dominante,
            abstencion_prob,
            debe_abstenerse,
            topic_shift_score,
            active_experts,
        })
    }
}
