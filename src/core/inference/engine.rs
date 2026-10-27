use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use ndarray::{Array2, Array3, ArrayView3};
use ort::environment::Environment;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tracing::info;

use crate::core::inference::moe::{MoeDecisionOutput, MoeStagesExecutor};
use crate::core::inference::tokenizer::{E5Tokenizer, DEFAULT_MAX_QUERY_LEN};
use crate::shared::config::AppConfig;
use crate::shared::error::AppError;

pub const EMBED_DIM: usize = 384;

/// Motor central de inferencia matemática ONNX desacoplado de la capa de transporte.
pub struct InferenceEngine {
    _env: Arc<Environment>,
    session_e5: Mutex<Session>,
    moe_executor: MoeStagesExecutor,
    tokenizer: E5Tokenizer,
    models_path: PathBuf,
    default_model_size: String,
}

impl InferenceEngine {
    /// Inicializa las sesiones neuronales de e5-small y remtk-juicio-moe con mmap.
    pub fn new(config: &AppConfig) -> Result<Self, AppError> {
        let env = Arc::new(
            ort::init()
                .with_name("remtk-core-engine")
                .build()
                .map_err(|e| AppError::ModelLoad(format!("ORT Environment build error: {}", e)))?,
        );

        let models_dir = &config.models_path;
        let e5_model_file = models_dir.join("multilingual-e5-small-onnx/model.onnx");
        let e5_tokenizer_file = models_dir.join("multilingual-e5-small-onnx/tokenizer.json");
        let jev_quantized = models_dir.join("remtk-juicio-jev-moe-int8/model_quantized.onnx");
        let moe_quantized = models_dir.join("remtk-juicio-moe-int8/model_quantized.onnx");
        let moe_fp32 = models_dir.join("remtk-juicio-moe-int8/model_fp32.onnx");
        let moe_model_file = if jev_quantized.exists() {
            info!("InferenceEngine: Detectado modelo JEV MoE en {}", jev_quantized.display());
            jev_quantized
        } else if moe_quantized.exists() {
            moe_quantized
        } else {
            moe_fp32
        };

        info!("InferenceEngine: Cargando tokenizer desde {}", e5_tokenizer_file.display());
        let tokenizer = E5Tokenizer::from_file(&e5_tokenizer_file)?;

        info!("InferenceEngine: Cargando e5-small ONNX desde {}", e5_model_file.display());
        let session_e5 = Session::builder(&env)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .with_intra_threads(config.intra_op_threads)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .with_inter_threads(config.inter_op_threads)
            .map_err(|e| AppError::ModelLoad(e.to_string()))?
            .commit_from_file(&e5_model_file)
            .map_err(|e| AppError::ModelLoad(format!("Error cargando e5-small: {}", e)))?;

        info!("InferenceEngine: Cargando MoE INT8 desde {}", moe_model_file.display());
        let moe_executor = MoeStagesExecutor::new(
            &env,
            &moe_model_file,
            config.intra_op_threads,
            config.inter_op_threads,
            config.abstencion_threshold,
        )?;

        Ok(Self {
            _env: env,
            session_e5: Mutex::new(session_e5),
            moe_executor,
            tokenizer,
            models_path: models_dir.clone(),
            default_model_size: config.model_size.clone(),
        })
    }

    /// Extrae el tensor de tokens de consulta (1, seq_len, 384) normalizado L2 por token con máscara aplicada.
    pub fn extract_query_tokens(&self, text: &str, max_len: usize) -> Result<Array3<f32>, AppError> {
        let tokenized = self.tokenizer.encode_query(text, max_len)?;
        self.run_e5_tokens_normalized(&tokenized.input_ids, &tokenized.attention_mask, tokenized.seq_len)
    }

    /// Extrae el tensor de tokens de contexto histórico o intentContext normalizado L2 por token.
    pub fn extract_context_tokens(&self, text: &str, max_len: usize) -> Result<Array3<f32>, AppError> {
        let tokenized = self.tokenizer.encode_passage(text, max_len)?;
        self.run_e5_tokens_normalized(&tokenized.input_ids, &tokenized.attention_mask, tokenized.seq_len)
    }

    /// Extrae los tokens de una herramienta normalizados L2 por token para su pre-cálculo en caché.
    pub fn extract_tool_tokens(&self, text: &str, max_len: usize) -> Result<Array2<f32>, AppError> {
        let tokenized = self.tokenizer.encode_passage(text, max_len)?;
        let tokens_3d = self.run_e5_tokens_normalized(&tokenized.input_ids, &tokenized.attention_mask, tokenized.seq_len)?;
        
        let seq_len = tokenized.seq_len;
        let mut result = Array2::<f32>::zeros((seq_len, EMBED_DIM));
        for i in 0..seq_len {
            for d in 0..EMBED_DIM {
                result[[i, d]] = tokens_3d[[0, i, d]];
            }
        }
        Ok(result)
    }

    /// Ejecuta e5-small y normaliza cada token individualmente con L2 + máscara de atención (estándar JEV MoE).
    pub fn run_e5_tokens_normalized(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
        seq_len: usize,
    ) -> Result<Array3<f32>, AppError> {
        let mut array = self.run_e5_tokens(input_ids, attention_mask, seq_len)?;
        for i in 0..seq_len {
            let mask_val = attention_mask[[0, i]] as f32;
            if mask_val > 0.0 {
                let mut norm_sq = 0.0f32;
                for d in 0..EMBED_DIM {
                    let v = array[[0, i, d]];
                    norm_sq += v * v;
                }
                let norm = norm_sq.sqrt().max(1e-12);
                for d in 0..EMBED_DIM {
                    array[[0, i, d]] = (array[[0, i, d]] / norm) * mask_val;
                }
            } else {
                for d in 0..EMBED_DIM {
                    array[[0, i, d]] = 0.0;
                }
            }
        }
        Ok(array)
    }

    /// Ejecuta el cabezal MoE recibiendo tensores 3D en formato Zero-Copy ArrayView3.
    pub fn evaluate_moe(
        &self,
        query_tokens: ArrayView3<f32>,
        context_tokens: ArrayView3<f32>,
        tools_tokens: ArrayView3<f32>,
    ) -> Result<MoeDecisionOutput, AppError> {
        self.moe_executor.evaluate(query_tokens, context_tokens, tools_tokens)
    }

    /// Ejecuta la sesión e5-small para obtener el tensor last_hidden_state (1, seq_len, 384).
    pub fn run_e5_tokens(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
        seq_len: usize,
    ) -> Result<Array3<f32>, AppError> {
        let ids_vec: Vec<i64> = input_ids.iter().copied().collect();
        let mask_vec: Vec<i64> = attention_mask.iter().copied().collect();
        let type_ids_vec = vec![0i64; seq_len];

        let tensor_ids = Tensor::from_array(([1, seq_len], ids_vec))
            .map_err(|e| AppError::Inference(format!("Value input_ids error: {}", e)))?;
        let tensor_mask = Tensor::from_array(([1, seq_len], mask_vec))
            .map_err(|e| AppError::Inference(format!("Value attention_mask error: {}", e)))?;
        let tensor_type_ids = Tensor::from_array(([1, seq_len], type_ids_vec))
            .map_err(|e| AppError::Inference(format!("Value token_type_ids error: {}", e)))?;

        let mut session = self.session_e5.lock().map_err(|_| {
            AppError::Internal("Fallo al adquirir cerrojo de sesión e5-small".into())
        })?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => tensor_ids,
                "attention_mask" => tensor_mask,
                "token_type_ids" => tensor_type_ids,
            ])
            .map_err(|e| AppError::Inference(format!("Error en sesión e5-small: {}", e)))?;

        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::Inference(format!("Error extrayendo tensor e5: {}", e)))?;

        let array = ndarray::ArrayView3::from_shape((1, seq_len, EMBED_DIM), data)
            .map_err(|e| AppError::Inference(format!("Error reformateando salida e5: {}", e)))?
            .to_owned();

        Ok(array)
    }

    /// Genera un vector normalizado L2 de 384 dimensiones a partir de texto crudo usando mean pooling.
    pub fn embed_vector(&self, text: &str) -> Result<Vec<f32>, AppError> {
        let tokenized = self.tokenizer.encode_query(text, DEFAULT_MAX_QUERY_LEN)?;
        let seq_len = tokenized.seq_len;
        let hidden_states = self.run_e5_tokens(&tokenized.input_ids, &tokenized.attention_mask, seq_len)?;

        let mut pooled = vec![0.0f32; EMBED_DIM];
        let mut mask_sum = 0.0f32;

        for i in 0..seq_len {
            let mask_val = tokenized.attention_mask[[0, i]] as f32;
            if mask_val > 0.0 {
                mask_sum += mask_val;
                for d in 0..EMBED_DIM {
                    pooled[d] += hidden_states[[0, i, d]] * mask_val;
                }
            }
        }

        if mask_sum > 0.0 {
            for d in 0..EMBED_DIM {
                pooled[d] /= mask_sum;
            }
        }

        crate::shared::tensor::l2_normalize(&mut pooled);
        Ok(pooled)
    }

    pub fn tokenizer(&self) -> &E5Tokenizer {
        &self.tokenizer
    }

    pub fn default_model_size(&self) -> &str {
        &self.default_model_size
    }

    pub fn models_path(&self) -> &Path {
        &self.models_path
    }
}
