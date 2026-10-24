# Arquitectura 03: Pipeline MoE de 3 Etapas e Integración con `e5-small` en Rust

## 1. Visión Matemática del Pipeline de 2 Llamadas Unificadas

El motor de predicción ejecuta la inferencia mediante **dos llamadas secuenciales y deterministas a ONNX Runtime**, eliminando por completo los tres modelos antiguos dispersos (`ms-marco`, `mDeBERTa`, y múltiples llamadas NLI redundantes).

```
ENTRADA HTTP: text, keywords, intentContext, history, K candidatas
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LLAMADA 1: multilingual-e5-small (Extracción de Tensores a nivel de Token) │
│                                                                             │
│ • Entrada:  query: {text} + {keywords}  ──► (1, Lq, 384)                    │
│             query: {intentContext}      ──► (1, Lc, 384)                    │
│ • Salida:   Representaciones densas de tokens continuas en el espacio d=384 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                      query_tokens (1, Lq, 384)
                      context_tokens (1, Lc, 384)
                      tools_tokens (K, 96, 384) [Desde Caché RAM DashMap]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LLAMADA 2: remtk-juicio-moe-int8 (Cabezal Decisor de 3 Etapas en Grafo)     │
│                                                                             │
│ [ETAPA 1]: Compuerta Episódica NLI (TRANSICION / CONTINUACION / etc.)       │
│            Calcula peso alfa de fusión y limpia historial obsoleto.         │
│                                      │                                      │
│                                      ▼                                      │
│ [ETAPA 2]: Enrutador Top-2 + 6 Expertos de Interacción                      │
│            (Acción, Catálogo, Vago, Especificidad, Código, Adversarial)    │
│            Operador MaxSim bilineal aprendido (Einsum bld,ktd->bklt).       │
│                                      │                                      │
│                                      ▼                                      │
│ [ETAPA 3]: Set-Attention entre las K Herramientas                           │
│            Margen Delta 1-2, Entropía Shannon, Abstención, Scores [0.8-0.98]│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
RESPUESTA JSON: rankedScores, estadoDominante, debeAbstenerse, activeExperts
```

---

## 2. Topología Matemática del Grafo `remtk-juicio-moe-int8`

### 2.1. Etapa 1: Compuerta Episódica y Estado Conversacional NLI
Recibe $\mathbf{Q} \in \mathbb{R}^{1 \times L_q \times 384}$ y $\mathbf{C} \in \mathbb{R}^{1 \times L_c \times 384}$.
1. **Convolución Gated 1D (`kernel_size=3`) + LayerNorm:**
   $$\tilde{\mathbf{Q}} = \text{LayerNorm}\left(\mathbf{Q} + \text{GLU}(\text{Conv1D}(\mathbf{Q}))\right)$$
   $$\tilde{\mathbf{C}} = \text{LayerNorm}\left(\mathbf{C} + \text{GLU}(\text{Conv1D}(\mathbf{C}))\right)$$
2. **Atención Cruzada Query-Contexto:**
   $$\mathbf{A}_{qc} = \text{Softmax}\left(\frac{(\tilde{\mathbf{Q}}\mathbf{W}_q)(\tilde{\mathbf{C}}\mathbf{W}_k)^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{1 \times L_q \times L_c}$$
   $$\mathbf{M}_{qc} = \mathbf{A}_{qc}(\tilde{\mathbf{C}}\mathbf{W}_v) \in \mathbb{R}^{1 \times L_q \times 384}$$
3. **Clasificación NLI de 4 Estados:**
   $$\mathbf{u}_{\text{nli}} = \left[\mathbf{q}_{\text{pool}} \;\|\; \mathbf{c}_{\text{pool}} \;\|\; |\mathbf{q}_{\text{pool}} - \mathbf{c}_{\text{pool}}| \;\|\; (\mathbf{q}_{\text{pool}} \odot \mathbf{m}_{\text{pool}})\right] \in \mathbb{R}^{1536}$$
   $$\mathbf{p}_{\text{estado}} = \text{Softmax}(\mathbf{W}_{\text{nli}}\,\text{GELU}(\mathbf{W}_1 \mathbf{u}_{\text{nli}})) \in \mathbb{R}^4$$
   Donde las clases corresponden a `TRANSICION` ($p_0$), `CONTINUACION` ($p_1$), `CORRECCION` ($p_2$), `ABANDONO` ($p_3$).
4. **Compuerta Diferenciable de Fusión ($\alpha$):**
   $$\alpha = \sigma(\mathbf{w}_g^T \mathbf{p}_{\text{estado}} + b_g) \in [0, 1]$$
   $$\mathbf{Q}^* = \text{LayerNorm}(\tilde{\mathbf{Q}} + \alpha \cdot \mathbf{M}_{qc})$$

### 2.2. Etapa 2: Router Top-2 y 6 Expertos de Interacción
1. **Enrutamiento Top-2:**
   $$\mathbf{g} = \text{Top2Softmax}\left(\mathbf{W}_{\text{router}} [\mathbf{q}^* \;\|\; \mathbf{p}_{\text{estado}}]\right) \in \mathbb{R}^6$$
2. **Especialización de los 6 Expertos:**
   * **Experto 1:** Acción e Imperativos (APIs, creación, ejecución).
   * **Experto 2:** Catálogo y Búsqueda (filtros, directorios comerciales).
   * **Experto 3:** Intención Vaga e Implícita (mapeo de síntomas a diagnósticos técnicos).
   * **Experto 4:** Especificidad y Validación de Parámetros (penalización de tools genéricas).
   * **Experto 5:** Orquestación Multi-Herramienta y Código (flujos continuos en DAG).
   * **Experto 6:** Filtro Adversarial y Chitchat (bloqueo de negaciones y texto no operativo).
3. **Cálculo MaxSim Vectorizado dentro de cada Experto Activo:**
   Para $e \in \text{Top2}$:
   $$\mathbf{S}_e(k, i, j) = \frac{\langle \mathbf{Q}_e(1, i), \, \mathbf{T}_e(k, j) \rangle}{\|\mathbf{Q}_e(1, i)\|_2 \, \|\mathbf{T}_e(k, j)\|_2}$$
   $$\mathbf{h}_k = \sum_{e \in \text{Top2}} g_e \cdot \text{MLP}_e([\mathbf{q}_e^{\text{pool}} \;\|\; \mathbf{t}_{e,k}^{\text{pool}} \;\|\; \mu_{\text{fwd}, k} \;\|\; \mu_{\text{bwd}, k}]) \in \mathbb{R}^{128}$$

### 2.3. Etapa 3: Set-Transformer y Cabeza de Abstención
1. **Auto-atención entre Herramientas Candidatas:**
   $$\tilde{\mathbf{H}} = \text{MultiHeadSelfAttention}(\mathbf{H}, \mathbf{H}, \mathbf{H}) \in \mathbb{R}^{K \times 128}$$
2. **Scores Calibrados y Probabilidad de Abstención:**
   $$s_k = \sigma(\mathbf{w}_{\text{score}}^T \tilde{\mathbf{h}}_k + b_{\text{score}}) \in [0.80, 0.98]$$
   $$p_{\text{abstencion}} = \sigma\left(\text{MLP}_{\text{abs}}([s_{(1)}, \; \Delta_{1,2}, \; \mathcal{H}(\mathbf{s}), \; \mathbf{p}_{\text{estado}}])\right)$$

---

## 3. Implementación Completa en Rust con `ort` v2.0+

```rust
// src/engine/moe_engine.rs
use std::path::Path;
use std::sync::Arc;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, Session, SessionBuilder, Value};

pub struct MoeOutputs {
    pub ranked_scores: Vec<f32>,
    pub estado_probs: [f32; 4],
    pub estado_dominante: &'static str,
    pub abstencion_prob: f32,
    pub debe_abstenerse: bool,
    pub topic_shift_score: f32,
    pub active_experts: Vec<f32>,
}

pub struct InferenceEngine {
    _env: Arc<Environment>,
    session_e5: Session,
    session_moe: Session,
    abstencion_threshold: f32,
}

impl InferenceEngine {
    pub fn new(models_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let env = Arc::new(
            Environment::builder()
                .with_name("remtk-inference")
                .with_log_level(LoggingLevel::Warning)
                .build()?,
        );

        let e5_path = Path::new(models_dir).join("multilingual-e5-small-onnx/model.onnx");
        let moe_path = Path::new(models_dir).join("remtk-juicio-moe-int8/model_quantized.onnx");

        // Configuración de sesión con mmap y optimizaciones SIMD AVX2
        let session_e5 = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_inter_threads(1)?
            .with_model_from_file(&e5_path)?;

        let session_moe = SessionBuilder::new(&env)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .with_inter_threads(1)?
            .with_model_from_file(&moe_path)?;

        Ok(Self {
            _env: env,
            session_e5,
            session_moe,
            abstencion_threshold: 0.52,
        })
    }

    /// Ejecuta la Llamada 1: Extracción de tokens de e5-small
    pub fn extract_e5_tokens(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        seq_len: usize,
    ) -> Result<Array3<f32>, Box<dyn std::error::Error>> {
        let ids_view = ArrayView2::from_shape((1, seq_len), input_ids)?;
        let mask_view = ArrayView2::from_shape((1, seq_len), attention_mask)?;

        let ids_tensor = Value::from_array(ids_view)?;
        let mask_tensor = Value::from_array(mask_view)?;

        let outputs = self.session_e5.run(vec![ids_tensor, mask_tensor])?;
        
        // Extraer last_hidden_state de forma (1, seq_len, 384)
        let hidden_tensor: ort::OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let array = hidden_tensor.view().to_owned().into_shape((1, seq_len, 384))?;

        Ok(array)
    }

    /// Ejecuta la Llamada 2: Cabezal Decisor MoE de 3 Etapas
    pub fn evaluate_moe(
        &self,
        query_tokens: ArrayView3<f32>,
        context_tokens: ArrayView3<f32>,
        tools_tokens: ArrayView3<f32>,
    ) -> Result<MoeOutputs, Box<dyn std::error::Error>> {
        let q_tensor = Value::from_array(query_tokens)?;
        let c_tensor = Value::from_array(context_tokens)?;
        let t_tensor = Value::from_array(tools_tokens)?;

        let results = self.session_moe.run(vec![q_tensor, c_tensor, t_tensor])?;

        // 1. Extraer ranked_scores (forma: [num_tools])
        let scores_raw: ort::OrtOwnedTensor<f32, _> = results[0].try_extract()?;
        let ranked_scores: Vec<f32> = scores_raw.view().iter().copied().collect();

        // 2. Extraer probabilidades de estado NLI (forma: [1, 4])
        let estado_raw: ort::OrtOwnedTensor<f32, _> = results[1].try_extract()?;
        let mut estado_probs = [0.0f32; 4];
        for (i, val) in estado_raw.view().iter().enumerate().take(4) {
            estado_probs[i] = *val;
        }

        let max_idx = estado_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let estado_dominante = match max_idx {
            0 => "TRANSICION",
            1 => "CONTINUACION",
            2 => "CORRECCION",
            _ => "ABANDONO",
        };

        // 3. Extraer probabilidad de abstención (forma: [1, 1])
        let abs_raw: ort::OrtOwnedTensor<f32, _> = results[2].try_extract()?;
        let abstencion_prob = abs_raw.view().iter().copied().next().unwrap_or(0.0);
        let debe_abstenerse = abstencion_prob >= self.abstencion_threshold;

        // 4. Extraer topic_shift_score (forma: [1, 1])
        let ts_raw: ort::OrtOwnedTensor<f32, _> = results[3].try_extract()?;
        let topic_shift_score = ts_raw.view().iter().copied().next().unwrap_or(0.0);

        // 5. Extraer pesos de los 6 expertos (forma: [1, 6])
        let exp_raw: ort::OrtOwnedTensor<f32, _> = results[4].try_extract()?;
        let active_experts: Vec<f32> = exp_raw.view().iter().copied().collect();

        Ok(MoeOutputs {
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
```
