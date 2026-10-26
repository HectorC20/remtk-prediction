use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;
use half::f16;
use ndarray::Array3;

use crate::core::inference::engine::{InferenceEngine, EMBED_DIM};
use crate::core::inference::tokenizer::DEFAULT_MAX_TOOL_LEN;
use crate::shared::dto::ToolDefinitionDto;
use crate::shared::error::AppError;
use crate::shared::tensor::cosine_similarity_normalized;

/// Herramienta cacheada en memoria con tensor en precisión FP16 (73.7 KB) y embedding de referencia E5.
#[derive(Clone)]
pub struct CachedTool {
    pub metadata: ToolDefinitionDto,
    /// Vector de 96 * 384 = 36,864 flotantes FP16 (73,728 bytes).
    pub tokens_fp16: Vec<f16>,
    /// Longitud efectiva de tokens antes de padding.
    pub seq_len: usize,
    /// Vector normalizado L2 de 384 dimensiones para similitud semántica directa e5-small.
    pub embedding: Vec<f32>,
}

/// Registro en memoria de herramientas para un inquilino.
pub struct ToolRegistry {
    pub tools: Vec<CachedTool>,
    pub name_to_index: HashMap<String, usize>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            name_to_index: HashMap::new(),
        }
    }

    pub fn count(&self) -> usize {
        self.tools.len()
    }

    pub fn get_by_name(&self, name: &str) -> Option<&CachedTool> {
        self.name_to_index.get(name).map(|&idx| &self.tools[idx])
    }

    /// Ensambla el tensor tridimensional de herramientas candidatas (K, 96, 384) en FP32.
    pub fn assemble_candidates_tensor(&self, indices: &[usize]) -> Array3<f32> {
        let k = indices.len();
        let mut tensor = Array3::<f32>::zeros((k, DEFAULT_MAX_TOOL_LEN, EMBED_DIM));

        for (row, &tool_idx) in indices.iter().enumerate() {
            if tool_idx < self.tools.len() {
                let tool = &self.tools[tool_idx];
                let limit = tool.tokens_fp16.len().min(DEFAULT_MAX_TOOL_LEN * EMBED_DIM);
                for i in 0..tool.seq_len {
                    for d in 0..EMBED_DIM {
                        let flat_idx = i * EMBED_DIM + d;
                        if flat_idx < limit {
                            tensor[[row, i, d]] = tool.tokens_fp16[flat_idx].to_f32();
                        }
                    }
                }
            }
        }

        tensor
    }

    /// Búsqueda semántica densa vectorial sobre el catálogo para fase de recall (Bi-Encoder E5).
    pub fn retrieve_candidates_semantic(&self, query_emb: &[f32], limit: usize) -> Vec<usize> {
        if self.tools.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .tools
            .iter()
            .enumerate()
            .map(|(idx, tool)| {
                let sim = cosine_similarity_normalized(query_emb, &tool.embedding);
                (idx, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let effective_limit = limit.min(self.tools.len());
        scored.iter().take(effective_limit).map(|(idx, _)| *idx).collect()
    }

    /// Búsqueda y filtrado rápido léxico sobre el catálogo para fase de recall.
    pub fn retrieve_candidates(&self, query: &str, limit: usize) -> Vec<usize> {
        if self.tools.is_empty() {
            return Vec::new();
        }

        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| s.len() > 2)
            .collect();

        if query_terms.is_empty() {
            return (0..self.tools.len().min(limit)).collect();
        }

        let mut scored: Vec<(usize, f32)> = self
            .tools
            .iter()
            .enumerate()
            .map(|(idx, tool)| {
                let mut score = 0.0f32;
                let name_lower = tool.metadata.name.to_lowercase();
                let desc_lower = tool.metadata.description.to_lowercase();
                let summary_lower = tool.metadata.intent_summary.to_lowercase();

                for term in &query_terms {
                    if name_lower.contains(term) {
                        score += 3.0;
                    }
                    if summary_lower.contains(term) {
                        score += 2.0;
                    }
                    if desc_lower.contains(term) {
                        score += 1.0;
                    }
                }

                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let effective_limit = limit.min(self.tools.len());
        let nonzero_matches = scored.iter().filter(|(_, s)| *s > 0.0).count();
        let top_lexical: Vec<String> = scored
            .iter()
            .take(5)
            .filter(|(_, s)| *s > 0.0)
            .map(|(idx, s)| format!("{}({:.1})", self.tools[*idx].metadata.name, s))
            .collect();

        tracing::info!(
            target: "prediction_trace",
            "[trace:recall] catalog_size={} terms={:?} nonzero_lexical={} limit={} top_lexical=[{}]",
            self.tools.len(),
            query_terms,
            nonzero_matches,
            effective_limit,
            top_lexical.join(", ")
        );

        scored.iter().take(effective_limit).map(|(idx, _)| *idx).collect()
    }
}

/// Almacenamiento concurrente global indexado por TenantId.
pub type GlobalToolCache = Arc<DashMap<String, Arc<ToolRegistry>>>;

/// Servicio de soporte para administración de la caché multi-tenant.
pub struct TenantCache {
    storage: GlobalToolCache,
}

impl TenantCache {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(DashMap::new()),
        }
    }

    pub fn storage(&self) -> GlobalToolCache {
        self.storage.clone()
    }

    pub fn count_tools(&self, tenant_id: &str) -> usize {
        self.storage
            .get(tenant_id)
            .map(|r| r.count())
            .unwrap_or(0)
    }

    /// Indexa herramientas en memoria comprimiendo los tensores a media precisión (FP16).
    pub fn upsert_tools(
        &self,
        tenant_id: &str,
        definitions: &[ToolDefinitionDto],
        engine: &InferenceEngine,
    ) -> Result<usize, AppError> {
        let mut new_tools = Vec::with_capacity(definitions.len());
        let mut name_to_index = HashMap::with_capacity(definitions.len());

        for (idx, meta) in definitions.iter().enumerate() {
            let canonical_text = format!(
                "passage: {} | {} / {} | {} | {} | tags: {}",
                meta.name,
                meta.group,
                meta.category,
                meta.intent_summary,
                meta.description,
                meta.tags.join(", ")
            );

            let tool_emb = engine.embed_vector(&canonical_text)?;
            let tokens_2d = engine.extract_tool_tokens(&canonical_text, DEFAULT_MAX_TOOL_LEN)?;
            let seq_len = tokens_2d.shape()[0].min(DEFAULT_MAX_TOOL_LEN);

            let mut fp16_buffer = Vec::with_capacity(DEFAULT_MAX_TOOL_LEN * EMBED_DIM);
            for i in 0..seq_len {
                for d in 0..EMBED_DIM {
                    fp16_buffer.push(f16::from_f32(tokens_2d[[i, d]]));
                }
            }
            while fp16_buffer.len() < DEFAULT_MAX_TOOL_LEN * EMBED_DIM {
                fp16_buffer.push(f16::ZERO);
            }

            new_tools.push(CachedTool {
                metadata: meta.clone(),
                tokens_fp16: fp16_buffer,
                seq_len,
                embedding: tool_emb,
            });

            name_to_index.insert(meta.name.clone(), idx);
        }

        let total_indexed = new_tools.len();
        let registry = ToolRegistry {
            tools: new_tools,
            name_to_index,
        };

        self.storage.insert(tenant_id.to_string(), Arc::new(registry));
        Ok(total_indexed)
    }
}
