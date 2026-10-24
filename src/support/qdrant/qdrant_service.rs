use std::sync::Arc;
use crate::core::inference::engine::InferenceEngine;
use crate::shared::dto::{ScoredToolDto, SearchToolsRequest, SearchToolsResponse, ToolDefinitionDto};
use crate::shared::error::AppError;
use crate::support::registry::TenantCache;

/// Servicio de soporte para interacción y proxy con el motor de búsqueda Qdrant y BM25.
pub struct QdrantService {
    cache: Arc<TenantCache>,
    engine: Arc<InferenceEngine>,
}

impl QdrantService {
    pub fn new(cache: Arc<TenantCache>, engine: Arc<InferenceEngine>) -> Self {
        Self { cache, engine }
    }

    /// Indexa herramientas en el catálogo y genera sus representaciones vectoriales.
    pub fn upsert_tools(&self, tenant_id: &str, tools: &[ToolDefinitionDto]) -> Result<usize, AppError> {
        self.cache.upsert_tools(tenant_id, tools, &self.engine)
    }

    /// Conteo de herramientas registradas para un tenant.
    pub fn count_tools(&self, tenant_id: &str) -> usize {
        self.cache.count_tools(tenant_id)
    }

    /// Búsqueda y filtrado de herramientas para un tenant.
    pub fn search_tools(&self, payload: &SearchToolsRequest) -> SearchToolsResponse {
        let storage = self.cache.storage();
        let registry_opt = storage.get(&payload.tenant);

        let tools = match registry_opt {
            Some(registry) => {
                let candidate_indices = registry.retrieve_candidates(&payload.text, payload.limit);
                candidate_indices
                    .into_iter()
                    .map(|idx| {
                        let tool = &registry.tools[idx];
                        ScoredToolDto {
                            name: tool.metadata.name.clone(),
                            score: 0.90,
                        }
                    })
                    .collect()
            }
            None => Vec::new(),
        };

        SearchToolsResponse { tools }
    }

    /// Expansión de sinónimos para consultas de búsqueda.
    pub fn expand_synonyms(&self, text: &str) -> String {
        text.trim().to_string()
    }
}
