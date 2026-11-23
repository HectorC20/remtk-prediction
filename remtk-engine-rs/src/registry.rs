use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use dashmap::DashMap;
use candle_core::{DType, Device, Result, Tensor};
use crate::types::ToolDefinitionDto;

#[derive(Debug, Clone)]
pub struct TenantCatalog {
    pub tool_ids: Vec<String>,
    pub tools_dict: HashMap<String, ToolDefinitionDto>,
    pub tool_tensors: Tensor, // [N, 64, 384]
    pub tool_pooled: Tensor,  // [N, 384]
}

pub struct TenantRegistry {
    pub tenants: DashMap<String, Arc<TenantCatalog>>,
    pub global_tool_ids: Vec<String>,
    pub global_catalog: HashMap<String, serde_json::Value>,
    pub global_id_to_idx: HashMap<String, usize>,
    pub global_tensors: Tensor, // [400, 64, 384]
    pub global_pooled: Tensor,  // [400, 384]
}

impl TenantRegistry {
    pub fn load_from_dir(model_dir: &Path, dev: &Device) -> Result<Self> {
        let meta_path = model_dir.join("catalog_metadata.json");
        let tensors_path = model_dir.join("catalog_e5_tensors.safetensors");

        println!("[Registry] Loading global catalog metadata from {:?}", meta_path);
        let meta_str = std::fs::read_to_string(&meta_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read metadata: {}", e)))?;
        let meta_json: serde_json::Value = serde_json::from_str(&meta_str)
            .map_err(|e| candle_core::Error::Msg(format!("Invalid metadata JSON: {}", e)))?;

        let tool_ids_raw = meta_json["all_tool_ids"].as_array()
            .ok_or_else(|| candle_core::Error::Msg("Missing 'all_tool_ids'".into()))?;
        let global_tool_ids: Vec<String> = tool_ids_raw
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let mut global_id_to_idx = HashMap::new();
        for (i, id) in global_tool_ids.iter().enumerate() {
            global_id_to_idx.insert(id.clone(), i);
        }

        let global_catalog_raw = meta_json["catalog"].as_object()
            .ok_or_else(|| candle_core::Error::Msg("Missing 'catalog'".into()))?;
        let mut global_catalog = HashMap::new();
        for (k, v) in global_catalog_raw {
            global_catalog.insert(k.clone(), v.clone());
        }

        println!("[Registry] Loading precalculated E5 tensors from {:?}", tensors_path);
        let sf = candle_core::safetensors::load(&tensors_path, dev)?;
        let global_tensors = sf.get("catalog_tool_tensors")
            .ok_or_else(|| candle_core::Error::Msg("Missing catalog_tool_tensors".into()))?
            .to_dtype(DType::F32)?; // [400, 64, 384]

        // Compute global pooled vectors
        // norm per token across dim=-1
        let norms = global_tensors.sqr()?.sum_keepdim(2)?.sqrt()?; // [400, 64, 1]
        let mask = norms.gt(1e-5)?.to_dtype(DType::F32)?;
        let one_scalar = Tensor::new(&[1.0f32], dev)?;
        let eps_scalar = Tensor::new(&[1e-6f32], dev)?;
        let lens = mask.sum_keepdim(1)?.broadcast_maximum(&one_scalar)?;
        let masked = global_tensors.broadcast_mul(&mask)?;
        let sum_pooled = masked.sum_keepdim(1)?.squeeze(1)?; // [400, 384]
        let mean_pooled = sum_pooled.broadcast_div(&lens.squeeze(1)?)?;
        let p_norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?.broadcast_maximum(&eps_scalar)?;
        let global_pooled = mean_pooled.broadcast_div(&p_norm)?; // [400, 384]

        println!("[Registry] Global catalog ready: {} tools indexed.", global_tool_ids.len());

        Ok(Self {
            tenants: DashMap::new(),
            global_tool_ids,
            global_catalog,
            global_id_to_idx,
            global_tensors,
            global_pooled,
        })
    }

    pub fn extract_scope_key(tenant: Option<&str>, agent_id: Option<&str>, header_key: Option<&str>) -> String {
        let t = tenant
            .or(header_key)
            .unwrap_or("default")
            .trim();
        let a = agent_id.unwrap_or("").trim();

        if t.contains("::") && a.is_empty() {
            return t.to_string();
        }
        if !a.is_empty() {
            let clean_t = t.split("::").next().unwrap_or("default").trim();
            return format!("{}::{}", clean_t, a);
        }
        format!("{}::", t)
    }

    pub fn register(&self, scope_key: &str, tools: Vec<ToolDefinitionDto>, dev: &Device) -> Result<usize> {
        let mut tool_ids = Vec::new();
        let mut tools_dict = HashMap::new();
        let mut tensor_indices = Vec::new();

        for t in tools {
            let tid = t.id.as_deref().unwrap_or(&t.name).trim().to_string();
            if tid.is_empty() {
                continue;
            }

            if let Some(&idx) = self.global_id_to_idx.get(&tid) {
                tensor_indices.push(idx);
            } else {
                // Default fallback to index 0 if not precomputed
                tensor_indices.push(0);
            }

            tools_dict.insert(tid.clone(), t);
            tool_ids.push(tid);
        }

        let n = tool_ids.len();
        if n == 0 {
            let empty_tensors = Tensor::zeros((0, 64, 384), DType::F32, dev)?;
            let empty_pooled = Tensor::zeros((0, 384), DType::F32, dev)?;
            self.tenants.insert(scope_key.to_string(), Arc::new(TenantCatalog {
                tool_ids,
                tools_dict,
                tool_tensors: empty_tensors,
                tool_pooled: empty_pooled,
            }));
            return Ok(0);
        }

        let idx_tensor = Tensor::from_slice(&tensor_indices.iter().map(|&i| i as u32).collect::<Vec<_>>(), (n,), dev)?;
        let tool_tensors = self.global_tensors.index_select(&idx_tensor, 0)?;
        let tool_pooled = self.global_pooled.index_select(&idx_tensor, 0)?;

        let catalog = TenantCatalog {
            tool_ids,
            tools_dict,
            tool_tensors,
            tool_pooled,
        };

        self.tenants.insert(scope_key.to_string(), Arc::new(catalog));
        println!("[Registry] Scope '{}' registered with {} tools.", scope_key, n);
        Ok(n)
    }

    pub fn get_tenant(&self, scope_key: &str) -> Option<Arc<TenantCatalog>> {
        if let Some(entry) = self.tenants.get(scope_key) {
            return Some(entry.clone());
        }
        let clean_t = scope_key.split("::").next().unwrap_or("").trim();
        let alt_key = format!("{}::", clean_t);
        self.tenants.get(&alt_key).map(|entry| entry.clone())
    }

    pub fn count_tools(&self, scope_key: &str) -> usize {
        if let Some(tenant) = self.get_tenant(scope_key) {
            tenant.tool_ids.len()
        } else {
            self.global_tool_ids.len()
        }
    }
}
