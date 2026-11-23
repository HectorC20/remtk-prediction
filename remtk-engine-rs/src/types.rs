use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinitionDto {
    pub id: Option<String>,
    pub name: String,
    pub group: Option<String>,
    pub category: Option<String>,
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub intent_summary: Option<String>,
    pub input_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub prerequisites: Vec<String>,
    #[serde(default)]
    pub conflicts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegisterToolsPayload {
    pub tenant: Option<String>,
    pub tools: Vec<ToolDefinitionDto>,
    pub agent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterToolsResponse {
    pub indexed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsCountResponse {
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsReadyResponse {
    pub ready: bool,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
    pub tools_count: usize,
    pub global_catalog_size: usize,
    pub tenant_isolated: bool,
    pub runtime: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LedgerItem {
    pub query: Option<String>,
    pub tool_id: Option<String>,
    #[serde(default)]
    pub is_mutable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictRequest {
    pub text: String,
    pub tenant: Option<String>,
    pub agent_id: Option<String>,
    pub context: Option<String>,
    #[serde(default)]
    pub history: Vec<HistoryMessage>,
    pub rejected_intent: Option<String>,
    #[serde(default)]
    pub ledger: Vec<LedgerItem>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictedToolOut {
    pub id: String,
    pub name: String,
    pub group: String,
    pub category: String,
    pub description: String,
    pub tags: Vec<String>,
    pub intent_summary: String,
    pub input_schema: serde_json::Value,
    pub prerequisites: Vec<String>,
    pub conflicts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDto {
    pub nodes: Vec<String>,
    pub edges: Vec<String>,
    pub execution_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FileCommandModeDto {
    pub enabled: bool,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictDebugDto {
    pub decision: String,
    pub top_tool: String,
    pub top_3_tools: Vec<String>,
    pub master_gate: f32,
    pub p_sink: f32,
    pub p_noul_intent: f32,
    pub p_noul_coverage: f32,
    pub redundancy_prob: f32,
    pub is_abstain: bool,
    pub latency_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictResponse {
    pub tools: Vec<PredictedToolOut>,
    pub complexity: String,
    pub model_size: String,
    pub ranked_scores: Vec<f32>,
    pub calibrated_scores: Vec<f32>,
    pub graph: GraphDto,
    pub file_command_mode: FileCommandModeDto,
    pub debug: PredictDebugDto,
}
