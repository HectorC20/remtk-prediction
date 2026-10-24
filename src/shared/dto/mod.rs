use serde::{Deserialize, Serialize};
use serde_json::Value;

// ============================================================================
// PUERTO 6776: PREDICT DTOs
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictRequest {
    pub tenant: String,
    #[serde(default)]
    pub session_id: String,
    pub text: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub key_remtk: Option<String>,
    #[serde(default)]
    pub prior_plan: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub intent_context: Option<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

fn default_source() -> String {
    "human".to_string()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RankedToolScore {
    pub name: String,
    pub score: f32,
    pub expert_affinity: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictResponse {
    pub tools: Vec<String>,
    pub complexity: String,
    pub model_size: String,
    pub ranked_scores: Vec<RankedToolScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_shift: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estado_nli: Option<String>,
}

// ----------------------------------------------------------------------------
// SPATIAL PREDICT DTOs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CursorCoords {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpatialPredictDto {
    pub query: String,
    #[serde(default)]
    pub cursor: Option<CursorCoords>,
    #[serde(default)]
    pub selected_id: Option<String>,
    #[serde(default)]
    pub view_bounds: Option<Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualToolPayload {
    pub tool: String,
    pub parameters: Value,
    pub confidence: f32,
    pub execution_time_ms: u64,
}

// ----------------------------------------------------------------------------
// FAST PATH DTOs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FastPathInput {
    pub tools: Vec<Value>,
    pub text: String,
    #[serde(default)]
    pub ranked_scores: Option<Vec<f32>>,
    #[serde(default)]
    pub has_pending_plan: Option<bool>,
    #[serde(default)]
    pub average_planner_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FastPathDecision {
    pub action: String,
    pub skip_planner: bool,
    pub reason: String,
    pub estimated_tokens_saved: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direct_tool: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_chat_context: Option<Value>,
}

// ----------------------------------------------------------------------------
// MEMORY PREDICT DTOs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryPredictRequest {
    #[serde(default, alias = "chatId")]
    pub session_id: String,
    #[serde(alias = "tenant")]
    pub user_id: String,
    pub text: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryItemDto {
    pub id: String,
    pub content: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryPredictResponse {
    pub memories: Vec<MemoryItemDto>,
    pub count: usize,
}

// ----------------------------------------------------------------------------
// INTEREST PREDICT DTOs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InterestPredictionInput {
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub tenant: Option<String>,
    pub text: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub anchors: Option<Vec<String>>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InterestMatch {
    pub anchor: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InterestPredictionResult {
    pub interest_score: f32,
    pub topic: Option<String>,
    pub topic_score: f32,
    pub intent_score: f32,
    pub matches: Vec<InterestMatch>,
    pub method: String,
}

// ----------------------------------------------------------------------------
// MEMORY JUDGE DTOs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct JudgeMemoryRequest {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JudgeMemoryResponse {
    pub is_noise: bool,
    pub noise_score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
}

// ----------------------------------------------------------------------------
// HISTORY REGIME DTOs
// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryMessageDto {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryRegimeRequest {
    pub prompt: String,
    pub history: Vec<HistoryMessageDto>,
    #[serde(default)]
    pub tenant: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryRegimeResponse {
    pub regime: String,
    pub topic_shift: f32,
    pub anaphora_detected: bool,
    pub dependency_score: f32,
    pub selected_indices: Vec<usize>,
    pub reason: String,
}

// ============================================================================
// PUERTO 6777: EMBED DTOs
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct EmbedRequest {
    pub text: String,
    #[serde(default = "default_embed_size")]
    pub size: String,
}

fn default_embed_size() -> String {
    "small".to_string()
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EmbedResponse {
    pub embedding: Vec<f32>,
    pub model: String,
    pub dim: usize,
    pub model_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfoResponse {
    pub model: String,
    pub dim: usize,
    pub model_path: String,
}

// ============================================================================
// PUERTO 6775: QDRANT / TOOLS SYNC DTOs
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDefinitionDto {
    pub name: String,
    #[serde(default)]
    pub group: String,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub intent_summary: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpsertToolsRequest {
    pub tenant: String,
    pub tools: Vec<ToolDefinitionDto>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpsertToolsResponse {
    pub indexed: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolsCountResponse {
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchToolsRequest {
    pub tenant: String,
    pub text: String,
    #[serde(default = "default_search_limit")]
    pub limit: usize,
}

fn default_search_limit() -> usize {
    20
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoredToolDto {
    pub name: String,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchToolsResponse {
    pub tools: Vec<ScoredToolDto>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct DebugResponse {
    pub status: &'static str,
    pub memory_rss_mb: f32,
    pub active_tenants: usize,
    pub total_cached_tools: usize,
    pub onnx_runtime_version: &'static str,
}
