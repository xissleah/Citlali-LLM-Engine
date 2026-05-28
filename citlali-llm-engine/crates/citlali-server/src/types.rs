// crates/citlali-server/src/types.rs
//! OpenAI Chat Completions API 兼容的请求/响应类型

use serde::{Deserialize, Serialize};

/// 聊天消息
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// POST /v1/chat/completions 请求体
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// 模型名称（当前忽略，只有一个模型）
    #[serde(default)]
    pub model: String,
    /// 消息列表
    pub messages: Vec<ChatMessage>,
    /// 最大生成 token 数
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// 是否流式返回
    #[serde(default)]
    pub stream: bool,
    /// Temperature: 0.0 = greedy, >0 增加随机性
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Top-P (Nucleus Sampling): 1.0 = 不限制
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-K: 0 = 不限制
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Repetition Penalty: 1.0 = 不惩罚
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

fn default_max_tokens() -> usize { 512 }
fn default_temperature() -> f32 { 0.7 }
fn default_top_p() -> f32 { 0.9 }
fn default_top_k() -> usize { 40 }
fn default_repetition_penalty() -> f32 { 1.1 }

/// 响应中的 choice
#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// 用量统计
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// POST /v1/chat/completions 响应体
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// SSE 流式响应中的 chunk
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// OpenAI 格式错误响应
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: Option<String>,
}
