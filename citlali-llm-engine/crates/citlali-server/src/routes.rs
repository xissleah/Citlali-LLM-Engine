// crates/citlali-server/src/routes.rs
//! HTTP route handlers.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use tokio_stream::wrappers::ReceiverStream;

use citlali_model::model::Qwen3Model;
use citlali_model::tokenizer::Tokenizer;
use citlali_runtime::config::EngineConfig;
use citlali_runtime::generate::{generate_with_cache, PromptCache};
use citlali_runtime::sampler::SamplerConfig;

use crate::types::*;

/// 推理超时时间（秒）
const INFERENCE_TIMEOUT_SECS: u64 = 300;

pub struct AppState {
    pub model: Qwen3Model,
    pub tokenizer: Tokenizer,
    pub engine_config: EngineConfig,
    pub model_name: String,
    /// 并发控制：同时只允许 1 个推理任务
    pub infer_semaphore: Semaphore,
    /// Prompt Cache：缓存上一次推理的 KV 状态，前缀复用
    pub prompt_cache: std::sync::Mutex<Option<PromptCache>>,
}

/// 统一错误类型，返回 OpenAI 格式 JSON
pub struct AppError {
    status: StatusCode,
    message: String,
}

impl AppError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: msg.into() }
    }
    fn internal(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: msg.into() }
    }
    fn service_unavailable(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::SERVICE_UNAVAILABLE, message: msg.into() }
    }
}

/// 使用 tokenizer 自动检测的 chat template 拼装 prompt
fn build_prompt(messages: &[ChatMessage], tokenizer: &Tokenizer) -> String {
    let msgs: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    tokenizer.format_chat(&msgs)
}

/// 非流式响应
async fn chat_completions_blocking(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Json<ChatCompletionResponse>, AppError> {
    let prompt = build_prompt(&req.messages, &state.tokenizer);
    let prompt_tokens = state.tokenizer.encode(&prompt).len();

    let sampler = SamplerConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        stop_tokens: state.tokenizer.stop_token_ids.clone(),
        ..Default::default()
    };

    let _permit = state.infer_semaphore.acquire().await
        .map_err(|_| AppError::service_unavailable("Server is shutting down"))?;

    let model_name = state.model_name.clone();
    let state_for_infer = Arc::clone(&state);

    let result = tokio::time::timeout(
        Duration::from_secs(INFERENCE_TIMEOUT_SECS),
        tokio::task::spawn_blocking(move || {
            let mut cache = state_for_infer.prompt_cache.lock().unwrap().take();
            let result = generate_with_cache(
                &state_for_infer.model,
                &state_for_infer.tokenizer,
                &prompt,
                &sampler,
                &state_for_infer.engine_config,
                &mut cache,
                |_, _| {},
                |_| {},
            );
            *state_for_infer.prompt_cache.lock().unwrap() = cache;
            result
        }),
    )
    .await
    .map_err(|_| AppError::internal("Inference timed out"))?
    .map_err(|_| AppError::internal("Inference task panicked"))?;

    let completion_tokens = result.token_ids.len();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH).unwrap().as_secs();

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: now,
        model: model_name,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: result.text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = ApiError {
            error: ApiErrorDetail {
                message: self.message,
                error_type: "invalid_request_error".to_string(),
                code: None,
            },
        };
        (self.status, Json(body)).into_response()
    }
}

pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "owned_by": "citlali"
        }]
    }))
}

/// 统一入口：根据 stream 字段分发
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    if req.messages.is_empty() {
        return AppError::bad_request("messages must not be empty").into_response();
    }

    if req.stream {
        match chat_completions_stream(state, req).await {
            Ok(sse) => sse.into_response(),
            Err(e) => e.into_response(),
        }
    } else {
        match chat_completions_blocking(state, req).await {
            Ok(json) => json.into_response(),
            Err(e) => e.into_response(),
        }
    }
}

/// 流式响应：通过 SSE 逐 token 推送
async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Sse<ReceiverStream<Result<Event, std::convert::Infallible>>>, AppError> {
    let prompt = build_prompt(&req.messages, &state.tokenizer);

    let sampler = SamplerConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        stop_tokens: state.tokenizer.stop_token_ids.clone(),
        ..Default::default()
    };

    let _permit = state.infer_semaphore.acquire().await
        .map_err(|_| AppError::service_unavailable("Server is shutting down"))?;

    let model_name = state.model_name.clone();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH).unwrap().as_secs();

    // 用 channel 桥接 blocking 线程和 SSE stream
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    let state_for_infer = Arc::clone(&state);
    let id_clone = id.clone();
    let model_name_clone = model_name.clone();

    tokio::task::spawn_blocking(move || {
        // permit 随 move 进来，推理结束后自动释放
        let _permit = _permit;
        // 第一个 chunk：发送 role
        let first_chunk = ChatCompletionChunk {
            id: id_clone.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_name_clone.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        let data = serde_json::to_string(&first_chunk).unwrap();
        let _ = tx.blocking_send(Ok(Event::default().data(data)));

        // 推理，每个 token 发一个 chunk（处理 UTF-8 边界）
        let tokenizer = &state_for_infer.tokenizer;
        let mut token_buf: Vec<u32> = Vec::new();
        let mut sent_bytes: usize = 0;
        let mut cache = state_for_infer.prompt_cache.lock().unwrap().take();
        generate_with_cache(
            &state_for_infer.model,
            tokenizer,
            &prompt,
            &sampler,
            &state_for_infer.engine_config,
            &mut cache,
            |_, _| {},
            |token_id| {
                token_buf.push(token_id);
                let bytes = tokenizer.decode_to_bytes(&token_buf);
                let new_bytes = &bytes[sent_bytes..];
                let valid_up_to = match std::str::from_utf8(new_bytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid_up_to > 0 {
                    let text = unsafe {
                        std::str::from_utf8_unchecked(&new_bytes[..valid_up_to])
                    }.to_string();
                    sent_bytes += valid_up_to;
                    let chunk = ChatCompletionChunk {
                        id: id_clone.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: now,
                        model: model_name_clone.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                    };
                    let data = serde_json::to_string(&chunk).unwrap();
                    let _ = tx.blocking_send(Ok(Event::default().data(data)));
                }
            },
        );
        *state_for_infer.prompt_cache.lock().unwrap() = cache;

        // 结束 chunk
        let done_chunk = ChatCompletionChunk {
            id: id_clone,
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_name_clone,
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta { role: None, content: None },
                finish_reason: Some("stop".to_string()),
            }],
        };
        let data = serde_json::to_string(&done_chunk).unwrap();
        let _ = tx.blocking_send(Ok(Event::default().data(data)));
        // 发送 [DONE] 标记
        let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
