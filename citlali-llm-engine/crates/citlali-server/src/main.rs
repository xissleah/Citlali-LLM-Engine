// crates/citlali-server/src/main.rs
//! Citlali HTTP Server — OpenAI 兼容 API
//! 用法: citlali-server <path-to-gguf> [--port 8080] [--host 127.0.0.1]

mod routes;
mod types;

use axum::{
    routing::{get, post},
    Router,
};
use std::env;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};

use citlali_model::model::Qwen3Model;
use citlali_runtime::config::EngineConfig;
use routes::AppState;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: citlali-server <path-to-gguf> [--port 8080] [--host 127.0.0.1]");
        std::process::exit(1);
    }

    let path = &args[1];
    let port = parse_arg(&args, "--port").unwrap_or(8080u16);
    let host = parse_str_arg(&args, "--host")
        .unwrap_or_else(|| "127.0.0.1".to_string());

    println!("Citlali Server 启动中...");
    println!("加载模型: {} ...", path);

    let start = Instant::now();
    let (model, tokenizer) =
        Qwen3Model::load_with_tokenizer(path).expect("加载模型失败");
    println!("模型加载完成 ({:.2}s)", start.elapsed().as_secs_f32());

    let state = Arc::new(AppState {
        model,
        tokenizer,
        engine_config: EngineConfig::default_resident(),
        model_name: "citlali-qwen3".to_string(),
        infer_semaphore: Semaphore::new(1),
        prompt_cache: std::sync::Mutex::new(None),
    });

    // CORS: 允许浏览器跨域请求
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/v1/models", get(routes::list_models))
        .route("/v1/chat/completions", post(routes::chat_completions))
        .layer(cors)
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    println!("服务监听: http://{}", addr);
    println!("API 端点:");
    println!("  GET  /v1/models");
    println!("  POST /v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str) -> Option<T> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
}

fn parse_str_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}
