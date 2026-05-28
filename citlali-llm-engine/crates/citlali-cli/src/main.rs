// crates/citlali-cli/src/main.rs
//! Citlali CLI — 交互式命令行推理工具
//! 用法: citlali <path-to-gguf> [--mmap] [--drop]

use citlali_model::mmap_model::MmapQwen3Model;
use citlali_model::model::Qwen3Model;
use citlali_model::tokenizer::{ChatOptions, Tokenizer};
use citlali_runtime::config::EngineConfig;
use citlali_runtime::generate::{
    generate_mmap_stream, generate_with_cache, PromptCache,
};
use citlali_runtime::sampler::SamplerConfig;
use std::env;
use std::io::{self, BufRead, Write};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("用法: citlali <path-to-gguf> [--mmap] [--drop] [--greedy] [--no-think]");
        eprintln!("  --mmap      内存映射模式（低内存）");
        eprintln!("  --drop      KV Cache 淘汰时直接丢弃");
        eprintln!("  --greedy    贪心采样（确定性输出，无重复惩罚）");
        eprintln!("  --no-think  禁用 Qwen3 thinking mode");
        std::process::exit(1);
    }

    let path = &args[1];
    let use_mmap = args.iter().any(|a| a == "--mmap");
    let use_drop = args.iter().any(|a| a == "--drop");
    let use_greedy = args.iter().any(|a| a == "--greedy");
    let no_think = args.iter().any(|a| a == "--no-think");

    let chat_options = ChatOptions { no_think };

    let engine = match (use_mmap, use_drop) {
        (false, false) => EngineConfig::default_resident(),
        (true, false) => EngineConfig::low_memory(),
        (false, true) => EngineConfig::oneshot(),
        (true, true) => EngineConfig::minimal(),
    };

    println!("Citlali LLM Engine — 交互模式");
    println!("配置: {:?}", engine);
    println!("加载模型: {} ...", path);

    let start = Instant::now();

    if use_mmap {
        let (model, tokenizer) = MmapQwen3Model::load_with_tokenizer(path).expect("加载模型失败");
        println!("加载完成 ({:.2}s)\n", start.elapsed().as_secs_f32());
        repl_mmap(&model, &tokenizer, &engine, use_greedy, &chat_options);
    } else {
        let (model, tokenizer) = Qwen3Model::load_with_tokenizer(path).expect("加载模型失败");
        println!("加载完成 ({:.2}s)\n", start.elapsed().as_secs_f32());
        repl_resident(&model, &tokenizer, &engine, use_greedy, &chat_options);
    }
}

fn repl_resident(model: &Qwen3Model, tokenizer: &Tokenizer, engine: &EngineConfig, greedy: bool, chat_options: &ChatOptions) {
    let mut sampler = if greedy { SamplerConfig::greedy() } else { SamplerConfig::default() };
    sampler.stop_tokens = tokenizer.stop_token_ids.clone();
    let stdin = io::stdin();
    let mut messages: Vec<(String, String)> = Vec::new(); // (role, content)
    let mut cache: Option<PromptCache> = None;

    println!("多轮对话模式（输入 /clear 清除历史）");

    loop {
        print!(">>> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() || input == "exit" || input == "quit" {
            break;
        }
        if input == "/clear" {
            messages.clear();
            cache = None;
            println!("[已清除对话历史]\n");
            continue;
        }

        // 追加用户消息
        messages.push(("user".to_string(), input.to_string()));

        // 使用模型自带的 chat template 拼装 prompt
        let prompt = tokenizer.format_chat_with_options(&messages, chat_options);

        let t = Instant::now();
        let mut token_buf: Vec<u32> = Vec::new();
        let mut printed_bytes = 0;

        let result = generate_with_cache(
            model, tokenizer, &prompt, &sampler, engine, &mut cache,
            |done, total| {
                eprint!("\r[prefill {}/{}]", done, total);
                if done == total { eprint!("\r                    \r"); }
            },
            |tok_id| {
                token_buf.push(tok_id);
                let bytes = tokenizer.decode_to_bytes(&token_buf);
                // 找到从 printed_bytes 开始的最长合法 UTF-8 前缀
                let new_bytes = &bytes[printed_bytes..];
                let valid_up_to = match std::str::from_utf8(new_bytes) {
                    Ok(s) => s.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid_up_to > 0 {
                    let s = unsafe { std::str::from_utf8_unchecked(&new_bytes[..valid_up_to]) };
                    print!("{}", s);
                    io::stdout().flush().unwrap();
                    printed_bytes += valid_up_to;
                }
            },
        );

        // 追加 assistant 回复到历史
        messages.push(("assistant".to_string(), result.text.clone()));

        let elapsed = t.elapsed().as_secs_f32();
        println!();
        eprintln!(
            "[{} tokens, {:.2}s, {:.1} tok/s]\n",
            result.token_ids.len(),
            elapsed,
            result.token_ids.len() as f32 / elapsed
        );
    }
}

fn repl_mmap(model: &MmapQwen3Model, tokenizer: &Tokenizer, engine: &EngineConfig, greedy: bool, chat_options: &ChatOptions) {
    let mut sampler = if greedy { SamplerConfig::greedy() } else { SamplerConfig::default() };
    sampler.stop_tokens = tokenizer.stop_token_ids.clone();
    let stdin = io::stdin();

    loop {
        print!(">>> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() || input == "exit" || input == "quit" {
            break;
        }

        // 使用模型自带的 chat template
        let messages = vec![("user".to_string(), input.to_string())];
        let prompt = tokenizer.format_chat_with_options(&messages, chat_options);

        let t = Instant::now();
        let mut token_buf: Vec<u32> = Vec::new();
        let mut printed_bytes = 0;

        let result = generate_mmap_stream(model, tokenizer, &prompt, &sampler, engine, |tok_id| {
            token_buf.push(tok_id);
            let bytes = tokenizer.decode_to_bytes(&token_buf);
            let new_bytes = &bytes[printed_bytes..];
            let valid_up_to = match std::str::from_utf8(new_bytes) {
                Ok(s) => s.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_up_to > 0 {
                let s = unsafe { std::str::from_utf8_unchecked(&new_bytes[..valid_up_to]) };
                print!("{}", s);
                io::stdout().flush().unwrap();
                printed_bytes += valid_up_to;
            }
        });

        let elapsed = t.elapsed().as_secs_f32();
        println!();
        eprintln!(
            "[{} tokens, {:.2}s, {:.1} tok/s]\n",
            result.token_ids.len(),
            elapsed,
            result.token_ids.len() as f32 / elapsed
        );
    }
}
