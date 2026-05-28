// crates/citlali-runtime/src/generate.rs
//! 文本生成引擎：prefill + decode loop
//! 支持两种模式：Resident（常驻内存）和 MemoryMapped（按需反量化）
//! 支持两种 KV 淘汰策略：Swap（写磁盘）和 Drop（直接丢弃）

use crate::config::{EngineConfig, KvEvictionPolicy};
use crate::sampler::SamplerConfig;
use citlali_compute::arena::Arena;
use citlali_compute::kv_cache::{KvCacheConfig, PagedKvCache};
use citlali_model::forward::{compute_arena_size, forward_into, forward_mmap_into, prefill_batch};
use citlali_model::mmap_model::MmapQwen3Model;
use citlali_model::model::Qwen3Model;
use citlali_model::tokenizer::Tokenizer;

/// Prompt Cache：缓存上一次请求的 KV 状态，支持前缀复用
pub struct PromptCache {
    /// 缓存中已 prefill 的完整 token 序列（prompt + generated）
    pub tokens: Vec<u32>,
    /// 对应的 KV Cache
    pub kv_cache: PagedKvCache,
    /// 复用的 Arena（避免重复分配）
    pub arena: Arena,
}

/// 生成结果
pub struct GenerateResult {
    /// 生成的 token IDs（不含 prompt）
    pub token_ids: Vec<u32>,
    /// 解码后的文本
    pub text: String,
    /// Arena 峰值使用量（字节），用于调优
    pub arena_peak_bytes: usize,
}

/// 根据 EngineConfig 构建 KvCacheConfig
/// 当 kv_max_pages=0 时，自动计算所需页数
fn build_kv_config(
    model_config: &citlali_model::model::ModelConfig,
    engine: &EngineConfig,
    max_seq_len: usize,
) -> KvCacheConfig {
    let swap_path = match &engine.kv_eviction {
        KvEvictionPolicy::Swap { path } => Some(path.clone()),
        KvEvictionPolicy::Drop => None,
    };

    let kv_dim = model_config.num_kv_heads * model_config.head_dim;
    let positions_per_page = 16384 / (kv_dim * 4); // PAGE_SIZE / (kv_dim * sizeof(f32))
    let num_layers = model_config.num_layers;

    // 自动计算：每个 position 需要 2 页（K+V）× num_layers，
    // 同时预留下一个 block 的空间避免频繁触发淘汰
    let auto_pages = if engine.kv_max_pages == 0 {
        let pages_per_position = 2 * num_layers;
        let pages_for_seq =
            (max_seq_len + positions_per_page - 1) / positions_per_page * pages_per_position;
        // +20% 余量，最少 512 页
        (pages_for_seq * 12 / 10).max(512)
    } else {
        engine.kv_max_pages
    };

    KvCacheConfig {
        kv_dim,
        num_layers,
        max_pages_in_memory: auto_pages,
        swap_path,
    }
}

/// Resident 模式生成（模型常驻内存）
pub fn generate(
    model: &Qwen3Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    sampler: &SamplerConfig,
    engine: &EngineConfig,
) -> GenerateResult {
    generate_stream(model, tokenizer, prompt, sampler, engine, |_, _| {}, |_| {})
}

/// Resident 模式流式生成
/// on_prefill_progress(done, total): prefill 进度回调，调用方可显示进度条
/// on_token(token_id): 每生成一个 token 回调
pub fn generate_stream(
    model: &Qwen3Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    sampler: &SamplerConfig,
    engine: &EngineConfig,
    mut on_prefill_progress: impl FnMut(usize, usize),
    mut on_token: impl FnMut(u32),
) -> GenerateResult {
    let prompt_tokens = tokenizer.encode(prompt);
    let max_seq = prompt_tokens.len() + sampler.max_tokens;
    let kv_config = build_kv_config(&model.config, engine, max_seq);
    let mut kv_cache = PagedKvCache::new(kv_config);

    let arena_size = compute_arena_size(&model.config, max_seq);
    let mut arena = Arena::new(arena_size);

    let mut logits = vec![0.0f32; model.config.vocab_size];

    // Prefill (batch)
    prefill_batch(model, &prompt_tokens, &mut kv_cache, &mut arena, &mut logits, &mut on_prefill_progress);

    // Decode loop
    let mut generated = Vec::new();
    let mut current_token = sampler.sample(&logits, &generated);
    if sampler.stop_tokens.contains(&current_token) {
        return GenerateResult {
            token_ids: generated,
            text: String::new(),
            arena_peak_bytes: arena.peak_usage(),
        };
    }
    generated.push(current_token);
    on_token(current_token);

    for _ in 1..sampler.max_tokens {
        forward_into(model, current_token, &mut kv_cache, &mut arena, &mut logits);
        let next = sampler.sample(&logits, &generated);
        if sampler.stop_tokens.contains(&next) {
            break;
        }
        generated.push(next);
        on_token(next);
        current_token = next;
    }

    let text = tokenizer.decode(&generated);
    GenerateResult {
        token_ids: generated,
        text,
        arena_peak_bytes: arena.peak_usage(),
    }
}

/// 计算两个 token 序列的最长公共前缀长度
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// 带 Prompt Cache 的流式生成（Resident 模式）
/// 复用上一次请求的 KV Cache 前缀，只 prefill 新增部分
pub fn generate_with_cache(
    model: &Qwen3Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    sampler: &SamplerConfig,
    engine: &EngineConfig,
    cache: &mut Option<PromptCache>,
    mut on_prefill_progress: impl FnMut(usize, usize),
    mut on_token: impl FnMut(u32),
) -> GenerateResult {
    let prompt_tokens = tokenizer.encode(prompt);
    let max_seq = prompt_tokens.len() + sampler.max_tokens;

    // 计算前缀复用长度
    let prefix_len = cache
        .as_ref()
        .map(|c| common_prefix_len(&c.tokens, &prompt_tokens))
        .unwrap_or(0);

    // 决定是复用还是重建
    let arena_size_needed = compute_arena_size(&model.config, max_seq);
    let (mut kv_cache, mut arena) = if prefix_len > 0 {
        let mut c = cache.take().unwrap();
        // 截断到前缀位置
        c.kv_cache.truncate_to(prefix_len);
        // Arena 容量不够则重建，否则 reset 复用
        let arena = if c.arena.capacity() >= arena_size_needed {
            c.arena.reset();
            c.arena
        } else {
            drop(c.arena);
            Arena::new(arena_size_needed)
        };
        (c.kv_cache, arena)
    } else {
        // 丢弃旧缓存，从头来
        drop(cache.take());
        let kv_config = build_kv_config(&model.config, engine, max_seq);
        (PagedKvCache::new(kv_config), Arena::new(arena_size_needed))
    };

    let mut logits = vec![0.0f32; model.config.vocab_size];

    // Prefill 仅新增部分 (batch)
    let new_tokens = &prompt_tokens[prefix_len..];
    if !new_tokens.is_empty() {
        prefill_batch(model, new_tokens, &mut kv_cache, &mut arena, &mut logits, &mut on_prefill_progress);
    } else if prefix_len > 0 {
        // prompt 完全匹配缓存，重新计算最后一个 token 的 logits
        // 截断最后一个 position，重新 forward 它
        kv_cache.truncate_to(prefix_len - 1);
        let last_tok = prompt_tokens[prefix_len - 1];
        forward_into(model, last_tok, &mut kv_cache, &mut arena, &mut logits);
    }

    // Decode loop
    let mut generated = Vec::new();
    let mut current_token = sampler.sample(&logits, &generated);
    if sampler.stop_tokens.contains(&current_token) {
        // 保存缓存（仅 prompt 部分）
        *cache = Some(PromptCache {
            tokens: prompt_tokens,
            kv_cache,
            arena,
        });
        return GenerateResult {
            token_ids: generated,
            text: String::new(),
            arena_peak_bytes: 0,
        };
    }
    generated.push(current_token);
    on_token(current_token);

    for _ in 1..sampler.max_tokens {
        forward_into(model, current_token, &mut kv_cache, &mut arena, &mut logits);
        let next = sampler.sample(&logits, &generated);
        if sampler.stop_tokens.contains(&next) {
            break;
        }
        generated.push(next);
        on_token(next);
        current_token = next;
    }

    let text = tokenizer.decode(&generated);

    // 保存缓存：prompt + generated 的完整 token 序列
    let mut cached_tokens = prompt_tokens;
    cached_tokens.extend_from_slice(&generated);
    *cache = Some(PromptCache {
        tokens: cached_tokens,
        kv_cache,
        arena,
    });

    GenerateResult {
        token_ids: generated,
        text,
        arena_peak_bytes: 0,
    }
}

/// MemoryMapped 模式生成（按层反量化，低内存）
pub fn generate_mmap(
    model: &MmapQwen3Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    sampler: &SamplerConfig,
    engine: &EngineConfig,
) -> GenerateResult {
    generate_mmap_stream(model, tokenizer, prompt, sampler, engine, |_| {})
}

/// MemoryMapped 模式流式生成
pub fn generate_mmap_stream(
    model: &MmapQwen3Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    sampler: &SamplerConfig,
    engine: &EngineConfig,
    mut on_token: impl FnMut(u32),
) -> GenerateResult {
    let prompt_tokens = tokenizer.encode(prompt);
    let max_seq = prompt_tokens.len() + sampler.max_tokens;
    let kv_config = build_kv_config(&model.config, engine, max_seq);
    let mut kv_cache = PagedKvCache::new(kv_config);

    let arena_size = compute_arena_size(&model.config, max_seq);
    let mut arena = Arena::new(arena_size);

    let mut logits = vec![0.0f32; model.config.vocab_size];

    // Prefill
    for &tok in &prompt_tokens {
        forward_mmap_into(model, tok, &mut kv_cache, &mut arena, &mut logits);
    }

    // Decode loop
    let mut generated = Vec::new();
    let mut current_token = sampler.sample(&logits, &generated);
    if sampler.stop_tokens.contains(&current_token) {
        return GenerateResult {
            token_ids: generated,
            text: String::new(),
            arena_peak_bytes: arena.peak_usage(),
        };
    }
    generated.push(current_token);
    on_token(current_token);

    for _ in 1..sampler.max_tokens {
        forward_mmap_into(model, current_token, &mut kv_cache, &mut arena, &mut logits);
        let next = sampler.sample(&logits, &generated);
        if sampler.stop_tokens.contains(&next) {
            break;
        }
        generated.push(next);
        on_token(next);
        current_token = next;
    }

    let text = tokenizer.decode(&generated);
    GenerateResult {
        token_ids: generated,
        text,
        arena_peak_bytes: arena.peak_usage(),
    }
}
