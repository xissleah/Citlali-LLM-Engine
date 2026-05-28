// crates/citlali-model/src/forward.rs
//! Qwen3-0.6B 前向推理（单 token 自回归）
//! 使用 Linear Arena 管理算子暂存区，每层 reset 复用
//! 使用 PagedKvCache 管理 KV Cache（固定页 + LRU + Swap）

use crate::mmap_model::MmapQwen3Model;
use crate::model::{LayerWeights, ModelConfig, Qwen3Model};
use citlali_compute::arena::Arena;
use citlali_compute::kv_cache::{KvCacheConfig, PagedKvCache};
use citlali_compute::ops::{
    apply_rope_with_type, mat_mat_mul_into, mat_vec_mul_into, rms_norm_inplace, rms_norm_into,
    silu, tiled_attention,
};
use rayon::prelude::*;

/// Prefill 分块大小：每次批量处理的 token 数
/// 太大会增加 arena 内存需求，太小则无法充分利用 mat-mat 的缓存优势
const PREFILL_CHUNK_SIZE: usize = 32;

/// 根据模型配置计算单层所需的 Arena 容量（字节）
/// 同时考虑 batch prefill 的额外缓冲区需求
pub fn compute_arena_size(config: &ModelConfig, max_seq_len: usize) -> usize {
    let f = std::mem::size_of::<f32>();
    let hidden = config.hidden_dim;
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_dim;

    // 单 token decode 模式的缓冲区
    let attn_buffers =
        (hidden + heads * head_dim + kv_heads * head_dim * 2 + heads * head_dim + hidden) * f;
    let ffn_buffers = (hidden + inter * 3 + hidden) * f;
    // tiled_attention 使用栈上 [f32; 64]，不需要 arena 分配 scores
    let kv_prefetch = 2 * kv_heads * head_dim * max_seq_len * f;
    let layer_total = attn_buffers + ffn_buffers + kv_prefetch;

    // Batch prefill 模式的缓冲区（PREFILL_CHUNK_SIZE 个 token 同时处理）
    let bs = PREFILL_CHUNK_SIZE;
    let batch_hidden = bs * hidden * f;           // input batch embeddings
    let batch_q = bs * heads * head_dim * f;      // Q projections
    let batch_k = bs * kv_heads * head_dim * f;   // K projections
    let batch_v = bs * kv_heads * head_dim * f;   // V projections
    let batch_attn_out = bs * heads * head_dim * f;
    let batch_proj = bs * hidden * f;             // output projection
    let batch_ffn = bs * (hidden + inter * 3 + hidden) * f;
    // tiled_attention 使用栈上分配，不需要 arena 中的 scores 缓冲区
    let batch_kv_prefetch = kv_prefetch; // KV prefetch 不随 batch 增长
    let batch_total = batch_hidden + batch_q + batch_k + batch_v
        + batch_attn_out + batch_proj + batch_ffn
        + batch_kv_prefetch;

    // LM head 需要 hidden_dim + vocab_size 的缓冲区
    let lm_head = (config.hidden_dim + config.vocab_size) * f;
    let raw = layer_total.max(lm_head).max(batch_total);
    (raw * 12 / 10 + 4096) & !63
}

/// 创建 PagedKvCache 的配置
pub fn default_kv_cache_config(config: &ModelConfig, max_pages: usize) -> KvCacheConfig {
    KvCacheConfig {
        kv_dim: config.num_kv_heads * config.head_dim,
        num_layers: config.num_layers,
        max_pages_in_memory: max_pages,
        swap_path: Some("citlali_kv_swap.bin".to_string()),
    }
}

/// 单 token 前向传播（写入预分配的 logits 缓冲区，零堆分配）
pub fn forward_into(
    model: &Qwen3Model,
    token_id: u32,
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
    logits_out: &mut [f32],
) {
    let config = &model.config;
    let pos = kv_cache.seq_len();

    // 1. Embedding lookup
    let start = token_id as usize * config.hidden_dim;
    let mut hidden: Vec<f32> = model.token_embd[start..start + config.hidden_dim].to_vec();

    // 2. Transformer layers
    for layer_idx in 0..config.num_layers {
        arena.reset();
        transformer_block(
            &mut hidden,
            &model.layers[layer_idx],
            config,
            layer_idx,
            pos,
            kv_cache,
            arena,
        );
    }

    // 3. 所有层写入完毕，推进序列长度
    kv_cache.advance_seq();

    // 4. Final RMS norm + LM head
    arena.reset();
    let normed = arena.alloc_slice::<f32>(config.hidden_dim);
    rms_norm_into(&hidden, &model.output_norm, config.rms_norm_eps, normed);

    // 使用独立的 output.weight（如果有），否则 tied 到 token_embd
    let lm_head = model.output_weight.as_deref().unwrap_or(&model.token_embd);
    mat_vec_mul_into(
        lm_head,
        normed,
        config.vocab_size,
        config.hidden_dim,
        logits_out,
    );
}

/// 单 token 前向传播，返回 logits [vocab_size]（兼容旧接口）
pub fn forward(
    model: &Qwen3Model,
    token_id: u32,
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; model.config.vocab_size];
    forward_into(model, token_id, kv_cache, arena, &mut logits);
    logits
}

/// 单层 Transformer Block（Pre-Norm + GQA Attention + SwiGLU FFN）
/// 所有临时缓冲区从 arena 分配，KV 写入 PagedKvCache
fn transformer_block(
    hidden: &mut Vec<f32>,
    layer: &LayerWeights,
    config: &ModelConfig,
    layer_idx: usize,
    pos: usize,
    kv_cache: &mut PagedKvCache,
    arena: &Arena,
) {
    let h = config.hidden_dim;
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = kv_heads * head_dim;
    let inter = config.intermediate_dim;

    // ─── Attention ───────────────────────────────────

    // 1. Pre-norm
    let normed = arena.alloc_slice::<f32>(h);
    rms_norm_into(hidden, &layer.attn_norm, config.rms_norm_eps, normed);

    // 2. QKV projections
    let q = arena.alloc_slice::<f32>(heads * head_dim);
    mat_vec_mul_into(&layer.attn_q, normed, heads * head_dim, h, q);

    let k = arena.alloc_slice::<f32>(kv_dim);
    mat_vec_mul_into(&layer.attn_k, normed, kv_dim, h, k);

    let v = arena.alloc_slice::<f32>(kv_dim);
    mat_vec_mul_into(&layer.attn_v, normed, kv_dim, h, v);

    // 3. QK-Norm (per-head RMS norm) — only if model has it (Qwen3)
    if let Some(ref q_norm) = layer.attn_q_norm {
        for head in 0..heads {
            let q_head = &mut q[head * head_dim..(head + 1) * head_dim];
            rms_norm_inplace(q_head, q_norm, config.rms_norm_eps);
        }
    }
    if let Some(ref k_norm) = layer.attn_k_norm {
        for head in 0..kv_heads {
            let k_head = &mut k[head * head_dim..(head + 1) * head_dim];
            rms_norm_inplace(k_head, k_norm, config.rms_norm_eps);
        }
    }

    // 4. RoPE (支持 partial RoPE via config.rope_dim, rope_type 区分 Llama/Qwen)
    apply_rope_with_type(q, head_dim, heads, pos, config.rope_theta, config.rope_dim, config.rope_type);
    apply_rope_with_type(k, head_dim, kv_heads, pos, config.rope_theta, config.rope_dim, config.rope_type);

    // 5. 写入 KV Cache（PagedKvCache 管理分页和 LRU）
    kv_cache.push(layer_idx, k, v);

    // 6. Attention scores + weighted sum (GQA)
    // 预取当前层所有 position 的 K、V 到 arena（零堆分配）
    let seq_len = pos + 1;
    let heads_per_kv = heads / kv_heads;

    let k_all = arena.alloc_slice::<f32>(seq_len * kv_dim);
    let v_all = arena.alloc_slice::<f32>(seq_len * kv_dim);
    kv_cache.batch_copy_k_into(layer_idx, seq_len, k_all);
    kv_cache.batch_copy_v_into(layer_idx, seq_len, v_all);

    let attn_out = arena.alloc_zeroed_slice::<f32>(heads * head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    attn_out
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(head, out_head)| {
            let kv_head = head / heads_per_kv;
            let q_head = &q[head * head_dim..(head + 1) * head_dim];
            tiled_attention(
                q_head, k_all, v_all, kv_dim, kv_head, head_dim, seq_len, scale, out_head,
            );
        });

    // 7. Output projection
    let attn_proj = arena.alloc_slice::<f32>(h);
    mat_vec_mul_into(&layer.attn_output, attn_out, h, heads * head_dim, attn_proj);

    // 8. Residual connection (attention)
    for i in 0..h {
        hidden[i] += attn_proj[i];
    }

    // ─── FFN (SwiGLU) ────────────────────────────────

    // 9. Pre-norm for FFN
    let ffn_input = arena.alloc_slice::<f32>(h);
    rms_norm_into(hidden, &layer.ffn_norm, config.rms_norm_eps, ffn_input);

    // 10. Gate + Up projections
    let gate = arena.alloc_slice::<f32>(inter);
    mat_vec_mul_into(&layer.ffn_gate, ffn_input, inter, h, gate);

    let up = arena.alloc_slice::<f32>(inter);
    mat_vec_mul_into(&layer.ffn_up, ffn_input, inter, h, up);

    // 11. SiLU(gate) * up
    let ffn_hidden = arena.alloc_slice::<f32>(inter);
    for i in 0..inter {
        ffn_hidden[i] = silu(gate[i]) * up[i];
    }

    // 12. Down projection
    let down = arena.alloc_slice::<f32>(h);
    mat_vec_mul_into(&layer.ffn_down, ffn_hidden, h, inter, down);

    // 13. Residual connection (FFN)
    for i in 0..h {
        hidden[i] += down[i];
    }
}

/// 批量 Prefill：一次处理多个 token，QKV 投影使用 mat-mat 提高缓存利用率
/// 返回最后一个 token 的 logits
///
/// 策略：
/// - QKV 投影：mat-mat（权重矩阵只读一次，N 个输入向量同时计算）
/// - RoPE / QK-Norm：逐 token 应用（位置相关，无法批量）
/// - Attention：逐 token 顺序计算（因果 mask，每个 token 只看前面的）
/// - FFN：mat-mat 批量计算（token 间独立）
///
/// on_progress: 回调 (已完成 token 数, 总 token 数)
pub fn prefill_batch(
    model: &Qwen3Model,
    tokens: &[u32],
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
    logits_out: &mut [f32],
    mut on_progress: impl FnMut(usize, usize),
) {
    if tokens.is_empty() {
        return;
    }

    let config = &model.config;
    let total = tokens.len();
    let h = config.hidden_dim;
    let mut last_hidden = vec![0.0f32; h];

    // 分块处理
    for chunk_start in (0..total).step_by(PREFILL_CHUNK_SIZE) {
        let chunk_end = (chunk_start + PREFILL_CHUNK_SIZE).min(total);
        let chunk = &tokens[chunk_start..chunk_end];
        let chunk_size = chunk.len();

        // 1. Embedding lookup for chunk
        let mut hidden_batch = vec![0.0f32; chunk_size * h];
        for (i, &tok) in chunk.iter().enumerate() {
            let src = tok as usize * h;
            let dst = i * h;
            hidden_batch[dst..dst + h]
                .copy_from_slice(&model.token_embd[src..src + h]);
        }

        // 2. Transformer layers
        for layer_idx in 0..config.num_layers {
            arena.reset();
            prefill_transformer_block_batch(
                &mut hidden_batch,
                chunk_size,
                &model.layers[layer_idx],
                config,
                layer_idx,
                kv_cache,
                arena,
            );
        }

        // 3. 推进 seq_len
        for _ in 0..chunk_size {
            kv_cache.advance_seq();
        }

        // 4. 保存最后一个 token 的 hidden state
        let last_offset = (chunk_size - 1) * h;
        last_hidden.copy_from_slice(&hidden_batch[last_offset..last_offset + h]);

        on_progress(chunk_end, total);
    }

    // 5. Final RMS norm + LM head
    arena.reset();
    let normed = arena.alloc_slice::<f32>(h);
    rms_norm_into(&last_hidden, &model.output_norm, config.rms_norm_eps, normed);
    let lm_head = model.output_weight.as_deref().unwrap_or(&model.token_embd);
    mat_vec_mul_into(
        lm_head,
        normed,
        config.vocab_size,
        h,
        logits_out,
    );
}

/// 批量 Transformer Block：对 chunk_size 个 token 同时处理
/// QKV 投影用 mat-mat，attention 逐 token 顺序计算，FFN 用 mat-mat
fn prefill_transformer_block_batch(
    hidden_batch: &mut [f32],  // [chunk_size, hidden_dim]
    chunk_size: usize,
    layer: &LayerWeights,
    config: &ModelConfig,
    layer_idx: usize,
    kv_cache: &mut PagedKvCache,
    arena: &Arena,
) {
    let h = config.hidden_dim;
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = kv_heads * head_dim;
    let inter = config.intermediate_dim;
    let q_dim = heads * head_dim;

    // ─── Batch RMS Norm ───
    let normed_batch = arena.alloc_slice::<f32>(chunk_size * h);
    for i in 0..chunk_size {
        let src = &hidden_batch[i * h..(i + 1) * h];
        let dst = &mut normed_batch[i * h..(i + 1) * h];
        rms_norm_into(src, &layer.attn_norm, config.rms_norm_eps, dst);
    }

    // ─── Batch QKV Projections (mat-mat) ───
    let q_batch = arena.alloc_slice::<f32>(chunk_size * q_dim);
    mat_mat_mul_into(&layer.attn_q, normed_batch, q_dim, h, chunk_size, q_batch);

    let k_batch = arena.alloc_slice::<f32>(chunk_size * kv_dim);
    mat_mat_mul_into(&layer.attn_k, normed_batch, kv_dim, h, chunk_size, k_batch);

    let v_batch = arena.alloc_slice::<f32>(chunk_size * kv_dim);
    mat_mat_mul_into(&layer.attn_v, normed_batch, kv_dim, h, chunk_size, v_batch);

    // ─── Per-token: QK-Norm + RoPE + KV push + Attention ───
    let base_pos = kv_cache.seq_len();
    let attn_out_batch = arena.alloc_zeroed_slice::<f32>(chunk_size * q_dim);

    // 预分配 KV prefetch 缓冲区（最大 seq_len = base_pos + chunk_size）
    let max_seq = base_pos + chunk_size;
    let mut k_all = vec![0.0f32; max_seq * kv_dim];
    let mut v_all = vec![0.0f32; max_seq * kv_dim];

    let heads_per_kv = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // 增量预取：先一次性读取 [0..base_pos) 的历史 KV（不随 token 变化）
    if base_pos > 0 {
        kv_cache.batch_copy_k_into(layer_idx, base_pos, &mut k_all[..base_pos * kv_dim]);
        kv_cache.batch_copy_v_into(layer_idx, base_pos, &mut v_all[..base_pos * kv_dim]);
    }

    for t in 0..chunk_size {
        let pos = base_pos + t;
        let q = &mut q_batch[t * q_dim..(t + 1) * q_dim];
        let k = &mut k_batch[t * kv_dim..(t + 1) * kv_dim];
        let v = &v_batch[t * kv_dim..(t + 1) * kv_dim];

        // QK-Norm
        if let Some(ref q_norm) = layer.attn_q_norm {
            for head in 0..heads {
                let q_head = &mut q[head * head_dim..(head + 1) * head_dim];
                rms_norm_inplace(q_head, q_norm, config.rms_norm_eps);
            }
        }
        if let Some(ref k_norm) = layer.attn_k_norm {
            for head in 0..kv_heads {
                let k_head = &mut k[head * head_dim..(head + 1) * head_dim];
                rms_norm_inplace(k_head, k_norm, config.rms_norm_eps);
            }
        }

        // RoPE (支持 partial RoPE + rope_type)
        apply_rope_with_type(q, head_dim, heads, pos, config.rope_theta, config.rope_dim, config.rope_type);
        apply_rope_with_type(k, head_dim, kv_heads, pos, config.rope_theta, config.rope_dim, config.rope_type);

        // Push KV at explicit position (batch prefill 不能用 push，因为 seq_len 还没推进)
        kv_cache.push_at(layer_idx, pos, k, v);

        // 增量预取：只读取刚 push 的新 position
        kv_cache.copy_k_into(layer_idx, pos, &mut k_all[pos * kv_dim..(pos + 1) * kv_dim]);
        kv_cache.copy_v_into(layer_idx, pos, &mut v_all[pos * kv_dim..(pos + 1) * kv_dim]);

        // Attention: this token attends to positions [0..=pos]
        let seq_len = pos + 1;
        let attn_out = &mut attn_out_batch[t * q_dim..(t + 1) * q_dim];
        let q_ref: &[f32] = q; // 共享引用，允许 rayon 并行读取
        let k_ref: &[f32] = &k_all;
        let v_ref: &[f32] = &v_all;

        // 并行化 attention heads
        attn_out
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(head, out_head)| {
                let kv_head = head / heads_per_kv;
                let q_head = &q_ref[head * head_dim..(head + 1) * head_dim];
                tiled_attention(
                    q_head, k_ref, v_ref, kv_dim, kv_head, head_dim, seq_len, scale, out_head,
                );
            });
    }

    // ─── Batch Output Projection (mat-mat) ───
    let proj_batch = arena.alloc_slice::<f32>(chunk_size * h);
    mat_mat_mul_into(&layer.attn_output, attn_out_batch, h, q_dim, chunk_size, proj_batch);

    // ─── Residual (attention) ───
    for i in 0..chunk_size * h {
        hidden_batch[i] += proj_batch[i];
    }

    // ─── Batch FFN ───
    let ffn_input = arena.alloc_slice::<f32>(chunk_size * h);
    for i in 0..chunk_size {
        let src = &hidden_batch[i * h..(i + 1) * h];
        let dst = &mut ffn_input[i * h..(i + 1) * h];
        rms_norm_into(src, &layer.ffn_norm, config.rms_norm_eps, dst);
    }

    let gate_batch = arena.alloc_slice::<f32>(chunk_size * inter);
    mat_mat_mul_into(&layer.ffn_gate, ffn_input, inter, h, chunk_size, gate_batch);

    let up_batch = arena.alloc_slice::<f32>(chunk_size * inter);
    mat_mat_mul_into(&layer.ffn_up, ffn_input, inter, h, chunk_size, up_batch);

    // SiLU(gate) * up——向量化友好的循环
    let ffn_hidden = arena.alloc_slice::<f32>(chunk_size * inter);
    for i in (0..chunk_size * inter).step_by(8) {
        let end = (i + 8).min(chunk_size * inter);
        for j in i..end {
            ffn_hidden[j] = silu(gate_batch[j]) * up_batch[j];
        }
    }

    let down_batch = arena.alloc_slice::<f32>(chunk_size * h);
    mat_mat_mul_into(&layer.ffn_down, ffn_hidden, h, inter, chunk_size, down_batch);

    // Residual (FFN)
    for i in 0..chunk_size * h {
        hidden_batch[i] += down_batch[i];
    }
}

/// 单 token 前向传播（mmap 模式，写入预分配缓冲区）
pub fn forward_mmap_into(
    model: &MmapQwen3Model,
    token_id: u32,
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
    logits_out: &mut [f32],
) {
    let config = &model.config;
    let pos = kv_cache.seq_len();

    // 1. Embedding lookup
    let start = token_id as usize * config.hidden_dim;
    let mut hidden: Vec<f32> = model.token_embd[start..start + config.hidden_dim].to_vec();

    // 2. Transformer layers（每层按需反量化）
    for layer_idx in 0..config.num_layers {
        arena.reset();
        // 按需反量化当前层权重（临时分配，本次循环结束后释放）
        let layer_weights = model.dequant_layer(layer_idx);
        transformer_block(
            &mut hidden,
            &layer_weights,
            config,
            layer_idx,
            pos,
            kv_cache,
            arena,
        );
        // layer_weights 在此 drop，释放 ~50MB
    }

    // 3. 推进序列长度
    kv_cache.advance_seq();

    // 4. Final RMS norm + LM head
    arena.reset();
    let normed = arena.alloc_slice::<f32>(config.hidden_dim);
    rms_norm_into(&hidden, &model.output_norm, config.rms_norm_eps, normed);

    let lm_head = model.output_weight.as_deref().unwrap_or(&model.token_embd);
    mat_vec_mul_into(
        lm_head,
        normed,
        config.vocab_size,
        config.hidden_dim,
        logits_out,
    );
}

/// 单 token 前向传播（mmap 模式，兼容旧接口）
pub fn forward_mmap(
    model: &MmapQwen3Model,
    token_id: u32,
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; model.config.vocab_size];
    forward_mmap_into(model, token_id, kv_cache, arena, &mut logits);
    logits
}

/// 单 token 前向传播（mmap 量化模式，零整层反量化）
/// 大矩阵直接在量化数据上逐 block 解压 + 点积，内存仅需 ~256 f32 栈缓冲/线程
/// norm 权重（很小，hidden_dim 或 head_dim 大小）照常 dequant
pub fn forward_mmap_quantized_into(
    model: &MmapQwen3Model,
    token_id: u32,
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
    logits_out: &mut [f32],
) {
    use crate::quantized_ops::raw_mat_vec_mul_into;

    let config = &model.config;
    let pos = kv_cache.seq_len();

    // 1. Embedding lookup
    let start = token_id as usize * config.hidden_dim;
    let mut hidden: Vec<f32> = model.token_embd[start..start + config.hidden_dim].to_vec();

    // 2. Transformer layers
    for layer_idx in 0..config.num_layers {
        arena.reset();
        quantized_transformer_block(
            &mut hidden, model, config, layer_idx, pos, kv_cache, arena,
        );
    }

    // 3. 推进序列长度
    kv_cache.advance_seq();

    // 4. Final RMS norm + LM head
    arena.reset();
    let normed = arena.alloc_slice::<f32>(config.hidden_dim);
    rms_norm_into(&hidden, &model.output_norm, config.rms_norm_eps, normed);

    // LM head: 使用独立的 output.weight（如果有），否则 tied 到 token_embd
    let lm_head = model.output_weight.as_deref().unwrap_or(&model.token_embd);
    mat_vec_mul_into(
        lm_head,
        normed,
        config.vocab_size,
        config.hidden_dim,
        logits_out,
    );
}

/// 单 token 前向传播（mmap 量化模式，兼容旧接口）
pub fn forward_mmap_quantized(
    model: &MmapQwen3Model,
    token_id: u32,
    kv_cache: &mut PagedKvCache,
    arena: &mut Arena,
) -> Vec<f32> {
    let mut logits = vec![0.0f32; model.config.vocab_size];
    forward_mmap_quantized_into(model, token_id, kv_cache, arena, &mut logits);
    logits
}

/// 量化 Transformer Block：大矩阵用 quantized matmul，norm 权重 dequant
fn quantized_transformer_block(
    hidden: &mut Vec<f32>,
    model: &MmapQwen3Model,
    config: &ModelConfig,
    layer_idx: usize,
    pos: usize,
    kv_cache: &mut PagedKvCache,
    arena: &Arena,
) {
    use crate::quantized_ops::raw_mat_vec_mul_into;

    let h = config.hidden_dim;
    let heads = config.num_heads;
    let kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let kv_dim = kv_heads * head_dim;
    let inter = config.intermediate_dim;
    let prefix = format!("blk.{}", layer_idx);

    // Norm 权重很小，照常 dequant
    let attn_norm = model.dequant_tensor(&format!("{}.attn_norm.weight", prefix));
    let attn_q_norm = model.try_dequant_tensor(&format!("{}.attn_q_norm.weight", prefix));
    let attn_k_norm = model.try_dequant_tensor(&format!("{}.attn_k_norm.weight", prefix));
    let ffn_norm = model.dequant_tensor(&format!("{}.ffn_norm.weight", prefix));

    // 获取大矩阵的原始字节 + dtype
    let (raw_q, dtype_q) = model.get_raw_tensor(&format!("{}.attn_q.weight", prefix));
    let (raw_k, dtype_k) = model.get_raw_tensor(&format!("{}.attn_k.weight", prefix));
    let (raw_v, dtype_v) = model.get_raw_tensor(&format!("{}.attn_v.weight", prefix));
    let (raw_o, dtype_o) = model.get_raw_tensor(&format!("{}.attn_output.weight", prefix));
    let (raw_gate, dtype_gate) = model.get_raw_tensor(&format!("{}.ffn_gate.weight", prefix));
    let (raw_up, dtype_up) = model.get_raw_tensor(&format!("{}.ffn_up.weight", prefix));
    let (raw_down, dtype_down) = model.get_raw_tensor(&format!("{}.ffn_down.weight", prefix));

    // ─── Attention ───────────────────────────────

    // 1. Pre-norm
    let normed = arena.alloc_slice::<f32>(h);
    rms_norm_into(hidden, &attn_norm, config.rms_norm_eps, normed);

    // 2. QKV projections (quantized matmul)
    let q = arena.alloc_slice::<f32>(heads * head_dim);
    raw_mat_vec_mul_into(raw_q, normed, heads * head_dim, h, dtype_q, q);

    let k = arena.alloc_slice::<f32>(kv_dim);
    raw_mat_vec_mul_into(raw_k, normed, kv_dim, h, dtype_k, k);

    let v = arena.alloc_slice::<f32>(kv_dim);
    raw_mat_vec_mul_into(raw_v, normed, kv_dim, h, dtype_v, v);

    // 3. QK-Norm
    if let Some(ref q_norm) = attn_q_norm {
        for head in 0..heads {
            let q_head = &mut q[head * head_dim..(head + 1) * head_dim];
            rms_norm_inplace(q_head, q_norm, config.rms_norm_eps);
        }
    }
    if let Some(ref k_norm) = attn_k_norm {
        for head in 0..kv_heads {
            let k_head = &mut k[head * head_dim..(head + 1) * head_dim];
            rms_norm_inplace(k_head, k_norm, config.rms_norm_eps);
        }
    }

    // 4. RoPE (支持 partial RoPE + rope_type)
    apply_rope_with_type(q, head_dim, heads, pos, config.rope_theta, config.rope_dim, config.rope_type);
    apply_rope_with_type(k, head_dim, kv_heads, pos, config.rope_theta, config.rope_dim, config.rope_type);

    // 5. KV Cache
    kv_cache.push(layer_idx, k, v);

    // 6. Attention scores + weighted sum (GQA)
    let seq_len = pos + 1;
    let heads_per_kv = heads / kv_heads;

    let k_all = arena.alloc_slice::<f32>(seq_len * kv_dim);
    let v_all = arena.alloc_slice::<f32>(seq_len * kv_dim);
    kv_cache.batch_copy_k_into(layer_idx, seq_len, k_all);
    kv_cache.batch_copy_v_into(layer_idx, seq_len, v_all);

    let attn_out = arena.alloc_zeroed_slice::<f32>(heads * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    attn_out
        .par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(head, out_head)| {
            let kv_head = head / heads_per_kv;
            let q_head = &q[head * head_dim..(head + 1) * head_dim];
            tiled_attention(
                q_head, k_all, v_all, kv_dim, kv_head, head_dim, seq_len, scale, out_head,
            );
        });

    // 7. Output projection (quantized matmul)
    let attn_proj = arena.alloc_slice::<f32>(h);
    raw_mat_vec_mul_into(raw_o, attn_out, h, heads * head_dim, dtype_o, attn_proj);

    // 8. Residual
    for i in 0..h {
        hidden[i] += attn_proj[i];
    }

    // ─── FFN (SwiGLU) ────────────────────────────

    // 9. Pre-norm for FFN
    let ffn_input = arena.alloc_slice::<f32>(h);
    rms_norm_into(hidden, &ffn_norm, config.rms_norm_eps, ffn_input);

    // 10. Gate + Up (quantized matmul)
    let gate = arena.alloc_slice::<f32>(inter);
    raw_mat_vec_mul_into(raw_gate, ffn_input, inter, h, dtype_gate, gate);

    let up = arena.alloc_slice::<f32>(inter);
    raw_mat_vec_mul_into(raw_up, ffn_input, inter, h, dtype_up, up);

    // 11. SiLU(gate) * up
    let ffn_hidden = arena.alloc_slice::<f32>(inter);
    for i in 0..inter {
        ffn_hidden[i] = silu(gate[i]) * up[i];
    }

    // 12. Down projection (quantized matmul)
    let down = arena.alloc_slice::<f32>(h);
    raw_mat_vec_mul_into(raw_down, ffn_hidden, h, inter, dtype_down, down);

    // 13. Residual
    for i in 0..h {
        hidden[i] += down[i];
    }
}
