// crates/citlali-compute/src/ops.rs
//! 基础数学算子：matmul, rmsnorm, rope, softmax, silu

use rayon::prelude::*;

const PAR_MATVEC_OUT_DIM_THRESHOLD: usize = 256;

/// RMS Normalization: y = x * weight / rms(x)
/// x: [n], weight: [n], eps: 小常数
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (ss + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(xi, wi)| xi / rms * wi)
        .collect()
}

/// RMS Normalization (in-place)
pub fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (ss + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(weight.iter()) {
        *xi = *xi / rms * wi;
    }
}

/// 矩阵-向量乘法: output = weight @ input
/// weight: [out_dim, in_dim] (row-major), input: [in_dim]
/// output: [out_dim]
pub fn mat_vec_mul(weight: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; out_dim];
    mat_vec_mul_into(weight, input, out_dim, in_dim, &mut output);
    output
}

#[inline]
pub fn dot_product(row: &[f32], input: &[f32]) -> f32 {
    dot_product_dispatch(row, input)
}

/// 运行时分发：优先 AVX2 FMA，回退到标量 8 路展开
#[inline]
fn dot_product_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2_fma(a, b) };
        }
    }
    dot_product_scalar(a, b)
}

/// 标量 8 路展开（fallback）
#[inline]
fn dot_product_scalar(row: &[f32], input: &[f32]) -> f32 {
    let n = input.len();
    let chunks = n / 8;
    let remainder = n % 8;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    let base = chunks * 8;
    for i in 0..chunks {
        let j = i * 8;
        s0 += row[j] * input[j];
        s1 += row[j + 1] * input[j + 1];
        s2 += row[j + 2] * input[j + 2];
        s3 += row[j + 3] * input[j + 3];
        s4 += row[j + 4] * input[j + 4];
        s5 += row[j + 5] * input[j + 5];
        s6 += row[j + 6] * input[j + 6];
        s7 += row[j + 7] * input[j + 7];
    }
    for j in base..base + remainder {
        s0 += row[j] * input[j];
    }
    (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7)
}

/// RoPE (Rotary Position Embedding) 应用于 query/key
/// x: [num_heads, head_dim], pos: token 位置
/// rope_dim: 实际旋转的维度数（与 llama.cpp n_rot 对应）
pub fn apply_rope(x: &mut [f32], head_dim: usize, num_heads: usize, pos: usize, theta_base: f32) {
    apply_rope_partial(x, head_dim, num_heads, pos, theta_base, head_dim);
}

/// RoPE with partial rotation (仅前 rope_dim 维度旋转)
/// rope_dim = 0 或 head_dim 表示 full RoPE
/// 默认使用 split/neox 风格 (type=2)
pub fn apply_rope_partial(
    x: &mut [f32],
    head_dim: usize,
    num_heads: usize,
    pos: usize,
    theta_base: f32,
    rope_dim: usize,
) {
    apply_rope_with_type(x, head_dim, num_heads, pos, theta_base, rope_dim, 2);
}

/// RoPE with configurable pairing type
/// rope_type: 0 = adjacent/norm (Llama GGUF), 2 = split/neox (Qwen GGUF)
pub fn apply_rope_with_type(
    x: &mut [f32],
    head_dim: usize,
    num_heads: usize,
    pos: usize,
    theta_base: f32,
    rope_dim: usize,
    rope_type: u32,
) {
    let n_rot = if rope_dim == 0 || rope_dim >= head_dim {
        head_dim
    } else {
        rope_dim
    };
    let half_rot = n_rot / 2;
    let pos_f = pos as f32;

    for h in 0..num_heads {
        let offset = h * head_dim;
        for i in 0..half_rot {
            // theta = pos / base^(2*i/n_rot)
            let theta = pos_f * theta_base.powf(-2.0 * i as f32 / n_rot as f32);
            let cos_t = theta.cos();
            let sin_t = theta.sin();

            if rope_type == 0 {
                // Adjacent/Norm style (Llama): pairs (2i, 2i+1)
                let idx0 = offset + 2 * i;
                let idx1 = offset + 2 * i + 1;
                let x0 = x[idx0];
                let x1 = x[idx1];
                x[idx0] = x0 * cos_t - x1 * sin_t;
                x[idx1] = x0 * sin_t + x1 * cos_t;
            } else {
                // Split/NeoX style (Qwen): pairs (i, i + half_rot)
                let idx0 = offset + i;
                let idx1 = offset + i + half_rot;
                let x0 = x[idx0];
                let x1 = x[idx1];
                x[idx0] = x0 * cos_t - x1 * sin_t;
                x[idx1] = x0 * sin_t + x1 * cos_t;
            }
        }
        // 超出 rope_dim 的维度保持不变（partial RoPE）
    }
}

/// Softmax (in-place)
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

// ============================================================
// 激活函数
// ============================================================

/// SiLU activation: x * sigmoid(x)
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU activation (tanh 近似)
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
#[inline]
pub fn gelu(x: f32) -> f32 {
    let c = 0.7978845608_f32; // sqrt(2/pi)
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// SwiGLU: 对 gate 和 up 向量执行 SiLU(gate) * up
/// gate: [dim], up: [dim], output: [dim]
#[inline]
pub fn swiglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        output[i] = silu(gate[i]) * up[i];
    }
}

/// GeGLU: 对 gate 和 up 向量执行 GELU(gate) * up
/// gate: [dim], up: [dim], output: [dim]
#[inline]
pub fn geglu(gate: &[f32], up: &[f32], output: &mut [f32]) {
    for i in 0..output.len() {
        output[i] = gelu(gate[i]) * up[i];
    }
}

// ============================================================
// Arena-backed 算子变体
// 结果写入调用者提供的 output 切片，不产生堆分配
// ============================================================

/// RMS Norm → 写入 output
pub fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rms = (ss + eps).sqrt();
    for i in 0..n {
        output[i] = x[i] / rms * weight[i];
    }
}

/// 矩阵-向量乘法 → 写入 output
pub fn mat_vec_mul_into(
    weight: &[f32],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(output.len(), out_dim);

    if out_dim >= PAR_MATVEC_OUT_DIM_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let row = &weight[i * in_dim..(i + 1) * in_dim];
            *out = dot_product(row, input);
        });
    } else {
        for i in 0..out_dim {
            let row = &weight[i * in_dim..(i + 1) * in_dim];
            output[i] = dot_product(row, input);
        }
    }
}

/// 批量矩阵-向量乘法 (Mat-Mat): output = weight @ input_batch
/// weight: [out_dim, in_dim] (row-major)
/// input_batch: [batch_size, in_dim] (row-major，每行一个输入向量)
/// output: [batch_size, out_dim] (row-major，每行一个输出向量)
///
/// 并行策略：按 weight 行并行（out_dim 通常 >> batch_size）
/// 每个线程处理若干行，对所有 batch 元素计算 dot product
/// 相比 batch_size 次 mat_vec_mul_into，权重矩阵的缓存利用率更高
pub fn mat_mat_mul_into(
    weight: &[f32],
    input_batch: &[f32],
    out_dim: usize,
    in_dim: usize,
    batch_size: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(input_batch.len(), batch_size * in_dim);
    debug_assert_eq!(output.len(), batch_size * out_dim);

    if out_dim >= PAR_MATVEC_OUT_DIM_THRESHOLD {
        // 按 out_dim 行并行：每个线程处理一行 weight，计算该行与所有 batch 的 dot
        // 使用 unsafe 指针写入避免借用冲突（output 按 [batch, out_dim] 布局）
        let out_ptr = output.as_mut_ptr();
        let out_len = output.len();
        // Safety: 每个 row 写入不同的位置 output[b * out_dim + row]，不重叠
        let out_raw = UnsafeSlice { ptr: out_ptr, len: out_len };
        (0..out_dim).into_par_iter().for_each(|row| {
            let w_row = &weight[row * in_dim..(row + 1) * in_dim];
            for b in 0..batch_size {
                let inp = &input_batch[b * in_dim..(b + 1) * in_dim];
                unsafe { out_raw.write(b * out_dim + row, dot_product(w_row, inp)); }
            }
        });
    } else {
        for b in 0..batch_size {
            let inp = &input_batch[b * in_dim..(b + 1) * in_dim];
            let out_row = &mut output[b * out_dim..(b + 1) * out_dim];
            for row in 0..out_dim {
                let w_row = &weight[row * in_dim..(row + 1) * in_dim];
                out_row[row] = dot_product(w_row, inp);
            }
        }
    }
}

/// 用于并行写入不重叠位置的包装器
struct UnsafeSlice {
    ptr: *mut f32,
    len: usize,
}

unsafe impl Send for UnsafeSlice {}
unsafe impl Sync for UnsafeSlice {}

impl UnsafeSlice {
    /// Safety: 调用方保证不同线程写入不同的 index
    #[inline]
    unsafe fn write(&self, index: usize, value: f32) {
        debug_assert!(index < self.len);
        *self.ptr.add(index) = value;
    }
}

// ============================================================
// AVX2 + FMA SIMD 点积
// ============================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let chunks = n / 32;
    let mut i = 0usize;
    for _ in 0..chunks {
        let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
        let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
        let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);
        let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
        i += 32;
    }
    // 处理剩余的 8 元素块
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
        i += 8;
    }
    // 合并 4 个累加器
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    // 水平求和 256-bit → 标量
    let hi = _mm256_extractf128_ps(sum0, 1);
    let lo = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);
    // 标量处理尾部
    while i < n {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    result
}

// ============================================================
// Online Softmax Attention (Flash Attention 核心思想)
// 单趟遍历 KV，零额外 scores 分配
// ============================================================

/// Online softmax attention for a single head (decode mode)
/// q_head: [head_dim], k_all: KV 缓存中所有 position 的 K
/// v_all: KV 缓存中所有 position 的 V
/// kv_dim: 每个 position 的完整 KV 维度 (num_kv_heads * head_dim)
/// kv_head: 当前 head 对应的 kv_head 索引
/// head_dim: 每个 head 的维度
/// seq_len: 当前序列长度
/// scale: 1/sqrt(head_dim)
/// out_head: [head_dim] 输出
pub fn online_softmax_attention(
    q_head: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    kv_dim: usize,
    kv_head: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    out_head: &mut [f32],
) {
    let mut max_score = f32::NEG_INFINITY;
    let mut sum_exp = 0.0f32;
    // out_head 已经是零初始化的

    for t in 0..seq_len {
        let k_offset = t * kv_dim + kv_head * head_dim;
        let k_head_t = &k_all[k_offset..k_offset + head_dim];
        let score = dot_product(q_head, k_head_t) * scale;

        if score > max_score {
            // 修正之前累积的结果
            let correction = (max_score - score).exp();
            for d in 0..head_dim {
                out_head[d] *= correction;
            }
            sum_exp *= correction;
            max_score = score;
        }

        let w = (score - max_score).exp();
        sum_exp += w;

        let v_offset = t * kv_dim + kv_head * head_dim;
        let v_head_t = &v_all[v_offset..v_offset + head_dim];
        for d in 0..head_dim {
            out_head[d] += w * v_head_t[d];
        }
    }

    // 最终归一化
    if sum_exp > 0.0 {
        let inv_sum = 1.0 / sum_exp;
        for d in 0..head_dim {
            out_head[d] *= inv_sum;
        }
    }
}

// ============================================================
// Tiled Attention (CPU Flash Attention)
// 分块处理 KV，scores 子矩阵始终在 L2 cache 内
// ============================================================

/// KV tile 大小：每次处理 64 个 position
/// 对于 head_dim=128，每个 tile 的 scores 仅 64*4=256 bytes
const TILE_KV: usize = 64;

/// Tiled online softmax attention for a single head
/// 与 online_softmax_attention 语义相同，但按 TILE_KV 分块处理
/// 每个 tile 内先计算局部 max 和 sum，再与全局状态合并
/// 优势：K/V 的访问更连续，减少 TLB miss
pub fn tiled_attention(
    q_head: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    kv_dim: usize,
    kv_head: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    out_head: &mut [f32],
) {
    let mut global_max = f32::NEG_INFINITY;
    let mut global_sum = 0.0f32;
    // out_head 已经是零初始化的

    let mut t = 0;
    while t < seq_len {
        let tile_end = (t + TILE_KV).min(seq_len);
        let tile_len = tile_end - t;

        // Phase 1: 计算 tile 内的 scores 并找到 tile_max
        // 使用栈上数组避免堆分配（TILE_KV=64，仅 256 bytes）
        let mut tile_scores = [0.0f32; TILE_KV];
        let mut tile_max = f32::NEG_INFINITY;

        for i in 0..tile_len {
            let pos = t + i;
            let k_offset = pos * kv_dim + kv_head * head_dim;
            let k_head = &k_all[k_offset..k_offset + head_dim];
            let s = dot_product(q_head, k_head) * scale;
            tile_scores[i] = s;
            if s > tile_max {
                tile_max = s;
            }
        }

        // Phase 2: 合并 tile 与全局状态
        if tile_max > global_max {
            // 修正之前的累积结果
            let correction = (global_max - tile_max).exp();
            for d in 0..head_dim {
                out_head[d] *= correction;
            }
            global_sum *= correction;
            global_max = tile_max;
        }

        // Phase 3: 累加 tile 内的加权 V
        for i in 0..tile_len {
            let w = (tile_scores[i] - global_max).exp();
            global_sum += w;
            let pos = t + i;
            let v_offset = pos * kv_dim + kv_head * head_dim;
            let v_head = &v_all[v_offset..v_offset + head_dim];
            for d in 0..head_dim {
                out_head[d] += w * v_head[d];
            }
        }

        t = tile_end;
    }

    // 最终归一化
    if global_sum > 0.0 {
        let inv_sum = 1.0 / global_sum;
        for d in 0..head_dim {
            out_head[d] *= inv_sum;
        }
    }
}
