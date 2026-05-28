// crates/citlali-model/src/quantized_ops.rs
//! 量化矩阵-向量乘法：逐 block 解压 + 点积，不分配整个 f32 权重
//!
//! 内存占用：仅需一个 BLOCK_SIZE (256) 的栈缓冲区/线程
//! 代价：每次 matmul 都要解压（无缓存），但省去了整层 ~300MB 的 f32 分配

use citlali_gguf::dequant::{
    Q4_K_BLOCK_SIZE, Q4_K_BYTES_PER_BLOCK, Q5_K_BLOCK_SIZE, Q5_K_BYTES_PER_BLOCK,
    Q6_K_BLOCK_SIZE, Q6_K_BYTES_PER_BLOCK, vec_dot_q4_k, vec_dot_q5_k, vec_dot_q6_k,
};
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 256;

/// 量化矩阵-向量乘法: output = quantized_weight @ input
/// raw_weight: 量化后的原始字节 [out_dim rows, 每行 in_dim 个元素]
/// input: f32 输入向量 [in_dim]
/// output: f32 输出向量 [out_dim]
pub fn quantized_mat_vec_mul_into(
    raw_weight: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    dtype: u32,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), in_dim);
    debug_assert_eq!(output.len(), out_dim);

    let (block_size, bytes_per_block) = match dtype {
        12 => (Q4_K_BLOCK_SIZE, Q4_K_BYTES_PER_BLOCK),
        13 => (Q5_K_BLOCK_SIZE, Q5_K_BYTES_PER_BLOCK),
        14 => (Q6_K_BLOCK_SIZE, Q6_K_BYTES_PER_BLOCK),
        _ => panic!("quantized_mat_vec_mul_into: unsupported dtype {}", dtype),
    };

    let blocks_per_row = in_dim / block_size;
    let row_bytes = blocks_per_row * bytes_per_block;

    if out_dim >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let row_data = &raw_weight[i * row_bytes..(i + 1) * row_bytes];
            *out = quantized_dot_product(row_data, input, dtype, blocks_per_row, block_size, bytes_per_block);
        });
    } else {
        for i in 0..out_dim {
            let row_data = &raw_weight[i * row_bytes..(i + 1) * row_bytes];
            output[i] = quantized_dot_product(row_data, input, dtype, blocks_per_row, block_size, bytes_per_block);
        }
    }
}

/// 通用分发：根据 dtype 选择 quantized 或 f32 路径
pub fn raw_mat_vec_mul_into(
    raw_weight: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    dtype: u32,
    output: &mut [f32],
) {
    match dtype {
        0 => f32_mat_vec_mul_into(raw_weight, input, out_dim, in_dim, output),
        12 | 13 | 14 => quantized_mat_vec_mul_into(raw_weight, input, out_dim, in_dim, dtype, output),
        _ => panic!("raw_mat_vec_mul_into: unsupported dtype {}", dtype),
    }
}

/// Fused 点积：直接从量化数据计算点积，零中间缓冲区
/// 利用代数恒等式避免逐元素 dequant：
///   Σ dequant(q)*x = d*scale*Σ(nibble*x) - dmin*min*Σ(x)
#[inline]
fn quantized_dot_product(
    row_data: &[u8],
    input: &[f32],
    dtype: u32,
    blocks_per_row: usize,
    _block_size: usize,
    _bytes_per_block: usize,
) -> f32 {
    match dtype {
        12 => vec_dot_q4_k(row_data, input, blocks_per_row),
        13 => vec_dot_q5_k(row_data, input, blocks_per_row),
        14 => vec_dot_q6_k(row_data, input, blocks_per_row),
        _ => unreachable!(),
    }
}
pub fn f32_mat_vec_mul_into(
    raw_weight: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    output: &mut [f32],
) {
    let row_bytes = in_dim * 4;
    if out_dim >= PAR_THRESHOLD {
        output.par_iter_mut().enumerate().for_each(|(i, out)| {
            let row = &raw_weight[i * row_bytes..(i + 1) * row_bytes];
            let mut sum = 0.0f32;
            for j in 0..in_dim {
                let b = [row[j*4], row[j*4+1], row[j*4+2], row[j*4+3]];
                sum += f32::from_le_bytes(b) * input[j];
            }
            *out = sum;
        });
    } else {
        for i in 0..out_dim {
            let row = &raw_weight[i * row_bytes..(i + 1) * row_bytes];
            let mut sum = 0.0f32;
            for j in 0..in_dim {
                let b = [row[j*4], row[j*4+1], row[j*4+2], row[j*4+3]];
                sum += f32::from_le_bytes(b) * input[j];
            }
            output[i] = sum;
        }
    }
}
