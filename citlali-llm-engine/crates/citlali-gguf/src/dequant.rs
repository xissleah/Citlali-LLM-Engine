// crates/citlali-gguf/src/dequant.rs

/// 将 f16 的两个字节（小端）转为 f32
#[inline]
fn f16_to_f32(bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(bytes);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut val = frac as f32 / 1024.0;
        val *= 2.0f32.powi(-14);
        if sign == 1 {
            -val
        } else {
            val
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | 0x7F800000)
        } else {
            f32::NAN
        }
    } else {
        let new_exp = (exp as i32 - 15 + 127) as u32;
        let new_frac = frac << 13;
        f32::from_bits((sign << 31) | (new_exp << 23) | new_frac)
    }
}

/// Q4_K block 大小
pub const Q4_K_BLOCK_SIZE: usize = 256;
/// Q4_K 每个 block 的字节数
pub const Q4_K_BYTES_PER_BLOCK: usize = 144;

/// 反量化一个 Q4_K block（144 字节 → 256 个 f32）
/// 参考 llama.cpp dequantize_row_q4_K:
/// 4 groups × 64 values, 每 group 用 32 字节 qs
/// 低 nibble → values[0..32], 高 nibble → values[32..64]
pub fn dequantize_q4_k_block(block: &[u8], output: &mut [f32]) {
    assert!(block.len() >= Q4_K_BYTES_PER_BLOCK);
    assert!(output.len() >= Q4_K_BLOCK_SIZE);

    let d = f16_to_f32([block[0], block[1]]);
    let dmin = f16_to_f32([block[2], block[3]]);
    let scales_raw = &block[4..16]; // 12 bytes
    let qs = &block[16..144]; // 128 bytes

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    decode_q4k_scales(scales_raw, &mut scales, &mut mins);

    for j in 0..4 {
        let d1 = d * scales[2 * j] as f32;
        let m1 = dmin * mins[2 * j] as f32;
        let d2 = d * scales[2 * j + 1] as f32;
        let m2 = dmin * mins[2 * j + 1] as f32;

        let q = &qs[j * 32..(j + 1) * 32];
        let base = j * 64;

        for l in 0..32 {
            output[base + l] = d1 * (q[l] & 0x0F) as f32 - m1;
            output[base + l + 32] = d2 * ((q[l] >> 4) & 0x0F) as f32 - m2;
        }
    }
}

/// 解码 Q4_K 的 12 字节 scales 为 8 个 scale 和 8 个 min
fn decode_q4k_scales(raw: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for i in 0..4 {
        scales[i] = raw[i] & 63;
        mins[i] = raw[i + 4] & 63;
    }
    for i in 0..4 {
        let low_sc = raw[8 + i] & 0x0F;
        let low_mn = (raw[8 + i] >> 4) & 0x0F;
        let hi_sc = (raw[i] >> 6) & 0x03;
        let hi_mn = (raw[i + 4] >> 6) & 0x03;
        scales[4 + i] = low_sc | (hi_sc << 4);
        mins[4 + i] = low_mn | (hi_mn << 4);
    }
}

/// Q5_K block 大小
pub const Q5_K_BLOCK_SIZE: usize = 256;
/// Q5_K 每个 block 的字节数
pub const Q5_K_BYTES_PER_BLOCK: usize = 176;

/// 反量化一个 Q5_K block（176 字节 → 256 个 f32）
/// 布局: d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
/// 与 Q4_K 相同的 split 布局：每 group 64 值 = 前 32 (low nibble) + 后 32 (high nibble)
/// 额外的第 5 bit 来自 qh 字节
pub fn dequantize_q5_k_block(block: &[u8], output: &mut [f32]) {
    assert!(block.len() >= Q5_K_BYTES_PER_BLOCK);
    assert!(output.len() >= Q5_K_BLOCK_SIZE);

    let d = f16_to_f32([block[0], block[1]]);
    let dmin = f16_to_f32([block[2], block[3]]);
    let scales_raw = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    decode_q4k_scales(scales_raw, &mut scales, &mut mins);

    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for j in 0..4 {
        let d1 = d * scales[2 * j] as f32;
        let m1 = dmin * mins[2 * j] as f32;
        let d2 = d * scales[2 * j + 1] as f32;
        let m2 = dmin * mins[2 * j + 1] as f32;

        let q = &qs[j * 32..(j + 1) * 32];
        let base = j * 64;

        for l in 0..32 {
            let hbit1 = if qh[l] & u1 != 0 { 16u8 } else { 0u8 };
            let hbit2 = if qh[l] & u2 != 0 { 16u8 } else { 0u8 };
            // Q5_K: asymmetric quantization, same as Q4_K
            // dequant = d * q5_value - m  (NO zero point subtraction)
            // q5_value = low_4bits + high_bit*16, range 0..31
            let val1 = ((q[l] & 0x0F) + hbit1) as f32;
            let val2 = ((q[l] >> 4) + hbit2) as f32;
            output[base + l] = d1 * val1 - m1;
            output[base + l + 32] = d2 * val2 - m2;
        }

        u1 <<= 2;
        u2 <<= 2;
    }
}

/// 反量化一整个 tensor 的数据 (Q5_K)
pub fn dequantize_q5_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / Q5_K_BLOCK_SIZE;
    let mut output = vec![0.0f32; num_elements];

    for b in 0..num_blocks {
        let block = &data[b * Q5_K_BYTES_PER_BLOCK..(b + 1) * Q5_K_BYTES_PER_BLOCK];
        let out = &mut output[b * Q5_K_BLOCK_SIZE..(b + 1) * Q5_K_BLOCK_SIZE];
        dequantize_q5_k_block(block, out);
    }
    output
}

/// Q6_K block 大小
pub const Q6_K_BLOCK_SIZE: usize = 256;
/// Q6_K 每个 block 的字节数
pub const Q6_K_BYTES_PER_BLOCK: usize = 210;

/// 反量化一个 Q6_K block（210 字节 → 256 个 f32）
/// 参考 llama.cpp dequantize_row_q6_K:
/// ql[0..128] 低4位, qh[0..64] 高2位, scales[0..16] i8, d (f16)
pub fn dequantize_q6_k_block(block: &[u8], output: &mut [f32]) {
    assert!(block.len() >= Q6_K_BYTES_PER_BLOCK);
    assert!(output.len() >= Q6_K_BLOCK_SIZE);

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32([block[208], block[209]]);

    for n in 0..2 {
        let ql_off = n * 64;
        let qh_off = n * 32;
        let sc_off = n * 8;
        let y_off = n * 128;

        for l in 0..32 {
            let is = l / 16;

            let q1 = (ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4);
            let q2 = (ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4);
            let q3 = ((ql[ql_off + l] >> 4) & 0xF) | (((qh[qh_off + l] >> 4) & 3) << 4);
            let q4 = ((ql[ql_off + l + 32] >> 4) & 0xF) | (((qh[qh_off + l] >> 6) & 3) << 4);

            let sc1 = (scales[sc_off + is] as i8) as f32;
            let sc2 = (scales[sc_off + is + 2] as i8) as f32;
            let sc3 = (scales[sc_off + is + 4] as i8) as f32;
            let sc4 = (scales[sc_off + is + 6] as i8) as f32;

            output[y_off + l] = d * sc1 * (q1 as i8 - 32) as f32;
            output[y_off + l + 32] = d * sc2 * (q2 as i8 - 32) as f32;
            output[y_off + l + 64] = d * sc3 * (q3 as i8 - 32) as f32;
            output[y_off + l + 96] = d * sc4 * (q4 as i8 - 32) as f32;
        }
    }
}

/// 反量化一整个 tensor 的数据 (Q4_K)
pub fn dequantize_q4_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / Q4_K_BLOCK_SIZE;
    let mut output = vec![0.0f32; num_elements];

    for b in 0..num_blocks {
        let block = &data[b * Q4_K_BYTES_PER_BLOCK..(b + 1) * Q4_K_BYTES_PER_BLOCK];
        let out = &mut output[b * Q4_K_BLOCK_SIZE..(b + 1) * Q4_K_BLOCK_SIZE];
        dequantize_q4_k_block(block, out);
    }
    output
}

// ============================================================
// Fused vec_dot 内核：直接从量化数据计算点积，零中间缓冲区
// 核心思想：dequant(q) = d * scale * nibble - dmin * min
// 点积 = Σ dequant(q) * x = d*scale*Σ(nibble*x) - dmin*min*Σ(x)
// ============================================================

/// Fused Q4_K 整行点积：row_data 是一行量化数据，input 是 f32 输入向量
/// blocks_per_row = in_dim / 256
#[inline]
pub fn vec_dot_q4_k(row_data: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block = &row_data[b * Q4_K_BYTES_PER_BLOCK..(b + 1) * Q4_K_BYTES_PER_BLOCK];
        let inp = &input[b * Q4_K_BLOCK_SIZE..];
        sum += vec_dot_q4_k_block(block, inp);
    }
    sum
}

/// Fused Q4_K 单 block 点积（144 字节 block × 256 个 f32 输入）
/// 不分配中间 f32 缓冲区，直接在寄存器中完成 dequant + 累加
#[inline]
fn vec_dot_q4_k_block(block: &[u8], input: &[f32]) -> f32 {
    let d = f16_to_f32([block[0], block[1]]);
    let dmin = f16_to_f32([block[2], block[3]]);
    let scales_raw = &block[4..16];
    let qs = &block[16..144];

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    decode_q4k_scales(scales_raw, &mut scales, &mut mins);

    let mut sum = 0.0f32;

    for j in 0..4 {
        let d1 = d * scales[2 * j] as f32;
        let m1 = dmin * mins[2 * j] as f32;
        let d2 = d * scales[2 * j + 1] as f32;
        let m2 = dmin * mins[2 * j + 1] as f32;

        let q = &qs[j * 32..(j + 1) * 32];
        let base = j * 64;

        // 4路展开：nibble*x 累加 + x 累加，全在寄存器里
        let mut sumi1_a = 0.0f32;
        let mut sumi1_b = 0.0f32;
        let mut sumx1_a = 0.0f32;
        let mut sumx1_b = 0.0f32;
        let mut sumi2_a = 0.0f32;
        let mut sumi2_b = 0.0f32;
        let mut sumx2_a = 0.0f32;
        let mut sumx2_b = 0.0f32;

        let inp_lo = &input[base..base + 32];
        let inp_hi = &input[base + 32..base + 64];

        let chunks = 32 / 2;
        for l in 0..chunks {
            let l0 = l * 2;
            let l1 = l0 + 1;

            let x1_0 = inp_lo[l0];
            let x1_1 = inp_lo[l1];
            let x2_0 = inp_hi[l0];
            let x2_1 = inp_hi[l1];

            sumi1_a += (q[l0] & 0x0F) as f32 * x1_0;
            sumi1_b += (q[l1] & 0x0F) as f32 * x1_1;
            sumx1_a += x1_0;
            sumx1_b += x1_1;

            sumi2_a += ((q[l0] >> 4) & 0x0F) as f32 * x2_0;
            sumi2_b += ((q[l1] >> 4) & 0x0F) as f32 * x2_1;
            sumx2_a += x2_0;
            sumx2_b += x2_1;
        }

        let sumi1 = sumi1_a + sumi1_b;
        let sumx1 = sumx1_a + sumx1_b;
        let sumi2 = sumi2_a + sumi2_b;
        let sumx2 = sumx2_a + sumx2_b;

        sum += d1 * sumi1 - m1 * sumx1 + d2 * sumi2 - m2 * sumx2;
    }
    sum
}

/// 反量化一整个 tensor 的数据 (Q6_K)
pub fn dequantize_q6_k(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / Q6_K_BLOCK_SIZE;
    let mut output = vec![0.0f32; num_elements];

    for b in 0..num_blocks {
        let block = &data[b * Q6_K_BYTES_PER_BLOCK..(b + 1) * Q6_K_BYTES_PER_BLOCK];
        let out = &mut output[b * Q6_K_BLOCK_SIZE..(b + 1) * Q6_K_BLOCK_SIZE];
        dequantize_q6_k_block(block, out);
    }
    output
}

/// Fused Q6_K 整行点积
#[inline]
pub fn vec_dot_q6_k(row_data: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block = &row_data[b * Q6_K_BYTES_PER_BLOCK..(b + 1) * Q6_K_BYTES_PER_BLOCK];
        let inp = &input[b * Q6_K_BLOCK_SIZE..];
        sum += vec_dot_q6_k_block(block, inp);
    }
    sum
}

/// Fused Q6_K 单 block 点积（210 字节 block × 256 个 f32 输入）
/// Q6_K: dequant = d * scale * (q6 - 32)
/// 点积 = d * Σ scale * (q6 - 32) * x = d * (Σ scale*q6*x - 32*Σ scale*x)
#[inline]
fn vec_dot_q6_k_block(block: &[u8], input: &[f32]) -> f32 {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32([block[208], block[209]]);

    let mut sum = 0.0f32;

    for n in 0..2 {
        let ql_off = n * 64;
        let qh_off = n * 32;
        let sc_off = n * 8;
        let y_off = n * 128;

        for l in 0..32 {
            let is = l / 16;

            let q1 = (ql[ql_off + l] & 0xF) | (((qh[qh_off + l] >> 0) & 3) << 4);
            let q2 = (ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4);
            let q3 = ((ql[ql_off + l] >> 4) & 0xF) | (((qh[qh_off + l] >> 4) & 3) << 4);
            let q4 = ((ql[ql_off + l + 32] >> 4) & 0xF) | (((qh[qh_off + l] >> 6) & 3) << 4);

            let sc1 = (scales[sc_off + is] as i8) as f32;
            let sc2 = (scales[sc_off + is + 2] as i8) as f32;
            let sc3 = (scales[sc_off + is + 4] as i8) as f32;
            let sc4 = (scales[sc_off + is + 6] as i8) as f32;

            let x1 = input[y_off + l];
            let x2 = input[y_off + l + 32];
            let x3 = input[y_off + l + 64];
            let x4 = input[y_off + l + 96];

            sum += sc1 * (q1 as i8 - 32) as f32 * x1;
            sum += sc2 * (q2 as i8 - 32) as f32 * x2;
            sum += sc3 * (q3 as i8 - 32) as f32 * x3;
            sum += sc4 * (q4 as i8 - 32) as f32 * x4;
        }
    }
    sum * d
}

/// Fused Q5_K 整行点积
#[inline]
pub fn vec_dot_q5_k(row_data: &[u8], input: &[f32], blocks_per_row: usize) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..blocks_per_row {
        let block = &row_data[b * Q5_K_BYTES_PER_BLOCK..(b + 1) * Q5_K_BYTES_PER_BLOCK];
        let inp = &input[b * Q5_K_BLOCK_SIZE..];
        sum += vec_dot_q5_k_block(block, inp);
    }
    sum
}

/// Fused Q5_K 单 block 点积（176 字节 block × 256 个 f32 输入）
#[inline]
fn vec_dot_q5_k_block(block: &[u8], input: &[f32]) -> f32 {
    let d = f16_to_f32([block[0], block[1]]);
    let dmin = f16_to_f32([block[2], block[3]]);
    let scales_raw = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    decode_q4k_scales(scales_raw, &mut scales, &mut mins);

    let mut sum = 0.0f32;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;

    for j in 0..4 {
        let d1 = d * scales[2 * j] as f32;
        let m1 = dmin * mins[2 * j] as f32;
        let d2 = d * scales[2 * j + 1] as f32;
        let m2 = dmin * mins[2 * j + 1] as f32;

        let q = &qs[j * 32..(j + 1) * 32];
        let base = j * 64;
        let inp_lo = &input[base..base + 32];
        let inp_hi = &input[base + 32..base + 64];

        let mut sumi1 = 0.0f32;
        let mut sumx1 = 0.0f32;
        let mut sumi2 = 0.0f32;
        let mut sumx2 = 0.0f32;

        for l in 0..32 {
            let hbit1 = if qh[l] & u1 != 0 { 16u8 } else { 0u8 };
            let hbit2 = if qh[l] & u2 != 0 { 16u8 } else { 0u8 };
            let val1 = ((q[l] & 0x0F) + hbit1) as f32;
            let val2 = ((q[l] >> 4) + hbit2) as f32;

            sumi1 += val1 * inp_lo[l];
            sumx1 += inp_lo[l];
            sumi2 += val2 * inp_hi[l];
            sumx2 += inp_hi[l];
        }

        sum += d1 * sumi1 - m1 * sumx1 + d2 * sumi2 - m2 * sumx2;
        u1 <<= 2;
        u2 <<= 2;
    }
    sum
}
