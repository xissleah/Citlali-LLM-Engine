// crates/citlali-model/src/mmap_model.rs
//! 内存映射模型加载：mmap GGUF 文件，按需反量化
//!
//! 与 Resident 模式的区别：
//! - Resident: 启动时全量反量化到 Vec<f32>，内存 ≈ 2.4GB
//! - MemoryMapped: mmap 文件 + 按层反量化缓冲，内存 ≈ 50MB 工作集
//!
//! 代价：每层 forward 需要实时反量化，推理速度下降 ~20-40%

use std::collections::HashMap;
use std::fs::File;

use citlali_gguf::dequant::{
    dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, Q4_K_BLOCK_SIZE, Q4_K_BYTES_PER_BLOCK,
    Q5_K_BLOCK_SIZE, Q5_K_BYTES_PER_BLOCK, Q6_K_BLOCK_SIZE, Q6_K_BYTES_PER_BLOCK,
};
use citlali_gguf::GgufFile;
use memmap2::Mmap;

use crate::model::ModelConfig;
use crate::tokenizer::Tokenizer;

/// Tensor 在 GGUF 文件中的位置信息
#[derive(Debug, Clone)]
struct TensorLocation {
    /// 相对于 data_offset 的偏移
    offset: u64,
    /// 元素总数
    num_elements: usize,
    /// 量化类型 (0=F32, 12=Q4_K, 14=Q6_K)
    dtype: u32,
    /// 原始数据字节数
    data_size: usize,
}

/// 内存映射模型：持有 mmap 句柄，按需反量化
pub struct MmapQwen3Model {
    pub config: ModelConfig,
    /// mmap 映射的 GGUF 文件
    mmap: Mmap,
    /// 数据段起始偏移
    data_offset: u64,
    /// tensor name → 位置信息
    tensor_locations: HashMap<String, TensorLocation>,
    /// token embedding（常驻，因为每次 forward 都需要）
    pub token_embd: Vec<f32>,
    /// output norm（常驻，每次 forward 都需要）
    pub output_norm: Vec<f32>,
    /// LM head 权重：如果模型有独立的 output.weight 则使用它，否则 tied 到 token_embd
    pub output_weight: Option<Vec<f32>>,
}

impl MmapQwen3Model {
    /// 从 GGUF 文件创建 mmap 模型
    pub fn load(path: &str) -> std::io::Result<Self> {
        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf);

        println!("模型配置: {:?}", config);
        println!("内存映射模式加载...");

        // mmap 整个文件
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // 构建 tensor 位置索引
        let mut tensor_locations = HashMap::new();
        for t in &gguf.tensors {
            let num_elements: u64 = t.dimensions.iter().product();
            let (data_size, _) = compute_tensor_size(t.dtype, num_elements as usize);
            tensor_locations.insert(
                t.name.clone(),
                TensorLocation {
                    offset: t.offset,
                    num_elements: num_elements as usize,
                    dtype: t.dtype,
                    data_size,
                },
            );
        }

        // 反量化常驻 tensor（embedding + output_norm + output_weight）
        let token_embd = dequant_from_mmap(
            &mmap,
            gguf.data_offset,
            &tensor_locations,
            "token_embd.weight",
        )?;
        let output_norm = dequant_from_mmap(
            &mmap,
            gguf.data_offset,
            &tensor_locations,
            "output_norm.weight",
        )?;

        // 加载独立的 output.weight（如果存在）
        let output_weight = if tensor_locations.contains_key("output.weight") {
            let w = dequant_from_mmap(
                &mmap,
                gguf.data_offset,
                &tensor_locations,
                "output.weight",
            )?;
            println!("  检测到独立 output.weight (untied embeddings)");
            Some(w)
        } else {
            println!("  使用 tied embeddings (token_embd = LM head)");
            None
        };

        println!("内存映射加载完成！仅常驻 embedding + norm");

        Ok(Self {
            config,
            mmap,
            data_offset: gguf.data_offset,
            tensor_locations,
            token_embd,
            output_norm,
            output_weight,
        })
    }

    /// 加载模型 + tokenizer
    pub fn load_with_tokenizer(path: &str) -> std::io::Result<(Self, Tokenizer)> {
        let gguf = GgufFile::open(path)?;
        let tokenizer = Tokenizer::from_gguf(&gguf);
        drop(gguf);
        let model = Self::load(path)?;
        Ok((model, tokenizer))
    }

    /// 按需反量化指定层的权重（调用方负责缓存/释放）
    pub fn dequant_layer(&self, layer_idx: usize) -> crate::model::LayerWeights {
        let prefix = format!("blk.{}", layer_idx);
        let q_norm_name = format!("{}.attn_q_norm.weight", prefix);
        let k_norm_name = format!("{}.attn_k_norm.weight", prefix);
        crate::model::LayerWeights {
            attn_norm: self.dequant_tensor(&format!("{}.attn_norm.weight", prefix)),
            attn_q: self.dequant_tensor(&format!("{}.attn_q.weight", prefix)),
            attn_k: self.dequant_tensor(&format!("{}.attn_k.weight", prefix)),
            attn_v: self.dequant_tensor(&format!("{}.attn_v.weight", prefix)),
            attn_output: self.dequant_tensor(&format!("{}.attn_output.weight", prefix)),
            attn_q_norm: self.try_dequant_tensor(&q_norm_name),
            attn_k_norm: self.try_dequant_tensor(&k_norm_name),
            ffn_norm: self.dequant_tensor(&format!("{}.ffn_norm.weight", prefix)),
            ffn_gate: self.dequant_tensor(&format!("{}.ffn_gate.weight", prefix)),
            ffn_up: self.dequant_tensor(&format!("{}.ffn_up.weight", prefix)),
            ffn_down: self.dequant_tensor(&format!("{}.ffn_down.weight", prefix)),
        }
    }

    /// 从 mmap 中反量化单个 tensor
    pub(crate) fn dequant_tensor(&self, name: &str) -> Vec<f32> {
        let loc = self
            .tensor_locations
            .get(name)
            .unwrap_or_else(|| panic!("tensor not found: {}", name));
        let start = (self.data_offset + loc.offset) as usize;
        let raw = &self.mmap[start..start + loc.data_size];
        dequant_raw(raw, loc.dtype, loc.num_elements)
    }

    /// 尝试反量化 tensor，不存在则返回 None
    pub(crate) fn try_dequant_tensor(&self, name: &str) -> Option<Vec<f32>> {
        if self.tensor_locations.contains_key(name) {
            Some(self.dequant_tensor(name))
        } else {
            None
        }
    }

    /// 获取 tensor 的原始字节切片（不反量化）+ dtype
    /// 用于 quantized matmul 直接在量化数据上计算
    pub fn get_raw_tensor(&self, name: &str) -> (&[u8], u32) {
        let loc = self
            .tensor_locations
            .get(name)
            .unwrap_or_else(|| panic!("tensor not found: {}", name));
        let start = (self.data_offset + loc.offset) as usize;
        (&self.mmap[start..start + loc.data_size], loc.dtype)
    }
}

// ─── 辅助函数 ───────────────────────────────────

/// 计算 tensor 的原始数据大小和反量化函数
fn compute_tensor_size(dtype: u32, num_elements: usize) -> (usize, u32) {
    match dtype {
        12 => {
            // Q4_K
            let num_blocks = num_elements / Q4_K_BLOCK_SIZE;
            (num_blocks * Q4_K_BYTES_PER_BLOCK, 12)
        }
        13 => {
            // Q5_K
            let num_blocks = num_elements / Q5_K_BLOCK_SIZE;
            (num_blocks * Q5_K_BYTES_PER_BLOCK, 13)
        }
        14 => {
            // Q6_K
            let num_blocks = num_elements / Q6_K_BLOCK_SIZE;
            (num_blocks * Q6_K_BYTES_PER_BLOCK, 14)
        }
        0 => {
            // F32
            (num_elements * 4, 0)
        }
        _ => (0, dtype),
    }
}

/// 从原始字节反量化
fn dequant_raw(raw: &[u8], dtype: u32, num_elements: usize) -> Vec<f32> {
    match dtype {
        12 => dequantize_q4_k(raw, num_elements),
        13 => dequantize_q5_k(raw, num_elements),
        14 => dequantize_q6_k(raw, num_elements),
        0 => {
            let mut out = vec![0.0f32; num_elements];
            for i in 0..num_elements {
                let bytes = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                out[i] = f32::from_le_bytes(bytes);
            }
            out
        }
        _ => panic!("unsupported dtype: {}", dtype),
    }
}

/// 从 mmap 中反量化指定 tensor
fn dequant_from_mmap(
    mmap: &Mmap,
    data_offset: u64,
    locations: &HashMap<String, TensorLocation>,
    name: &str,
) -> std::io::Result<Vec<f32>> {
    let loc = locations.get(name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("tensor not found: {}", name),
        )
    })?;
    let start = (data_offset + loc.offset) as usize;
    let raw = &mmap[start..start + loc.data_size];
    Ok(dequant_raw(raw, loc.dtype, loc.num_elements))
}
