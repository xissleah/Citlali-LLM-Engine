// crates/citlali-model/src/model.rs
//! Qwen3-0.6B 模型定义与权重加载

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use citlali_gguf::dequant::{
    dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, Q4_K_BLOCK_SIZE, Q4_K_BYTES_PER_BLOCK,
    Q5_K_BLOCK_SIZE, Q5_K_BYTES_PER_BLOCK, Q6_K_BLOCK_SIZE, Q6_K_BYTES_PER_BLOCK,
};
use citlali_gguf::GgufFile;

/// 模型超参数（支持 Qwen3 / Llama 等架构）
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// 是否有 QK-Norm（Qwen3 有，Llama 没有）
    pub has_qk_norm: bool,
    /// RoPE 旋转维度数（0 表示 full RoPE = head_dim）
    pub rope_dim: usize,
    /// RoPE 类型：0 = adjacent/norm (Llama), 2 = split/neox (Qwen)
    pub rope_type: u32,
}

impl ModelConfig {
    /// 从 GGUF metadata 中提取配置（自动检测架构）
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        // 检测架构前缀
        let arch = gguf
            .get_metadata("general.architecture")
            .and_then(|v| match v {
                citlali_gguf::metadata::MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "qwen3".to_string());

        let get_u32 = |key: &str, default: u32| -> u32 {
            gguf.get_metadata(key)
                .and_then(|v| match v {
                    citlali_gguf::metadata::MetadataValue::Uint32(n) => Some(*n),
                    _ => None,
                })
                .unwrap_or(default)
        };
        let get_f32 = |key: &str, default: f32| -> f32 {
            gguf.get_metadata(key)
                .and_then(|v| match v {
                    citlali_gguf::metadata::MetadataValue::Float32(n) => Some(*n),
                    _ => None,
                })
                .unwrap_or(default)
        };

        let num_heads = get_u32(&format!("{}.attention.head_count", arch), 32) as usize;
        let num_kv_heads = get_u32(
            &format!("{}.attention.head_count_kv", arch),
            num_heads as u32,
        ) as usize;
        let hidden_dim = get_u32(&format!("{}.embedding_length", arch), 2048) as usize;
        let num_layers = get_u32(&format!("{}.block_count", arch), 16) as usize;
        let intermediate_dim =
            get_u32(&format!("{}.feed_forward_length", arch), 8192) as usize;
        let head_dim = get_u32(
            &format!("{}.attention.key_length", arch),
            (hidden_dim / num_heads) as u32,
        ) as usize;
        let rms_norm_eps = get_f32(
            &format!("{}.attention.layer_norm_rms_epsilon", arch),
            1e-5,
        );
        let rope_theta = get_f32(&format!("{}.rope.freq_base", arch), 500_000.0);

        // vocab_size 从 token_embd tensor 维度推断
        // GGUF 遵循 ggml 约定：dimensions[0] = 最内层（hidden_dim），dimensions[1] = 外层（vocab_size）
        let vocab_size = gguf
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .map(|t| {
                if t.dimensions.len() >= 2 {
                    t.dimensions[1] as usize
                } else {
                    t.dimensions[0] as usize
                }
            })
            .unwrap_or(128256);

        // QK-Norm: Qwen3 有，Llama 没有
        let has_qk_norm = gguf
            .tensors
            .iter()
            .any(|t| t.name == "blk.0.attn_q_norm.weight");

        // RoPE dimension count (partial RoPE support)
        // Default to head_dim (full RoPE) if not specified
        let rope_dim = get_u32(
            &format!("{}.rope.dimension_count", arch),
            head_dim as u32,
        ) as usize;

        // RoPE type: 根据架构决定配对方式
        // Llama 系列: GGUF 中 Q/K 权重已被 permute，使用 adjacent pairs (type=0)
        // Qwen 系列: GGUF 中 Q/K 权重未 permute，使用 split pairs (type=2)
        let rope_type = match arch.as_str() {
            "llama" => 0u32,  // GGML_ROPE_TYPE_NORM: adjacent (2i, 2i+1)
            _ => 2u32,        // GGML_ROPE_TYPE_NEOX: split (i, i+dim/2)
        };

        println!(
            "检测到架构: {} (has_qk_norm={}, rope_dim={}, rope_type={})",
            arch, has_qk_norm, rope_dim, rope_type
        );

        Self {
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            rms_norm_eps,
            rope_theta,
            has_qk_norm,
            rope_dim,
            rope_type,
        }
    }
}

/// 一个 Transformer 层的权重（已反量化为 f32）
pub struct LayerWeights {
    // Attention
    pub attn_norm: Vec<f32>,   // [hidden_dim]
    pub attn_q: Vec<f32>,      // [num_heads * head_dim, hidden_dim]
    pub attn_k: Vec<f32>,      // [num_kv_heads * head_dim, hidden_dim]
    pub attn_v: Vec<f32>,      // [num_kv_heads * head_dim, hidden_dim]
    pub attn_output: Vec<f32>, // [hidden_dim, num_heads * head_dim]
    pub attn_q_norm: Option<Vec<f32>>, // [head_dim] — Qwen3 only
    pub attn_k_norm: Option<Vec<f32>>, // [head_dim] — Qwen3 only

    // FFN (SwiGLU)
    pub ffn_norm: Vec<f32>, // [hidden_dim]
    pub ffn_gate: Vec<f32>, // [intermediate_dim, hidden_dim]
    pub ffn_up: Vec<f32>,   // [intermediate_dim, hidden_dim]
    pub ffn_down: Vec<f32>, // [hidden_dim, intermediate_dim]
}

/// 完整模型权重
pub struct Qwen3Model {
    pub config: ModelConfig,
    pub token_embd: Vec<f32>,  // [vocab_size, hidden_dim]
    pub output_norm: Vec<f32>, // [hidden_dim]
    pub layers: Vec<LayerWeights>,
    /// LM head 权重：如果模型有独立的 output.weight 则使用它，否则 tied 到 token_embd
    pub output_weight: Option<Vec<f32>>,
}

impl Qwen3Model {
    /// 从 GGUF 文件加载模型（反量化所有权重到 f32）
    pub fn load(path: &str) -> std::io::Result<Self> {
        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf);

        println!("模型配置: {:?}", config);
        println!("加载权重...");

        // 建立 tensor name → TensorInfo 的映射
        let tensor_map: HashMap<&str, &citlali_gguf::tensor::TensorInfo> =
            gguf.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut file = File::open(path)?;

        // 加载 embedding
        let token_embd = load_tensor(&mut file, &tensor_map, &gguf, "token_embd.weight")?;
        println!("  token_embd: {} 元素", token_embd.len());

        // 加载 output norm
        let output_norm = load_tensor(&mut file, &tensor_map, &gguf, "output_norm.weight")?;

        // 加载 output weight（如果存在独立的 LM head）
        let output_weight = try_load_tensor(&mut file, &tensor_map, &gguf, "output.weight")?;
        if output_weight.is_some() {
            println!("  检测到独立 output.weight (untied embeddings)");
        } else {
            println!("  使用 tied embeddings (token_embd = LM head)");
        }

        // 加载各层
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            print!("  加载层 {}/{}...\r", i + 1, config.num_layers);
            let layer = load_layer(&mut file, &tensor_map, &gguf, i)?;
            layers.push(layer);
        }
        println!("\n权重加载完成！");

        Ok(Self {
            config,
            token_embd,
            output_norm,
            layers,
            output_weight,
        })
    }
    /// 加载模型 + tokenizer
    pub fn load_with_tokenizer(path: &str) -> std::io::Result<(Self, crate::tokenizer::Tokenizer)> {
        let gguf = GgufFile::open(path)?;
        let config = ModelConfig::from_gguf(&gguf);
        let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf);

        println!("模型配置: {:?}", config);
        println!("加载权重...");

        let tensor_map: HashMap<&str, &citlali_gguf::tensor::TensorInfo> =
            gguf.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        let mut file = File::open(path)?;

        let token_embd = load_tensor(&mut file, &tensor_map, &gguf, "token_embd.weight")?;
        println!("  token_embd: {} 元素", token_embd.len());

        let output_norm = load_tensor(&mut file, &tensor_map, &gguf, "output_norm.weight")?;

        // 加载 output weight（如果存在独立的 LM head）
        let output_weight = try_load_tensor(&mut file, &tensor_map, &gguf, "output.weight")?;
        if output_weight.is_some() {
            println!("  检测到独立 output.weight (untied embeddings)");
        } else {
            println!("  使用 tied embeddings (token_embd = LM head)");
        }

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            print!("  加载层 {}/{}...\r", i + 1, config.num_layers);
            let layer = load_layer(&mut file, &tensor_map, &gguf, i)?;
            layers.push(layer);
        }
        println!("\n权重加载完成！");

        let model = Self {
            config,
            token_embd,
            output_norm,
            layers,
            output_weight,
        };

        Ok((model, tokenizer))
    }
}

/// 加载单个 tensor 并反量化
fn load_tensor(
    file: &mut File,
    tensor_map: &HashMap<&str, &citlali_gguf::tensor::TensorInfo>,
    gguf: &GgufFile,
    name: &str,
) -> std::io::Result<Vec<f32>> {
    let info = tensor_map.get(name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("tensor not found: {}", name),
        )
    })?;

    let num_elements: u64 = info.dimensions.iter().product();
    let dtype = info.dtype;

    // 计算原始数据大小
    let (data_size, dequant_fn): (usize, fn(&[u8], usize) -> Vec<f32>) = match dtype {
        12 => {
            // Q4K
            let num_blocks = num_elements as usize / Q4_K_BLOCK_SIZE;
            (num_blocks * Q4_K_BYTES_PER_BLOCK, dequantize_q4_k)
        }
        13 => {
            // Q5_K
            let num_blocks = num_elements as usize / Q5_K_BLOCK_SIZE;
            (num_blocks * Q5_K_BYTES_PER_BLOCK, dequantize_q5_k)
        }
        14 => {
            // Q6K
            let num_blocks = num_elements as usize / Q6_K_BLOCK_SIZE;
            (num_blocks * Q6_K_BYTES_PER_BLOCK, dequantize_q6_k)
        }
        0 => {
            // F32
            (num_elements as usize * 4, |data: &[u8], n: usize| {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let bytes = [
                        data[i * 4],
                        data[i * 4 + 1],
                        data[i * 4 + 2],
                        data[i * 4 + 3],
                    ];
                    out[i] = f32::from_le_bytes(bytes);
                }
                out
            })
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!("unsupported dtype {} for tensor {}", dtype, name),
            ));
        }
    };

    file.seek(SeekFrom::Start(gguf.data_offset + info.offset))?;
    let mut raw = vec![0u8; data_size];
    file.read_exact(&mut raw)?;

    Ok(dequant_fn(&raw, num_elements as usize))
}

/// 尝试加载 tensor，不存在则返回 None
fn try_load_tensor(
    file: &mut File,
    tensor_map: &HashMap<&str, &citlali_gguf::tensor::TensorInfo>,
    gguf: &GgufFile,
    name: &str,
) -> std::io::Result<Option<Vec<f32>>> {
    if tensor_map.contains_key(name) {
        Ok(Some(load_tensor(file, tensor_map, gguf, name)?))
    } else {
        Ok(None)
    }
}

/// 加载一个 transformer 层的所有权重
fn load_layer(
    file: &mut File,
    tensor_map: &HashMap<&str, &citlali_gguf::tensor::TensorInfo>,
    gguf: &GgufFile,
    layer_idx: usize,
) -> std::io::Result<LayerWeights> {
    let prefix = format!("blk.{}", layer_idx);

    let attn_norm = load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.attn_norm.weight", prefix),
    )?;
    let attn_q = load_tensor(file, tensor_map, gguf, &format!("{}.attn_q.weight", prefix))?;
    let attn_k = load_tensor(file, tensor_map, gguf, &format!("{}.attn_k.weight", prefix))?;
    let attn_v = load_tensor(file, tensor_map, gguf, &format!("{}.attn_v.weight", prefix))?;
    let attn_output = load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.attn_output.weight", prefix),
    )?;
    let attn_q_norm = try_load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.attn_q_norm.weight", prefix),
    )?;
    let attn_k_norm = try_load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.attn_k_norm.weight", prefix),
    )?;
    let ffn_norm = load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.ffn_norm.weight", prefix),
    )?;
    let ffn_gate = load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.ffn_gate.weight", prefix),
    )?;
    let ffn_up = load_tensor(file, tensor_map, gguf, &format!("{}.ffn_up.weight", prefix))?;
    let ffn_down = load_tensor(
        file,
        tensor_map,
        gguf,
        &format!("{}.ffn_down.weight", prefix),
    )?;

    Ok(LayerWeights {
        attn_norm,
        attn_q,
        attn_k,
        attn_v,
        attn_output,
        attn_q_norm,
        attn_k_norm,
        ffn_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
    })
}
