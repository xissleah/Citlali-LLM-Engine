// crates/citlali-runtime/src/config.rs
//! 引擎配置：用户可自定义的运行时行为
//!
//! 两个核心选项：
//! - 模型权重加载策略（常驻内存 vs 内存映射）
//! - KV Cache 淘汰策略（Swap 到磁盘 vs 直接丢弃）

/// 模型权重加载策略
#[derive(Debug, Clone)]
pub enum WeightStrategy {
    /// 全部反量化到内存（默认）
    /// 优点：推理速度最快，无额外开销
    /// 缺点：内存占用大（Qwen3-0.6B Q4_K ≈ 2.4GB f32）
    Resident,

    /// 内存映射 GGUF 文件，按层按需反量化
    /// 优点：内存占用极低（仅当前层的反量化缓冲 ≈ 50MB）
    /// 缺点：每层需要实时反量化，推理速度下降 ~20-40%
    MemoryMapped,
}

/// KV Cache 淘汰策略
#[derive(Debug, Clone)]
pub enum KvEvictionPolicy {
    /// 淘汰的页写入 swap 文件，需要时可读回（默认）
    /// 适用于：长上下文对话，需要回溯历史
    Swap {
        /// Swap 文件路径
        path: String,
    },

    /// 直接丢弃淘汰的页，数据不可恢复
    /// 适用于：流式生成、一次性推理、内存极度受限
    /// 注意：如果 attention 需要访问已丢弃的 position，会 panic
    Drop,
}

/// 引擎运行时配置
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// 模型权重加载策略
    pub weight_strategy: WeightStrategy,

    /// KV Cache 淘汰策略
    pub kv_eviction: KvEvictionPolicy,

    /// KV Cache 内存页数上限
    /// 每页 16KB，默认 512 页 = 8MB
    /// 设为 0 表示自动计算（根据 max_tokens 预留足够页数）
    pub kv_max_pages: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            weight_strategy: WeightStrategy::Resident,
            // 默认 Drop 策略：KV Cache 满了直接丢弃，不写磁盘
            // （Swap 仅用于极长上下文场景，且需要足够大的 kv_max_pages）
            kv_eviction: KvEvictionPolicy::Drop,
            // 0 = 自动计算（根据 max_tokens + 模型配置）
            kv_max_pages: 0,
        }
    }
}

impl EngineConfig {
    /// 常驻内存 + Drop（默认配置，适合大多数场景）
    pub fn default_resident() -> Self {
        Self::default()
    }

    /// 内存映射 + Swap（低内存设备）
    pub fn low_memory() -> Self {
        Self {
            weight_strategy: WeightStrategy::MemoryMapped,
            kv_eviction: KvEvictionPolicy::Swap {
                path: "citlali_kv_swap.bin".to_string(),
            },
            kv_max_pages: 256,
        }
    }

    /// 常驻内存 + Drop（一次性推理，不需要长上下文）
    pub fn oneshot() -> Self {
        Self {
            weight_strategy: WeightStrategy::Resident,
            kv_eviction: KvEvictionPolicy::Drop,
            kv_max_pages: 0, // 自动计算
        }
    }

    /// 极限低内存：mmap + Drop
    pub fn minimal() -> Self {
        Self {
            weight_strategy: WeightStrategy::MemoryMapped,
            kv_eviction: KvEvictionPolicy::Drop,
            kv_max_pages: 0, // 自动计算
        }
    }
}
