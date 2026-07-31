#include "citlali/model/qwen_config.h"

#include "citlali/common.h"

namespace citlali::model
{
    namespace
    {
        //根据 metadata key 查询 GGUF 中保存的值，并按目标类型返回
        uint32_t read_u32_any_prefix(
            const io::GgufFile& gguf,
            const std::string& arch,
            const std::string& suffix,
            uint32_t fallback=0)
        {
            const std::string key = arch + "." + suffix;
            return static_cast<uint32_t>(gguf.get_u64(key,fallback));
        }

        float read_f32_any_prefix(
            const io::GgufFile& gguf,
            const std::string& arch,
            const std::string& suffix,
            float fallback = 0.0f
            )
        {
            const std::string key = arch + "." + suffix;
            return gguf.get_f32(key,fallback);
        }
    }

    QwenConfig QwenConfig::from_gguf(const io::GgufFile& gguf)
    {
        QwenConfig config;
        // 创建空配置
        config.architecture = gguf.get_string("general.architecture", "qwen3");
        require(
            config.architecture.find("qwen") != std::string::npos,
            "Only Qwen/Qwen3 dense GGUF is supported in this engine"
        );

        const std::string prefix = config.architecture;
        // 保存架构前缀
        config.vocab_size = static_cast<uint32_t>(gguf.get_string_array("tokenizer.ggml.tokens").size());
        // 读取词表大小
        config.context_length = read_u32_any_prefix(gguf,prefix,"context_length",2048);
        // 上下文长度
        config.embedding_length = read_u32_any_prefix(gguf,prefix,"embedding_length");
        // embedding维度 这个字段通常表示模型隐藏维度
        config.block_count = read_u32_any_prefix(gguf,prefix,"block_count");
        // Transformer 层数
        config.feed_forward_length = read_u32_any_prefix(gguf,prefix,"feed_forward_length");
        // mlp中间层维度
        config.head_count = read_u32_any_prefix(gguf,prefix,"attention.head_count");
        // Query attention head 的数量
        config.head_count_kv = read_u32_any_prefix(gguf,prefix,"attention.head_count_kv",config.head_count);
        // kv head 数量
        config.head_dim = read_u32_any_prefix(gguf,prefix,"attention.key_length",0);
        // QK head的维度数量
        config.value_dim = read_u32_any_prefix(gguf,prefix,"attention.value_length",0);
        // V head的维度数量
        config.rms_norm_eps = read_f32_any_prefix(gguf,prefix,"attention.layer_norm_rms_epsilon",1e-6f);
        // 查找rms归一化时所用的eps（防止除0用），找不到默认用eps=1e-6f
        config.rope_theta = read_f32_any_prefix(gguf,prefix,"rope.freq_base",1000000.0f);
        // RoPE的θ的数值，找不到就默认用1000000.0f

        if (config.embedding_length == 0) {
            config.embedding_length = read_u32_any_prefix(gguf, "qwen2", "embedding_length");
        }
        if (config.block_count == 0) {
            config.block_count = read_u32_any_prefix(gguf, "qwen2", "block_count");
        }
        if (config.feed_forward_length == 0) {
            config.feed_forward_length = read_u32_any_prefix(gguf, "qwen2", "feed_forward_length");
        }
        if (config.head_count == 0) {
            config.head_count = read_u32_any_prefix(gguf, "qwen2", "attention.head_count");
        }
        if (config.head_count_kv == 0) {
            config.head_count_kv = read_u32_any_prefix(gguf, "qwen2", "attention.head_count_kv", config.head_count);
        }
        if (config.head_dim == 0) {
            config.head_dim = read_u32_any_prefix(gguf, "qwen2", "attention.key_length", 0);
        }
        if (config.value_dim == 0) {
            config.value_dim = read_u32_any_prefix(gguf, "qwen2", "attention.value_length", 0);
        }
        // qwen2的兼容

        require(config.vocab_size > 0, "failed to read vocab size from GGUF");
        require(config.embedding_length > 0, "failed to read embedding_length from GGUF");
        require(config.block_count > 0, "failed to read block_count from GGUF");
        require(config.feed_forward_length > 0, "failed to read feed_forward_length from GGUF");
        require(config.head_count > 0, "failed to read attention.head_count from GGUF");
        require(config.head_count_kv > 0, "failed to read attention.head_count_kv from GGUF");
        if (config.head_dim == 0) {
            if (const auto* q = gguf.find_tensor("blk.0.attn_q.weight")) {
                if (q->dims.size() >= 2 && q->dims[1] % config.head_count == 0) {
                    config.head_dim = static_cast<uint32_t>(q->dims[1] / config.head_count);
                }
            }
        }
        // 如果 metadata 没有 head_dim，就从 Q 权重形状推导
        if (config.value_dim == 0) {
            if (const auto* v = gguf.find_tensor("blk.0.attn_v.weight")) {
                if (v->dims.size() >= 2 && v->dims[1] % config.head_count_kv == 0) {
                    config.value_dim = static_cast<uint32_t>(v->dims[1] / config.head_count_kv);
                }
            }
        }
        // 如果 metadata 没有 value_dim，就从 V 权重形状推导
        if (config.head_dim == 0) {
            require(config.embedding_length % config.head_count == 0, "embedding_length must be divisible by head_count when attention.key_length is absent");
            config.head_dim = config.embedding_length / config.head_count;
        }
        // 如果仍然没有 head_dim，就用 embedding_length 推导
        if (config.value_dim == 0) {
            config.value_dim = config.head_dim;
        }
        require(config.head_dim > 0, "failed to read or derive attention.key_length/head_dim from GGUF");
        require(config.value_dim == config.head_dim, "this minimal attention path currently requires attention.value_length == attention.key_length");
        return config;
        // 如果仍然没有 value_dim，就使用 head_dim
    }
    // 从 GGUF 文件的 metadata 和 tensor 信息中读取或推导出 Qwen 模型的结构配置，并返回一个完整的 QwenConfig
    /*
    GGUF 文件
    ↓ 查 metadata
    读取模型配置
    ↓ 必要时查 tensor 形状推导
    补齐 head_dim / value_dim
    ↓
    返回 QwenConfig
    */
}
