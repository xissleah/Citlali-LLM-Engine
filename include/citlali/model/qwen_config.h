#pragma once

#include "citlali/io/gguf_reader.h"

#include <cstdint>
#include <string>

namespace citlali::model {

    struct QwenConfig {
        std::string architecture;           // 模型架构名称
        uint32_t vocab_size = 0;            // 词表大小，也就是 tokenizer一共有多少个token
        uint32_t context_length = 0;        // 上下文长度
        uint32_t embedding_length = 0;      // 向量有多少维
        uint32_t block_count = 0;           // 模型层数
        uint32_t feed_forward_length = 0;   // FFN中间层维度
        uint32_t head_count = 0;            // 注意力头数
        uint32_t head_count_kv = 0;         // Key/Value的head数量
        uint32_t head_dim = 0;              // 一个头有多少维度
        uint32_t value_dim = 0;             // 一个value有多少维度
        float rms_norm_eps = 1e-6f;         // 归一化防止除0
        float rope_theta = 1000000.0f;      // rope的一个参数

        static QwenConfig from_gguf(const io::GgufFile& gguf);
    };

} // namespace citlali::model
