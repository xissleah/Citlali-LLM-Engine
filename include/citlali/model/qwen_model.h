#pragma once

#include "citlali/compute/cuda/tensor.h"
#include "citlali/io/gguf_reader.h"
#include "citlali/model/linear_backend.h"
#include "citlali/model/qwen_config.h"
#include "citlali/model/weight_loader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace citlali::model
{
    struct QwenLayerWeights
    {
        WeightHandle attn_norm;        // 注意力前置RMS Norm
        WeightHandle attn_q, attn_q_bias; // Q投影权重+偏置
        WeightHandle attn_k, attn_k_bias; // K投影权重+偏置
        WeightHandle attn_v, attn_v_bias; // V投影权重+偏置
        WeightHandle attn_o, attn_o_bias; // 输出O投影权重+偏置
        WeightHandle attn_q_norm, attn_k_norm; // Qwen特有：Q/K RMS归一化
        WeightHandle ffn_norm;         // FFN前置Norm
        WeightHandle ffn_gate, ffn_gate_bias; // Gate分支
        WeightHandle ffn_up, ffn_up_bias;     // Up分支
        WeightHandle ffn_down, ffn_down_bias;  // Down输出投影
        compute::DeviceBufferPtr k_cache; // 当前层所有历史token的K向量，存在GPU显存
        compute::DeviceBufferPtr v_cache; // 当前层所有历史token的V向量
    };

    class QwenModel
    {
        public:
        QwenModel() = default; //空构造

        void load(const io::GgufFile& gguf, uint32_t max_context = 0);
        // 加载模型
        int32_t forward_token(int32_t token_id, uint32_t position, bool compute_logits = true);
        // 单次生成一个 token 的完整前向传播
        void synchronize() const;
        // gpu同步等待

        const QwenConfig& config() const { return config_; }
        uint32_t max_context() const { return max_context_; }
        // 读取模型超参数、最大上下文长度

        private:
        // 根据max_context分配KV缓存、中间计算临时显存
        void allocate_runtime_buffers();
        // 从最后一层logits中贪心采样，选出概率最大的token id
        int32_t sample_greedy_from_logits();

        QwenConfig config_;
        uint32_t max_context_ = 0;

        WeightHandle token_embedding_; // 词嵌入权重，token id → 向量
        WeightHandle output_norm_;     // 输出前置归一化
        WeightHandle output_weight_;  // 输出层权重，向量转词表logits
        std::vector<QwenLayerWeights> layers_;
        // 数组长度 = 模型 block 层数，每一个元素对应一层权重 + KV 缓存

        //推理临时显存缓冲区 （DeviceBufferPtr = GPU 显存指针）
        compute::DeviceBufferPtr x_;        // 当前层输入向量
        compute::DeviceBufferPtr norm_;     // Norm输出
        compute::DeviceBufferPtr q_,k_,v_;  // QKV投影结果
        compute::DeviceBufferPtr attn_out_; // Attention输出
        compute::DeviceBufferPtr attn_proj_;// O投影
        compute::DeviceBufferPtr gate_, up_, ff_act_, ff_out_; // FFN中间缓存
        compute::DeviceBufferPtr logits_;   // 最终词表概率数组
        compute::DeviceBufferPtr scores_;   // Attention打分矩阵
        std::unique_ptr<LinearBackend> linear_backend_; // 量化矩阵计算后端
    };
}
