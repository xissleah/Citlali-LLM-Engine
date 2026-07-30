#pragma once

#include <cstdint>
#include <cstddef>

namespace citlali::compute {

    void cuda_zero(void* data, std::size_t bytes);
    //  显存清理
    void launch_embed(const uint16_t* embedding, int32_t token_id, uint16_t* out, int hidden);
    //  词嵌入查表：根据 token_id，从词嵌入权重里取出对应向量，写入输出显存
    void launch_copy(const uint16_t* input, uint16_t* output, int n);
    //  复制 n 个 FP16 元素：
    void launch_add_inplace(uint16_t* target, const uint16_t* delta, int n);
    //  原地加法：target = target + delta，残差连接专用
    void launch_add_bias_inplace(uint16_t* target, const uint16_t* bias, int n);
    //  原地加偏置：target = target + bias，线性层后加 bias
    void launch_rms_norm(const uint16_t* input, const uint16_t* weight, uint16_t* output, int n, float eps);
    //  普通单向量 RMS 归一化
    void launch_rms_norm_heads(const uint16_t* input, const uint16_t* weight, uint16_t* output, int heads, int head_dim, float eps);
    //  多头专用 RMSNorm，Qwen 特有 Q/K 分归一化
    void launch_matvec_fp16(const uint16_t* weight, const uint16_t* input, uint16_t* output, int in_features, int out_features);
    //  FP16 稠密权重矩阵乘向量 out = input × W，QKV/FFN 投影核心计算
    void launch_matvec_fp16_to_float(const uint16_t* weight, const uint16_t* input, float* output, int in_features, int out_features);
    //  矩阵乘结果直接输出 float，用于注意力分数计算（softmax 需要浮点）
    void launch_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* output, int n);
    //  激活函数swiglu
    void launch_rope(uint16_t* data, int heads, int head_dim, int position, float theta);
    //  旋转编码RoPE
    void launch_store_kv(const uint16_t* k, const uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, int position, int kv_size);
    //  kv缓存写入
    void launch_attention(const uint16_t* q,
                          const uint16_t* k_cache,
                          const uint16_t* v_cache,
                          float* scores,
                          uint16_t* output,
                          int position,
                          int max_context,
                          int heads,
                          int kv_heads,
                          int head_dim);
    //  launch_attention(Q向量,
    //                  K缓存,
    //                  V缓存,
    //                  浮点分数缓存,
    //                  输出,
    //                  当前位置,
    //                  最大上下文,
    //                  头数,
    //                  KV头数,
    //                  单头维度
    //  );
    //  多头自注意力计算（完整 Attention 核）
} // namespace citlali::compute
