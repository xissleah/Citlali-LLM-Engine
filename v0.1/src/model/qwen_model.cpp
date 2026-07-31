#include "citlali/model/qwen_model.h"

#include "citlali/common.h"
#include "citlali/compute/common/dtype.h"
#include "citlali/compute/cuda/cuda_utils.h"
#include "citlali/compute/cuda/kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
/*
v0.1改进：
forward_token 新增bool参数 compute_logits，从原先每个token都要计算logits转变为仅最后一个prompt token计算logits
*/
namespace citlali::model {
    namespace {

        std::string layer_name(uint32_t layer, const std::string& suffix) {
            return "blk." + std::to_string(layer) + "." + suffix;
        }
        // 返回层的名字

        int as_int(uint64_t value, const std::string& label) {
            require(value <= static_cast<uint64_t>(std::numeric_limits<int>::max()), label + " is too large for this minimal kernel path");
            return static_cast<int>(value);
        }
        // 把 uint64_t 安全地转换成 int

        void require_shape(const WeightHandle& weight, uint64_t cols, uint64_t rows) {
            require(weight.cols() == cols && weight.rows() == rows,
                    "unexpected tensor shape for " + weight.name + ": got [" +
                    std::to_string(weight.cols()) + ", " + std::to_string(weight.rows()) +
                    "], expected [" + std::to_string(cols) + ", " + std::to_string(rows) + "]");
        }
        // 校验一个权重张量 weight 的形状是否符合预期

        void add_bias_if_present(const WeightHandle& bias, const compute::DeviceBufferPtr& target, uint64_t elements) {
            if (!bias.buffer) return;
            require_shape(bias, elements, 1);
            compute::launch_add_bias_inplace(target->half_data(), bias.device_half_data(), as_int(elements, bias.name + " elements"));
        }
        // 如果某个层存在 bias，就把 bias 加到目标张量 target 上，如果没有 bias，则什么也不做
    }

    void QwenModel::load(const io::GgufFile& gguf, uint32_t max_context) {
        config_ = QwenConfig::from_gguf(gguf);
        // 读取模型配置
        max_context_ = max_context == 0 ? std::min<uint32_t>(config_.context_length, 2048) : std::min(max_context, config_.context_length);
        // 最大上下文
        require(max_context_ > 0, "max_context must be positive");

        WeightLoader loader(gguf, WeightLoadPolicy::DequantizeToFp16);
        linear_backend_ = make_fp16_linear_backend();
        // 创建权重加载器和fp16计算后端

        std::cerr << "Loading token embedding...\n";
        token_embedding_ = loader.load_required("token_embd.weight");
        output_norm_ = loader.load_required("output_norm.weight");
        output_weight_ = loader.load_optional("output.weight");
        // 加载全局权重
        if (!output_weight_.buffer) {
            output_weight_ = token_embedding_;
        }
        // 输出权重缺失时复用embedding权重

        require(config_.head_count % config_.head_count_kv == 0,
                "attention head_count must be divisible by head_count_kv");
        // 要求query头数 可以均匀分配到 KV头上

        const uint64_t hidden = config_.embedding_length;
        // 计算隐藏层维度
        const uint64_t q_size = static_cast<uint64_t>(config_.head_count) * config_.head_dim;
        // query 维度 = 头数 * 每个头维度
        const uint64_t kv_size = static_cast<uint64_t>(config_.head_count_kv) * config_.head_dim;
        // kv 维度 = kv头数 * 每个头的唯独
        const uint64_t ff = config_.feed_forward_length;
        // 前馈神经网络中间层维度
        require_shape(token_embedding_, hidden, config_.vocab_size);
        require_shape(output_norm_, hidden, 1);
        require_shape(output_weight_, hidden, config_.vocab_size);
        // 检查模型文件的实际权重尺寸和预期是否一致

        layers_.resize(config_.block_count);
        // 根据层数创建对应数量的layer对象
        for (uint32_t i = 0; i < config_.block_count; ++i) {
            // 逐层加载
            std::cerr << "Loading layer " << (i + 1) << "/" << config_.block_count << "\n";
            auto& layer = layers_[i];
            layer.attn_norm = loader.load_required(layer_name(i, "attn_norm.weight"));
            // 注意力前的归一化权重
            layer.attn_q = loader.load_required(layer_name(i, "attn_q.weight"));
            layer.attn_q_bias = loader.load_optional(layer_name(i, "attn_q.bias"));
            // query权重及其偏置
            layer.attn_k = loader.load_required(layer_name(i, "attn_k.weight"));
            layer.attn_k_bias = loader.load_optional(layer_name(i, "attn_k.bias"));
            // key权重及其偏置
            layer.attn_v = loader.load_required(layer_name(i, "attn_v.weight"));
            layer.attn_v_bias = loader.load_optional(layer_name(i, "attn_v.bias"));
            // value权重及其偏置
            layer.attn_o = loader.load_required(layer_name(i, "attn_output.weight"));
            layer.attn_o_bias = loader.load_optional(layer_name(i, "attn_output.bias"));
            // 注意力输出投影权重及其偏置
            layer.attn_q_norm = loader.load_optional(layer_name(i, "attn_q_norm.weight"));
            layer.attn_k_norm = loader.load_optional(layer_name(i, "attn_k_norm.weight"));
            // 这是对每个 Query/Key 头进行额外归一化的权重，属于可选项
            layer.ffn_norm = loader.load_required(layer_name(i, "ffn_norm.weight"));
            // 前馈神经网络之前的归一化权重
            layer.ffn_gate = loader.load_required(layer_name(i, "ffn_gate.weight"));
            layer.ffn_gate_bias = loader.load_optional(layer_name(i, "ffn_gate.bias"));
            // gate 权重及其偏置
            layer.ffn_up = loader.load_required(layer_name(i, "ffn_up.weight"));
            layer.ffn_up_bias = loader.load_optional(layer_name(i, "ffn_up.bias"));
            // up分支的权重及其偏置
            layer.ffn_down = loader.load_required(layer_name(i, "ffn_down.weight"));
            layer.ffn_down_bias = loader.load_optional(layer_name(i, "ffn_down.bias"));
            // down分支的权重及其偏置

            require_shape(layer.attn_norm, hidden, 1);
            require_shape(layer.attn_q, hidden, q_size);
            if (layer.attn_q_bias.buffer) require_shape(layer.attn_q_bias, q_size, 1);
            require_shape(layer.attn_k, hidden, kv_size);
            if (layer.attn_k_bias.buffer) require_shape(layer.attn_k_bias, kv_size, 1);
            require_shape(layer.attn_v, hidden, kv_size);
            if (layer.attn_v_bias.buffer) require_shape(layer.attn_v_bias, kv_size, 1);
            require_shape(layer.attn_o, q_size, hidden);
            if (layer.attn_o_bias.buffer) require_shape(layer.attn_o_bias, hidden, 1);
            if (layer.attn_q_norm.buffer) require_shape(layer.attn_q_norm, config_.head_dim, 1);
            if (layer.attn_k_norm.buffer) require_shape(layer.attn_k_norm, config_.head_dim, 1);
            require_shape(layer.ffn_norm, hidden, 1);
            require_shape(layer.ffn_gate, hidden, ff);
            if (layer.ffn_gate_bias.buffer) require_shape(layer.ffn_gate_bias, ff, 1);
            require_shape(layer.ffn_up, hidden, ff);
            if (layer.ffn_up_bias.buffer) require_shape(layer.ffn_up_bias, ff, 1);
            require_shape(layer.ffn_down, ff, hidden);
            if (layer.ffn_down_bias.buffer) require_shape(layer.ffn_down_bias, hidden, 1);
            // 检查每层权重的形状

            layer.k_cache = compute::make_device_half_buffer(static_cast<size_t>(max_context_) * static_cast<size_t>(kv_size));
            layer.v_cache = compute::make_device_half_buffer(static_cast<size_t>(max_context_) * static_cast<size_t>(kv_size));
            // 为每层创建KV cache
        }

        allocate_runtime_buffers();
        // 分配运行时临时缓冲区
    }
    // 从 GGUF 模型文件中读取 Qwen 模型的配置和权重，检查形状，创建每一层的 KV Cache，最后分配推理时需要的临时缓冲区

    void QwenModel::allocate_runtime_buffers() {
        const uint64_t hidden = config_.embedding_length;
        // embedding向量的长度
        const uint64_t q = static_cast<uint64_t>(config_.head_count) * config_.head_dim;
        // query的总长度
        const uint64_t kv = static_cast<uint64_t>(config_.head_count_kv) * config_.head_dim;
        // kv总长度
        const uint64_t ff = config_.feed_forward_length;
        // 前馈神经网络的长度

        x_ = compute::make_device_half_buffer(static_cast<size_t>(hidden));
        // 保存当前 token 的隐藏状态
        norm_ = compute::make_device_half_buffer(static_cast<size_t>(hidden));
        // 保存 RMSNorm 等归一化操作的输出
        q_ = compute::make_device_half_buffer(static_cast<size_t>(q));
        k_ = compute::make_device_half_buffer(static_cast<size_t>(kv));
        v_ = compute::make_device_half_buffer(static_cast<size_t>(kv));
        // qkv缓冲区
        /*
        注意：
        k_、v_ 只保存当前正在处理的 token
        k_cache、v_cache 保存从第一个 token 到当前 token 的历史数据
        */
        attn_out_ = compute::make_device_half_buffer(static_cast<size_t>(q));
        // 保存多头注意力计算完成后的结果
        attn_proj_ = compute::make_device_half_buffer(static_cast<size_t>(hidden));
        // 保存注意力输出经过输出投影后的结果
        gate_ = compute::make_device_half_buffer(static_cast<size_t>(ff));
        up_ = compute::make_device_half_buffer(static_cast<size_t>(ff));
        // FFN 中间缓冲区
        ff_act_ = compute::make_device_half_buffer(static_cast<size_t>(ff));
        // 保存门控激活结果 例如ff_act_ = SiLU(gate_) × up_
        ff_out_ = compute::make_device_half_buffer(static_cast<size_t>(hidden));
        /*
        ff_act_
            ↓
        ffn_down.weight
            ↓
        ff_out_
        维度变化：ff → hidden
        随后进行第二次残差连接 x_ = x_ + ff_out_
        */
        logits_ = compute::make_device_buffer(static_cast<size_t>(config_.vocab_size) * sizeof(float));
        // logits_ 保存每一个词表 token 对应的输出分数
        scores_ = compute::make_device_buffer(static_cast<size_t>(config_.head_count) * max_context_ * sizeof(float));
        // 保存 Query 与历史 Key 做点积后得到的注意力分数
    }
    // 一次性在 GPU 上分配模型推理过程中需要反复使用的临时缓冲区，避免每经过一层都重复 cudaMalloc/cudaFree
    // 这些缓冲区是所有层共用的，例如 当第 1 层完成后，中间结果通常不再需要，所以第 2 层可以覆盖这些临时 buffer，这能够显著节省 GPU 显存

    int32_t QwenModel::forward_token(int32_t token_id, uint32_t position, bool compute_logits) {
        require(position < max_context_, "position exceeds max_context");
        require(token_id >= 0 && token_id < static_cast<int32_t>(config_.vocab_size), "token id out of vocab range");

        const int hidden = as_int(config_.embedding_length, "hidden size");
        // embedding向量长度
        const int head_dim = as_int(config_.head_dim, "head_dim");
        // 每一个注意力头的维度
        const int heads = as_int(config_.head_count, "head_count");
        // query头数量
        const int kv_heads = as_int(config_.head_count_kv, "head_count_kv");
        // kv头数量
        const int kv_size = kv_heads * head_dim;
        // kv拼接后的总维度
        const int ff = as_int(config_.feed_forward_length, "feed_forward_length");
        // FFN 中间层维度

        compute::launch_embed(token_embedding_.device_half_data(), token_id, x_->half_data(), hidden);
        // token ID -> 向量，这一步从 embedding 矩阵中取出 token_id 对应的向量，写入 x_
        /*
        GPU 上的 embedding 权重
                  ↓
              token_id 查表
                  ↓
        GPU 上的 x_，长度为 hidden
        */

        for (auto& layer : layers_) {
            // 逐层执行transformer
            compute::launch_rms_norm(x_->half_data(), layer.attn_norm.device_half_data(), norm_->half_data(), hidden, config_.rms_norm_eps);
            /*
            输入：x_       [hidden]
            权重：attn_norm [hidden]
            输出：norm_    [hidden]

            注意 x_ 没有被覆盖，归一化结果写到 norm_，因为后面还要保留原来的 x_ 用于残差连接
            */

            linear_backend_->forward(layer.attn_q, norm_, q_);
            // q_ = attn_q × norm_
            add_bias_if_present(layer.attn_q_bias, q_, static_cast<uint64_t>(heads) * head_dim);
            // 添加可选bias
            linear_backend_->forward(layer.attn_k, norm_, k_);
            add_bias_if_present(layer.attn_k_bias, k_, static_cast<uint64_t>(kv_heads) * head_dim);
            linear_backend_->forward(layer.attn_v, norm_, v_);
            add_bias_if_present(layer.attn_v_bias, v_, static_cast<uint64_t>(kv_heads) * head_dim);
            // kv同上

            if (layer.attn_q_norm.buffer) {
                compute::launch_rms_norm_heads(q_->half_data(), layer.attn_q_norm.device_half_data(), q_->half_data(), heads, head_dim, config_.rms_norm_eps);
            }
            if (layer.attn_k_norm.buffer) {
                compute::launch_rms_norm_heads(k_->half_data(), layer.attn_k_norm.device_half_data(), k_->half_data(), kv_heads, head_dim, config_.rms_norm_eps);
            }
            // 对qk每个头单独做RMSNorm

            compute::launch_rope(q_->half_data(), heads, head_dim, static_cast<int>(position), config_.rope_theta);
            compute::launch_rope(k_->half_data(), kv_heads, head_dim, static_cast<int>(position), config_.rope_theta);
            // RoPE位置编码
            compute::launch_store_kv(k_->half_data(), v_->half_data(), layer.k_cache->half_data(), layer.v_cache->half_data(), static_cast<int>(position), kv_size);
            // 把当前kv写入kV cache
            compute::launch_attention(q_->half_data(), layer.k_cache->half_data(), layer.v_cache->half_data(), static_cast<float*>(scores_->data()), attn_out_->half_data(), static_cast<int>(position), static_cast<int>(max_context_), heads, kv_heads, head_dim);
            /*
            这一步完成注意力计算，大致是
            scores = Q × Kᵀ / sqrt(head_dim)
            weights = softmax(scores)
            attn_out = weights × V
            其中
            q_        当前 token 的 Query
            k_cache   当前层所有历史 token 的 Key
            v_cache   当前层所有历史 token 的 Value
            scores_   注意力分数，FP32
            attn_out_ 注意力输出，FP16
            */
            linear_backend_->forward(layer.attn_o, attn_out_, attn_proj_);
            // 将注意力输出从 heads × head_dim 投影回 hidden 也就是 attn_proj_ = attn_o × attn_out_
            add_bias_if_present(layer.attn_o_bias, attn_proj_, hidden);
            // 加偏置
            compute::launch_add_inplace(x_->half_data(), attn_proj_->half_data(), hidden);
            // 残差连接 x_ = x_ + attn_proj_

            compute::launch_rms_norm(x_->half_data(), layer.ffn_norm.device_half_data(), norm_->half_data(), hidden, config_.rms_norm_eps);
            // FFN前的RMSNorm
            linear_backend_->forward(layer.ffn_gate, norm_, gate_);
            // gate_ = ffn_gate × norm_
            add_bias_if_present(layer.ffn_gate_bias, gate_, ff);
            // 加偏置
            linear_backend_->forward(layer.ffn_up, norm_, up_);
            // up_ = ffn_up × norm_
            add_bias_if_present(layer.ffn_up_bias, up_, ff);
            // 加偏置
            compute::launch_swiglu(gate_->half_data(), up_->half_data(), ff_act_->half_data(), ff);
            // swiglu激活函数
            linear_backend_->forward(layer.ffn_down, ff_act_, ff_out_);
            // 将 FFN 中间结果从 ff 维投影回 hidden 维
            add_bias_if_present(layer.ffn_down_bias, ff_out_, hidden);
            // 加偏置
            compute::launch_add_inplace(x_->half_data(), ff_out_->half_data(), hidden);
            // FFN残差连接
        }

        if (!compute_logits) {
            return -1;
        }
        // v0.1改进：如果当前token是中间token时，这个预测结果很快会被下一个token覆盖，所以只需要最后一个prompt token才需要计算logits

        compute::launch_rms_norm(x_->half_data(), output_norm_.device_half_data(), norm_->half_data(), hidden, config_.rms_norm_eps);
        // 所有层完成后得到最终隐藏状态，再对其进行RMSNorm
        compute::launch_matvec_fp16_to_float(output_weight_.device_half_data(),
                                             norm_->half_data(),
                                             static_cast<float*>(logits_->data()),
                                             hidden,
                                             as_int(config_.vocab_size, "vocab_size"));
        // 这一步将隐藏向量映射到整个词表 logits = output_weight(权重) × norm_(输入) -> logits_每一个词表 token 都得到一个分数
        return sample_greedy_from_logits();
        // 贪心采样并返回结果
    }
    // 输入一个 token 和它在序列中的位置，执行一次完整的 Transformer 前向计算，得到下一个 token
    /*
    token_id
       ↓
    Embedding
       ↓
    x_
       ↓
    逐层 Transformer
       ├── Attention
       └── FFN
       ↓
    最终 RMSNorm
       ↓
    输出层
       ↓
    logits
       ↓
    贪心采样
       ↓
    返回下一个 token
    */

    void QwenModel::synchronize() const {
        CITLALI_CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 等待当前 GPU 设备上此前提交的 CUDA 任务全部执行完成，并检查执行期间是否发生 CUDA 错误

    int32_t QwenModel::sample_greedy_from_logits() {
        synchronize();
        // 先等待 GPU 完成计算
        std::vector<float> host(static_cast<size_t>(config_.vocab_size));
        logits_->copy_to_host(host.data(), host.size() * sizeof(float));
        // 将 logits 从 GPU 复制到 CPU

        float best = -std::numeric_limits<float>::infinity();
        int32_t best_id = 0;
        for (int32_t i = 0; i < static_cast<int32_t>(host.size()); ++i) {
            const float value = host[static_cast<size_t>(i)];
            if (std::isfinite(value) && value > best) {
                best = value;
                best_id = i;
            }
        }
        return best_id;
    }
    // 将 GPU 上的 logits_ 复制到 CPU，找出分数最大的 token ID，并返回它

}

/*
推理过程大致数据流
token ID
   ↓
token embedding
   ↓
  x_ [hidden]
   ↓
norm_
   ├───────────────┬───────────────┐
   ↓               ↓               ↓
  q_               k_              v_
[q]               [kv]            [kv]
   │               ↓               ↓
   │          K Cache         V Cache
   │               │               │
   └───────────────┴───────────────┘
                   ↓
              scores_ [heads, context]
                   ↓
              attn_out_ [q]
                   ↓
              attn_proj_ [hidden]
                   ↓
             残差加到 x_
                   ↓
                 norm_
              ┌────┴────┐
              ↓         ↓
           gate_       up_
             [ff]       [ff]
              └────┬────┘
                   ↓
               ff_act_ [ff]
                   ↓
               ff_out_ [hidden]
                   ↓
             残差加到 x_
                   ↓
               下一层
                   ↓
              output norm
                   ↓
             logits_ [vocab_size]
*/
