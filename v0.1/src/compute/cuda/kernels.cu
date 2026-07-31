#include "citlali/compute/cuda/kernels.h"

#include "citlali/compute/cuda/cuda_utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace citlali::compute {
    namespace {
        // __device__ 表示这个函数运行在GPU上，并且只能被 __global__ kernel / 其他 __device__ 函数 / 某些带 __host__ __device__ 的函数中的设备端部分 代码调用
        __device__ float load_half(const uint16_t* ptr, int index) {
            const half* h = reinterpret_cast<const half*>(ptr);
            // 指针类型转换
            return __half2float(h[index]);
            // fp16 转 fp32
        }
        // 从 uint16_t 类型的显存数据中读取一个 FP16（half）元素，并转换成 float 返回

        __device__ void store_half(uint16_t* ptr, int index, float value) {
            half* h = reinterpret_cast<half*>(ptr);
            h[index] = __float2half(value);
        }
        // 在 GPU 上把一个 float 值转换为 FP16（half），再写入由 uint16_t* 指向的显存中

        __global__ void embed_kernel(const uint16_t* embedding, int32_t token_id, uint16_t* out, int hidden) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            // 在cuda里，gird下分block，block下分thread
            // blockIdx.x指当前block在grid里的编号，blockDim.x指每一个block在x方向的线程数量，threadIdx.x指当前线程在所属block里的编号
            if (i < hidden) {
                out[i] = embedding[token_id * hidden + i];
                // 在标准transformer里，所有token的向量长度是一样的。而hidden代表列数，同时embedding数组是一维的，因此某token的起始位置应该是token_ID * hidden，i则是这个向量某个元素的位置
            }
        }
        // 从 embedding 表中取出某个 token 对应的 embedding 向量，并复制到输出缓冲区
        // 这个函数一次性只处理一个token_id，所以要循环调用
        // 为什么一个 token 的向量元素不需要手动循环？ 原因是虽然每一个线程只处理一个元素，但是同时会启动多个线程

        __global__ void copy_kernel(const uint16_t* input, uint16_t* output, int n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) output[i] = input[i];
            // 启动线程数量通常会向上取整
        }
        //  GPU 上的并行数组复制 kernel
        /*
        等价于cpu代码：
        for (int i = 0; i < n; ++i) {
            output[i] = input[i];
        }
        */

        __global__ void add_inplace_kernel(uint16_t* target, const uint16_t* delta, int n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                const float v = load_half(target, i) + load_half(delta, i);
                store_half(target, i, v);
            }
        }
        // 将 delta 中的 FP16 数据逐元素加到 target 上，并将结果原地写回 target
        // 这类操作通常用于残差连接

        __global__ void add_bias_inplace_kernel(uint16_t* target, const uint16_t* bias, int n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                const float v = load_half(target, i) + load_half(bias, i);
                store_half(target, i, v);
            }
        }
        // 将 bias 数组逐元素加到 target 数组上，并把结果原地写回 target
        // 例如进激活函数前加偏置

        __global__ void rms_norm_kernel(const uint16_t* input, const uint16_t* weight, uint16_t* output, int n, float eps) {
            __shared__ float scratch[256];
            float sum = 0.0f;
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                const float v = load_half(input, i);
                sum += v * v;
            }
            scratch[threadIdx.x] = sum;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
                __syncthreads();
            }

            const float inv = rsqrtf(scratch[0] / static_cast<float>(n) + eps);
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                store_half(output, i, load_half(input, i) * inv * load_half(weight, i));
            }
        }
        // rmsnorm归一化

        __global__ void rms_norm_heads_kernel(const uint16_t* input, const uint16_t* weight, uint16_t* output, int head_dim, float eps) {
            __shared__ float scratch[256];
            // 共享内存
            const int head = blockIdx.x;
            const int base = head * head_dim;
            // 起始位置
            float sum = 0.0f;
            for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
                const float v = load_half(input, base + i);
                sum += v * v;
            }
            // 使用线程并行计算平方和
            scratch[threadIdx.x] = sum;
            // 结果写入共享内存
            __syncthreads();
            // 等待所有线程写完

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
                __syncthreads();
            }
            // 并行求总平方和

            const float inv = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
            // 计算rms的倒数
            for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
                store_half(output, base + i, load_half(input, base + i) * inv * load_half(weight, i));
            }
            // 对每一个元素完成归一化和缩放 每个线程再次处理自己负责的元素
        }
        // 对 每个 attention head 单独执行 RMSNorm

        __global__ void matvec_fp16_kernel(const uint16_t* weight, const uint16_t* input, uint16_t* output, int in_features) {
            // output = weight × input
            __shared__ float scratch[256];
            const int row = blockIdx.x;
            // 一个Block负责计算1行
            float sum = 0.0f;
            const uint16_t* row_weight = weight + row * in_features;
            // 找到当前行的权重
            for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
                sum += load_half(row_weight, i) * load_half(input, i);
            }
            /*
            Grid
            ├── block 0 → 计算 weight 第 0 行和 input 的点积
            ├── block 1 → 计算 weight 第 1 行和 input 的点积
            ├── block 2 → 计算 weight 第 2 行和 input 的点积
            └── ...
            */
            scratch[threadIdx.x] = sum;
            // 存此线程计算出来的局部sum
            __syncthreads();
            // 等待所有线程同步

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride)
                {
                    scratch[threadIdx.x] += scratch[threadIdx.x + stride];
                }
                __syncthreads();
            }
            // block 内并行归约，把所有线程的局部 sum 相加，最终得到当前矩阵行的完整点积
            /*
            比如
            scratch = [a, b, c, d, e, f, g, h]

            其中
            scratch[0] = 线程 0 的局部和
            scratch[1] = 线程 1 的局部和
            ...
            scratch[7] = 线程 7 的局部和

            第一轮 stride = 4
            线程 0：scratch[0] += scratch[4] → a + e
            线程 1：scratch[1] += scratch[5] → b + f
            线程 2：scratch[2] += scratch[6] → c + g
            线程 3：scratch[3] += scratch[7] → d + h

            第二轮 stride = 2
            线程 0：scratch[0] += scratch[2]
                    → (a+e) + (c+g)

            线程 1：scratch[1] += scratch[3]
                    → (b+f) + (d+h)
            即
            scratch[0] = a + c + e + g
            scratch[1] = b + d + f + h

            第三轮 stride = 1
            scratch[0] += scratch[1];
            即
            scratch[0] = a + b + c + d + e + f + g + h

            最后输出scratch[0]即可
            */
            if (threadIdx.x == 0) {
                store_half(output, row, scratch[0]);
            }
            // 只留 threadIdx.x == 0 这一个线程输出结果
        }
        // 计算一个 FP16 矩阵和 FP16 向量的乘法

        __global__ void matvec_fp16_to_float_kernel(const uint16_t* weight, const uint16_t* input, float* output, int in_features) {
            __shared__ float scratch[256];
            const int row = blockIdx.x;
            float sum = 0.0f;
            const uint16_t* row_weight = weight + row * in_features;
            for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
                sum += load_half(row_weight, i) * load_half(input, i);
            }
            scratch[threadIdx.x] = sum;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                output[row] = scratch[0];
            }
        }
        // 和上面那个函数基本相同，只是 output类型 变成了 float*

        __global__ void swiglu_kernel(const uint16_t* gate, const uint16_t* up, uint16_t* output, int n) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                const float g = load_half(gate, i);
                const float u = load_half(up, i);
                const float silu = g / (1.0f + expf(-g));
                store_half(output, i, silu * u);
            }
        }
        // swiglu激活函数 output[i] = SiLU(gate[i]) * up[i]

        __global__ void rope_kernel(uint16_t* data, int head_dim, int position, float theta) {
            const int head = blockIdx.x;
            const int i = threadIdx.x;
            const int half_dim = head_dim / 2;
            if (i >= half_dim) return;
            /*
            一个 block → 一个 head
            一个线程 → 该 head 中的一组旋转维度
            */

            const int base = head * head_dim;
            //当前head起始位置
            const float inv_freq = powf(theta, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
            // 计算 RoPE 频率，数学形式是 inv_freq_i = theta^(-2i / head_dim)
            const float angle = static_cast<float>(position) * inv_freq;
            // 计算当前位置的旋转角度
            float s;
            float c;
            sincosf(angle, &s, &c);
            // 同时计算正余弦
            /*
            CUDA 的 sincosf 会同时计算
            s = sin(angle)
            c = cos(angle)
            */

            const float x0 = load_half(data, base + i);
            const float x1 = load_half(data, base + half_dim + i);
            // 加载 FP16 数据
            store_half(data, base + i, x0 * c - x1 * s);
            store_half(data, base + half_dim + i, x0 * s + x1 * c);
            /*
            执行二维旋转
            x0' = x0 * c - x1 * s
            x1' = x0 * s + x1 * c
            */
        }
        // 这是我经常忘记怎么写的RoPE P_P

        __global__ void store_kv_kernel(const uint16_t* k, const uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, int position, int kv_size) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            // 计算线程的全局索引
            if (i < kv_size) {
                const int offset = position * kv_size + i;
                // 计算在 kv cache 中的偏移
                k_cache[offset] = k[i];
                v_cache[offset] = v[i];
            }
        }
        // 把当前 token 计算出的 Key 和 Value 写入 KV Cache
        /*
        参数含义
        k        当前 token 的 K 向量
        v        当前 token 的 V 向量
        k_cache  所有历史 token 的 K 缓存
        v_cache  所有历史 token 的 V 缓存
        position 当前 token 在序列中的位置
        kv_size  K/V 向量长度
        */

        __global__ void attention_scores_kernel(const uint16_t* q,
                                                const uint16_t* k_cache,
                                                float* scores,
                                                int position,
                                                int max_context,
                                                int heads,
                                                int kv_heads,
                                                int head_dim)
        {
            __shared__ float scratch[256];
            const int head = blockIdx.x;
            const int t = blockIdx.y;
            // 一个 block 负责一个 (query head, token position) 对
            /*
            blockIdx.x → Query head
            blockIdx.y → 历史 token 位置 t
            */
            if (head >= heads || t > position) return;

            const int kv_group = heads / kv_heads;
            const int kv_head = head / kv_group;
            // 这里处理的是 Grouped Query Attention(多个 Query head 共享同一个 KV head)
            /*
            例如
            heads = 8
            kv_heads = 2

            则
            kv_group = 8 / 2 = 4

            映射关系是
            Query head 0, 1, 2, 3 → KV head 0
            Query head 4, 5, 6, 7 → KV head 1
            */
            const int q_base = head * head_dim;
            // 计算 Query 的起始位置
            const int k_base = (t * kv_heads + kv_head) * head_dim;
            // 计算历史 K 的起始位置

            float sum = 0.0f;
            for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
                sum += load_half(q, q_base + i) * load_half(k_cache, k_base + i);
            }
            scratch[threadIdx.x] = sum;
            __syncthreads();
            // 每个线程开始计算各部分点积

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
                __syncthreads();
            }
            // 并行归约

            if (threadIdx.x == 0) {
                scores[head * max_context + t] = scratch[0] * rsqrtf(static_cast<float>(head_dim));
            }
            // 输出
        }
        // 计算注意力中的 Q 和历史 K 的缩放点积，生成 attention scores

        __global__ void attention_values_kernel(const float* scores,
                                                const uint16_t* v_cache,
                                                uint16_t* output,
                                                int position,
                                                int max_context,
                                                int heads,
                                                int kv_heads,
                                                int head_dim) {
            const int head = blockIdx.x;
            // 一个 block 负责一个 Query head
            const int dim = threadIdx.x;
            if (dim >= head_dim) return;

            const int kv_group = heads / kv_heads;
            const int kv_head = head / kv_group;
            // GQA 中选择对应的 KV head
            const float* head_scores = scores + head * max_context;
            // 找到当前 head 的 scores

            float max_score = -3.402823466e+38f;
            for (int t = 0; t <= position; ++t) {
                max_score = fmaxf(max_score, head_scores[t]);
            }
            // 计算 softmax 的最大值

            float denom = 0.0f;
            for (int t = 0; t <= position; ++t) {
                denom += expf(head_scores[t] - max_score);
            }
            // 计算 softmax 分母
            // denom = Σₜ exp(score[head][t] - max_score)

            float acc = 0.0f;
            for (int t = 0; t <= position; ++t) {
                const float p = expf(head_scores[t] - max_score) / denom;
                // 计算注意力权重
                const int v_base = (t * kv_heads + kv_head) * head_dim;
                // 找到 Value Cache 的地址
                acc += p * load_half(v_cache, v_base + dim);
                // 对所有历史 Value 做加权求和
            }
            // 计算加权 Value

            store_half(output, head * head_dim + dim, acc);
        }
        // 计算注意力的 加权 Value 输出
        //output[head][d] = Σₜ softmax(score[head][t]) × V_cache[t][kv_head][d]

    } // namespace

    void cuda_zero(void* data, std::size_t bytes) {
        CITLALI_CUDA_CHECK(cudaMemset(data, 0, bytes));
    }
    // 把 GPU 上的一段内存清零

    void launch_embed(const uint16_t* embedding, int32_t token_id, uint16_t* out, int hidden) {
        const int block = 256;
        // 每个 CUDA block 启动 256 个线程
        const int grid = (hidden + block - 1) / block;
        // 用数量为 grid 个 block
        embed_kernel<<<grid, block>>>(embedding, token_id, out, hidden);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 启动一个 CUDA kernel，把某个 token 对应的 embedding 向量复制到 out

    void launch_copy(const uint16_t* input, uint16_t* output, int n) {
        /*
        input  输入数组
        output 写入目标
        n      复制数量
        */
        const int block = 256;
        const int grid = (n + block - 1) / block;
        // 启用 grid 个 block，每个 block 启用256个线程
        copy_kernel<<<grid, block>>>(input, output, n);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 在 GPU 上并行复制一段 uint16_t 数组

    void launch_add_inplace(uint16_t* target, const uint16_t* delta, int n) {
        /*
        target  被修改的目标数组
        delta   要加到目标数组上的增量数组
        n       元素数量
        */
        const int block = 256;
        const int grid = (n + block - 1) / block;
        // 启用 grid 个 block，每个 block 启用256个线程
        add_inplace_kernel<<<grid, block>>>(target, delta, n);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 在 GPU 上执行逐元素原地加法

    void launch_add_bias_inplace(uint16_t* target, const uint16_t* bias, int n) {
        /*
        target  被修改的目标数组
        bias    偏置数组
        n       元素数量
        */
        const int block = 256;
        const int grid = (n + block - 1) / block;
        // 启用 grid 个 block，每个 block 启用256个线程
        add_bias_inplace_kernel<<<grid, block>>>(target, bias, n);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 在 GPU 上对 target 原地加上 bias

    void launch_rms_norm(const uint16_t* input, const uint16_t* weight, uint16_t* output, int n, float eps) {
        rms_norm_kernel<<<1, 256>>>(input, weight, output, n, eps);
        // 一个 block 256个线程
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 启动 RMSNorm

    void launch_rms_norm_heads(const uint16_t* input, const uint16_t* weight, uint16_t* output, int heads, int head_dim, float eps) {
        /*

        input       输入数据，通常布局为 [heads, head_dim]
        weight      RMSNorm 的缩放权重，通常长度为 head_dim ，所有 head 共享
        output      输出数据，布局通常也是 [heads, head_dim]
        heads       head 的数量，也就是需要独立归一化的向量数量
        head_dim    每个 head 的向量维度
        eps         防止除零、提高数值稳定性的小常数
        */
        rms_norm_heads_kernel<<<heads, 256>>>(input, weight, output, head_dim, eps);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 对多个 attention head 分别执行 RMSNorm

    void launch_matvec_fp16(const uint16_t* weight, const uint16_t* input, uint16_t* output, int in_features, int out_features) {
        /*
        weight          权重矩阵，通常形状为 [out_features, in_features]
        input           输入向量，长度为 in_features
        output          输出向量，长度为 out_features
        in_features     输入特征数量，也就是矩阵每一行的长度
        out_features    输出特征数量，也就是矩阵的行数

        一个 block → 一个输出元素
        一个线程 → 输入维度的一部分
        */
        matvec_fp16_kernel<<<out_features, 256>>>(weight, input, output, in_features);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 启动一个 FP16 矩阵-向量乘法    output = weight × input

    void launch_matvec_fp16_to_float(const uint16_t* weight, const uint16_t* input, float* output, int in_features, int out_features) {
        matvec_fp16_to_float_kernel<<<out_features, 256>>>(weight, input, output, in_features);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 和上面那个基本相同，区别是 输入和权重使用 FP16 存储，输出使用 float 存储

    void launch_swiglu(const uint16_t* gate, const uint16_t* up, uint16_t* output, int n) {
        /*
        gate    门控分支输入，长度为n
        up      上投影分支输入，长度为n
        output  输出数组，长度为n
        n       元素数量
        */
        const int block = 256;
        const int grid = (n + block - 1) / block;
        // 启用 grid 个 block，每个 block 启用256个线程
        swiglu_kernel<<<grid, block>>>(gate, up, output, n);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 执行逐元素的 SwiGLU

    void launch_rope(uint16_t* data, int heads, int head_dim, int position, float theta) {
        /*
        data        需要添加 RoPE 的数据，通常是 Q 或 K，布局一般为 [heads, head_dim]
        heads       head数量
        head_dim    每个 head 的维度
        position    当前 token 在序列中的位置
        theta       RoPE 的基数，常见默认值为 10000.0f
        */
        rope_kernel<<<heads, head_dim / 2>>>(data, head_dim, position, theta);
        // block 数量 = heads 每个 block 的线程数 = head_dim / 2
        // 因为通常 一个block处理一个head 一个线程处理一对维度
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 对多个 attention head 的数据执行 RoPE

    void launch_store_kv(const uint16_t* k, const uint16_t* v, uint16_t* k_cache, uint16_t* v_cache, int position, int kv_size) {
        /*
        k       	当前 token 的 K 向量
        v           当前 token 的 V 向量
        k_cache     保存历史 K 的缓存
        v_cache     保存历史 V 的缓存
        position    当前 token 在序列中的位置
        kv_size     当前 K/V 向量的总元素数量
        */
        const int block = 256;
        const int grid = (kv_size + block - 1) / block;
        // 启用 grid 个 block，每个 block 启用256个线程
        store_kv_kernel<<<grid, block>>>(k, v, k_cache, v_cache, position, kv_size);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 把当前 token 的 K、V 写入 KV Cache

    void launch_attention(const uint16_t* q,
                          const uint16_t* k_cache,
                          const uint16_t* v_cache,
                          float* scores,
                          uint16_t* output,
                          int position,
                          int max_context,
                          int heads,
                          int kv_heads,
                          int head_dim)
    /*
    q         当前 token 的 Query
    k_cache   历史 Key Cache
    v_cache   历史 Value Cache
    scores    Q 与历史 K 的注意力分数
    output    attention 输出
    position  当前 token 的位置
    max_context 最大上下文长度
    heads     Query head 数量
    kv_heads  KV head 数量
    head_dim  每个 head 的维度
    */
    {
        dim3 score_grid(heads, position + 1, 1);
        // 启动 score kernel
        attention_scores_kernel<<<score_grid, 256>>>(q, k_cache, scores, position, max_context, heads, kv_heads, head_dim);
        CITLALI_CUDA_CHECK(cudaGetLastError());
        // 检查 kernel 的启动错误
        attention_values_kernel<<<heads, head_dim>>>(scores, v_cache, output, position, max_context, heads, kv_heads, head_dim);
        CITLALI_CUDA_CHECK(cudaGetLastError());
    }
    // 依次启动两个 attention kernel

} // namespace citlali::compute
