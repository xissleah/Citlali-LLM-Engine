#include "citlali_remote/vulkan_context.h"
#include "vulkan_context_internal.h"

namespace citlali::remote {
using namespace detail;

    /*
    关键参数：
    first_layer / last_layer    要执行的层范围
    position                    当前 token 在序列中的位置
    max_context                 KV cache 最大长度
    input                       hidden state，FP16 bit，长度 hidden
    ffn_elements                FFN 中间层维度
    heads                       Query head 数量
    kv_heads                    Key/Value head 数量
    head_dim                    每个 head 的维度
    rms_norm_epsilon            RMSNorm epsilon
    rope_theta                  RoPE theta
    */
VulkanContext::MatVecResult VulkanContext::run_layer_range(
    std::uint32_t first_layer, std::uint32_t last_layer,
    std::uint32_t position, std::uint32_t max_context,
    const std::vector<std::uint16_t>& input, std::uint32_t ffn_elements,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
    float rms_norm_epsilon, float rope_theta)
{
    const std::uint32_t hidden = static_cast<std::uint32_t>(input.size());
    // 维度 input元素数
    const std::uint32_t q_elements = heads * head_dim;
    const std::uint32_t kv_elements = kv_heads * head_dim;
    if (first_layer > last_layer || hidden == 0 || position >= max_context ||
        kv_heads == 0 || heads % kv_heads != 0 || head_dim == 0 ||
        head_dim > 256 || (head_dim & 1U) != 0 ||
        rms_norm_epsilon <= 0.0F)
    {
        throw std::runtime_error("GPU layer range dimensions are invalid");
    }
    // 维度推导以及参数的校验

    ensure_execution_runtime(hidden, ffn_elements, q_elements, kv_elements, heads, max_context);
    // 执行运行时资源保障
    /*
    ensure_execution_runtime 确保所有 GPU 工作缓冲区
    x, norm, q, k, v, attention, projection, gate, up, activation, ffn_output, scores
    以及 descriptor pool、command pool、fence、query pool 已按当前维度分配好

    如果维度变化，会销毁旧资源并重新创建
    */
    const VkDeviceSize hidden_bytes = VkDeviceSize(hidden) * 2U;
    const VkDeviceSize q_bytes = VkDeviceSize(q_elements) * 2U;
    const VkDeviceSize kv_bytes = VkDeviceSize(kv_elements) * 2U;
    const VkDeviceSize ffn_bytes = VkDeviceSize(ffn_elements) * 2U;
    const VkDeviceSize cache_bytes = VkDeviceSize(max_context) * kv_bytes;
    // 字节数计算，乘2是因为每个元素占2字节，cache_bytes是整个 kv cache的字节数（max_context 个位置，每个位置 kv_bytes 大小）

    void* mapped = nullptr;
    // vulkan内存映射的目标指针
    check(vkMapMemory(device_, execution_.x.memory, 0, hidden_bytes, 0, &mapped), "vkMapMemory(layer range input)");
    // 调用 vkMapMemory 将 GPU 设备内存 execution_.x.memory 映射到 CPU 地址空间，映射范围从偏移 0 开始，长度 hidden_bytes。&mapped 接收映射后的 CPU 侧指针
    std::memcpy(mapped, input.data(), static_cast<std::size_t>(hidden_bytes));
    // 把input的data（fp16）拷贝到映射的GPU内存里
    vkUnmapMemory(device_, execution_.x.memory);
    // 解除映射

    // 输入数据上传到GPU

    check(vkResetDescriptorPool(device_, execution_.descriptor_pool, 0),
          "vkResetDescriptorPool(layer range)");
    /*
    重置描述符池 execution_.descriptor_pool。参数 0 表示 VK_DESCRIPTOR_POOL_RESET_NONE，即没有特殊标志
    这会释放池中所有之前分配的 descriptor set，使它们可以被下一轮重新分配。
    因为 run_layer_range 每次调用都会重新分配 descriptor set，所以必须在开始前重置池，避免池耗尽
    */
    check(vkResetCommandPool(device_, execution_.command_pool, 0),
          "vkResetCommandPool(layer range)");
    /*
    重置命令池 execution_.command_pool
    这会释放池中所有之前分配的 command buffer 及其录制的命令
    使 execution_.command_buffer 回到初始状态，可以重新开始录制
    */
    check(vkResetFences(device_, 1, &execution_.fence),
          "vkResetFences(layer range)");
    /*
    重置围栏 execution_.fence 为 unsignaled 状态
    1 表示重置 1 个 fence
    &execution_.fence 指向该 fence
    这个 fence 在函数末尾用于等待 GPU 完成提交（vkWaitForFences），所以必须在提交前确保它处于未信号状态
    */

    auto make_set = [&](VkBuffer a, VkDeviceSize a_bytes,
                        VkBuffer b, VkDeviceSize b_bytes,
                        VkBuffer c, VkDeviceSize c_bytes) {
        // 接受3个 (VkBuffer, VkDeviceSize) 对，分别对应 binding 0、1、2 的缓冲区和它们的字节大小，返回一个 VkDescriptorSet
        // 每个 binding 指向一个实际的 GPU 资源
        // 为什么是三个，因为三个对应的角色分别是 weight input output
        VkDescriptorSetAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocate.descriptorPool = execution_.descriptor_pool;
        // execution_.descriptor_pool刚重置过
        allocate.descriptorSetCount = 1;
        // 分配一个set
        allocate.pSetLayouts = &q4k_descriptor_layout_;
        // q4k_descriptor_layout_是VulkanContext 类的一个成员变量
        VkDescriptorSet set = VK_NULL_HANDLE;
        check(vkAllocateDescriptorSets(device_, &allocate, &set),
              "vkAllocateDescriptorSets(layer range)");
        // 调用 vkAllocateDescriptorSets 从池中分配一个 descriptor set，分配成功则 set 被赋值
        std::array<VkDescriptorBufferInfo, 3> infos{{
            {a, 0, a_bytes}, {b, 0, b_bytes}, {c, 0, c_bytes}}
        };
        // 构造 3 个 VkDescriptorBufferInfo

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t i = 0; i < writes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        /*
        构造 3 个 VkWriteDescriptorSet。循环中为每个 binding 设置
        dstSet = 刚分配的 set
        dstBinding = i（0、1、2）
        descriptorCount = 1（每个 binding 只写一个 descriptor）
        descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        pBufferInfo 指向对应的 VkDescriptorBufferInfo
        */
        vkUpdateDescriptorSets(device_, writes.size(), writes.data(), 0, nullptr);
        // 调用 vkUpdateDescriptorSets 将 3 个 write 操作应用到 descriptor set
        // 0 和 nullptr 表示没有 copy 操作
        // 这之后，set 就绑定好了三个 storage buffer

        return set;
        // 返回分配的 descriptor set
        // 注意：make_set 每次调用都分配一个新的 descriptor set，这就是为什么需要在第 49 行重置 descriptor pool
    };

    VkCommandBuffer command = execution_.command_buffer;
    // 从 execution_ 中取出预分配的 command buffer 句柄，赋给局部变量 command，方便后续使用
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    // 设置flags 为 VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    // 即这个 command buffer 只提交一次就被回收，驱动程序可以针对此做优化

    check(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(layer range)");
    // 开始录制vulkan命令缓冲区，所有后续的vkcmd*函数调用会被记录在这个buffer里，不会立即执行
    vkCmdResetQueryPool(command, execution_.query_pool, 0, 2);
    // 重置 query pool 中的 2 个 query（索引 0 和 1），为新的 GPU 时间戳测量做准备
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, execution_.query_pool, 0);
    // 在 GPU 管线的起始阶段，写入 timestamp 到 query 0。这是 GPU 计时的起点

    VkMemoryBarrier compute_memory{};
    compute_memory.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    // 这里指定 VK_ACCESS_SHADER_WRITE_BIT，意思是：等待所有 compute shader 的写入操作完成
    compute_memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_SHADER_WRITE_BIT |
                                   VK_ACCESS_TRANSFER_READ_BIT;
    // 确保 barrier 之前的所有 shader 写入完成，并且 barrier 之后的 shader 读/写操作和 copy 操作都能看到那些写入的数据
    auto compute_barrier = [&]() {
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &compute_memory, 0, nullptr, 0, nullptr);
    };
    // 计算着色器内存屏障，在命令缓冲区中插入一个管线屏障，确保所有计算着色器的写入操作完成，然后后续的计算或传输操作才能开始
    /*
    command：当前录制的命令缓冲区
    srcStageMask：VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT — 等待计算着色器阶段完成
    dstStageMask：VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT — 后续的计算着色器和数据传输操作等待这个屏障
    dependencyFlags：0 — 无特殊依赖
    memoryBarrierCount：1 — 使用1个内存屏障
    pMemoryBarriers：指向 compute_memory（前面定义的屏障）
    bufferMemoryBarrierCount：0 — 不使用缓冲区级屏障
    imageMemoryBarrierCount：0 — 不使用图像级屏障
    */

    auto transfer_barrier = [&]() {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &barrier, 0, nullptr, 0, nullptr);
    };
    // 传输操作内存屏障，确保传输操作写入完成然后计算着色器才能读取这些数据
    auto dispatch = [&](VkPipeline pipeline, VkDescriptorSet set,
                        const void* push, std::uint32_t push_bytes,
                        std::uint32_t x, std::uint32_t y = 1U) {
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        // 绑定计算管线
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                q4k_pipeline_layout_, 0, 1, &set, 0, nullptr);
        // 绑定描述符集
        if (push_bytes != 0) {
            vkCmdPushConstants(command, q4k_pipeline_layout_,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes,
                               push);
        }
        // 若有 push constant 数据，则把它推到布局的偏移 0 处、作用于计算着色器阶段。这是把每次调用不同的"小参数"（维度、position、epsilon、theta 等）传给着色器的方式——描述符集管大块 buffer，push constant 管少量标量
        vkCmdDispatch(command, x, y, 1);
        // 派发计算任务
    };
    // 分发计算任务
    auto matvec = [&](const StoredTensor& weight, VkBuffer in,
                      VkDeviceSize in_bytes, VkBuffer out,
                      VkDeviceSize out_bytes, std::uint32_t in_features,
                      std::uint32_t out_features)
    {
        VkDescriptorSet set = make_set(weight.buffer, weight.info.data_bytes,
                                       in, in_bytes, out, out_bytes);
        // 调用前面定义的 make_set 分配一个描述符集：binding 0 = 量化权重数据，binding 1 = 输入向量，binding 2 = 输出向量
        const std::array<std::uint32_t, 2> push{in_features, out_features};
        // push constant 只推两个维度。着色器拿到 in_features 才知道量化块怎么排布、知道要做多少次迭代，拿到 out_features 才知道输出怎么落位
        dispatch(weight.info.gguf_type == 12 ? q4k_pipeline_ : q6k_pipeline_,
                 set, push.data(), sizeof(push), (out_features + 1U) / 2U);
        // 按gguf量化类型选择管线
    };
    auto require_matrix = [](const StoredTensor& tensor,
                             std::uint32_t columns, std::uint32_t rows,
                             const char* label)
    {
        if ((tensor.info.gguf_type != 12 && tensor.info.gguf_type != 14) ||
            tensor.dims.size() != 2 || tensor.dims[0] != columns ||
            tensor.dims[1] != rows) {
            throw std::runtime_error(std::string("GPU layer tensor mismatch: ") + label);
        }
        // 任意一条不满足就抛异常
    };
    // 张量形状校验

    for (std::uint32_t layer_index = first_layer; layer_index <= last_layer; ++layer_index)
    {
        const std::string prefix = "blk." + std::to_string(layer_index) + ".";
        auto tensor = [&](const char* suffix) -> const StoredTensor& {
            const auto name = tensor_names_.find(prefix + suffix);
            // 记录该层名字
            if (name == tensor_names_.end())
                throw std::runtime_error("remote layer tensor is not loaded: " + prefix + suffix);
            // 没找到抛异常
            return tensors_.at(name->second);
            // 指向下一层
        };
        const StoredTensor& attn_norm = tensor("attn_norm.weight");
        const StoredTensor& attn_q = tensor("attn_q.weight");
        const StoredTensor& attn_k = tensor("attn_k.weight");
        const StoredTensor& attn_v = tensor("attn_v.weight");
        const StoredTensor& attn_o = tensor("attn_output.weight");
        const StoredTensor& q_norm = tensor("attn_q_norm.weight");
        const StoredTensor& k_norm = tensor("attn_k_norm.weight");
        const StoredTensor& ffn_norm = tensor("ffn_norm.weight");
        const StoredTensor& ffn_gate = tensor("ffn_gate.weight");
        const StoredTensor& ffn_up = tensor("ffn_up.weight");
        const StoredTensor& ffn_down = tensor("ffn_down.weight");
        require_matrix(attn_q, hidden, q_elements, "Q");
        require_matrix(attn_k, hidden, kv_elements, "K");
        require_matrix(attn_v, hidden, kv_elements, "V");
        require_matrix(attn_o, q_elements, hidden, "O");
        require_matrix(ffn_gate, hidden, ffn_elements, "gate");
        require_matrix(ffn_up, hidden, ffn_elements, "up");
        require_matrix(ffn_down, ffn_elements, hidden, "down");
        // 取这一层的11个张量，然后逐个校验维度

        LayerRuntime& runtime = layer_runtimes_[layer_index];
        if (runtime.max_context != max_context ||
            runtime.kv_elements != kv_elements) {
            runtime.max_context = max_context;
            runtime.kv_elements = kv_elements;
            ensure_gpu_buffer(runtime.gpu_k_cache, cache_bytes);
            ensure_gpu_buffer(runtime.gpu_v_cache, cache_bytes);
        }
        // 每一层的kvcache只在上下文长度或者kv维度发生变化时才重新分配，这是懒分配，同一维度下重复调用不会白白销毁重建

        struct RmsPush { std::uint32_t elements; float epsilon; };
        const RmsPush rms{hidden, rms_norm_epsilon};
        // 定义RMSNorm 的 push constant：元素个数 + epsilon
        dispatch(rms_norm_pipeline_,
                 make_set(execution_.x.buffer, hidden_bytes, attn_norm.buffer,
                          attn_norm.info.data_bytes, execution_.norm.buffer,
                          hidden_bytes),
                 &rms, sizeof(rms), 1);
        compute_barrier();
        // 读 execution_.x（当前 hidden state），乘 attn_norm 权重，输出到 execution_.norm
        // x=1 只派 1 个 workgroup——对一个 hidden 向量做 RMSNorm（求均方→归一化）可以用一个 workgroup 内的共享内存完成
        // 之后 compute_barrier() 保证 norm 写完后，后续矩阵乘才能读到

        matvec(attn_q, execution_.norm.buffer, hidden_bytes,
               execution_.q.buffer, q_bytes, hidden, q_elements);
        matvec(attn_k, execution_.norm.buffer, hidden_bytes,
               execution_.k.buffer, kv_bytes, hidden, kv_elements);
        matvec(attn_v, execution_.norm.buffer, hidden_bytes,
               execution_.v.buffer, kv_bytes, hidden, kv_elements);
        compute_barrier();
        /*
        让ai画一下流程图吧
                 ┌── Wq ──▶ q  (维度 q_elements = heads × head_dim)
        norm ────┼── Wk ──▶ k  (维度 kv_elements = kv_heads × head_dim)
         (hidden)│
                 └── Wv ──▶ v  (维度 kv_elements = kv_heads × head_dim)

        注意一下：
        RMSNorm归一化是按行归一化的，这里因为一次只处理一个token所以这一行恰好是全部元素
        */

        struct HeadNormPush { std::uint32_t heads; std::uint32_t head_dim; float epsilon; };
        // 信息结构体，包含的信息有：head数量、每个head多少维、eps用多少
        const HeadNormPush qn{heads, head_dim, rms_norm_epsilon};
        const HeadNormPush kn{kv_heads, head_dim, rms_norm_epsilon};
        // qn和kn不一样
        dispatch(head_rms_norm_pipeline_,
                 make_set(execution_.q.buffer, q_bytes, q_norm.buffer,
                          q_norm.info.data_bytes, execution_.q.buffer, q_bytes),
                 &qn, sizeof(qn), heads);
        // 对q逐头归一化
        dispatch(head_rms_norm_pipeline_,
                 make_set(execution_.k.buffer, kv_bytes, k_norm.buffer,
                          k_norm.info.data_bytes, execution_.k.buffer, kv_bytes),
                 &kn, sizeof(kn), kv_heads);
        // 对k逐头归一化
        compute_barrier();
        // 逐头 RMSNorm
        /*
        norm ──Wq──▶ q ──▶ [逐头RMSNorm(q_norm)] ──▶ RoPE ──▶ projection（旋转后的Q）
        norm ──Wk──▶ k ──▶ [逐头RMSNorm(k_norm)] ──▶ RoPE ──▶ attention（旋转后的K）
        norm ──Wv──▶ v（V 不经过这里，直接写 cache）
        */

        struct RopePush { std::uint32_t heads; std::uint32_t head_dim; std::uint32_t position; float theta; };
        /*
        heads	    头数	                    知道怎么把一维数组按 head 分段
        head_dim	每个head的维度	        决定每个维度对的旋转频率
        position	当前token在序列中的位置	    旋转角度的大小由它决定
        theta	    基频（如10000）	        控制频率衰减速度
        */
        const RopePush qr{heads, head_dim, position, rope_theta};
        const RopePush kr{kv_heads, head_dim, position, rope_theta};
        dispatch(rope_pipeline_,
                 make_set(execution_.q.buffer, q_bytes,
                            // binding 0：输入Q
                         execution_.q.buffer, q_bytes,
                            // binding 1：占位
                          execution_.projection.buffer, q_bytes),
                            // binding 2：输出到projection
                 &qr, sizeof(qr), (q_elements / 2U + 255U) / 256U);
        // 对Q施加RoPE
        dispatch(rope_pipeline_,
                 make_set(execution_.k.buffer, kv_bytes, execution_.k.buffer,
                          kv_bytes, execution_.attention.buffer, kv_bytes),
                 &kr, sizeof(kr), (kv_elements / 2U + 255U) / 256U);
        // 对kv施加RoPE
        compute_barrier();

        const VkDeviceSize cache_offset = VkDeviceSize(position) * kv_bytes;
        // 第position个槽的偏移
        VkBufferCopy copy{0, cache_offset, kv_bytes};
        // 拷贝描述结构体，源偏移0，目标偏移cache_offset，长度kv_bytes
        vkCmdCopyBuffer(command, execution_.attention.buffer,
                        runtime.gpu_k_cache.buffer, 1, &copy);
        // 把旋转后的k拷贝入本层k cache
        vkCmdCopyBuffer(command, execution_.v.buffer,
                        runtime.gpu_v_cache.buffer, 1, &copy);
        // v也是
        transfer_barrier();

        struct AttentionPush {
            std::uint32_t position, max_context, heads, kv_heads, head_dim;
            // 位置、最长上下文、q头数、kv头数、头的维度
            float scale;
            // 缩放点积注意力系数
            // Q·K 点积的方差随 head_dim 增长，除 √head_dim 把方差拉回 1，softmax 才不会饱和
        } attention_push{position, max_context, heads, kv_heads, head_dim,
                         1.0F / std::sqrt(static_cast<float>(head_dim))};
        dispatch(attention_scores_pipeline_,
                 make_set(execution_.projection.buffer, q_bytes,
                            // 旋转后的Q
                          runtime.gpu_k_cache.buffer, cache_bytes,
                            // 整层 k cache
                          execution_.scores.buffer,
                            // 分数矩阵
                          VkDeviceSize(heads) * max_context * sizeof(float)),
                 &attention_push, sizeof(attention_push), position + 1U,
                 heads);
        // 注意力打分S = Q · K（转置） · scale
        compute_barrier();
        dispatch(attention_value_pipeline_,
                 make_set(execution_.scores.buffer, VkDeviceSize(heads) * max_context * sizeof(float),
                            // 分数
                          runtime.gpu_v_cache.buffer, cache_bytes,
                            // v cache
                          execution_.attention.buffer, q_bytes),
                            // 注意力结果（输出）
                 &attention_push, 5U * sizeof(std::uint32_t), heads);
        // softmax + 加权求和
        compute_barrier();

        matvec(attn_o, execution_.attention.buffer, q_bytes,
               execution_.projection.buffer, hidden_bytes, q_elements, hidden);
        // 输出投影：attention_out → hidden
        compute_barrier();
        const std::uint32_t residual_push = hidden;
        dispatch(residual_add_pipeline_,
                 make_set(execution_.x.buffer, hidden_bytes,
                            // 输入：x原值
                          execution_.projection.buffer, hidden_bytes,
                            // 加数：注意力输出
                          execution_.x.buffer, hidden_bytes),
                            // 输出x+attention
                 &residual_push, sizeof(residual_push),
                 ((hidden + 1U) / 2U + 255U) / 256U);
        // 残差 x = x + attention
        compute_barrier();

        dispatch(rms_norm_pipeline_,
                 make_set(execution_.x.buffer, hidden_bytes, ffn_norm.buffer,
                          ffn_norm.info.data_bytes, execution_.norm.buffer,
                          hidden_bytes),
                 &rms, sizeof(rms), 1);
        compute_barrier();
        matvec(ffn_gate, execution_.norm.buffer, hidden_bytes,
               execution_.gate.buffer, ffn_bytes, hidden, ffn_elements);
        matvec(ffn_up, execution_.norm.buffer, hidden_bytes,
               execution_.up.buffer, ffn_bytes, hidden, ffn_elements);
        compute_barrier();
        // FFN上半：norm → gate/up

        const std::uint32_t swiglu_push = ffn_elements;
        dispatch(swiglu_pipeline_,
                 make_set(execution_.gate.buffer, ffn_bytes,
                          execution_.up.buffer, ffn_bytes,
                          execution_.activation.buffer, ffn_bytes),
                 &swiglu_push, sizeof(swiglu_push),
                 ((ffn_elements + 1U) / 2U + 255U) / 256U);
        compute_barrier();
        // swiglu激活

        matvec(ffn_down, execution_.activation.buffer, ffn_bytes,
               execution_.ffn_output.buffer, hidden_bytes, ffn_elements,
               hidden);
        compute_barrier();
        dispatch(residual_add_pipeline_,
                 make_set(execution_.x.buffer, hidden_bytes,
                          execution_.ffn_output.buffer, hidden_bytes,
                          execution_.x.buffer, hidden_bytes),
                 &residual_push, sizeof(residual_push),
                 ((hidden + 1U) / 2U + 255U) / 256U);
        compute_barrier();
        // FFN下半：down 投影 + 第二个残差
    }
    // 遍历要执行的层范围，每层完整执行一次 transformer block：注意力（norm→QKV→头部 norm→RoPE→KV 缓存写入→打分/加权求和→O 投影→残差）＋ FFN（norm→gate/up→SiLU→down→残差）

    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        execution_.query_pool, 1);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer(layer range)");
    // 在管线最末端写入结束时间戳
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(compute_queue_, 1, &submit, execution_.fence),
          "vkQueueSubmit(layer range)");
    // 提交后开始计算
    check(vkWaitForFences(device_, 1, &execution_.fence, VK_TRUE,
                          std::numeric_limits<std::uint64_t>::max()),
          "vkWaitForFences(layer range)");
    // 等待同步
    std::array<std::uint64_t, 2> timestamps{};
    check(vkGetQueryPoolResults(device_, execution_.query_pool, 0, 2,
                                sizeof(timestamps), timestamps.data(),
                                sizeof(std::uint64_t),
                                VK_QUERY_RESULT_64_BIT |
                                    VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults(layer range)");
    // 把两个 64 位时间戳读回 CPU

    check(vkMapMemory(device_, execution_.x.memory, 0, hidden_bytes, 0,
                      &mapped), "vkMapMemory(layer range output)");
    MatVecResult result;
    result.output.resize(hidden);
    std::memcpy(result.output.data(), mapped,
                static_cast<std::size_t>(hidden_bytes));
    vkUnmapMemory(device_, execution_.x.memory);
    // 把最终 hidden state（FP16）拷回 CPU 侧
    result.gpu_time_ns = static_cast<std::uint64_t>(
        static_cast<double>(timestamps[1] - timestamps[0]) *
        timestamp_period_ns_);
    return result;
    // 用设备的 timestamp period 把"时钟 tick 差"换算成纳秒，连同结果一起返回
}
    /*
    QwenModel
      -> RemoteClient::run_layer_range
      -> TCP RUN_LAYER_RANGE 请求
      -> TcpServer
      -> VulkanContext::run_layer_range
      -> Vulkan compute shaders
      -> 返回 hidden state
    */

} // namespace citlali::remote

