#include "citlali_remote/vulkan_context.h"
#include "vulkan_context_internal.h"

namespace citlali::remote {
using namespace detail;

VulkanContext::VulkanContext(const std::string& shader_path) {
    std::uint32_t loader_version = VK_API_VERSION_1_0;
    // 初始化vulkan版本为1.0
    const auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
            vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion")
            );
    // 从Vulkan驱动中获取函数指针
    // 作用：动态获取查询Vulkan版本的能力（1.1之前这个函数不存在）
    if (enumerate_instance_version != nullptr) {
        check(enumerate_instance_version(&loader_version),
              "vkEnumerateInstanceVersion");
    }
    // 如果成功获取函数指针，调用它查询系统支持的vulkan版本

    VkApplicationInfo application_info{};
    // 创建配置信息结构体
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "Citlali Remote Server";
    application_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.pEngineName = "Citlali";
    application_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application_info.apiVersion = loader_version;
    // 写入对应信息

    VkInstanceCreateInfo instance_info{};
    // 创建此结构体用于关联应用程序信息
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application_info;
    check(vkCreateInstance(&instance_info, nullptr, &instance_),
          "vkCreateInstance");
    /*
    vkCreateInstance：创建Vulkan实例（与驱动建立连接）
    instance_：存储创建好的实例句柄（成员变量）
    nullptr：不使用自定义分配器
    */

    std::uint32_t device_count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
          "vkEnumeratePhysicalDevices(count)");
    // 枚举物理设备（显卡）
    if (device_count == 0) {
        throw std::runtime_error("no Vulkan physical device found");
    }
    // 没有就抛异常

    std::vector<VkPhysicalDevice> devices(device_count);
    check(vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
          "vkEnumeratePhysicalDevices(devices)");
    // 传入数组指针，获取所有物理设备的句柄

    std::uint32_t compute_queue_family = 0;
    std::uint32_t compute_queue_count = 0;
    // 初始化计算队列族索引和队列数量
    for (const VkPhysicalDevice device : devices) {
        std::uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queues.data());
        /*
        遍历每个物理设备
        第一次调用 vkGetPhysicalDeviceQueueFamilyProperties：获取队列族数量
        创建 queues 向量，第二次调用：获取所有队列族的属性
        */

        for (std::uint32_t i = 0; i < queue_family_count; ++i) {
            if ((queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 &&
                queues[i].queueCount > 0) {
                physical_device_ = device;
                compute_queue_family = i;
                compute_queue_count = queues[i].queueCount;
                break;
            }
        }
        // 遍历每个队列族，检查是否支持计算队列，找到就记录设备、队列族索引和队列数量然后break
        if (physical_device_ != VK_NULL_HANDLE) {
            break;
        }
        // 找到设备就跳出循环
    }
    if (physical_device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("no Vulkan compute queue found");
    }
    // 没找到支持计算的设备就抛异常

    compute_queue_family_ = compute_queue_family;

    const auto get_properties2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceProperties2"));
    const auto get_features2 =
        reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
            vkGetInstanceProcAddr(instance_, "vkGetPhysicalDeviceFeatures2"));
    /*
    动态获取 vkGetPhysicalDeviceProperties2 和 vkGetPhysicalDeviceFeatures2
    这两个函数在Vulkan 1.0/1.1 中可能不存在，所以需要动态加载
    如果获取失败，会回退到旧版函数
    */
    VkPhysicalDeviceSubgroupProperties subgroup{};
    // 用于查询设备属性
    subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    if (get_properties2 != nullptr) {
        properties.pNext = &subgroup;
        get_properties2(physical_device_, &properties);
    }
    else {
        vkGetPhysicalDeviceProperties(physical_device_, &properties.properties);
    }
    /*
    如果支持 vkGetPhysicalDeviceProperties2，将 subgroup 链到 pNext 上以获取子组信息
    否则使用旧版 vkGetPhysicalDeviceProperties（但不包含子组信息）
    properties.properties 包含设备名称、驱动版本、限制等
    */

    VkPhysicalDeviceShaderFloat16Int8Features float16{};
    // 用于查询是否支持fp16
    float16.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    VkPhysicalDevice16BitStorageFeatures storage16{};
    // 用于查询是否支持fp16存储
    storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16.pNext = &float16;

    VkPhysicalDeviceFeatures2 features{};
    features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    if (get_features2 != nullptr) {
        features.pNext = &storage16;
        get_features2(physical_device_, &features);
    } else {
        vkGetPhysicalDeviceFeatures(physical_device_, &features.features);
    }
    /*
    如果支持 vkGetPhysicalDeviceFeatures2，将 storage16 链到 pNext 上
    否则使用旧版 vkGetPhysicalDeviceFeatures
    */

    const float queue_priority = 1.0F;
    // 队列优先级，范围0~1,1.0最高
    VkDeviceQueueCreateInfo queue_info{};
    // 用于指定队列族索引、队列数量和优先级
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = compute_queue_family_;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = get_features2 != nullptr ? &storage16 : nullptr;
    // 支持如果支持Vulkan 1.2，链接到 storage16 以启用FP16特性
    device_info.queueCreateInfoCount = 1;
    // 使用1个队列创建信息
    device_info.pQueueCreateInfos = &queue_info;
    // 指向队列创建信息
    device_info.pEnabledFeatures = &features.features;
    // 启用查询到的设备特性
    check(vkCreateDevice(physical_device_, &device_info, nullptr, &device_),
          "vkCreateDevice");
    // 创建逻辑设备
    vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
    // 从设备获取计算队列句柄，用于提交GPU命令

    q4k_shader_ = create_shader_module(device_, shader_path);
        // q4k矩阵向量乘法
    q6k_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "q6k_matvec.spv"));
        // q6k
    rms_norm_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "rms_norm.spv"));
        // RMSNorm归一化
    swiglu_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "swiglu.spv"));
        // swiglu
    head_rms_norm_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "head_rms_norm.spv"));
        // head的RMSNorm归一化
    rope_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "rope.spv"));
        // RoPE
    attention_scores_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "attention_scores.spv"));
        // 注意力分数
    attention_value_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "attention_value.spv"));
        // 注意力值
    residual_add_shader_ = create_shader_module(
        device_, sibling_shader_path(shader_path, "residual_add.spv"));
        // 残差连接
    // 加载算子

    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    // 3个绑定点用于GPU缓冲区绑定
    for (std::uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // 存储缓冲区类型（buffer）
        bindings[i].descriptorCount = 1;
        // 每个绑定一个缓冲区
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        // 在计算阶段可用
    }
    // 创建描述符集布局

    VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
    descriptor_layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_info.bindingCount = bindings.size();
    descriptor_layout_info.pBindings = bindings.data();
    check(vkCreateDescriptorSetLayout(device_, &descriptor_layout_info, nullptr, &q4k_descriptor_layout_),
          "vkCreateDescriptorSetLayout");
    // 描述GPU如何访问缓冲区

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = 8 * sizeof(std::uint32_t);
    // 创建管线布局
    /*
    push_range：Push常量范围（用于传递小数据，如维度参数）
    size = 32字节，可传递8个32位整数
    */

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    // 1个描述符集
    pipeline_layout_info.pSetLayouts = &q4k_descriptor_layout_;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    check(vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &q4k_pipeline_layout_),
          "vkCreatePipelineLayout");
    // 组合描述符集和Push常量

    q4k_pipeline_ =
        create_compute_pipeline(device_, q4k_shader_, q4k_pipeline_layout_);
    q6k_pipeline_ =
        create_compute_pipeline(device_, q6k_shader_, q4k_pipeline_layout_);
    rms_norm_pipeline_ = create_compute_pipeline(
        device_, rms_norm_shader_, q4k_pipeline_layout_);
    swiglu_pipeline_ = create_compute_pipeline(
        device_, swiglu_shader_, q4k_pipeline_layout_);
    head_rms_norm_pipeline_ = create_compute_pipeline(
        device_, head_rms_norm_shader_, q4k_pipeline_layout_);
    rope_pipeline_ = create_compute_pipeline(
        device_, rope_shader_, q4k_pipeline_layout_);
    attention_scores_pipeline_ = create_compute_pipeline(
        device_, attention_scores_shader_, q4k_pipeline_layout_);
    attention_value_pipeline_ = create_compute_pipeline(
        device_, attention_value_shader_, q4k_pipeline_layout_);
    residual_add_pipeline_ = create_compute_pipeline(
        device_, residual_add_shader_, q4k_pipeline_layout_);
    // 为每个着色器（算子）创建计算管线，所有管线统一使用同一个管线布局q4k_pipeline_layout_

    capabilities_.selected_version = kProtocolVersion;
    capabilities_.backend = BackendType::VulkanGpu;
    capabilities_.vulkan_api_version = properties.properties.apiVersion;
    capabilities_.driver_version = properties.properties.driverVersion;
    capabilities_.vendor_id = properties.properties.vendorID;
    capabilities_.device_id = properties.properties.deviceID;
    capabilities_.max_storage_buffer_bytes =
        properties.properties.limits.maxStorageBufferRange;
    capabilities_.max_compute_workgroup_invocations =
        properties.properties.limits.maxComputeWorkGroupInvocations;
    capabilities_.subgroup_size = subgroup.subgroupSize;
    capabilities_.compute_queue_family = compute_queue_family;
    capabilities_.compute_queue_count = compute_queue_count;
    capabilities_.shader_float16 = float16.shaderFloat16 == VK_TRUE;
    capabilities_.storage_buffer_16bit =
        storage16.storageBuffer16BitAccess == VK_TRUE;
    capabilities_.uniform_and_storage_buffer_16bit =
        storage16.uniformAndStorageBuffer16BitAccess == VK_TRUE;
    capabilities_.device_name = properties.properties.deviceName;
    timestamp_period_ns_ = properties.properties.limits.timestampPeriod;
    // 填充能力信息
}

VulkanContext::~VulkanContext() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        // 等待GPU完成所有工作，这是安全清理的前提
        for (auto& entry : tensors_) {
            if (entry.second.buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, entry.second.buffer, nullptr);
            }
            if (entry.second.memory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, entry.second.memory, nullptr);
            }
        }
        tensors_.clear();
        // 清理张量数据

        for (auto& entry : layer_runtimes_) {
            destroy_gpu_buffer(entry.second.gpu_k_cache);
            destroy_gpu_buffer(entry.second.gpu_v_cache);
        }
        // 清理层，销毁kvcache
        destroy_gpu_buffer(execution_.x);
        destroy_gpu_buffer(execution_.norm);
        destroy_gpu_buffer(execution_.q);
        destroy_gpu_buffer(execution_.k);
        destroy_gpu_buffer(execution_.v);
        destroy_gpu_buffer(execution_.attention);
        destroy_gpu_buffer(execution_.projection);
        destroy_gpu_buffer(execution_.gate);
        destroy_gpu_buffer(execution_.up);
        destroy_gpu_buffer(execution_.activation);
        destroy_gpu_buffer(execution_.ffn_output);
        destroy_gpu_buffer(execution_.scores);
        // 清理GPU所有缓冲区
        if (execution_.fence != VK_NULL_HANDLE) vkDestroyFence(device_, execution_.fence, nullptr);
        if (execution_.query_pool != VK_NULL_HANDLE) vkDestroyQueryPool(device_, execution_.query_pool, nullptr);
        if (execution_.command_pool != VK_NULL_HANDLE) vkDestroyCommandPool(device_, execution_.command_pool, nullptr);
        if (execution_.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, execution_.descriptor_pool, nullptr);
        // 销毁栅栏、查询池（用于性能计时）、命令池、描述符池

        if (q4k_pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, q4k_pipeline_, nullptr);
        }
        if (q6k_pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, q6k_pipeline_, nullptr);
        }
        if (rms_norm_pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, rms_norm_pipeline_, nullptr);
        }
        if (swiglu_pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, swiglu_pipeline_, nullptr);
        }
        // 销毁管线

        const std::array<VkPipeline, 5> extra_pipelines = {
            head_rms_norm_pipeline_, rope_pipeline_, attention_scores_pipeline_,
            attention_value_pipeline_, residual_add_pipeline_};
        for (VkPipeline pipeline : extra_pipelines) {
            if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, pipeline, nullptr);
        }
        // 用数组批量销毁五个额外管线

        if (q4k_pipeline_layout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, q4k_pipeline_layout_, nullptr);
        }
        if (q4k_descriptor_layout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, q4k_descriptor_layout_,
                                         nullptr);
        }
        // 销毁管线布局和描述符集布局
        if (q4k_shader_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, q4k_shader_, nullptr);
        }
        if (q6k_shader_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, q6k_shader_, nullptr);
        }
        if (rms_norm_shader_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, rms_norm_shader_, nullptr);
        }
        if (swiglu_shader_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, swiglu_shader_, nullptr);
        }
        const std::array<VkShaderModule, 5> extra_shaders = {
            head_rms_norm_shader_, rope_shader_, attention_scores_shader_,
            attention_value_shader_, residual_add_shader_};
        for (VkShaderModule shader : extra_shaders) {
            if (shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, shader, nullptr);
        }
        // 销毁着色器（算子）

        vkDestroyDevice(device_, nullptr);
        // 销毁逻辑设备
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        // 销毁vulkan实例
    }
}

void VulkanContext::destroy_gpu_buffer(GpuBuffer& buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device_, buffer.buffer, nullptr);
    if (buffer.memory != VK_NULL_HANDLE) vkFreeMemory(device_, buffer.memory, nullptr);
    buffer = {};
}
    // 销毁缓冲区工具函数

void VulkanContext::ensure_gpu_buffer(GpuBuffer& buffer, VkDeviceSize bytes) {
    if (buffer.buffer != VK_NULL_HANDLE && buffer.capacity >= bytes) return;
    // 缓冲区存在且足够直接返回
    destroy_gpu_buffer(buffer);
    // 否则销毁旧的缓冲区

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    /*
    usage 标志：
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT：用于着色器存储（buffer 类型）
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT：可作为数据传输源（读）
        VK_BUFFER_USAGE_TRANSFER_DST_BIT：可作为数据传输目标（写）
        sharingMode = VK_SHARING_MODE_EXCLUSIVE：只被一个队列族使用（性能最佳）
    */
    check(vkCreateBuffer(device_, &info, nullptr, &buffer.buffer),
          "vkCreateBuffer(persistent)");
    // 创建vulkan缓冲区对象
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer.buffer, &requirements);
    // 查询缓冲区内存需求（大小，对齐，支持的内存类型）
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex =
        find_host_memory_type(physical_device_, requirements.memoryTypeBits);
    // 从支持的内存类型中找到 Host Visible + Host Coherent 的类型（用于CPU上传数据）
    check(vkAllocateMemory(device_, &allocation, nullptr, &buffer.memory),
          "vkAllocateMemory(persistent)");
    // 分配GPU内存
    check(vkBindBufferMemory(device_, buffer.buffer, buffer.memory, 0),
          "vkBindBufferMemory(persistent)");
    // 缓冲区绑定在分配的显存上
    buffer.capacity = bytes;
    // 记录缓冲区容量
}
    // 确保GPU缓冲区存在并有足够数量

void VulkanContext::ensure_execution_runtime(
    std::uint32_t hidden, std::uint32_t ffn, std::uint32_t q_elements,
    std::uint32_t kv_elements, std::uint32_t heads,
    std::uint32_t max_context)
    /*
    hidden：隐藏层维度（hidden_size）
    ffn：前馈网络维度（ffn_size）
    q_elements：Q矩阵元素数（heads * head_dim）
    kv_elements：KV矩阵元素数（kv_heads * head_dim）
    heads：注意力头数
    max_context：最大上下文长度
    */
{
    const VkDeviceSize hidden_bytes = VkDeviceSize(hidden) * 2U;
    const VkDeviceSize q_bytes = VkDeviceSize(q_elements) * 2U;
    const VkDeviceSize kv_bytes = VkDeviceSize(kv_elements) * 2U;
    const VkDeviceSize ffn_bytes = VkDeviceSize(ffn) * 2U;
    // 计算各缓冲区大小
    ensure_gpu_buffer(execution_.x, hidden_bytes);
    ensure_gpu_buffer(execution_.norm, hidden_bytes);
    ensure_gpu_buffer(execution_.q, q_bytes);
    ensure_gpu_buffer(execution_.k, kv_bytes);
    ensure_gpu_buffer(execution_.v, kv_bytes);
    ensure_gpu_buffer(execution_.attention, q_bytes);
    ensure_gpu_buffer(execution_.projection, hidden_bytes > q_bytes ? hidden_bytes : q_bytes);
    ensure_gpu_buffer(execution_.gate, ffn_bytes);
    ensure_gpu_buffer(execution_.up, ffn_bytes);
    ensure_gpu_buffer(execution_.activation, ffn_bytes);
    ensure_gpu_buffer(execution_.ffn_output, hidden_bytes);
    ensure_gpu_buffer(execution_.scores,
                      VkDeviceSize(heads) * max_context * sizeof(float));
    // 确保所有缓冲区已分配

    execution_.hidden = hidden;
    execution_.ffn = ffn;
    execution_.q_elements = q_elements;
    execution_.kv_elements = kv_elements;
    execution_.heads = heads;
    execution_.max_context = max_context;
    // 记录模型参数

    if (execution_.descriptor_pool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize size{};
        size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        size.descriptorCount = 4096;
        // 可分配4096个存储缓冲区描述符
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 1365;
        // 最多可创建1365个描述符集（4096/3≈1365）
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        check(vkCreateDescriptorPool(device_, &pool, nullptr, &execution_.descriptor_pool),
              "vkCreateDescriptorPool(persistent)");
    }
    // 创建描述符池

    if (execution_.command_pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        // VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT：允许单独重置命令缓冲区
        pool.queueFamilyIndex = compute_queue_family_;
        // queueFamilyIndex：与计算队列关联
        check(vkCreateCommandPool(device_, &pool, nullptr, &execution_.command_pool),
              "vkCreateCommandPool(persistent)");

        VkCommandBufferAllocateInfo allocate{};
        allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate.commandPool = execution_.command_pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &allocate, &execution_.command_buffer),
              "vkAllocateCommandBuffers(persistent)");
        // 分配1个主命令缓冲区（可直接提交到队列）

        VkFenceCreateInfo fence{};
        fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        check(vkCreateFence(device_, &fence, nullptr, &execution_.fence),
              "vkCreateFence(persistent)");
        // 创建栅栏，用于CPU等待GPU完成工作

        VkQueryPoolCreateInfo query{};
        query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query.queryCount = 2;
        // queryCount = 2：需要2个查询（开始和结束时间戳）
        check(vkCreateQueryPool(device_, &query, nullptr,&execution_.query_pool),
              "vkCreateQueryPool(persistent)");
        // 创建时间戳查询池，用于测量GPU执行时间
    }
    // 创建命令池和命令缓冲区
}
    // 确保执行运行时已准备完毕

} // namespace citlali::remote

