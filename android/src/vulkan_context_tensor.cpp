#include "citlali_remote/vulkan_context.h"
#include "vulkan_context_internal.h"

namespace citlali::remote {
using namespace detail;

VulkanContext::StoredTensorInfo VulkanContext::store_tensor(
    const LoadTensorPayload& tensor) {
    constexpr std::uint32_t kGgufTypeQ4K = 12;
    // GGUF格式中Q4_K量化类型的标识符
    constexpr std::uint64_t kQ4KBlockElements = 256;
    // Q4_K每个块包含256个元素
    constexpr std::uint64_t kQ4KBlockBytes = 144;
    // Q4_K每个块占用144字节
    constexpr std::uint64_t kQ6KBlockBytes = 210;
    // Q6_K每个块占用210字节

    if (device_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Vulkan logical device is not initialized");
    }
    // vulkan设备是否初始化，没有就抛异常
    std::uint64_t expected_bytes = 0;
    if (tensor.gguf_type == kGgufTypeQ4K) {
        expected_bytes =
            ((tensor.element_count + kQ4KBlockElements - 1) /
             kQ4KBlockElements) *
            kQ4KBlockBytes;
    }
    // Q4_K
    else if (tensor.gguf_type == 14) {
        expected_bytes =
            ((tensor.element_count + kQ4KBlockElements - 1) /
             kQ4KBlockElements) *
            kQ6KBlockBytes;
    }
    // Q6_K
    else if (tensor.gguf_type == 0) {
        expected_bytes = tensor.element_count * sizeof(float);
    }
    // f32
    else if (tensor.gguf_type == 1) {
        expected_bytes = tensor.element_count * sizeof(std::uint16_t);
    }
    // f16
    else {
        throw std::runtime_error(
            "first FFN version only accepts F32, F16, Q4_K and Q6_K tensors");
    }
    // 为定义的类型抛异常

    if (tensor.data.size() != expected_bytes) {
        throw std::runtime_error("tensor byte count does not match elements");
    }
    // 验证实际大小和预期大小是否一致
    if (tensor.data.size() > capabilities_.max_storage_buffer_bytes) {
        throw std::runtime_error("tensor exceeds maxStorageBufferRange");
    }
    // 校验数据大小是否超限
    const auto existing_name = tensor_names_.find(tensor.name);
    // 检查张量是否存在
    if (existing_name != tensor_names_.end()) {
        const auto existing_tensor = tensors_.find(existing_name->second);
        if (existing_tensor == tensors_.end()) {
            throw std::runtime_error("remote tensor index is inconsistent");
        }
        const StoredTensor& stored = existing_tensor->second;
        if (stored.info.gguf_type != tensor.gguf_type ||
            stored.info.data_bytes != tensor.data.size() ||
            stored.info.content_hash != tensor.content_hash ||
            stored.dims != tensor.dims ||
            stored.element_count != tensor.element_count) {
            throw std::runtime_error(
                "tensor name is already loaded with different content: " +
                tensor.name);
        }
        // 张量存在，校验元数据
        return stored.info;
        // 已有缓存，避免重复操作，提前退出
    }
    if (next_tensor_id_ == 0) {
        throw std::runtime_error("remote tensor id space exhausted");
    }
    // 张量id为0表示id空间耗尽

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    try {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = static_cast<VkDeviceSize>(tensor.data.size());
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer),
              "vkCreateBuffer");
        /*
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT：用作着色器存储缓冲区（GPU计算核心用途）
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT：可作为传输源（用于读取或复制）
        VK_BUFFER_USAGE_TRANSFER_DST_BIT：可作为传输目标（用于更新数据）
        VK_SHARING_MODE_EXCLUSIVE：只由一个队列族使用（性能最佳）
        */

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer, &requirements);
        // 分配GPU内存

        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
        const VkMemoryPropertyFlags required_flags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // 获取物理设备的内存属性
        /*
        HOST_VISIBLE：CPU可以直接访问（用于上传数据）
        HOST_COHERENT：CPU和GPU缓存一致（无需显式刷新）
        */
        std::uint32_t memory_type_index =
            std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1U << i)) == 0) {
                continue;
            }
            // 检查缓冲区兼容性
            // 不同内存类型存在一个数组里，通过逐个下标匹配就能知道支持什么内存类型
            const VkMemoryPropertyFlags flags = memory_properties.memoryTypes[i].propertyFlags;
            if ((flags & required_flags) != required_flags) {
                continue;
            }
            // 不支持HOST_VISIBLE和HOST_COHERENT这两个属性就跳过
            memory_type_index = i;
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
                break;
            }
            // 找到最合适的内存类型就停止
        }
        // 选择内存类型
        if (memory_type_index == std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "no host-visible coherent Vulkan memory type for tensor");
        }
        // 找不到合适的内存类型就抛异常

        VkMemoryAllocateInfo allocation_info{};
        allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation_info.allocationSize = requirements.size;
        allocation_info.memoryTypeIndex = memory_type_index;
        check(vkAllocateMemory(device_, &allocation_info, nullptr, &memory),
              "vkAllocateMemory");
        check(vkBindBufferMemory(device_, buffer, memory, 0),
              "vkBindBufferMemory");
        // 分配GPU内存，并将缓冲区绑定到这块内存

        void* mapped = nullptr;
        check(vkMapMemory(device_, memory, 0, tensor.data.size(), 0, &mapped),
              "vkMapMemory");
        std::memcpy(mapped, tensor.data.data(), tensor.data.size());
        vkUnmapMemory(device_, memory);
        // 上传数据到GPU

        StoredTensor stored;
        stored.info.tensor_id = next_tensor_id_++;
        stored.info.gguf_type = tensor.gguf_type;
        stored.info.data_bytes = tensor.data.size();
        stored.info.allocation_bytes = requirements.size;
        stored.info.content_hash = tensor.content_hash;
        stored.info.name = tensor.name;
        stored.dims = tensor.dims;
        stored.element_count = tensor.element_count;
        stored.buffer = buffer;
        stored.memory = memory;

        const StoredTensorInfo result = stored.info;
        tensor_names_.emplace(stored.info.name, stored.info.tensor_id);
        tensors_.emplace(stored.info.tensor_id, std::move(stored));
        // 存储张量元数据
        return result;
    }
    catch (...) {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer, nullptr);
        }
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, memory, nullptr);
        }
        throw;
    }
    // 异常处理
}
    // 存储张量到vulkan上下文

} // namespace citlali::remote

