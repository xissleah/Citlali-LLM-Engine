#pragma once

#include "citlali_remote/protocol.h"

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace citlali::remote {

class VulkanContext {
public:
    struct StoredTensorInfo {
        std::uint32_t tensor_id = 0;
        std::uint32_t gguf_type = 0;
        std::uint64_t data_bytes = 0;
        std::uint64_t allocation_bytes = 0;
        std::uint64_t content_hash = 0;
        std::string name;
    };

    struct MatVecResult {
        std::vector<std::uint16_t> output;
        std::uint64_t gpu_time_ns = 0;
    };

    explicit VulkanContext(const std::string& shader_path);
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    const CapabilitiesPayload& capabilities() const {
        return capabilities_;
    }

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice device() const { return device_; }

    StoredTensorInfo store_tensor(const LoadTensorPayload& tensor);
    MatVecResult run_layer_range(std::uint32_t first_layer,
                                 std::uint32_t last_layer,
                                 std::uint32_t position,
                                 std::uint32_t max_context,
                                 const std::vector<std::uint16_t>& input,
                                 std::uint32_t ffn_elements,
                                 std::uint32_t heads,
                                 std::uint32_t kv_heads,
                                 std::uint32_t head_dim,
                                 float rms_norm_epsilon,
                                 float rope_theta);

private:
    struct GpuBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize capacity = 0;
    };

    struct StoredTensor {
        StoredTensorInfo info;
        std::vector<std::uint64_t> dims;
        std::uint64_t element_count = 0;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    struct LayerRuntime {
        std::uint32_t max_context = 0;
        std::uint32_t kv_elements = 0;
        std::vector<std::uint16_t> k_cache;
        std::vector<std::uint16_t> v_cache;
        GpuBuffer gpu_k_cache;
        GpuBuffer gpu_v_cache;
    };

    struct ExecutionRuntime {
        std::uint32_t hidden = 0;
        std::uint32_t ffn = 0;
        std::uint32_t q_elements = 0;
        std::uint32_t kv_elements = 0;
        std::uint32_t heads = 0;
        std::uint32_t max_context = 0;
        GpuBuffer x;
        GpuBuffer norm;
        GpuBuffer q;
        GpuBuffer k;
        GpuBuffer v;
        GpuBuffer attention;
        GpuBuffer projection;
        GpuBuffer gate;
        GpuBuffer up;
        GpuBuffer activation;
        GpuBuffer ffn_output;
        GpuBuffer scores;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VkQueryPool query_pool = VK_NULL_HANDLE;
    };

    void ensure_gpu_buffer(GpuBuffer& buffer, VkDeviceSize bytes);
    void destroy_gpu_buffer(GpuBuffer& buffer);
    void ensure_execution_runtime(std::uint32_t hidden,
                                  std::uint32_t ffn,
                                  std::uint32_t q_elements,
                                  std::uint32_t kv_elements,
                                  std::uint32_t heads,
                                  std::uint32_t max_context);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    std::uint32_t compute_queue_family_ = 0;
    float timestamp_period_ns_ = 0.0F;
    VkShaderModule q4k_shader_ = VK_NULL_HANDLE;
    VkShaderModule q6k_shader_ = VK_NULL_HANDLE;
    VkShaderModule rms_norm_shader_ = VK_NULL_HANDLE;
    VkShaderModule swiglu_shader_ = VK_NULL_HANDLE;
    VkShaderModule head_rms_norm_shader_ = VK_NULL_HANDLE;
    VkShaderModule rope_shader_ = VK_NULL_HANDLE;
    VkShaderModule attention_scores_shader_ = VK_NULL_HANDLE;
    VkShaderModule attention_value_shader_ = VK_NULL_HANDLE;
    VkShaderModule residual_add_shader_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout q4k_descriptor_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout q4k_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline q4k_pipeline_ = VK_NULL_HANDLE;
    VkPipeline q6k_pipeline_ = VK_NULL_HANDLE;
    VkPipeline rms_norm_pipeline_ = VK_NULL_HANDLE;
    VkPipeline swiglu_pipeline_ = VK_NULL_HANDLE;
    VkPipeline head_rms_norm_pipeline_ = VK_NULL_HANDLE;
    VkPipeline rope_pipeline_ = VK_NULL_HANDLE;
    VkPipeline attention_scores_pipeline_ = VK_NULL_HANDLE;
    VkPipeline attention_value_pipeline_ = VK_NULL_HANDLE;
    VkPipeline residual_add_pipeline_ = VK_NULL_HANDLE;
    std::uint32_t next_tensor_id_ = 1;
    CapabilitiesPayload capabilities_;
    std::unordered_map<std::uint32_t, StoredTensor> tensors_;
    std::unordered_map<std::string, std::uint32_t> tensor_names_;
    std::unordered_map<std::uint32_t, LayerRuntime> layer_runtimes_;
    ExecutionRuntime execution_;
};

} // namespace citlali::remote
