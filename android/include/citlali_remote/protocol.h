#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace citlali::remote {

constexpr std::uint32_t kProtocolMagic = 0x314C5443U; // "CTL1" in LE
constexpr std::uint16_t kProtocolVersion = 1;
constexpr std::size_t kFrameHeaderBytes = 24;
constexpr std::uint64_t kMaxPayloadBytes = 64ULL * 1024ULL * 1024ULL;

enum class MessageType : std::uint16_t {
    Ping = 1,
    Pong = 2,
    Hello = 3,
    Capabilities = 4,
    LoadTensor = 5,
    TensorLoaded = 6,
    RunLayerRange = 13,
    LayerRangeResult = 14,
    Error = 255,
};

enum class BackendType : std::uint16_t {
    Auto = 0,
    VulkanGpu = 1,
    Npu = 2,
    Hybrid = 3,
};

struct HelloPayload {
    std::uint16_t minimum_version = kProtocolVersion;
    std::uint16_t maximum_version = kProtocolVersion;
    BackendType requested_backend = BackendType::VulkanGpu;
};

struct CapabilitiesPayload {
    std::uint16_t selected_version = kProtocolVersion;
    BackendType backend = BackendType::VulkanGpu;
    std::uint32_t vulkan_api_version = 0;
    std::uint32_t driver_version = 0;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint64_t max_storage_buffer_bytes = 0;
    std::uint32_t max_compute_workgroup_invocations = 0;
    std::uint32_t subgroup_size = 0;
    std::uint32_t compute_queue_family = 0;
    std::uint32_t compute_queue_count = 0;
    bool shader_float16 = false;
    bool storage_buffer_16bit = false;
    bool uniform_and_storage_buffer_16bit = false;
    std::string device_name;
};

struct LoadTensorPayload {
    std::uint32_t gguf_type = 0;
    std::vector<std::uint64_t> dims;
    std::uint64_t element_count = 0;
    std::uint64_t content_hash = 0;
    std::string name;
    std::vector<std::uint8_t> data;
};

struct TensorLoadedPayload {
    std::uint32_t tensor_id = 0;
    std::uint32_t gguf_type = 0;
    std::uint64_t data_bytes = 0;
    std::uint64_t allocation_bytes = 0;
    std::uint64_t content_hash = 0;
    std::string name;
};

struct RunLayerRangePayload {
    std::uint32_t first_layer = 0;
    std::uint32_t last_layer = 0;
    std::uint32_t position = 0;
    std::uint32_t max_context = 0;
    std::uint32_t hidden_elements = 0;
    std::uint32_t ffn_elements = 0;
    std::uint32_t heads = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim = 0;
    float rms_norm_epsilon = 0.0F;
    float rope_theta = 0.0F;
    std::uint64_t input_hash = 0;
    std::vector<std::uint16_t> input;
};

struct LayerRangeResultPayload {
    std::uint32_t first_layer = 0;
    std::uint32_t last_layer = 0;
    std::uint32_t position = 0;
    std::uint64_t output_hash = 0;
    std::uint64_t gpu_time_ns = 0;
    std::vector<std::uint16_t> output;
};

struct Frame {
    MessageType type = MessageType::Error;
    std::uint64_t request_id = 0;
    std::vector<std::uint8_t> payload;
};

bool receive_frame(int socket_fd, Frame& frame, std::string& error);
bool send_frame(int socket_fd, const Frame& frame, std::string& error);
bool decode_hello_payload(const std::vector<std::uint8_t>& bytes,
                          HelloPayload& hello, std::string& error);
std::vector<std::uint8_t> encode_capabilities_payload(
    const CapabilitiesPayload& capabilities);
bool decode_load_tensor_payload(const std::vector<std::uint8_t>& bytes,
                                LoadTensorPayload& tensor,
                                std::string& error);
std::vector<std::uint8_t> encode_tensor_loaded_payload(
    const TensorLoadedPayload& tensor);
bool decode_run_layer_range_payload(const std::vector<std::uint8_t>& bytes,
                                    RunLayerRangePayload& request,
                                    std::string& error);
std::vector<std::uint8_t> encode_layer_range_result_payload(
    const LayerRangeResultPayload& result);
std::uint64_t content_hash64(const void* data, std::size_t bytes);

} // namespace citlali::remote
