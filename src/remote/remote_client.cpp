#include "citlali/remote/remote_client.h"

#include "citlali/compute/common/dtype.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace citlali::remote {
namespace {

constexpr std::uint32_t kProtocolMagic = 0x314C5443U;
constexpr std::uint16_t kProtocolVersion = 1;
constexpr std::size_t kFrameHeaderBytes = 24;   // 帧头固定 24 字节
constexpr std::uint64_t kMaxPayloadBytes = 64ULL * 1024ULL * 1024ULL;   // 64MB上限
// 协议常量与消息类型

enum class MessageType : std::uint16_t {
    Hello = 3,              // 客户端握手
    Capabilities = 4,       // 服务端回复能力
    LoadTensor = 5,         // 上传张量
    TensorLoaded = 6,       // 服务器确认
    RunLayerRange = 13,     // 请求跑一层区间
    LayerRangeResult = 14,  // 层区间结果
    Error = 255,            // 错误
};

struct Frame {
    MessageType type = MessageType::Error;  // 消息类型
    std::uint64_t request_id = 0;           // 对应哪一次请求
    std::vector<std::uint8_t> payload;      // 具体内容
};

void write_u16_le(std::uint8_t* output, std::uint16_t value) {
    output[0] = static_cast<std::uint8_t>(value);
    output[1] = static_cast<std::uint8_t>(value >> 8U);
}

void write_u32_le(std::uint8_t* output, std::uint32_t value) {
    for (unsigned int i = 0; i < 4; ++i) {
        output[i] = static_cast<std::uint8_t>(value >> (i * 8U));
    }
}

void write_u64_le(std::uint8_t* output, std::uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        output[i] = static_cast<std::uint8_t>(value >> (i * 8U));
    }
}

std::uint16_t read_u16_le(const std::uint8_t* input) {
    return static_cast<std::uint16_t>(input[0]) |
           (static_cast<std::uint16_t>(input[1]) << 8U);
}

std::uint32_t read_u32_le(const std::uint8_t* input) {
    std::uint32_t value = 0;
    for (unsigned int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(input[i]) << (i * 8U);
    }
    return value;
}

std::uint64_t read_u64_le(const std::uint8_t* input) {
    std::uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(input[i]) << (i * 8U);
    }
    return value;
}

void append_u16_le(std::vector<std::uint8_t>& output, std::uint16_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + 2);
    write_u16_le(output.data() + offset, value);
}

void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + 4);
    write_u32_le(output.data() + offset, value);
}

void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + 8);
    write_u64_le(output.data() + offset, value);
}

std::uint64_t content_hash64(const void* data, std::size_t bytes) {
    constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* input = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = kOffsetBasis;
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= input[i];
        hash *= kPrime;
    }
    return hash;
}
    // FNV-1a哈希，用于端到端完整性校验

std::uint16_t backend_wire_value(Backend backend) {
    switch (backend) {
        case Backend::Auto: return 0;
        case Backend::Gpu: return 1;
        case Backend::Npu: return 2;
        case Backend::Hybrid: return 3;
    }
    return 0;
}

std::string layer_tensor_name(std::uint32_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

#ifdef _WIN32

class WinsockRuntime {
public:
    WinsockRuntime() {
        WSADATA data{};
        const int result = WSAStartup(MAKEWORD(2, 2), &data);
        if (result != 0) {
            throw std::runtime_error("WSAStartup failed: " +
                                     std::to_string(result));
        }
    }
    ~WinsockRuntime() { WSACleanup(); }
};

class SocketHandle {
public:
    SocketHandle() = default;
    explicit SocketHandle(SOCKET socket) : socket_(socket) {}
    ~SocketHandle() {
        if (socket_ != INVALID_SOCKET) closesocket(socket_);
    }
    SocketHandle(const SocketHandle&) = delete;
    SocketHandle& operator=(const SocketHandle&) = delete;
    SocketHandle(SocketHandle&& other) noexcept : socket_(other.socket_) {
        other.socket_ = INVALID_SOCKET;
    }
    SocketHandle& operator=(SocketHandle&& other) noexcept {
        if (this != &other) {
            if (socket_ != INVALID_SOCKET) closesocket(socket_);
            socket_ = other.socket_;
            other.socket_ = INVALID_SOCKET;
        }
        return *this;
    }
    SOCKET get() const { return socket_; }

private:
    SOCKET socket_ = INVALID_SOCKET;
};

void send_all(SOCKET socket, const void* data, std::size_t bytes) {
    const auto* cursor = static_cast<const char*>(data);
    std::size_t sent = 0;
    while (sent < bytes) {
        const int chunk = static_cast<int>(std::min<std::size_t>(
            bytes - sent,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        const int result = send(socket, cursor + sent, chunk, 0);
        if (result == SOCKET_ERROR) {
            throw std::runtime_error("remote send failed: " +
                                     std::to_string(WSAGetLastError()));
        }
        sent += static_cast<std::size_t>(result);
    }
}

void receive_exact(SOCKET socket, void* data, std::size_t bytes) {
    auto* cursor = static_cast<char*>(data);
    std::size_t received = 0;
    while (received < bytes) {
        const int chunk = static_cast<int>(std::min<std::size_t>(
            bytes - received,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
        const int result = recv(socket, cursor + received, chunk, 0);
        if (result == 0) {
            throw std::runtime_error("remote device closed the connection");
        }
        if (result == SOCKET_ERROR) {
            throw std::runtime_error("remote receive failed: " +
                                     std::to_string(WSAGetLastError()));
        }
        received += static_cast<std::size_t>(result);
    }
}

SocketHandle connect_tcp(const std::string& endpoint) {
    const std::size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= endpoint.size()) {
        throw std::runtime_error("--remote must use host:port format");
    }
    const std::string host = endpoint.substr(0, colon);
    const std::string port = endpoint.substr(colon + 1);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    addrinfo* addresses = nullptr;
    const int lookup = getaddrinfo(host.c_str(), port.c_str(), &hints,
                                   &addresses);
    if (lookup != 0) {
        throw std::runtime_error("remote getaddrinfo failed: " +
                                 std::to_string(lookup));
    }

    SOCKET connected = INVALID_SOCKET;
    for (addrinfo* address = addresses; address; address = address->ai_next) {
        SOCKET candidate = socket(address->ai_family, address->ai_socktype,
                                  address->ai_protocol);
        if (candidate == INVALID_SOCKET) continue;
        if (connect(candidate, address->ai_addr,
                    static_cast<int>(address->ai_addrlen)) == 0) {
            connected = candidate;
            break;
        }
        closesocket(candidate);
    }
    freeaddrinfo(addresses);
    if (connected == INVALID_SOCKET) {
        throw std::runtime_error("unable to connect to remote device at " +
                                 endpoint);
    }

    BOOL no_delay = TRUE;
    setsockopt(connected, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&no_delay), sizeof(no_delay));
    return SocketHandle(connected);
}

void send_frame(SOCKET socket, MessageType type, std::uint64_t request_id,
                const std::vector<std::uint8_t>& payload) {
    if (payload.size() > kMaxPayloadBytes) {
        throw std::runtime_error("remote request payload exceeds limit");
    }
    std::array<std::uint8_t, kFrameHeaderBytes> header{};
    write_u32_le(header.data(), kProtocolMagic);
    write_u16_le(header.data() + 4, kProtocolVersion);
    write_u16_le(header.data() + 6, static_cast<std::uint16_t>(type));
    write_u64_le(header.data() + 8, request_id);
    write_u64_le(header.data() + 16, payload.size());
    send_all(socket, header.data(), header.size());
    if (!payload.empty()) send_all(socket, payload.data(), payload.size());
}

Frame receive_frame(SOCKET socket) {
    std::array<std::uint8_t, kFrameHeaderBytes> header{};
    receive_exact(socket, header.data(), header.size());
    if (read_u32_le(header.data()) != kProtocolMagic) {
        throw std::runtime_error("invalid remote response magic");
    }
    if (read_u16_le(header.data() + 4) != kProtocolVersion) {
        throw std::runtime_error("unsupported remote protocol version");
    }
    const std::uint64_t payload_bytes = read_u64_le(header.data() + 16);
    if (payload_bytes > kMaxPayloadBytes) {
        throw std::runtime_error("remote response payload exceeds limit");
    }
    Frame frame;
    frame.type = static_cast<MessageType>(read_u16_le(header.data() + 6));
    frame.request_id = read_u64_le(header.data() + 8);
    frame.payload.resize(static_cast<std::size_t>(payload_bytes));
    if (!frame.payload.empty()) {
        receive_exact(socket, frame.payload.data(), frame.payload.size());
    }
    return frame;
}

#endif

void throw_if_error(const Frame& frame) {
    if (frame.type == MessageType::Error) {
        throw std::runtime_error("remote error: " +
            std::string(frame.payload.begin(), frame.payload.end()));
    }
}

std::vector<std::uint8_t> make_hello_payload(Backend backend) {
    std::vector<std::uint8_t> payload;
    append_u16_le(payload, kProtocolVersion);
    append_u16_le(payload, kProtocolVersion);
    append_u16_le(payload, backend_wire_value(backend));
    append_u16_le(payload, 0);
    return payload;
}

std::vector<std::uint8_t> make_load_tensor_payload(
    const io::GgufTensorInfo& tensor,
    const std::vector<std::uint8_t>& data,
    std::uint64_t hash) {
    if (tensor.dims.empty() || tensor.dims.size() > 8 ||
        tensor.name.empty() || tensor.name.size() > 1024) {
        throw std::runtime_error("remote tensor metadata exceeds limits");
    }
    std::vector<std::uint8_t> payload;
    payload.reserve(36 + tensor.dims.size() * 8 + tensor.name.size() +
                    data.size());
    append_u32_le(payload, static_cast<std::uint32_t>(tensor.type));
    append_u16_le(payload, static_cast<std::uint16_t>(tensor.dims.size()));
    append_u16_le(payload, 0);
    append_u64_le(payload, tensor.element_count());
    append_u64_le(payload, data.size());
    append_u64_le(payload, hash);
    append_u16_le(payload, static_cast<std::uint16_t>(tensor.name.size()));
    append_u16_le(payload, 0);
    for (const std::uint64_t dimension : tensor.dims) {
        append_u64_le(payload, dimension);
    }
    payload.insert(payload.end(), tensor.name.begin(), tensor.name.end());
    payload.insert(payload.end(), data.begin(), data.end());
    return payload;
}

std::vector<std::uint8_t> make_run_layer_range_payload(
    std::uint32_t first_layer, std::uint32_t last_layer,
    std::uint32_t position, std::uint32_t max_context,
    const std::vector<std::uint16_t>& input,
    std::uint32_t ffn_elements, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    float rms_norm_epsilon, float rope_theta) {
    std::uint32_t epsilon_bits = 0;
    std::uint32_t theta_bits = 0;
    std::memcpy(&epsilon_bits, &rms_norm_epsilon, sizeof(epsilon_bits));
    std::memcpy(&theta_bits, &rope_theta, sizeof(theta_bits));
    std::vector<std::uint8_t> payload;
    payload.reserve(52 + input.size() * sizeof(std::uint16_t));
    append_u32_le(payload, first_layer);
    append_u32_le(payload, last_layer);
    append_u32_le(payload, position);
    append_u32_le(payload, max_context);
    append_u32_le(payload, static_cast<std::uint32_t>(input.size()));
    append_u32_le(payload, ffn_elements);
    append_u32_le(payload, heads);
    append_u32_le(payload, kv_heads);
    append_u32_le(payload, head_dim);
    append_u32_le(payload, epsilon_bits);
    append_u32_le(payload, theta_bits);
    append_u64_le(payload,
                  content_hash64(input.data(), input.size() * sizeof(std::uint16_t)));
    for (const std::uint16_t value : input) append_u16_le(payload, value);
    return payload;
}

} // namespace


bool RemoteOptions::offloads_layer(std::uint32_t layer_index) const {
    return std::find(layers.begin(), layers.end(), layer_index) != layers.end();
}

class RemoteClient::Impl {
public:

    explicit Impl(const RemoteOptions& options)
#ifdef _WIN32
        : socket_(connect_tcp(options.endpoint))
#endif
    {
#ifndef _WIN32
        (void)options;
        throw std::runtime_error("remote execution is currently supported on Windows only");
#else
        const std::uint64_t request_id = next_request_id_++;
        send_frame(socket_.get(), MessageType::Hello, request_id,
                   make_hello_payload(options.backend));
        Frame response = receive_frame(socket_.get());
        throw_if_error(response);
        if (response.type != MessageType::Capabilities ||
            response.request_id != request_id || response.payload.size() < 4) {
            throw std::runtime_error("unexpected remote HELLO response");
        }
        const std::uint16_t selected_backend =
            read_u16_le(response.payload.data() + 2);
        std::cerr << "Remote device connected: endpoint=" << options.endpoint
                  << " transport="
                  << (options.transport == Transport::Wifi6 ? "wifi6" : "usb")
                  << " backend=" << selected_backend << "\n";
#endif
    }

    std::uint32_t upload_tensor(io::GgufTensorInfo info,
                                std::vector<std::uint8_t> bytes) {
#ifdef _WIN32
        const std::uint64_t hash = content_hash64(bytes.data(), bytes.size());
        const std::uint64_t request_id = next_request_id_++;
        send_frame(socket_.get(), MessageType::LoadTensor, request_id,
                   make_load_tensor_payload(info, bytes, hash));
        Frame response = receive_frame(socket_.get());
        throw_if_error(response);
        if (response.type != MessageType::TensorLoaded ||
            response.request_id != request_id || response.payload.size() < 36) {
            throw std::runtime_error("unexpected LOAD_TENSOR response for " +
                                     info.name);
        }
        const std::uint32_t tensor_id = read_u32_le(response.payload.data());
        const std::uint32_t gguf_type = read_u32_le(response.payload.data() + 4);
        const std::uint64_t data_bytes = read_u64_le(response.payload.data() + 8);
        const std::uint64_t returned_hash = read_u64_le(response.payload.data() + 24);
        const std::uint16_t name_bytes = read_u16_le(response.payload.data() + 32);
        if (response.payload.size() != 36ULL + name_bytes) {
            throw std::runtime_error("invalid TENSOR_LOADED payload");
        }
        const std::string returned_name(
            reinterpret_cast<const char*>(response.payload.data() + 36),
            name_bytes);
        if (gguf_type != static_cast<std::uint32_t>(info.type) ||
            data_bytes != bytes.size() || returned_hash != hash ||
            returned_name != info.name) {
            throw std::runtime_error("remote tensor verification failed: " +
                                     info.name);
        }
        std::cerr << "Remote tensor loaded: " << info.name
                  << " bytes=" << data_bytes << " id=" << tensor_id << "\n";
        return tensor_id;
#else
        (void)info;
        (void)bytes;
        return 0;
#endif
    }

    std::vector<std::uint16_t> run_layer_range(
        std::uint32_t first_layer, std::uint32_t last_layer,
        std::uint32_t position, std::uint32_t max_context,
        const std::vector<std::uint16_t>& input,
        std::uint32_t ffn_elements, std::uint32_t heads,
        std::uint32_t kv_heads, std::uint32_t head_dim,
        float rms_norm_epsilon, float rope_theta) {
        if (first_layer > last_layer) {
            throw std::runtime_error("remote layer range is invalid");
        }
        for (std::uint32_t layer = first_layer; layer <= last_layer; ++layer) {
            if (uploaded_layers_.find(layer) == uploaded_layers_.end()) {
                throw std::runtime_error("remote layer was not uploaded: " +
                                         std::to_string(layer));
            }
        }
#ifdef _WIN32
        const std::uint64_t request_id = next_request_id_++;
        send_frame(socket_.get(), MessageType::RunLayerRange, request_id,
                   make_run_layer_range_payload(
                       first_layer, last_layer, position, max_context, input,
                       ffn_elements, heads, kv_heads, head_dim,
                       rms_norm_epsilon, rope_theta));
        Frame response = receive_frame(socket_.get());
        throw_if_error(response);
        if (response.type != MessageType::LayerRangeResult ||
            response.request_id != request_id || response.payload.size() < 32) {
            throw std::runtime_error("unexpected RUN_LAYER_RANGE response");
        }
        const std::uint32_t returned_first = read_u32_le(response.payload.data());
        const std::uint32_t returned_last = read_u32_le(response.payload.data() + 4);
        const std::uint32_t returned_position = read_u32_le(response.payload.data() + 8);
        const std::uint32_t output_elements = read_u32_le(response.payload.data() + 12);
        const std::uint64_t output_hash = read_u64_le(response.payload.data() + 16);
        if (returned_first != first_layer || returned_last != last_layer ||
            returned_position != position || output_elements != input.size() ||
            response.payload.size() != 32ULL + output_elements * 2ULL) {
            throw std::runtime_error("invalid LAYER_RANGE_RESULT metadata");
        }
        std::vector<std::uint16_t> output(output_elements);
        for (std::uint32_t i = 0; i < output_elements; ++i) {
            output[i] = read_u16_le(response.payload.data() + 32 + i * 2ULL);
        }
        if (content_hash64(output.data(), output.size() * 2ULL) != output_hash) {
            throw std::runtime_error("LAYER_RANGE_RESULT hash mismatch");
        }
        return output;
#else
        (void)position;
        (void)max_context;
        (void)input;
        (void)ffn_elements;
        (void)heads;
        (void)kv_heads;
        (void)head_dim;
        (void)rms_norm_epsilon;
        (void)rope_theta;
        return {};
#endif
    }

    std::unordered_map<std::uint32_t, bool> uploaded_layers_;

private:
#ifdef _WIN32
    WinsockRuntime winsock_;
    SocketHandle socket_;
#endif
    std::uint64_t next_request_id_ = 1;
};

RemoteClient::RemoteClient(const RemoteOptions& options)
    : impl_(std::make_unique<Impl>(options)) {}

RemoteClient::~RemoteClient() = default;

void RemoteClient::upload_layer(const io::GgufFile& gguf,
                                std::uint32_t layer_index) {
    const std::array<const char*, 11> suffixes = {
        "attn_norm.weight", "attn_q.weight", "attn_k.weight",
        "attn_v.weight", "attn_output.weight", "attn_q_norm.weight",
        "attn_k_norm.weight", "ffn_norm.weight", "ffn_gate.weight",
        "ffn_up.weight", "ffn_down.weight"};
    for (const char* suffix : suffixes) {
        const std::string name = layer_tensor_name(layer_index, suffix);
        const io::GgufTensorInfo* source = gguf.find_tensor(name);
        if (!source) {
            throw std::runtime_error("missing remote layer tensor: " + name);
        }
        io::GgufTensorInfo wire = *source;
        std::vector<std::uint8_t> bytes = gguf.read_tensor_bytes(*source);
        if (source->type == compute::GgufTensorType::F32) {
            if (bytes.size() != source->element_count() * sizeof(float)) {
                throw std::runtime_error("invalid F32 remote layer tensor: " +
                                         name);
            }
            std::vector<std::uint16_t> half(source->element_count());
            for (std::size_t i = 0; i < half.size(); ++i) {
                float value = 0.0F;
                std::memcpy(&value, bytes.data() + i * sizeof(float),
                            sizeof(value));
                half[i] = compute::float_to_half_bits(value);
            }
            bytes.resize(half.size() * sizeof(std::uint16_t));
            std::memcpy(bytes.data(), half.data(), bytes.size());
            wire.type = compute::GgufTensorType::F16;
            wire.nbytes = bytes.size();
        } else if (source->type != compute::GgufTensorType::F16 &&
                   source->type != compute::GgufTensorType::Q4_K &&
                   source->type != compute::GgufTensorType::Q6_K) {
            throw std::runtime_error("unsupported remote layer tensor type: " +
                                     name);
        }
        impl_->upload_tensor(std::move(wire), std::move(bytes));
    }
    impl_->uploaded_layers_[layer_index] = true;
}

std::vector<std::uint16_t> RemoteClient::run_layer_range(
    std::uint32_t first_layer, std::uint32_t last_layer,
    std::uint32_t position, std::uint32_t max_context,
    const std::vector<std::uint16_t>& input,
    std::uint32_t ffn_elements, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t head_dim,
    float rms_norm_epsilon, float rope_theta) {
    return impl_->run_layer_range(
        first_layer, last_layer, position, max_context, input,
        ffn_elements, heads, kv_heads, head_dim, rms_norm_epsilon,
        rope_theta);
}

} // namespace citlali::remote

