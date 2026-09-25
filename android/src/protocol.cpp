#include "citlali_remote/protocol.h"

#include <array>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/socket.h>

namespace citlali::remote {
namespace {

enum class ReceiveStatus {
    Ok,
    Closed,
    Error,
};

void write_u16_le(std::uint8_t* output, std::uint16_t value) {
    output[0] = static_cast<std::uint8_t>(value);
    output[1] = static_cast<std::uint8_t>(value >> 8U);
}
    // 将u16的v写入output，每一个下标对应一个字节

void write_u32_le(std::uint8_t* output, std::uint32_t value) {
    for (unsigned int i = 0; i < 4; ++i) {
        output[i] = static_cast<std::uint8_t>(value >> (i * 8U));
    }
}
    // 将u32的v写入output

void write_u64_le(std::uint8_t* output, std::uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        output[i] = static_cast<std::uint8_t>(value >> (i * 8U));
    }
}
    // 将u64的v写入output

std::uint16_t read_u16_le(const std::uint8_t* input) {
    return static_cast<std::uint16_t>(input[0]) |
           (static_cast<std::uint16_t>(input[1]) << 8U);
}
    // 从input里读取一个u16

std::uint32_t read_u32_le(const std::uint8_t* input) {
    std::uint32_t value = 0;
    for (unsigned int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(input[i]) << (i * 8U);
    }
    return value;
}
    // 从input里读取一个u32

std::uint64_t read_u64_le(const std::uint8_t* input) {
    std::uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(input[i]) << (i * 8U);
    }
    return value;
}
    // 从input里读取一个u64

void append_u16_le(std::vector<std::uint8_t>& output, std::uint16_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + 2);
    write_u16_le(output.data() + offset, value);
}
    // 追加数据到output后面，u16占两个字节所以偏移+2

void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + 4);
    write_u32_le(output.data() + offset, value);
}
    // 追加数据到output后面，u32占两个字节所以偏移+4

void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    const std::size_t offset = output.size();
    output.resize(offset + 8);
    write_u64_le(output.data() + offset, value);
}
    // 追加数据到output后面，u64占两个字节所以偏移+8

ReceiveStatus receive_exact(int socket_fd, void* output, std::size_t bytes, std::string& error) {
    /*
    socket_fd：套接字文件描述符（已连接的 TCP socket）
    output：void* 指针，指向接收数据的缓冲区（调用者已分配好内存）
    bytes：期望接收的字节总数
    error：std::string& 引用，用于返回错误信息（当发生错误时）
    */
    auto* cursor = static_cast<std::uint8_t*>(output);
    // 初始化，转为u8*，方便后续操作
    std::size_t received = 0;
    // 记录已成功接受的字节数
    while (received < bytes) {
        // 持续接收直到收完
        const ssize_t result =
            recv(socket_fd, cursor + received, bytes - received, 0);
        // recv接受数据
        /*
        socket_fd           套接字描述符
        cursor+received     接收缓冲区地址，cursor为起点，received为偏移
        bytes-received      剩余要接收的字节数
        0                   无特殊标志
        */
        if (result == 0) {
            return ReceiveStatus::Closed;
        }
        // recv返回值0，连接提前关闭
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            // 可能是信号中断，continue重试
            error = std::string("recv failed: ") + std::strerror(errno);
            return ReceiveStatus::Error;
            // 其余为致命错误，无法尝试通过重试解决，直接报错
        }

        received += static_cast<std::size_t>(result);
        // 累加已接受的字节数
    }
    return ReceiveStatus::Ok;
    // 正常结束
}
    // 接收指定字节数的网络数据接收函数

bool send_all(int socket_fd, const void* input, std::size_t bytes,
              std::string& error) {
    const auto* cursor = static_cast<const std::uint8_t*>(input);
    // 初始化把void*转成u8*方便后续操作
    std::size_t sent = 0;
    // 成功发送的字节数
    while (sent < bytes) {
        // 持续发送直到发完
        const ssize_t result =
            send(socket_fd, cursor + sent, bytes - sent, MSG_NOSIGNAL);
        /*
        socket_fd       套接字描述符
        cursor+sent     发送缓冲区地址+偏移
        bytes-sent      还剩多少要发送
        MSG_NOSIGNAL    关键标志，防止发送失败时产生 SIGPIPE 信号

        MSG_NOSIGNAL 的重要性
        不加 MSG_NOSIGNAL	对端关闭连接时send会触发SIGPIPE信号      默认终止进程（不可行）
        加 MSG_NOSIGNAL	    对端关闭连接时send返回-1，errno=EPIPE	可以优雅处理错误，进程继续运行
        */
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            // 中断重试
            error = std::string("send failed: ") + std::strerror(errno);
            return false;
            // 致命错误需要报错
        }
        sent += static_cast<std::size_t>(result);
        // sent累加
    }
    return true;
    // 发送成功
}

} // namespace

bool receive_frame(int socket_fd, Frame& frame, std::string& error) {
    std::array<std::uint8_t, kFrameHeaderBytes> header{};
    // 创建一个固定大小的array用于存储帧头
    const ReceiveStatus header_status =
        receive_exact(socket_fd, header.data(), header.size(), error);
    // 调用 receive_exact 精确接收 kFrameHeaderBytes 字节的帧头并保存到header_status
    if (header_status == ReceiveStatus::Closed) {
        error.clear();
        return false;
    }
    // 对端关闭连接，不算协议错误，因此清空error，并返回false
    if (header_status == ReceiveStatus::Error) {
        return false;
    }
    // 发生系统错误直接返回false

    const std::uint32_t magic = read_u32_le(header.data());
    const std::uint16_t version = read_u16_le(header.data() + 4);
    const std::uint16_t type = read_u16_le(header.data() + 6);
    const std::uint64_t request_id = read_u64_le(header.data() + 8);
    const std::uint64_t payload_bytes = read_u64_le(header.data() + 16);
    /*
    从 header 缓冲区中按小端序读取各个字段：
    偏移 0：magic（4字节）— 协议魔数，用于验证协议标识
    偏移 4：version（2字节）— 协议版本号
    偏移 6：type（2字节）— 消息类型
    偏移 8：request_id（8字节）— 请求ID（用于请求-响应匹配）
    偏移 16：payload_bytes（8字节）— 负载长度
    */

    if (magic != kProtocolMagic) {
        error = "invalid protocol magic";
        return false;
    }
    // 魔数不匹配
    if (version != kProtocolVersion) {
        error = "unsupported protocol version: " + std::to_string(version);
        return false;
    }
    // 协议版本不匹配
    if (payload_bytes > kMaxPayloadBytes ||
        payload_bytes > std::numeric_limits<std::size_t>::max()) {
        error = "payload exceeds protocol limit";
        return false;
    }
    // 双重检查：是否超过协议规定最大值、是否超过当前平台的size_t

    frame.type = static_cast<MessageType>(type);
    frame.request_id = request_id;
    // 填充帧元数据
    frame.payload.resize(static_cast<std::size_t>(payload_bytes));
    // 接受负载数据
    if (!frame.payload.empty()) {
        const ReceiveStatus payload_status = receive_exact(
            socket_fd, frame.payload.data(), frame.payload.size(), error);
        // 负载数据写入frame.payload.data()
        if (payload_status != ReceiveStatus::Ok) {
            if (payload_status == ReceiveStatus::Closed) {
                error = "connection closed during payload";
            }
            return false;
        }
        // 异常处理
    }
    // 负载大小大于零才需要接收数据
    return true;
    // 成功接收完整帧
}
    // 协议帧接收函数

bool send_frame(int socket_fd, const Frame& frame, std::string& error) {
    if (frame.payload.size() > kMaxPayloadBytes) {
        error = "payload exceeds protocol limit";
        return false;
    }
    // 负载超限检查

    std::array<std::uint8_t, kFrameHeaderBytes> header{};
    // 构造帧头
    write_u32_le(header.data(), kProtocolMagic);
    // 写入魔数
    write_u16_le(header.data() + 4, kProtocolVersion);
    // 写入协议版本
    write_u16_le(header.data() + 6,
                 static_cast<std::uint16_t>(frame.type));
    // 写入消息类型
    write_u64_le(header.data() + 8, frame.request_id);
    // 写入请求id
    write_u64_le(header.data() + 16,
                 static_cast<std::uint64_t>(frame.payload.size()));
    // 写入负载数据大小，这里size_t是可以安全转成u64_t的，意味着32位系统可能出错

    if (!send_all(socket_fd, header.data(), header.size(), error)) {
        return false;
    }
    // 发送帧头并返回发送状态
    return frame.payload.empty() ||
           send_all(socket_fd, frame.payload.data(), frame.payload.size(), error);
    // 如果是空负载或者发送成功就返回成功态
}
    // 发送一个完整的协议帧

bool decode_hello_payload(const std::vector<std::uint8_t>& bytes,
                          HelloPayload& hello, std::string& error) {
    constexpr std::size_t kHelloPayloadBytes = 8;
    // 表示hello负载大小固定为8
    if (bytes.size() != kHelloPayloadBytes) {
        error = "HELLO payload must be 8 bytes";
        return false;
    }
    // 依旧先验证大小

    hello.minimum_version = read_u16_le(bytes.data());
    // 读取客户端允许的最小版本
    hello.maximum_version = read_u16_le(bytes.data() + 2);
    // 读取客户端允许的最大版本
    hello.requested_backend =
        static_cast<BackendType>(read_u16_le(bytes.data() + 4));
    // 表示客户端请求的后端服务类型
    const std::uint16_t reserved = read_u16_le(bytes.data() + 6);
    // 保留字段，便于扩展
    if (reserved != 0) {
        error = "HELLO reserved field must be zero";
        return false;
    }
    // 既然是保留自动自然必须是0了
    return true;
}
    // 解码hello消息

std::vector<std::uint8_t> encode_capabilities_payload(
    const CapabilitiesPayload& capabilities) {
    constexpr std::size_t kFixedBytes = 50;
    std::vector<std::uint8_t> output;
    output.reserve(kFixedBytes + capabilities.device_name.size());
    // 输出缓冲区，大小50

    append_u16_le(output, capabilities.selected_version);
    // 写入选中的版本协议
    append_u16_le(output, static_cast<std::uint16_t>(capabilities.backend));
    // 写入后端类型
    append_u32_le(output, capabilities.vulkan_api_version);
    // 写入vulkan api版本
    append_u32_le(output, capabilities.driver_version);
    // 写入驱动版本
    append_u32_le(output, capabilities.vendor_id);
    // 写入厂商id
    append_u32_le(output, capabilities.device_id);
    // 写入设备id
    append_u64_le(output, capabilities.max_storage_buffer_bytes);
    // 写入最大缓冲区的大小
    append_u32_le(output, capabilities.max_compute_workgroup_invocations);
    // 写入计算工作组最大调用数
    append_u32_le(output, capabilities.subgroup_size);
    // 写入子组大小
    append_u32_le(output, capabilities.compute_queue_family);
    // 写入计算队列族索引
    append_u32_le(output, capabilities.compute_queue_count);
    // 写入计算队列数量
    output.push_back(capabilities.shader_float16 ? 1U : 0U);
    // 写入是否支持fp16着色器
    output.push_back(capabilities.storage_buffer_16bit ? 1U : 0U);
    // 写入是否支持16位存储缓冲区
    output.push_back(capabilities.uniform_and_storage_buffer_16bit ? 1U : 0U);
    // 写入是否支持统一和存储缓冲区的16位访问
    output.push_back(0U);
    // 填充字节

    const std::size_t bounded_name_size =
        std::min<std::size_t>(capabilities.device_name.size(), 65535U);
    // 设备名称长度，最多不超过2字节
    append_u16_le(output, static_cast<std::uint16_t>(bounded_name_size));
    // 写入设备名称长度
    output.insert(output.end(), capabilities.device_name.begin(),
                  capabilities.device_name.begin() + bounded_name_size);
    // 追加设备名称
    return output;
}
    // 编码能力信息负载

std::uint64_t content_hash64(const void* data, std::size_t bytes) {
    constexpr std::uint64_t kOffsetBasis = 14695981039346656037ULL;
    // 偏移基数，FNV-1a 算法的初始哈希值
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    // FNV素数，FNV 算法的乘法因子
    const auto* input = static_cast<const std::uint8_t*>(data);
    std::uint64_t hash = kOffsetBasis;
    // 初始化哈希值
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= input[i];
        hash *= kPrime;
    }
    // 哈希操作
    return hash;
}
    // 计算内容的64位哈希值

bool decode_load_tensor_payload(const std::vector<std::uint8_t>& bytes,
                                LoadTensorPayload& tensor,
                                std::string& error) {
    /*
    bytes：const std::vector<uint8_t>& 引用，包含待解析的负载数据（只读）
    tensor：LoadTensorPayload& 引用，输出参数，存储解析后的张量数据
    error：std::string& 引用，用于返回错误信息
    */
    constexpr std::size_t kFixedBytes = 36;
    // 固定头部大小
    constexpr std::uint16_t kMaxRank = 8;
    // 张量最大维度（秩）限制
    constexpr std::uint16_t kMaxNameBytes = 1024;
    // 张量名称最大长度
    if (bytes.size() < kFixedBytes) {
        error = "LOAD_TENSOR payload is too short";
        return false;
    }
    // 负载至少36字节

    tensor.gguf_type = read_u32_le(bytes.data());
    // 解析张量数据类型
    const std::uint16_t rank = read_u16_le(bytes.data() + 4);
    // 解析张量的秩（维度数）
    const std::uint16_t reserved0 = read_u16_le(bytes.data() + 6);
    // 保留字段
    tensor.element_count = read_u64_le(bytes.data() + 8);
    // 解析元素总数
    const std::uint64_t data_bytes = read_u64_le(bytes.data() + 16);
    // 解析数据字节数
    tensor.content_hash = read_u64_le(bytes.data() + 24);
    // 解析内容哈希值
    const std::uint16_t name_bytes = read_u16_le(bytes.data() + 32);
    // 解析名称长度
    const std::uint16_t reserved1 = read_u16_le(bytes.data() + 34);
    // 保留字段

    if (reserved0 != 0 || reserved1 != 0) {
        error = "LOAD_TENSOR reserved fields must be zero";
        return false;
    }
    // 验证头部字段合法性
    if (rank == 0 || rank > kMaxRank) {
        error = "LOAD_TENSOR rank is invalid";
        return false;
    }
    // 秩为0或者超限就报错
    if (name_bytes == 0 || name_bytes > kMaxNameBytes) {
        error = "LOAD_TENSOR name length is invalid";
        return false;
    }
    // 张量名称太长报错
    if (data_bytes == 0 || data_bytes > kMaxPayloadBytes) {
        error = "LOAD_TENSOR data length is invalid";
        return false;
    }
    // 数据为空或者超限报错

    const std::uint64_t metadata_bytes =
        kFixedBytes + static_cast<std::uint64_t>(rank) * 8ULL + name_bytes;
    // 元数据长度
    if (metadata_bytes > bytes.size() ||
        data_bytes != bytes.size() - metadata_bytes) {
        error = "LOAD_TENSOR payload length does not match metadata";
        return false;
    }
    // 元数据不能超过负载大小，验证数据大小

    tensor.dims.resize(rank);
    // 预留张量空间
    std::uint64_t calculated_elements = 1;
    // 初始化计算出的元素总数为1
    std::size_t cursor = kFixedBytes;
    // 跳过固定头部
    for (std::uint16_t i = 0; i < rank; ++i) {
        const std::uint64_t dimension = read_u64_le(bytes.data() + cursor);
        // 读取维度数据
        cursor += 8;
        // 读一个u64得前进8字节
        if (dimension == 0 ||
            calculated_elements >
                std::numeric_limits<std::uint64_t>::max() / dimension) {
            error = "LOAD_TENSOR dimensions are invalid";
            return false;
        }
        // 空维度或者超限报错
        tensor.dims[i] = dimension;
        // 存储维度
        calculated_elements *= dimension;
        // 累加维度总数
    }
    // 循环读取每一个维度
    if (calculated_elements != tensor.element_count) {
        error = "LOAD_TENSOR element count does not match dimensions";
        return false;
    }

    tensor.name.assign(reinterpret_cast<const char*>(bytes.data() + cursor),
                       name_bytes);
    // 解析张量名称
    cursor += name_bytes;
    // 跳过名称长度，指向数据部分
    tensor.data.assign(bytes.begin() + static_cast<std::ptrdiff_t>(cursor),
                       bytes.end());
    // 解析张量数据

    if (content_hash64(tensor.data.data(), tensor.data.size()) !=
        tensor.content_hash) {
        error = "LOAD_TENSOR content hash mismatch";
        return false;
    }
    // 张量数据进行哈希校验
    return true;
}
    // 解码加载张量的负载

std::vector<std::uint8_t> encode_tensor_loaded_payload( const TensorLoadedPayload& tensor)
{
    constexpr std::size_t kFixedBytes = 36;
    // 固定头部36字节
    const std::size_t name_bytes = std::min<std::size_t>(tensor.name.size(), 65535U);
    // 张量名称长度

    std::vector<std::uint8_t> output;
    output.reserve(kFixedBytes + name_bytes);
    append_u32_le(output, tensor.tensor_id);
    append_u32_le(output, tensor.gguf_type);
    append_u64_le(output, tensor.data_bytes);
    append_u64_le(output, tensor.allocation_bytes);
    append_u64_le(output, tensor.content_hash);
    append_u16_le(output, static_cast<std::uint16_t>(name_bytes));
    append_u16_le(output, 0);
    output.insert(output.end(), tensor.name.begin(), tensor.name.begin() + name_bytes);
    // 写入对应负载数据，具体内容上面函数有说明
    return output;
}
    // 编码“张量已加载”的负载数据

bool decode_run_layer_range_payload(const std::vector<std::uint8_t>& bytes,
                                    RunLayerRangePayload& request,
                                    std::string& error) {
    constexpr std::size_t kFixedBytes = 52;
    if (bytes.size() < kFixedBytes) {
        error = "RUN_LAYER_RANGE payload is too short";
        return false;
    }
    // 依旧长度检查
    request.first_layer = read_u32_le(bytes.data());
    // 读取起始层索引
    request.last_layer = read_u32_le(bytes.data() + 4);
    // 读取结束层索引
    request.position = read_u32_le(bytes.data() + 8);
    // 读取当前token在序列中的位置
    request.max_context = read_u32_le(bytes.data() + 12);
    // 读取最大上下文长度
    request.hidden_elements = read_u32_le(bytes.data() + 16);
    // 读取隐藏层的元素数
    request.ffn_elements = read_u32_le(bytes.data() + 20);
    // 读取ffn的元素数
    request.heads = read_u32_le(bytes.data() + 24);
    // 读取注意力头数
    request.kv_heads = read_u32_le(bytes.data() + 28);
    // 读取kv头数
    request.head_dim = read_u32_le(bytes.data() + 32);
    // 读取头的维度数
    const std::uint32_t epsilon_bits = read_u32_le(bytes.data() + 36);
    // 读取RMS Norm Epsilon
    const std::uint32_t theta_bits = read_u32_le(bytes.data() + 40);
    // 读取RoPE Theta
    std::memcpy(&request.rms_norm_epsilon, &epsilon_bits, sizeof(float));
    std::memcpy(&request.rope_theta, &theta_bits, sizeof(float));
    // u32t转float
    /*
    为什么使用 memcpy 而不是 reinterpret_cast？
    reinterpret_cast 在严格别名规则下可能触发未定义行为
    memcpy 是安全的，编译器会优化掉实际拷贝操作
    */
    request.input_hash = read_u64_le(bytes.data() + 44);
    // 读取数据哈希值
    if (request.first_layer > request.last_layer ||
        request.max_context == 0 || request.position >= request.max_context ||
        request.hidden_elements == 0 || request.ffn_elements == 0 ||
        request.heads == 0 || request.kv_heads == 0 ||
        request.head_dim == 0 || request.heads % request.kv_heads != 0 ||
        request.rms_norm_epsilon <= 0.0F || request.rope_theta <= 0.0F) {
        error = "RUN_LAYER_RANGE parameters are invalid";
        return false;
    }
    // 参数合法性检查
    const std::uint64_t expected_bytes =
        kFixedBytes + static_cast<std::uint64_t>(request.hidden_elements) * 2ULL;
    if (expected_bytes != bytes.size()) {
        error = "RUN_LAYER_RANGE input byte count does not match hidden size";
        return false;
    }
    // 验证负载大小一致性
    request.input.resize(request.hidden_elements);
    // 准备存储输入数据
    for (std::uint32_t i = 0; i < request.hidden_elements; ++i) {
        request.input[i] = read_u16_le(bytes.data() + kFixedBytes + i * 2ULL);
    }
    // 存储输入数据，也就是单个token的向量
    // 单请求单token是为了流式处理，避免用户认为是卡死
    if (content_hash64(request.input.data(), request.input.size() * 2ULL) !=
        request.input_hash) {
        error = "RUN_LAYER_RANGE input hash mismatch";
        return false;
    }
    // 哈希校验
    return true;
}
    // 解码运行层范围的负载

std::vector<std::uint8_t> encode_layer_range_result_payload(
    const LayerRangeResultPayload& result) {
    constexpr std::size_t kFixedBytes = 32;
    // 头部大小固定32字节
    std::vector<std::uint8_t> output;
    output.reserve(kFixedBytes + result.output.size() * 2ULL);
    append_u32_le(output, result.first_layer);
    append_u32_le(output, result.last_layer);
    append_u32_le(output, result.position);
    append_u32_le(output, static_cast<std::uint32_t>(result.output.size()));
    append_u64_le(output, result.output_hash);
    append_u64_le(output, result.gpu_time_ns);
    // 写入头部信息
    for (const std::uint16_t value : result.output) {
        append_u16_le(output, value);
    }
    // 写入输出向量
    return output;
}
    // 编码层范围计算结果的负载

} // namespace citlali::remote

