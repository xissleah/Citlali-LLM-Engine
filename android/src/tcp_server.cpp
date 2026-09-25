#include "citlali_remote/tcp_server.h"

#include "citlali_remote/protocol.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

namespace citlali::remote {

TcpServer::TcpServer(std::uint16_t port, const std::string& shader_path, const std::string& bind_address)
    : port_(port), bind_address_(bind_address), vulkan_(shader_path) {}
    // 成员变量初始化为形参
// tcp构造

TcpServer::~TcpServer() {
    if (listen_fd_ >= 0) {
        close(listen_fd_);
    }
}
// tcp析构

int TcpServer::run() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    // 创建监听socket AF_INET表示ipv4 SOCK_STREAM表示使用面向连接的字节流socket也就是tcp 0表示让系统根据前两个参数自动选择协议
    if (listen_fd_ < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << "\n";
        return 1;
    }
    // 错误处理

    int reuse_address = 1;
    if (setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse_address, sizeof(reuse_address)) != 0) {
        std::cerr << "setsockopt failed: " << std::strerror(errno) << "\n";
        return 1;
    }
    // setsockopt()用于设置socket的选项，SOL_SOCKET表示这是socket层通用选项，SO_REUSEADDR表示允许地址复用，&reuse_address选项值的地址；值为 1 表示开启，sizeof(reuse_address)表示选项值所占字节数
    /*
    它通常用于解决这种情况：
    服务器退出
        ↓
    系统暂时保留旧连接状态（TIME_WAIT 等）
        ↓
    立刻重新启动服务器
        ↓
    bind() 可能报 “Address already in use”
    */

    sockaddr_in address{};
    // 准备ipv4地址结构体
    address.sin_family = AF_INET;
    // 表示使用ipv4
    if (inet_pton(AF_INET, bind_address_.c_str(), &address.sin_addr) != 1) {
        std::cerr << "invalid bind address: " << bind_address_ << "\n";
        return 1;
    }
    // inet_pton将ip字符串转成网络地址
    address.sin_port = htons(port_);
    // 设置端口号

    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
        std::cerr << "bind failed on " << bind_address_ << ":" << port_ << ": "
                  << std::strerror(errno) << "\n";
        return 1;
    }
    // bind将socket绑定到指定ip和端口，reinterpret_cast<sockaddr*>(&address)是因为adress的类型是sockaddr_in类型，但是bind要求是sockaddr*
    if (listen(listen_fd_, 4) != 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << "\n";
        return 1;
    }
    // 开始监听，listen将普通socket转成监听socket，调用listen后才能接收客户端连接
    // 第二个参数4是backing，表示内核等待队列允许积压的大致连接数量

    std::cout << "Citlali remote server listening on " << bind_address_ << ":" << port_
              << "\n";
    std::cout.flush();
    // 输出监听成功信息，这里'\n'+flush的操作等价endl，显式写方便查bug之类的，也可以直接写成endl

    while (true) {
        sockaddr_in peer{};
        socklen_t peer_size = sizeof(peer);
        const int client_fd =
            accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer), &peer_size);
        // accept从监听socket取出一个等待中的客户端连接，并创建一个新的socket文件描述符client_id
        /*
        listen_fd_：一直用于等待新连接
        client_fd ：只用于当前这个已经连接的客户端

        listen_fd_  → 服务端门口，持续接客
        client_fd   → 某位具体客户端的通信通道
        */
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "accept failed: " << std::strerror(errno) << "\n";
            return 1;
        }
        // 处理EINTR，EINTR表示accept() 等待连接时被系统信号中断，但是不一定意味着socket坏了，所以continue

        std::cout << "client connected\n";
        std::cout.flush();
        // 连接成功

        int no_delay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &no_delay, sizeof(no_delay));
        // 为客户端设置TCP_NODELAY，这会禁用TCP的Nagle算法，不过这里忘了写错误检查了，后续补上
        serve_client(client_fd);
        // 处理客户端
        close(client_fd);
        // 关闭连接
        std::cout << "client disconnected\n";
        std::cout.flush();
        // 输出关闭连接的提示信息
    }
    // 无限循环等待客户端
    /*
    等待连接
        ↓
    处理一个客户端
        ↓
    关闭该客户端
        ↓
    继续等待下一个客户端

    当前是单线程串行服务器。后续可升级
    */
}

bool TcpServer::serve_client(int client_fd) {
    bool handshake_complete = false;
    // 握手状态一开始为false
    while (true) {
        Frame request;
        std::string error;
        if (!receive_frame(client_fd, request, error)) {
            // receive_frame() 从 TCP 流中完整读取并解析一个协议帧，结果写入request
            if (!error.empty()) {
                std::cerr << "receive error: " << error << "\n";
            }
            return error.empty();
        }
        // 接受请求帧

        Frame response;
        // 创建响应并关联请求id
        response.request_id = request.request_id;
        // 每一个响应都复制请求的request_id
        if (request.type == MessageType::Ping) {
            const std::string text = "citlali-android-pong";
            response.type = MessageType::Pong;
            response.payload.assign(text.begin(), text.end());
            std::cout << "PING request_id=" << request.request_id << "\n";
        }
        // ping请求，连通性检测
        else if (request.type == MessageType::Hello) {
            HelloPayload hello;
            if (!decode_hello_payload(request.payload, hello, error)) {
                response.type = MessageType::Error;
                response.payload.assign(error.begin(), error.end());
            }
            // 如果 payload 格式非法，例如字段不足、版本字段损坏等，返回error
            else if (hello.minimum_version > kProtocolVersion ||
                       hello.maximum_version < kProtocolVersion) {
                const std::string text = "no compatible protocol version";
                response.type = MessageType::Error;
                response.payload.assign(text.begin(), text.end());
            }
            // 检查客户端和服务端版本兼容性
            else if (hello.requested_backend != BackendType::Auto &&
                       hello.requested_backend != BackendType::VulkanGpu) {
                const std::string text = "requested backend is not implemented";
                response.type = MessageType::Error;
                response.payload.assign(text.begin(), text.end());
            }
            // 当前服务端只接收客户端的auto和vulkanGpu计算后端
            else {
                response.type = MessageType::Capabilities;
                response.payload =
                    encode_capabilities_payload(vulkan_.capabilities());
                handshake_complete = true;
                std::cout << "HELLO request_id=" << request.request_id
                          << " backend=vulkan-gpu\n";
            }
            // 握手成功
        }
        // hello握手请求
        else if (request.type == MessageType::LoadTensor) {
            if (!handshake_complete) {
                const std::string text =
                    "HELLO required before LOAD_TENSOR";
                response.type = MessageType::Error;
                response.payload.assign(text.begin(), text.end());
            }
            // 握手没成功抛出异常
            else {
                LoadTensorPayload tensor;
                if (!decode_load_tensor_payload(request.payload, tensor, error)) {
                    response.type = MessageType::Error;
                    response.payload.assign(error.begin(), error.end());
                }
                // 校验tensor数据
                else {
                    try {
                        const VulkanContext::StoredTensorInfo stored = vulkan_.store_tensor(tensor);
                        TensorLoadedPayload loaded;
                        loaded.tensor_id = stored.tensor_id;
                        loaded.gguf_type = stored.gguf_type;
                        loaded.data_bytes = stored.data_bytes;
                        loaded.allocation_bytes = stored.allocation_bytes;
                        loaded.content_hash = stored.content_hash;
                        loaded.name = stored.name;
                        response.type = MessageType::TensorLoaded;
                        response.payload =
                            encode_tensor_loaded_payload(loaded);
                        std::cout << "LOAD_TENSOR request_id="
                                  << request.request_id
                                  << " tensor_id=" << stored.tensor_id
                                  << " name=" << stored.name
                                  << " bytes=" << stored.data_bytes
                                  << " allocation="
                                  << stored.allocation_bytes << "\n";
                    }
                    catch (const std::exception& exception) {
                        const std::string text = exception.what();
                        response.type = MessageType::Error;
                        response.payload.assign(text.begin(), text.end());
                    }
                    // 异常处理
                }
                // 加载tensor
            }
        }
        // 加载tensor到vulkan
        else if (request.type == MessageType::RunLayerRange) {
            if (!handshake_complete) {
                const std::string text =
                    "HELLO required before RUN_LAYER_RANGE";
                response.type = MessageType::Error;
                response.payload.assign(text.begin(), text.end());
            }
            // 握手没成功抛异常
            else {
                RunLayerRangePayload range;
                if (!decode_run_layer_range_payload(request.payload, range, error)) {
                    response.type = MessageType::Error;
                    response.payload.assign(error.begin(), error.end());
                }
                // 解析tensor
                else {
                    try {
                        VulkanContext::MatVecResult computed =
                            vulkan_.run_layer_range(
                                range.first_layer, range.last_layer,
                                range.position, range.max_context,
                                range.input, range.ffn_elements,
                                range.heads, range.kv_heads,
                                range.head_dim, range.rms_norm_epsilon,
                                range.rope_theta);
                        // 计算
                        LayerRangeResultPayload result;
                        result.first_layer = range.first_layer;
                        result.last_layer = range.last_layer;
                        result.position = range.position;
                        result.gpu_time_ns = computed.gpu_time_ns;
                        result.output = std::move(computed.output);
                        result.output_hash = content_hash64(
                            result.output.data(),
                            result.output.size() * sizeof(std::uint16_t));
                        response.type = MessageType::LayerRangeResult;
                        response.payload =
                            encode_layer_range_result_payload(result);
                        std::cout << "RUN_LAYER_RANGE request_id="
                                  << request.request_id
                                  << " layers=" << range.first_layer << "-"
                                  << range.last_layer
                                  << " position=" << range.position
                                  << " gpu_ns=" << result.gpu_time_ns << "\n";
                        // 封装计算结果
                    }
                    // 调用vulkan计算
                    catch (const std::exception& exception) {
                        const std::string text = exception.what();
                        response.type = MessageType::Error;
                        response.payload.assign(text.begin(), text.end());
                    }
                }
            }
        }
        // 执行一段transformer
        else {
            const std::string text = handshake_complete
                                         ? "unsupported message type"
                                         : "HELLO required before this message";
            response.type = MessageType::Error;
            response.payload.assign(text.begin(), text.end());
        }
        // 未知消息类型就返回报错

        if (!send_frame(client_fd, response, error)) {
            std::cerr << "send error: " << error << "\n";
            return false;
        }
        /*
        send_frame会
        生成 24 字节协议头
            ↓
        写入 magic
            ↓
        写入协议版本
            ↓
        写入 response.type
            ↓
        写入 request_id
            ↓
        写入 payload 长度
            ↓
        循环 send 完整发送 header
            ↓
        循环 send 完整发送 payload
        */
        std::cout.flush();
    }
}
    /*
    客户端已连接
        ↓
    handshake_complete = false
        ↓
    循环接收一个 Frame
        ↓
    根据 request.type 分发处理
        ↓
    构建并发送 response
        ↓
    继续处理下一帧
        ↓
    客户端关闭 / 网络错误
        ↓
    return
     */

} // namespace citlali::remote

