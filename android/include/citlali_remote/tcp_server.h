#pragma once

#include "citlali_remote/vulkan_context.h"

#include <cstdint>
#include <string>

namespace citlali::remote {

class TcpServer {
public:
    TcpServer(std::uint16_t port, const std::string& shader_path,
              const std::string& bind_address = "127.0.0.1");
    //  创建 TCP 服务器对象
    /*
    port 端口
    shader_path shader文件、CUDA/PTX文件或相关计算资源的路径
    bind_address 绑定的本地 IP 地址，默认只监听 127.0.0.1
    */
    ~TcpServer();
    //  析构函数，在TcpServer对象销毁时调用

    TcpServer(const TcpServer&) = delete;
    //  禁止拷贝构造，即不能复制服务器对象
    /*
    TCP server 往往拥有唯一资源，例如：

    监听 socket
    线程
    GPU context
    模型实例
    连接状态
    文件描述符

    这些资源通常不能安全地直接复制

    如果允许默认复制，可能造成：
    两个对象持有同一个 socket
    两个析构函数重复 close()
    两个对象竞争同一份连接状态
    */
    TcpServer& operator=(const TcpServer&) = delete;
    //  禁止拷贝赋值

    int run();
    //  启动事件

private:
    bool serve_client(int client_fd);
    //  用于处理一个已连接的客户端

    std::uint16_t port_;
    //  服务器监听端口
    std::string bind_address_;
    //  服务器绑定ip地址
    int listen_fd_ = -1;
    //  保存监听 socket 的文件描述符
    VulkanContext vulkan_;
    //  保存 Vulkan 后端上下文
};

} // namespace citlali::remote
