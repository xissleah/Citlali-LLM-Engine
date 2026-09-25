#include "citlali_remote/tcp_server.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    try {
        unsigned long parsed_port = 27183;
        // 端口号
        std::string shader_path = "/data/local/tmp/q4k_matvec.spv";
        // shader文件位置
        std::string bind_address = "127.0.0.1";
        // 绑定本机
        if (argc > 1) {
            parsed_port = std::stoul(argv[1]);
            // 读取端口号，并将字符串转为ull
        }
        if (argc > 2) {
            shader_path = argv[2];
            // 读取shader路径
        }
        if (argc > 3) {
            bind_address = argv[3];
            // 读取监听地址
        }
        if (parsed_port == 0 ||
            parsed_port > std::numeric_limits<std::uint16_t>::max()) {
            std::cerr << "invalid TCP port\n";
            // 校验端口范围 1~65535（uint16_t）
            return 1;
        }

        citlali::remote::TcpServer server(static_cast<std::uint16_t>(parsed_port), shader_path, bind_address);
        // 创建服务对象
        return server.run();
        // 运行
    }
    catch (const std::exception& error) {
        std::cerr << "server failed: " << error.what() << "\n";
        // 错误处理
        return 1;
    }
    /*
    创建 server
        ↓
    执行 server.run()
        ↓
    run() 返回
        ↓
    调用 server.~TcpServer()
        ↓
    main 返回
    */
}
