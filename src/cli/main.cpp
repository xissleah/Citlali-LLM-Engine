#include "citlali/runtime/session.h"

#include "citlali/compute/common/dtype.h"
#include "citlali/io/gguf_reader.h"
#include "citlali/model/qwen_config.h"
#include "citlali/tokenizer/tokenizer.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

struct CliArgs {
    std::string model_path = "";            // 模型路径
    std::string prompt;                     // 提示词
    uint32_t max_new_tokens = 256;          // 默认每次最多产出256新toekn
    uint32_t max_context = 2048;            // 默认最长上下文2048
    bool chat = true;                       // 是否套聊天模板
    bool interactive = false;               // 交互模式
    bool info = false;                      // 打印模型消息
    bool tokens = false;                    // 打印分词结果
    bool quantized = false;                 // 是否开启量化模式
    citlali::remote::RemoteOptions remote;  // 是否开启异构计算
};

void print_usage() {
    std::cout << "Citlali-LLM-Engine CUDA\n"
              << "\n"
              << "Usage:\n"
              << "  citlali_cli --model <path.gguf> --prompt <text> [--max-new-tokens N] [--ctx N]\n"
              << "  citlali_cli --model <path.gguf> --interactive\n"
              << "  citlali_cli --model <path.gguf> --info\n"
              << "  citlali_cli --model <path.gguf> --prompt <text> --tokens [--no-chat-template]\n"
              << "\n"
              << "Options:\n"
              << "  --model <path>          GGUF model path\n"
              << "  --prompt <text>         Prompt or user message\n"
              << "  --max-new-tokens <N>    Tokens to generate, default 256\n"
              << "  --ctx <N>               Max context, default 2048\n"
              << "  --no-chat-template      Treat --prompt as raw text\n"
              << "  --interactive           Multi-turn CLI chat\n"
              << "  --info                  Print GGUF/config summary without loading weights\n"
              << "  --tokens                Print prompt tokenization without loading weights\n"
              << "  --quantized             Choose the quantized mode\n";
    std::cout << "  --remote <host:port>    Remote device endpoint (USB adb forward uses 127.0.0.1)\n"
              << "  --remote-transport <usb|wifi6>  Communication transport, default usb\n"
              << "  --remote-backend <gpu|npu|hybrid|auto>\n"
              << "  --offload <layer:N[,N]>  Whole-layer offload, for example layer:0,1\n";
}
    // 纯帮助文本

void parse_offload(const std::string& value, citlali::remote::RemoteOptions& remote) {
    const bool whole_layer = value.rfind("layer:", 0) == 0;
    // 从位置0开始寻找“layer:”
    constexpr std::size_t prefix_bytes = 6;
    // 前缀长度常量。“layer:”长度为6
    if (!whole_layer || value.size() <= prefix_bytes) {
        throw std::runtime_error("--offload accepts layer:N[,N]");
    }
    // 格式不合法
    std::stringstream stream(value.substr(prefix_bytes));
    // 去除前缀后面的部分，准备切分
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("empty FFN layer in --offload");
        }
        // 都选择分摊了层数自然不能是0咯
        const uint32_t layer = static_cast<uint32_t>(std::stoul(item));
        // 把要分摊的每一层的序号转成u32并push到remote.layer里
        remote.layers.push_back(layer);
    }
}
    // 解析offloed，也就是异构分摊多少层

std::string escaped_piece(const std::string& piece) {
    std::ostringstream out;
    for (unsigned char c : piece) {
        // 使用unsigned char是因为后面第 c < 0x20 和 c == 0x7f 的比较、以及 static_cast<int>(c) 打印都要求 0~255 无符号值才正确
        // 同时用uc遍历的话，每个字节按0~255处理，高位字节和ascii字符都能够正确判断
        switch (c) {
            case '\\n': out << "\\\\n"; break;
            case '\\r': out << "\\\\r"; break;
            case '\\t': out << "\\\\t"; break;
            case '\\"': out << "\\\\\""; break;
            case '\\\\': out << "\\\\\\\\"; break;
            // 显式转义4个常见控制字符
            default:
                if (c < 0x20 || c == 0x7f) {
                    out << "\\\\x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::nouppercase;
                }
            // 防止打印控制字符，其中0x7F是DEL
                else {
                    out << static_cast<char>(c);
                }
            // 可打印的字符转为char并加入out流
                break;
        }
    }
    return out.str();
}
    // 把一个 token 的原始字节文本，转成可安全打印、可一眼看懂控制字符的转义形式

void print_prompt_tokens(const std::string& model_path, const std::string& prompt, bool chat) {
    citlali::io::GgufFile gguf;
    gguf.load(model_path);

    citlali::tokenizer::Tokenizer tokenizer;
    tokenizer.load_from_gguf(gguf);

    std::string prompt_text = prompt;
    if (chat) {
        prompt_text = tokenizer.apply_chat_template({{"user", prompt}}, true);
    }

    const auto tokens = tokenizer.encode(prompt_text, true);
    std::cout << "prompt_text: " << prompt_text << "\n"
              << "token_count: " << tokens.size() << "\n";
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int32_t id = tokens[i];
        std::cout << i << "\t" << id << "\tpiece=\"" << escaped_piece(tokenizer.token_piece(id))
                  << "\"\tdecoded=\"" << escaped_piece(tokenizer.decode_token(id)) << "\"\n";
    }
}

const char* stop_reason_name(citlali::runtime::StopReason reason) {
    switch (reason) {
        case citlali::runtime::StopReason::Eos: return "eos";
        case citlali::runtime::StopReason::MaxNewTokens: return "max_new_tokens";
        case citlali::runtime::StopReason::ContextLimit: return "context_limit";
    }
    return "unknown";
}

void print_generation_stats(const citlali::runtime::GenerationResult& result, double seconds) {
    const double tokens_per_second = seconds > 0.0 ? static_cast<double>(result.generated_tokens) / seconds : 0.0;
    std::ostringstream out;
    out << "\n[stats] tokens=" << result.generated_tokens
        << " time=" << std::fixed << std::setprecision(3) << seconds << "s"
        << " speed=" << std::setprecision(2) << tokens_per_second << " tok/s"
        << " stop=" << stop_reason_name(result.stop_reason) << "\n";
    std::cout << out.str();
}

void print_model_info(const std::string& model_path) {
    citlali::io::GgufFile gguf;
    gguf.load(model_path);
    const auto config = citlali::model::QwenConfig::from_gguf(gguf);

    std::map<citlali::compute::GgufTensorType, size_t> type_counts;
    uint64_t raw_bytes = 0;
    for (const auto& tensor : gguf.tensors()) {
        ++type_counts[tensor.type];
        raw_bytes += tensor.nbytes;
    }

    std::cout << "model: " << model_path << "\n"
              << "architecture: " << config.architecture << "\n"
              << "vocab: " << config.vocab_size << "\n"
              << "layers: " << config.block_count << "\n"
              << "hidden: " << config.embedding_length << "\n"
              << "ffn: " << config.feed_forward_length << "\n"
              << "heads: " << config.head_count << "\n"
              << "kv_heads: " << config.head_count_kv << "\n"
              << "head_dim: " << config.head_dim << "\n"
              << "value_dim: " << config.value_dim << "\n"
              << "context: " << config.context_length << "\n"
              << "rope_theta: " << config.rope_theta << "\n"
              << "rms_norm_eps: " << config.rms_norm_eps << "\n"
              << "tensors: " << gguf.tensors().size() << "\n"
              << "raw tensor bytes: " << raw_bytes << "\n"
              << "tensor types:\n";
    for (const auto& entry : type_counts) {
        std::cout << "  " << citlali::compute::to_string(entry.first) << ": " << entry.second << "\n";
    }
}

CliArgs parse_args(const std::vector<std::string>& argv) {
    CliArgs args;
    for (size_t i = 1; i < argv.size(); ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argv.size()) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            args.model_path = need_value(arg);
        } else if (arg == "--prompt") {
            args.prompt = need_value(arg);
        } else if (arg == "--max-new-tokens") {
            args.max_new_tokens = static_cast<uint32_t>(std::stoul(need_value(arg)));
        } else if (arg == "--ctx" || arg == "--context") {
            args.max_context = static_cast<uint32_t>(std::stoul(need_value(arg)));
        } else if (arg == "--no-chat-template") {
            args.chat = false;
        } else if (arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "--info") {
            args.info = true;
        } else if (arg == "--tokens") {
            args.tokens = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else if (arg == "--quantized"){
            args.quantized = true;
        } else if (arg == "--remote") {
            args.remote.endpoint = need_value(arg);
        } else if (arg == "--remote-transport") {
            const std::string transport = need_value(arg);
            if (transport == "usb") {
                args.remote.transport = citlali::remote::Transport::Usb;
            } else if (transport == "wifi6") {
                args.remote.transport = citlali::remote::Transport::Wifi6;
            } else {
                throw std::runtime_error(
                    "--remote-transport must be usb or wifi6");
            }
        } else if (arg == "--remote-backend") {
            const std::string backend = need_value(arg);
            if (backend == "gpu") args.remote.backend = citlali::remote::Backend::Gpu;
            else if (backend == "npu") args.remote.backend = citlali::remote::Backend::Npu;
            else if (backend == "hybrid") args.remote.backend = citlali::remote::Backend::Hybrid;
            else if (backend == "auto") args.remote.backend = citlali::remote::Backend::Auto;
            else throw std::runtime_error("--remote-backend must be gpu, npu, hybrid or auto");
        } else if (arg == "--offload") {
            parse_offload(need_value(arg), args.remote);
        }else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    const bool has_offload = !args.remote.layers.empty();
    if (has_offload && args.remote.endpoint.empty() &&
        args.remote.transport == citlali::remote::Transport::Usb) {
        args.remote.endpoint = "127.0.0.1:27183";
    }
    if (has_offload &&
        args.remote.transport == citlali::remote::Transport::Wifi6 &&
        args.remote.endpoint.empty()) {
        throw std::runtime_error(
            "wifi6 transport requires an endpoint; the launcher configures a wireless ADB tunnel automatically");
    }
    if (args.remote.endpoint.empty() != !has_offload) {
        throw std::runtime_error("--remote and --offload must be used together");
    }
    return args;
}

#ifdef _WIN32
std::string wide_to_utf8(const wchar_t* text) {
    if (!text) return {};
    const int needed = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (needed <= 1) return {};
    std::string out(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), needed, nullptr, nullptr);
    out.resize(static_cast<size_t>(needed - 1));
    return out;
}

void setup_console_utf8() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
}
#endif

int run_cli(const std::vector<std::string>& argv) {
    try {
        CliArgs args = parse_args(argv);
        citlali::runtime::GenerationOptions options;
        options.max_new_tokens = args.max_new_tokens;
        options.max_context = args.max_context;
        options.chat_template = args.chat;

        if (args.info) {
            print_model_info(args.model_path);
            return 0;
        }

        if (args.tokens) {
            if (args.prompt.empty()) {
                print_usage();
                return 1;
            }
            print_prompt_tokens(args.model_path, args.prompt, args.chat);
            return 0;
        }

        citlali::runtime::InferenceSession session;
        session.load_model(args.model_path, args.max_context, args.quantized,
                           args.remote);

        if (args.interactive) {
            std::cout << "Citlali CUDA chat. Type /exit to quit, /reset to clear history.\n";
            std::string line;
            while (true) {
                std::cout << "\n<user> ";
                if (!std::getline(std::cin, line)) break;
                if (line == "/exit" || line == "/quit") break;
                if (line == "/reset") {
                    session.reset_conversation();
                    std::cout << "history cleared\n";
                    continue;
                }
                std::cout << "<assistant> ";
                const auto start = std::chrono::steady_clock::now();
                const auto result = session.chat_once_result(line, options, [](const std::string& piece) {
                    std::cout << piece << std::flush;
                });
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                print_generation_stats(result, seconds);
            }
            return 0;
        }

        if (args.prompt.empty()) {
            print_usage();
            return 1;
        }

        citlali::runtime::GenerationResult result;
        const auto start = std::chrono::steady_clock::now();
        if (args.chat) {
            result = session.chat_once_result(args.prompt, options, [](const std::string& piece) {
                std::cout << piece << std::flush;
            });
        } else {
            result = session.generate_once_result(args.prompt, options, [](const std::string& piece) {
                std::cout << piece << std::flush;
            });
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        print_generation_stats(result, seconds);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}

} // namespace

#ifndef _WIN32
int main(int argc, char** argv) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }
    return run_cli(args);
}
#else
int wmain(int argc, wchar_t** argv) {
    setup_console_utf8();
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.push_back(wide_to_utf8(argv[i]));
    }
    return run_cli(args);
}
#endif
