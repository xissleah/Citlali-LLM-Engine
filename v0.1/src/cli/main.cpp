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
    std::string model_path = "";
    std::string prompt;
    uint32_t max_new_tokens = 256;
    uint32_t max_context = 2048;
    bool chat = true;
    bool interactive = false;
    bool info = false;
    bool tokens = false;
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
              << "  --tokens                Print prompt tokenization without loading weights\n";
}

std::string escaped_piece(const std::string& piece) {
    std::ostringstream out;
    for (unsigned char c : piece) {
        switch (c) {
            case '\\n': out << "\\\\n"; break;
            case '\\r': out << "\\\\r"; break;
            case '\\t': out << "\\\\t"; break;
            case '\\"': out << "\\\\\""; break;
            case '\\\\': out << "\\\\\\\\"; break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    out << "\\\\x" << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::nouppercase;
                } else {
                    out << static_cast<char>(c);
                }
                break;
        }
    }
    return out.str();
}

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
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
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
        session.load_model(args.model_path, args.max_context);

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
