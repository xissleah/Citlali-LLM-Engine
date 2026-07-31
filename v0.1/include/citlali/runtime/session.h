#pragma once

#include "citlali/io/gguf_reader.h"
#include "citlali/model/qwen_model.h"
#include "citlali/tokenizer/tokenizer.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace citlali::runtime
{
    // 生成参数
    struct GenerationOptions {
        uint32_t max_new_tokens = 256;  // 一次性最多多少token
        uint32_t max_context = 2048;    //  最长上下文
        bool chat_template = true;      //  是否启用chat_template
    };

    // 生成停止原因
    enum class StopReason {
        Eos,            // 出现终止符
        MaxNewTokens,   // 超过最大token数限制
        ContextLimit,   // 超过上下文限制
    };

    // 返回结果
    struct GenerationResult {
        std::string text;                                   //  返回文本
        uint32_t generated_tokens = 0;                      //  返回的token数
        StopReason stop_reason = StopReason::MaxNewTokens;  //  停止原因
    };

    class InferenceSession
    {
        public:
            void load_model(const std::string& model_path, uint32_t max_context = 2048);
            std::string generate_once(const std::string& prompt, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token = {});
            std::string chat_once(const std::string& user_message, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token = {});
            GenerationResult generate_once_result(const std::string& prompt, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token = {});
            GenerationResult chat_once_result(const std::string& user_message, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token = {});
            void reset_conversation();

            const model::QwenConfig& config() const{ return model_.config(); }
            const tokenizer::Tokenizer& tokenizer() const{ return tokenizer_; }

        private:
            GenerationResult generate_from_prompt_text(const std::string& prompt_text, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token);

            io::GgufFile gguf_;
            tokenizer::Tokenizer tokenizer_;
            model::QwenModel model_;
            std::vector<std::pair<std::string, std::string>> history_;
    };
}