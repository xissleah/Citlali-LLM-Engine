#include "citlali/runtime/session.h"

#include "citlali/common.h"

#include <algorithm>
#include <iostream>
/*
v0.1改进：
forward_token 新增bool参数 compute_logits，从原先每个token都要计算logits转变为仅最后一个prompt token计算logits
*/
namespace citlali::runtime {
    void InferenceSession::load_model(const std::string& model_path, uint32_t max_context){
        std::cerr << "Reading GGUF: " << model_path << std::endl;
        gguf_.load(model_path);
        tokenizer_.load_from_gguf(gguf_);
        model_.load(gguf_, max_context);
        history_.clear();
        // 初始化
        std::cerr << "Model loaded. vocab=" << config().vocab_size
              << " layers=" << config().block_count
              << " hidden=" << config().embedding_length
              << " heads=" << config().head_count
              << " kv_heads=" << config().head_count_kv
              << " head_dim=" << config().head_dim
              << " ctx=" << model_.max_context() << "\n";
        // 打印信息
    }
    // 加载一个 GGUF 模型，并初始化推理会话

    std::string InferenceSession::generate_once(const std::string& prompt, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token) {
        return generate_once_result(prompt, options, on_token).text;
    }
    // 37行的函数的包装函数

    std::string InferenceSession::chat_once(const std::string& user_message, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token) {
        return chat_once_result(user_message, options, on_token).text;
    }
    // 根据用户消息，自动套聊天模板后生成

    GenerationResult InferenceSession::generate_once_result(const std::string& prompt, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token) {
        return generate_from_prompt_text(prompt, options, on_token);
    }
    // 根据已有prompt生成

    GenerationResult InferenceSession::chat_once_result(const std::string& user_message, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token) {
        history_.push_back({"user", user_message});
        const std::string prompt = tokenizer_.apply_chat_template(history_, true);
        // 生成聊天模板
        GenerationResult result = generate_from_prompt_text(prompt, options, on_token);
        // 执行推理
        history_.push_back({"assistant", result.text});
        // 压入历史记录
        return result;
    }
    // 用户的输入压入历史 -> 给用户的输入套上聊天模板 -> 进行推理 -> 模型的回复压入历史 -> 返回模型的回复

    void InferenceSession::reset_conversation() {
        history_.clear();
    }
    // 清空历史记录

    GenerationResult InferenceSession::generate_from_prompt_text(const std::string& prompt_text, const GenerationOptions& options, const std::function<void(const std::string&)>& on_token) {
        std::vector<int32_t> tokens = tokenizer_.encode(prompt_text, true);
        // 用户的输入经过分词器编码
        require(!tokens.empty(), "prompt produced no tokens");
        // 检查输入是否是空的
        require(tokens.size() + options.max_new_tokens < model_.max_context(), "prompt plus generation exceeds max_context");
        // 检查最大上下文

        uint32_t position = 0; // 位置
        int32_t next = 0;      // 模型根据当前已输入内容预测出来的下一个token ID
        for (std::size_t i = 0; i < tokens.size(); i++)
        {
            next = model_.forward_token(tokens[i],position++,i == tokens.size()-1);
        }
        // prefill 调度

        GenerationResult result;
        for (uint32_t i = 0; i < options.max_new_tokens; ++i) {
            if (next == tokenizer_.eos_id()) {
                result.stop_reason = StopReason::Eos;
                break;
            }
            // 如果遇到EOS了，停止原因设置为EOS
            const std::string piece = tokenizer_.decode_token(next);
            // 解码token
            result.text += piece;
            // 拼接到结果里
            ++result.generated_tokens;
            // 统计数量
            if (on_token) on_token(piece);
            // 如果调用者提供回调就把当前片段传出去
            next = model_.forward_token(next, position++);
            // 把当前token送回模型，预测下一个token
            if (position >= model_.max_context()) {
                result.stop_reason = StopReason::ContextLimit;
                break;
            }
            // 上下文超限处理
        }
        // 自回归生成循环，每次生成一个 token
        model_.synchronize();
        // GPU 同步等待
        return result;
    }
    /*
    prompt
    → encode
    → 逐个 forward prompt token
    → 得到第一个待生成 token
    → 循环生成
    → decode
    → 回调输出
    */
}
