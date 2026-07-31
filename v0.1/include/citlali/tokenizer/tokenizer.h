#pragma once

#include "citlali/io/gguf_reader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace citlali::tokenizer
{
    class Tokenizer
    {
        public:
            void load_from_gguf(const io::GgufFile& gguf);
            std::vector<int32_t> encode(const std::string& text, bool add_bos) const;
            //输入一段自然语言，输出一串数字 id 给模型输入,add_bos=true 自动在开头拼接起始符
            std::string token_piece(int32_t token_id) const;
            //根据 id 取出原始词片段
            std::string decode_token(int32_t token_id) const;
            //处理字节级转义，还原可读字符
            std::string decode(const std::vector<int32_t>& tokens) const;
            //批量解码

            int32_t bos_id() const { return bos_id_; } // 起始符id <s>
            int32_t eos_id() const { return eos_id_; } // 结束符id </s>
            int32_t pad_id() const { return pad_id_; } // 填充符id
            int32_t vocab_size() const { return static_cast<int32_t>(vocab_.size()); } // 总词表数量
            std::string apply_chat_template(
                const std::vector<std::pair<std::string, std::string>>& messages,
                // 历史对话 <角色> <信息>
                bool add_generation_prompt
                // 如果为true，会追加助手回复前缀，方便模型续写
            ) const;
        private:
            struct MergeRule {
                std::string merged; // 合并后的子词
                int rank = 0;      // 合并优先级（BPE从小到大合并）
            };

            // 字节转BPE子词片段，处理不可见ASCII字节
            std::string byte_to_token_piece(uint8_t byte) const;
            // 将原始文本拆分为初始单字符片段（BPE第一步）
            std::vector<std::string> initial_pieces(const std::string& text) const;
            // 根据词片段查id，找不到返回unk_id未知标记
            int32_t token_to_id_or_unk(const std::string& token) const;
            // 解码字节级BPE片段，还原正常文字
            std::string decode_byte_level_piece(const std::string& token) const;

            std::vector<std::string> vocab_; // 下标=token_id，值是子词文本
            std::unordered_map<std::string, int32_t> token_to_id_; // 子词→id 反向查找
            std::unordered_map<std::string, MergeRule> merges_;    // BPE合并规则

            // 特殊标记id
            int32_t bos_id_ = 151643;   //开始字符
            int32_t eos_id_ = 151645;   //终止字符
            int32_t pad_id_ = 151643;   //填充字符
            int32_t unk_id_ = 0;        //未知字符

            bool add_bos_token_ = false; // 是否默认自动加BOS
            std::string chat_template_;   // GGUF读出的对话模板字符串
    };
}
