#include "citlali/tokenizer/tokenizer.h"

#include "citlali/common.h"

#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace citlali::tokenizer {
    namespace {
        std::vector<std::string> split_once_pair(const std::string& line) {
            const size_t pos = line.find(' ');
            // 查找第一个空格
            if (pos == std::string::npos) return {};
            // 没有就返回空
            return {line.substr(0, pos), line.substr(pos + 1)};
            // 否则返回以第一个空格为基准分割的两部分
            // 例如 <user> prompt 会被分割成 <user>和prompt
        }

        void append_utf8(std::string& out, uint32_t cp) {
            if (cp <= 0x7f) {
                out.push_back(static_cast<char>(cp));
            }
            // 1字节
            else if (cp <= 0x7ff) {
                out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
            }
            // 2字节
            else if (cp <= 0xffff) {
                out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
            }
            // 3字节
            else {
                out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
            }
            // 更大字节
        }
        // 把一个字符的 Unicode 编号转换成计算机常用的 UTF-8 字节，然后追加到字符串中
        /*
            unicode在二进制里最大是21位
            以3字节的utf-8为例子，格式规定为1110xxxx 10xxxxxx 10xxxxxx
            其中 1110是固定标记，表示这是三字节编码的第一个字节
            然后 10是固定标记，说明是后续字节
            然后 x是真正保存字符编号的数据

            首先 cp & 0x3f
            其中0x3f = 00111111
            所以按位与后取得是cp的末6位
            然后 0X80 | (cp & 0x3f)
            其中 0x80 = 10000000
            是为了给这6位的前面打上10的标记，表示是后续字节
        */

        const std::array<std::string, 256>& byte_encoder() {
            static const std::array<std::string, 256> table = [] {
                std::array<std::string, 256> t{};
                std::vector<int> bs;
                for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) bs.push_back(b);
                // 找出可以直接显示的字节，也就是会加入从ASCII中从!到~的字符,这通常是可打印的字符，范围是33~126
                for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
                for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
                // 中间跳过一些字符是因为不是所有字符都能够直接使用，这些push进去的是可以作为“可见字符”使用的字符
                std::unordered_set<int> used(bs.begin(), bs.end());
                int n = 0;
                for (int b = 0; b < 256; ++b) {
                    int cp = b;
                    if (used.find(b) == used.end()) {
                        cp = 256 + n++;
                    }
                    // 给不能直接使用的字符分配新的unicode码点
                    append_utf8(t[static_cast<size_t>(b)], static_cast<uint32_t>(cp));
                    // 给每个原始字节 b 分配一个 Unicode 字符编号 cp，然后把这个编号编码成 UTF-8，存进 t[b]
                }
                return t;
                // t是指单个字节范围内每一个数对应的utf-8的字符
            }();
            return table;
        }

        const std::unordered_map<std::string, uint8_t>& byte_decoder() {
            static const std::unordered_map<std::string, uint8_t> table = [] {
                std::unordered_map<std::string, uint8_t> t;
                const auto& encoder = byte_encoder();
                for (int b = 0; b < 256; ++b) {
                    t[encoder[static_cast<size_t>(b)]] = static_cast<uint8_t>(b);
                }
                return t;
            }();
            return table;
        }
        // 上一个函数的反向操作

        size_t utf8_char_size(unsigned char c) {
            if ((c & 0x80u) == 0) return 1;
            if ((c & 0xe0u) == 0xc0u) return 2;
            if ((c & 0xf0u) == 0xe0u) return 3;
            if ((c & 0xf8u) == 0xf0u) return 4;
            return 1;
        }
        // 判断一个utf8字符占几个字节

        bool is_special_at(const std::string& text, size_t pos, std::string& special) {
            static const std::vector<std::string> specials = {
                "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
                "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
                "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"
            };
            for (const auto& token : specials) {
                if (text.compare(pos, token.size(), token) == 0) {
                    special = token;
                    return true;
                }
            }
            return false;
        }
        // 用于判断 text 从位置 pos 开始，是否出现了某个预定义的特殊字符串

    }

    void Tokenizer::load_from_gguf(const io::GgufFile& gguf) {
        vocab_ = gguf.get_string_array("tokenizer.ggml.tokens");
        require(!vocab_.empty(), "GGUF tokenizer.ggml.tokens is missing");
        // 读取词表

        token_to_id_.clear();
        for (size_t i = 0; i < vocab_.size(); ++i) {
            token_to_id_[vocab_[i]] = static_cast<int32_t>(i);
        }
        // 建立 token->ID的映射

        bos_id_ = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.bos_token_id", bos_id_));
        // 读取BOS，即 Beginning Of Sequence开始token
        eos_id_ = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.eos_token_id", eos_id_));
        // 读取EOS，即结束token
        pad_id_ = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.padding_token_id", pad_id_));
        // 读取pad，即padding_token 补齐token
        unk_id_ = static_cast<int32_t>(gguf.get_u64("tokenizer.ggml.unknown_token_id", unk_id_));
        // 未知token
        add_bos_token_ = gguf.get_bool("tokenizer.ggml.add_bos_token", false);
        // 是否自动添加BOS，如果没有就默认false
        chat_template_ = gguf.get_string("tokenizer.chat_template", "");
        // 读取聊天模板，没有就默认空字符串

        merges_.clear();
        const auto merges = gguf.get_string_array("tokenizer.ggml.merges");
        // 获取相邻token合并规则
        int rank = 0;
        // 表示 merge 规则的优先级
        for (const std::string& merge : merges) {
            const auto pair = split_once_pair(merge);
            // 拆分左右两个token
            if (pair.size() != 2) continue;
            // 格式错误处理
            const std::string key = pair[0] + "\n" + pair[1];
            // 生成查找键
            merges_[key] = MergeRule{pair[0] + pair[1], rank++};
            // rank越小优先级越高，其实模型文件里官方已经把词表规则的优先级排好了，也就是越靠前的越优先
        }
        // BPE merge token合并
    }
    // 从 GGUF 文件中读取 tokenizer 的词表、特殊 token 配置、聊天模板和 BPE merge 规则，并保存到 Tokenizer 对象内部

    std::string Tokenizer::byte_to_token_piece(uint8_t byte) const {
        const auto& table = byte_encoder();
        // 获取字节编码表
        if (token_to_id_.find(table[byte]) != token_to_id_.end()) {
            return table[byte];
        }
        // 优先使用 byte encoder 的结果

        const std::string hex = "0123456789ABCDEF";
        std::string fallback = "<0x";
        fallback.push_back(hex[byte >> 4]);
        fallback.push_back(hex[byte & 15]);
        fallback.push_back('>');
        // 如果 table[byte] 不在词表中，就构造十六进制形式的fallback，查看是否是特殊token
        if (token_to_id_.find(fallback) != token_to_id_.end()) {
            return fallback;
        }
        return std::string(1, static_cast<char>(byte));
        // 也就是这个token查表都查不到，我认为应该写成return "<unk>"，而不是这句ai给出的兜底方案;
    }
    // 把一个原始字节 uint8_t 转换成词表中可使用的 token 字符串

    std::vector<std::string> Tokenizer::initial_pieces(const std::string& text) const {
        std::vector<std::string> pieces;
        // 结果数组
        pieces.reserve(text.size());
        // 预留空间
        for (size_t i = 0; i < text.size();) {
            // 注意这里没有写 ++i，因为下面遇到特殊 token 时，可能一次跳过多个字节
            std::string special;
            if (is_special_at(text, i, special) && token_to_id_.find(special) != token_to_id_.end()) {
                // 从当前位置 i 开始做前缀匹配，看看是不是一个特殊 token，如果检测到了，再确认这个特殊 token 确实存在于词表中
                pieces.push_back(special);
                // 例如遍历到了“<pad>”，那直接加入“<pad>”，同时跳过"<pad>".size()
                i += special.size();
                continue;
            }
            pieces.push_back(byte_to_token_piece(static_cast<uint8_t>(text[i])));
            // 普通字符
            ++i;
        }
        return pieces;
    }
    // 把输入预处理成一串初始 piece，并识别需要完整保留的特殊 token

    int32_t Tokenizer::token_to_id_or_unk(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) return it->second;
        return unk_id_;
        // 词表中不存在则返回 unk = 0
    }
    // 根据 token 字符串查找对应的 token ID

    std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
        std::vector<int32_t> ids;
        // 创建结果数组
        if (add_bos && add_bos_token_ && bos_id_ >= 0) {
            ids.push_back(bos_id_);
        }
        // 添加 BOS token
        // 这里要注意，BOS 是直接加入 ID 的，不经过普通文本分词

        std::vector<std::string> pieces = initial_pieces(text);
        // 生成初始pieces
        if (pieces.empty()) return ids;
        // 如果输入是空的就直接返回，这里的ids可能已经包含了BOS

        while (pieces.size() > 1) {
            // BPE合并循环，只要有至少两个piece就尝试合并相邻的两个piece
            int best_rank = std::numeric_limits<int>::max();
            size_t best_index = std::numeric_limits<size_t>::max();
            // 初始化
            /*
            best_rank = 最大整数
            best_index = 无效位置
             */
            std::string best_merged;
            // 查找当前所有相邻 pair 中优先级最高的合并，rank越小优先级越高

            for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                // 循环遍历相邻的pair
                if (!pieces[i].empty() && pieces[i][0] == '<') continue;
                if (!pieces[i + 1].empty() && pieces[i + 1][0] == '<') continue;
                // 如果某个piece以‘<’开头就不允许合并，这是为了防止某些特殊token被错误合并，例如<bos>
                const std::string key = pieces[i] + "\n" + pieces[i + 1];
                // 构造merge查找键
                /*
                为什么中间加换行？
                主要是为了把两个字符串明确分隔开，避免字符串边界不清晰。例如直接拼接：
                "a" + "bc" = "abc"
                "ab" + "c" = "abc"
                这两种pair都会产生同一个key
                所以加了'\n'就能区分了
                */
                auto it = merges_.find(key);
                if (it != merges_.end() && it->second.rank < best_rank) {
                    // 如果这个相邻pair存在合并表里，并且rank更小，就保存这次最佳合并
                    best_rank = it->second.rank;
                    best_index = i;
                    best_merged = it->second.merged;
                }
            }

            if (best_index == std::numeric_limits<size_t>::max()) {
                break;
            }
            // 如果没有任何pair可以合并了就停止

            pieces[best_index] = best_merged;
            // 执行最佳合并
            pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
            // 既然合并了那肯定得删除一个piece
        }

        for (const std::string& piece : pieces) {
            ids.push_back(token_to_id_or_unk(piece));
        }
        // 把最终piece转成token ID
        return ids;
    }
    // 当前方法时间复杂度约为O(n²)，主流工业化做法可以优化到O(nlogn)
    /*
    把用户输入最终转换成模型需要的 token ID 序列
    输入文本
      ↓
    可选添加 BOS
      ↓
    拆成初始 pieces
      ↓
    按照 merges_ 规则做 BPE 合并
      ↓
    每个 piece 查词表
      ↓
    查不到则使用 unk_id_
      ↓
    返回 token IDs
    */

    std::string Tokenizer::decode_byte_level_piece(const std::string& token) const {
        if (token.size() == 6 && token.rfind("<0x", 0) == 0 && token.back() == '>') {
            auto hex_value = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                return 0;
            };
            // 十六进制转换
            char b = static_cast<char>((hex_value(token[3]) << 4) | hex_value(token[4]));
            // (hex_value(token[3]) << 4)高4位，hex_value(token[4])低4位
            return std::string(1, b);
        }
        // 处理<0xXX>格式

        const auto& decoder = byte_decoder();
        // 获取反向映射表
        std::string bytes;
        bytes.reserve(token.size());
        for (size_t i = 0; i < token.size();) {
            const size_t n = utf8_char_size(static_cast<unsigned char>(token[i]));
            // 判断当前utf8字符的长度
            if (i + n > token.size()) return token;
            // 检查是否越界
            const std::string piece = token.substr(i, n);
            // 取出完整的utf8字符
            auto it = decoder.find(piece);
            // 查找该字符的原始字节
            if (it == decoder.end()) return token;
            // 查不到返回原token，上面191行那段代码是词表都找不到的时候写入原始token，这里直接返回原始token应该可以？
            bytes.push_back(static_cast<char>(it->second));
            // 写入还原后的原始字节
            i += n;
        }
        return bytes;
    }
    // 把一个 byte-level tokenizer 的 token 字符串还原成原始字节，是编码过程的逆操作
    /*
    token 字符串
      ↓
    如果是 <0xXX> 格式，解析十六进制
      ↓
    否则按照 UTF-8 字符查 byte decoder
      ↓
    恢复原始字节字符串
    */

    std::string Tokenizer::token_piece(int32_t token_id) const {
        if (token_id < 0 || token_id >= static_cast<int32_t>(vocab_.size())) return "";
        // 边界处理
        return vocab_[static_cast<size_t>(token_id)];
    }
    // 根据 token_id 获取对应的 token

    std::string Tokenizer::decode_token(int32_t token_id) const {
        const std::string token = token_piece(token_id);
        if (token.empty()) return "";
        if (token.rfind("<|", 0) == 0 && token.size() >= 3 && token.compare(token.size() - 2, 2, "|>") == 0) {
            return "";
        }
        //查找当前token是不是特殊token，是则用空字符串代替
        return decode_byte_level_piece(token);
    }
    // 把一个 token ID 解码成实际文本，同时过滤掉特殊 token

    std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
        std::string text;
        for (int32_t token : tokens) {
            text += decode_token(token);
        }
        return text;
    }
    // 把一组 token 编号解码成完整的字符串

    std::string Tokenizer::apply_chat_template(const std::vector<std::pair<std::string, std::string>>& messages, bool add_generation_prompt) const {
        (void)chat_template_;
        std::ostringstream out;
        for (const auto& message : messages) {
            out << "<|im_start|>" << message.first << "\n" << message.second << "<|im_end|>\n";
        }
        /*
        把模板硬编码成
        <|im_start|>角色
        内容<|im_end|>
        */
        if (add_generation_prompt) {
            out << "<|im_start|>assistant\n";
        }
        return out.str();
    }
    // 把多轮消息转换成模型需要的聊天模板字符串

}
