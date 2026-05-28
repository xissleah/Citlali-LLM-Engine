// crates/citlali-model/src/tokenizer.rs
//! BPE Tokenizer — 从 GGUF metadata 加载 vocab 和 merge 规则
//! 支持两种 vocab 编码格式：
//! - GPT-2 byte-level: token 字符串使用 GPT-2 的 byte↔unicode 映射
//! - Raw UTF-8: token 字符串直接存储为 UTF-8（Qwen3 等新模型）
//! 自动检测 chat template 和 stop tokens

use citlali_gguf::metadata::MetadataValue;
use citlali_gguf::GgufFile;
use fancy_regex::Regex;
use std::collections::HashMap;

/// Vocab 编码格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VocabEncoding {
    /// GPT-2 byte-level BPE: token 使用 byte↔unicode 映射
    Gpt2,
    /// Raw UTF-8: token 直接存储为 UTF-8 字符串
    RawUtf8,
}

/// Chat 模板风格
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatStyle {
    /// Qwen3 / ChatML: <|im_start|>role\ncontent<|im_end|>
    ChatML,
    /// Llama 3: <|begin_of_text|><|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    Llama3,
}

/// Chat 格式化选项
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    /// 禁用 Qwen3 thinking mode（预填充空 think 块）
    pub no_think: bool,
}

/// BPE Tokenizer
pub struct Tokenizer {
    /// token string → token id
    token_to_id: HashMap<String, u32>,
    /// token id → token string (原始存储格式)
    id_to_token: Vec<String>,
    /// BPE merge 规则，按优先级排列 (pair → rank)
    merges: HashMap<(String, String), u32>,
    /// vocab 编码格式
    encoding: VocabEncoding,
    /// GPT-2 unicode→byte 映射（仅 Gpt2 模式使用）
    unicode_to_byte: HashMap<char, u8>,
    /// GPT-2 byte→unicode 映射（仅 Gpt2 模式使用）
    byte_to_unicode: HashMap<u8, char>,
    /// 特殊 token 列表（按长度降序排列，用于 encode 时优先匹配）
    special_tokens: Vec<String>,
    /// Chat 模板风格
    pub chat_style: ChatStyle,
    /// Stop token IDs（遇到即停止生成）
    pub stop_token_ids: Vec<u32>,
    /// BOS token ID（某些模型 prompt 开头需要）
    pub bos_token_id: Option<u32>,
    /// Pre-tokenization regex（GPT-2/Llama 风格 BPE 需要）
    pre_tokenize_regex: Option<Regex>,
}

/// 构建 GPT-2 byte↔unicode 映射表
fn build_byte_to_unicode() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut n: u32 = 0;
    for b in 0u16..=255 {
        let b = b as u8;
        let c = match b {
            33..=126 | 161..=172 | 174..=255 => b as u32,
            _ => {
                let c = 256 + n;
                n += 1;
                c
            }
        };
        map.insert(b, char::from_u32(c).unwrap());
    }
    map
}

fn build_unicode_to_byte() -> HashMap<char, u8> {
    build_byte_to_unicode()
        .into_iter()
        .map(|(b, c)| (c, b))
        .collect()
}

/// 检测 vocab 是否使用 GPT-2 byte-level 编码
/// 通过检查第一批 token 是否包含 GPT-2 特征字符（U+0100+）
fn detect_encoding(tokens: &[String]) -> VocabEncoding {
    // GPT-2 映射会把字节 0x00..0x20 等映射到 U+0100..U+0120
    // 如果 vocab 中存在这些字符，说明是 GPT-2 编码
    let gpt2_marker_range = '\u{0100}'..='\u{0120}';
    let has_gpt2_markers = tokens.iter().take(512).any(|t| {
        t.chars().any(|c| gpt2_marker_range.contains(&c))
    });
    if has_gpt2_markers {
        VocabEncoding::Gpt2
    } else {
        VocabEncoding::RawUtf8
    }
}

impl Tokenizer {
    /// 从 GGUF metadata 加载 tokenizer
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        // 加载 vocab tokens
        let tokens: Vec<String> = match gguf.get_metadata("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
            _ => panic!("缺少 tokenizer.ggml.tokens metadata"),
        };

        // 检测编码格式
        let encoding = detect_encoding(&tokens);

        // 构建 token_to_id 映射
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        // 加载 merges
        let mut merges = HashMap::new();
        if let Some(MetadataValue::Array(arr)) = gguf.get_metadata("tokenizer.ggml.merges") {
            for (rank, v) in arr.iter().enumerate() {
                if let MetadataValue::String(s) = v {
                    if let Some((left, right)) = s.split_once(' ') {
                        merges.insert(
                            (left.to_string(), right.to_string()),
                            rank as u32,
                        );
                    }
                }
            }
        }

        let byte_to_unicode = build_byte_to_unicode();
        let unicode_to_byte = build_unicode_to_byte();

        // 收集特殊 token：匹配 <|...|> 模式的 token
        let mut special_tokens: Vec<String> = tokens
            .iter()
            .filter(|t| t.starts_with("<|") && t.ends_with("|>"))
            .cloned()
            .collect();
        // 按长度降序排列，确保优先匹配最长的特殊 token
        special_tokens.sort_by(|a, b| b.len().cmp(&a.len()));

        // 检测 Chat 模板风格和 Stop Tokens
        let (chat_style, stop_token_ids, bos_token_id) =
            detect_chat_style(&token_to_id, gguf);

        // 检测 pre-tokenization regex
        // 所有编码类型都需要检查 tokenizer.ggml.pre metadata
        // Llama、Qwen 等模型都依赖 pre-tokenize regex 来正确分词
        let pre_tokenize_regex = {
            let regex_str = gguf
                .get_metadata("tokenizer.ggml.pre")
                .and_then(|v| match v {
                    MetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                });
            let pattern = match regex_str.as_deref() {
                Some("llama-bpe") => {
                    Some(concat!(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                        r"|\p{N}{1,3}",
                        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                        r"|\s*[\r\n]+",
                        r"|\s+(?!\S)",
                        r"|\s+"
                    ))
                }
                Some("gpt2") => {
                    Some(concat!(
                        r"'s|'t|'re|'ve|'m|'ll|'d",
                        r"| ?\p{L}+",
                        r"| ?\p{N}+",
                        r"| ?[^\s\p{L}\p{N}]+",
                        r"|\s+(?!\S)",
                        r"|\s+"
                    ))
                }
                Some("qwen2") => {
                    Some(concat!(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                        r"|\p{N}{1,3}",
                        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                        r"|\s*[\r\n]+",
                        r"|\s+(?!\S)",
                        r"|\s+"
                    ))
                }
                Some("deepseek-llm") => {
                    Some(concat!(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                        r"|\p{N}{1,3}",
                        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                        r"|\s*[\r\n]+",
                        r"|\s+(?!\S)",
                        r"|\s+"
                    ))
                }
                Some("deepseek-coder") => {
                    Some(concat!(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                        r"|\p{N}{1,3}",
                        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                        r"|\s*[\r\n]+",
                        r"|\s+(?!\S)",
                        r"|\s+"
                    ))
                }
                Some(other) => {
                    // 未知的 pre-tokenizer 类型，使用 llama-bpe 兼容模式
                    eprintln!("[warn] 未知 pre-tokenizer 类型: {:?}, 使用默认 regex", other);
                    Some(concat!(
                        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                        r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                        r"|\p{N}{1,3}",
                        r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                        r"|\s*[\r\n]+",
                        r"|\s+(?!\S)",
                        r"|\s+"
                    ))
                }
                None => {
                    // 没有 pre metadata 时，GPT-2 编码默认用 llama-bpe regex
                    if encoding == VocabEncoding::Gpt2 {
                        Some(concat!(
                            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
                            r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                            r"|\p{N}{1,3}",
                            r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                            r"|\s*[\r\n]+",
                            r"|\s+(?!\S)",
                            r"|\s+"
                        ))
                    } else {
                        None
                    }
                }
            };
            match pattern {
                Some(pat) => match Regex::new(pat) {
                    Ok(re) => Some(re),
                    Err(e) => {
                        eprintln!("[warn] pre-tokenize regex 编译失败: {}", e);
                        None
                    }
                },
                None => None,
            }
        };

        println!(
            "检测到 chat_style: {:?}, stop_tokens: {:?}, pre_tokenize: {}",
            chat_style, stop_token_ids, pre_tokenize_regex.is_some()
        );

        Self {
            token_to_id,
            id_to_token: tokens,
            merges,
            encoding,
            unicode_to_byte,
            byte_to_unicode,
            special_tokens,
            chat_style,
            stop_token_ids,
            bos_token_id,
            pre_tokenize_regex,
        }
    }

    /// 将文本编码为 token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // 先按特殊 token 分割文本，然后对普通文本段做 BPE
        let segments = self.split_on_special_tokens(text);
        let mut result = Vec::new();
        for seg in segments {
            if let Some(&id) = self.token_to_id.get(&seg) {
                // 特殊 token 或刚好在 vocab 中的整个字符串
                if self.special_tokens.contains(&seg) {
                    result.push(id);
                    continue;
                }
            }
            // 普通文本段：先 pre-tokenize，再对每个片段做 BPE
            if let Some(ref re) = self.pre_tokenize_regex {
                // 用 regex 预分词，每个匹配片段独立做 BPE
                let mut search_start = 0;
                while search_start < seg.len() {
                    match re.find_from_pos(&seg, search_start) {
                        Ok(Some(m)) => {
                            result.extend(self.bpe_encode(m.as_str()));
                            search_start = m.end();
                        }
                        _ => break,
                    }
                }
            } else {
                // 无 pre-tokenize（Qwen3 等）：直接 BPE
                result.extend(self.bpe_encode(&seg));
            }
        }
        result
    }

    /// 按特殊 token 分割文本，返回交替的普通文本和特殊 token 片段
    fn split_on_special_tokens(&self, text: &str) -> Vec<String> {
        let mut segments = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            // 尝试匹配最长的特殊 token
            let mut matched = false;
            for st in &self.special_tokens {
                if remaining.starts_with(st.as_str()) {
                    segments.push(st.clone());
                    remaining = &remaining[st.len()..];
                    matched = true;
                    break;
                }
            }
            if !matched {
                // 找下一个特殊 token 的位置
                let mut next_special = remaining.len();
                for st in &self.special_tokens {
                    if let Some(pos) = remaining.find(st.as_str()) {
                        if pos < next_special {
                            next_special = pos;
                        }
                    }
                }
                // next_special 之前的都是普通文本
                segments.push(remaining[..next_special].to_string());
                remaining = &remaining[next_special..];
            }
        }
        segments
    }

    /// 对普通文本段进行 BPE 编码
    fn bpe_encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let encoded_text = self.text_to_vocab_str(text);
        let mut symbols: Vec<String> = encoded_text
            .chars()
            .map(|c| c.to_string())
            .collect();
        // BPE merge loop
        loop {
            if symbols.len() < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_idx = 0;
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self.merges.get(
                    &(symbols[i].clone(), symbols[i + 1].clone())
                ) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if best_rank == u32::MAX {
                break;
            }
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }
        symbols
            .iter()
            .map(|s| self.token_to_id.get(s).copied().unwrap_or(0))
            .collect()
    }

    /// 将 token IDs 解码为文本
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut pieces: Vec<&str> = Vec::with_capacity(tokens.len());
        for &id in tokens {
            if (id as usize) < self.id_to_token.len() {
                pieces.push(&self.id_to_token[id as usize]);
            }
        }
        let joined: String = pieces.concat();
        self.vocab_str_to_text(&joined)
    }

    /// 将原始文本转换为 vocab 空间的字符串
    fn text_to_vocab_str(&self, text: &str) -> String {
        match self.encoding {
            VocabEncoding::Gpt2 => {
                // GPT-2: 每个字节通过 byte→unicode 映射
                text.as_bytes()
                    .iter()
                    .map(|&b| self.byte_to_unicode[&b])
                    .collect()
            }
            VocabEncoding::RawUtf8 => {
                // Raw UTF-8: 直接使用原始文本
                text.to_string()
            }
        }
    }

    /// 将 vocab 空间的字符串转回原始文本
    fn vocab_str_to_text(&self, s: &str) -> String {
        match self.encoding {
            VocabEncoding::Gpt2 => {
                // GPT-2: 每个 unicode 字符通过 unicode→byte 映射还原
                let bytes: Vec<u8> = s
                    .chars()
                    .filter_map(|c| self.unicode_to_byte.get(&c).copied())
                    .collect();
                String::from_utf8_lossy(&bytes).into_owned()
            }
            VocabEncoding::RawUtf8 => {
                // Raw UTF-8: 处理 <0xHH> 形式的字节 token
                Self::decode_byte_tokens(s)
            }
        }
    }

    /// 处理 Raw UTF-8 模式下的 <0xHH> 字节 token
    /// Qwen3 等模型用 "<0xE4>" 这样的形式表示单字节 token
    fn decode_byte_tokens(s: &str) -> String {
        let mut result = Vec::new();
        let mut chars = s.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '<' {
                // 尝试解析 <0xHH> 模式
                let mut buf = String::new();
                let mut found_close = false;
                for next_c in chars.by_ref() {
                    if next_c == '>' {
                        found_close = true;
                        break;
                    }
                    buf.push(next_c);
                }
                if found_close && buf.starts_with("0x") && buf.len() == 4 {
                    if let Ok(byte) = u8::from_str_radix(&buf[2..], 16) {
                        result.push(byte);
                        continue;
                    }
                }
                // 不是字节 token，原样输出
                result.extend_from_slice("<".as_bytes());
                result.extend_from_slice(buf.as_bytes());
                if found_close {
                    result.push(b'>');
                }
            } else {
                let mut tmp = [0u8; 4];
                let encoded = c.encode_utf8(&mut tmp);
                result.extend_from_slice(encoded.as_bytes());
            }
        }
        String::from_utf8_lossy(&result).into_owned()
    }

    /// 将 token IDs 解码为原始字节（用于流式输出的 UTF-8 边界检测）
    pub fn decode_to_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut pieces = Vec::new();
        for &id in tokens {
            if (id as usize) < self.id_to_token.len() {
                let piece = &self.id_to_token[id as usize];
                match self.encoding {
                    VocabEncoding::Gpt2 => {
                        for c in piece.chars() {
                            if let Some(&b) = self.unicode_to_byte.get(&c) {
                                pieces.push(b);
                            }
                        }
                    }
                    VocabEncoding::RawUtf8 => {
                        // 处理 <0xHH> 字节 token
                        if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                            if let Ok(byte) = u8::from_str_radix(&piece[3..5], 16) {
                                pieces.push(byte);
                                continue;
                            }
                        }
                        pieces.extend_from_slice(piece.as_bytes());
                    }
                }
            }
        }
        pieces
    }

    /// 解码单个 token ID 为字符串（用于流式输出）
    pub fn decode_token(&self, id: u32) -> String {
        if (id as usize) < self.id_to_token.len() {
            let piece = &self.id_to_token[id as usize];
            self.vocab_str_to_text(piece)
        } else {
            String::new()
        }
    }

    /// 根据模型的 chat template 格式化对话
    /// messages: [(role, content), ...]
    pub fn format_chat(&self, messages: &[(String, String)]) -> String {
        self.format_chat_with_options(messages, &ChatOptions::default())
    }

    /// 根据模型的 chat template 格式化对话（带选项）
    /// messages: [(role, content), ...]
    pub fn format_chat_with_options(&self, messages: &[(String, String)], options: &ChatOptions) -> String {
        match self.chat_style {
            ChatStyle::ChatML => {
                let mut prompt = String::new();
                for (role, content) in messages {
                    prompt.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n", role, content
                    ));
                }
                prompt.push_str("<|im_start|>assistant\n");
                // Qwen3 thinking mode 控制：预填充空 think 块跳过思考
                if options.no_think {
                    prompt.push_str("<think>\n\n</think>\n\n");
                }
                prompt
            }
            ChatStyle::Llama3 => {
                let mut prompt = String::from("<|begin_of_text|>");
                for (role, content) in messages {
                    prompt.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, content
                    ));
                }
                prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                prompt
            }
        }
    }
}

/// 检测模型的 chat 风格和 stop tokens
fn detect_chat_style(
    token_to_id: &HashMap<String, u32>,
    gguf: &GgufFile,
) -> (ChatStyle, Vec<u32>, Option<u32>) {
    // 读取架构信息
    let arch = gguf
        .get_metadata("general.architecture")
        .and_then(|v| match v {
            MetadataValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default();

    // Llama 3 检测：查看是否有 Llama 3 特征 token
    if arch == "llama" {
        if let Some(&eot_id) = token_to_id.get("<|eot_id|>") {
            // Llama 3 风格
            let mut stop_tokens = vec![eot_id];
            if let Some(&end_of_text) = token_to_id.get("<|end_of_text|>") {
                stop_tokens.push(end_of_text);
            }
            let bos = token_to_id.get("<|begin_of_text|>").copied();
            return (ChatStyle::Llama3, stop_tokens, bos);
        }
    }

    // Qwen3 / ChatML 检测
    if let Some(&im_end) = token_to_id.get("<|im_end|>") {
        let mut stop_tokens = vec![im_end];
        if let Some(&eof) = token_to_id.get("<|endoftext|>") {
            stop_tokens.push(eof);
        }
        return (ChatStyle::ChatML, stop_tokens, None);
    }

    // Fallback: 尝试从 GGUF metadata 读取 EOS token
    let eos_id = gguf
        .get_metadata("tokenizer.ggml.eos_token_id")
        .and_then(|v| match v {
            MetadataValue::Uint32(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(2);
    let bos_id = gguf
        .get_metadata("tokenizer.ggml.bos_token_id")
        .and_then(|v| match v {
            MetadataValue::Uint32(n) => Some(*n),
            _ => None,
        });

    (ChatStyle::ChatML, vec![eos_id], bos_id)
}

