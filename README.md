<p align="center">
  <h1 align="center">🏔️ Citlali LLM Engine</h1>
  <p align="center"><em>A from-scratch LLM inference engine in pure Rust.<br>No Python. No C++. No CUDA bindings.</em></p>
</p>

---

<p align="center">
  <a href="https://github.com/xissleah/Citlali-LLM-Engine"><img alt="GitHub" src="https://img.shields.io/badge/github-xissleah%2FCitlali--LLM--Engine-blue?logo=github"></a>
  <img alt="Rust" src="https://img.shields.io/badge/rust-1.80+-orange?logo=rust">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## English

### Overview

Citlali is a **pure Rust** LLM inference runtime. Every component — GGUF parsing, dequantization, attention kernels, tokenizer, sampling, and HTTP serving — is hand-written from first principles with **zero external ML dependencies**. It runs on CPU alone, fits on consumer hardware, and is small enough to read in a weekend (~20 source files).

| | Citlali | llama.cpp (Rust bindings) |
|---|---|---|
| Language | Pure Rust | C++ with Rust FFI |
| GGUF parser | Built-in | ggml (C++) |
| Dequant | Hand-written kernels | ggml |
| Tokenizer | From GGUF metadata | SentencePiece / HF |
| Chat template | Auto-detected | Manual config |
| Compilation | `cargo build` | CMake + C++ toolchain |

### Architecture

```
┌─────────────────────────────────────────────┐
│         citlali-cli      citlali-server     │  ← Binary layer
├─────────────────────────────────────────────┤
│              citlali-runtime                │  ← Sampler / Prompt Cache / Generate loop
├─────────────────────────────────────────────┤
│              citlali-model                  │  ← Transformer forward / Tokenizer
├─────────────────────────────────────────────┤
│              citlali-compute                │  ← MatMul, RMSNorm, RoPE, Softmax, KV Cache
├─────────────────────────────────────────────┤
│              citlali-gguf                   │  ← GGUF parser, Q4_K/Q5_K/Q6_K dequant
└─────────────────────────────────────────────┘
```

### Features

- **GGUF v3 parser** — header, metadata KV pairs, tensor descriptors. No external library.
- **Dequantization** — Q4_K (4-bit), Q5_K (5-bit), Q6_K (6-bit) → f32, block-wise with per-block scales.
- **MatMul** — GQA-aware KV projection, grouped query attention support.
- **RoPE** — auto-detects Qwen (NEOX pairs) vs Llama (adjacent pairs) from GGUF metadata.
- **Paged KV Cache** — 16KB pages, Copy-on-Write eviction, optional disk swap.
- **Prompt Cache** — reuse KV prefixes across multi-turn conversations.
- **Sampling** — temperature, top-k, top-p, repetition penalty, greedy.
- **Streaming** — token-by-token output with correct UTF-8 boundary handling.
- **OpenAI-compatible HTTP API** — `/v1/chat/completions` with SSE streaming, `/v1/models`, CORS.
- **Qwen3 thinking mode** — controllable via `--no-think` flag or chat API.

### Quick Start

```bash
# 1. Clone
git clone https://github.com/xissleah/Citlali-LLM-Engine.git
cd Citlali-LLM-Engine

# 2. Build (release recommended)
cargo build --release -p citlali-cli
cargo build --release -p citlali-server

# 3. Download a GGUF model (e.g. Qwen3-0.6B-Q4_K_M.gguf) to the model/ directory

# 4. Run interactive CLI
./target/release/citlali model/Qwen3-0.6B-Q4_K_M.gguf

# 5. Or run HTTP server
./target/release/citlali-server model/Qwen3-0.6B-Q4_K_M.gguf --port 8080
```

### CLI Usage

```
citlali <path-to-gguf> [options]

Options:
  --mmap      Memory-mapped mode (low RAM, ~20-40% slower)
  --drop      Drop evicted KV pages (no disk swap)
  --greedy    Greedy sampling (deterministic output)
  --no-think  Disable Qwen3 thinking mode
```

### Server Usage

```bash
./target/release/citlali-server <path-to-gguf> [--port 8080] [--host 127.0.0.1]
```

```bash
# Test with curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "citlali-qwen3",
    "messages": [{"role": "user", "content": "What is Rust?"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  }'
```

### Engine Configurations

| Preset | Weight Strategy | KV Eviction | Use Case |
|--------|----------------|-------------|----------|
| `default_resident()` | Resident | Drop | General purpose, fastest |
| `low_memory()` | MMAP | Swap | Low-RAM devices, long context |
| `oneshot()` | Resident | Drop | Single-turn inference |
| `minimal()` | MMAP | Drop | Extreme memory constraints |

### Memory Profiles (Qwen3-0.6B Q4_K)

| Mode | RAM | Speed |
|------|-----|-------|
| Resident (default) | ~2.4 GB | Fastest |
| Resident + Drop | ~2.4 GB | Fastest |
| MMAP | ~50 MB | ~20-40% slower |
| MMAP + Drop | ~50 MB | Slowest |

### Project Structure

```
citlali-llm-engine/
├── Cargo.toml                      # Workspace
├── crates/
│   ├── citlali-gguf/               # GGUF parsing & dequantization
│   │   └── src/
│   │       ├── header.rs           # Magic, version, metadata/tensor counts
│   │       ├── metadata.rs         # Key-value metadata parser
│   │       ├── tensor.rs           # Tensor descriptor (name, dtype, dims, offset)
│   │       └── dequant.rs          # Q4_K, Q5_K, Q6_K → f32
│   ├── citlali-compute/            # Math kernels & memory management
│   │   └── src/
│   │       ├── ops.rs              # MatMul, RMSNorm, RoPE, Softmax, SwiGLU
│   │       ├── arena.rs            # Linear bump allocator
│   │       └── kv_cache.rs         # Paged KV cache with swap
│   ├── citlali-model/              # Transformer forward pass & tokenizer
│   │   └── src/
│   │       ├── model.rs            # Resident model (Qwen3, Llama)
│   │       ├── mmap_model.rs       # Memory-mapped model variant
│   │       ├── forward.rs          # Prefill + single-step decode
│   │       ├── quantized_ops.rs    # Quantized matmul kernels
│   │       └── tokenizer.rs        # Tokenizer from GGUF metadata
│   ├── citlali-runtime/            # Generation & sampling
│   │   └── src/
│   │       ├── generate.rs         # Prefill/decode loop, Prompt Cache
│   │       ├── sampler.rs          # Temperature, top-k, top-p, greedy
│   │       └── config.rs           # EngineConfig, WeightStrategy, KvPolicy
│   ├── citlali-cli/                # Interactive CLI binary
│   │   └── src/main.rs
│   └── citlali-server/             # HTTP server binary
│       └── src/
│           ├── main.rs             # Axum server
│           ├── routes.rs           # /v1/chat/completions, /v1/models
│           └── types.rs            # OpenAI-compatible request/response types
├── examples/
│   ├── inspect_gguf.rs             # Dump GGUF header + metadata
│   ├── dequant_test.rs             # Validate dequant against reference
│   └── generate.rs                 # Minimal inference example
└── model/                          # Drop your .gguf files here
```

### Technical Highlights

**Custom GGUF parser** — Parses GGUF v3 from scratch including header (magic, version, tensor/metadata counts), all metadata value types (string, integer, float, bool, array), and tensor descriptors with offset/dimensions/dtype.

**Hand-written dequantization** — Block-wise dequant for Q4_K (32-value blocks with 16×4-bit quants), Q5_K (32-value blocks with 16×5-bit), Q6_K (32-value blocks with 16×6-bit), each with per-block scaling and minimum values.

**Architecture auto-detection** — Reads `general.architecture` from GGUF metadata and selects the correct RoPE pair style: NEOX (Qwen: pairs `i` with `i+dim/2`) vs NORM (Llama: pairs `2i` with `2i+1`).

**Paged KV Cache** — Inspired by vLLM's PagedAttention. 16KB pages, page table mapping logical→physical positions, Copy-on-Write eviction, optional disk swap for long contexts.

**Prompt Cache** — In multi-turn conversations, the KV cache prefix from previous turns is reused. Only new tokens are prefill'd, making follow-up responses significantly faster.

**Tokenizer from GGUF** — Extracts tokenizer configuration (BPE merges, special tokens, chat template) directly from GGUF metadata — no external tokenizer files needed. Supports Qwen3's chat template with thinking mode control.

### Performance

Tested on Intel i7-12700H (laptop CPU), Qwen3-0.6B Q4_K:

| Mode | Prefill | Generation |
|------|---------|------------|
| Resident | ~80 tok/s | ~12 tok/s |
| MMAP | ~50 tok/s | ~9 tok/s |

### Roadmap

- [ ] SIMD acceleration (SSE/AVX for matmul & dequant)
- [ ] Metal / CUDA backend
- [ ] Speculative decoding
- [ ] LoRA adapter support
- [ ] Quantized KV cache (int8)
- [ ] Batch inference / continuous batching
- [ ] Web UI demo

### Contributing

Contributions are welcome! This is a learning-oriented project — if you're interested in how LLM inference works under the hood, there's no better way than reading and hacking on the code.

Areas where contributions are especially valuable: SIMD/AVX kernel optimization, additional architecture support (Mistral, Phi, Gemma), tests, benchmarks, and documentation.

### License

MIT © 2026 xissleah

---

## 中文

### 概述

Citlali 是一个**纯 Rust** 实现的 LLM 推理引擎。从 GGUF 解析、反量化、注意力计算、分词器、采样策略到 HTTP 服务，全部从零手写，**零外部 ML 依赖**。仅靠 CPU 即可运行，适合消费级硬件，代码量约 20 个源文件，一个周末即可通读。

| | Citlali | llama.cpp (Rust 绑定) |
|---|---|---|
| 语言 | 纯 Rust | C++ 加 Rust FFI |
| GGUF 解析 | 内置 | ggml (C++) |
| 反量化 | 手写 kernel | ggml |
| 分词器 | 从 GGUF 元数据提取 | SentencePiece / HF |
| 对话模板 | 自动检测 | 手动配置 |
| 编译 | `cargo build` | CMake + C++ 工具链 |

### 架构

```
┌─────────────────────────────────────────────┐
│         citlali-cli      citlali-server     │  ← 二进制层
├─────────────────────────────────────────────┤
│              citlali-runtime                │  ← 采样 / Prompt Cache / 生成循环
├─────────────────────────────────────────────┤
│              citlali-model                  │  ← Transformer 前向 / 分词器
├─────────────────────────────────────────────┤
│              citlali-compute                │  ← MatMul, RMSNorm, RoPE, Softmax, KV Cache
├─────────────────────────────────────────────┤
│              citlali-gguf                   │  ← GGUF 解析, Q4_K/Q5_K/Q6_K 反量化
└─────────────────────────────────────────────┘
```

### 功能特性

- **GGUF v3 解析器** — 完整解析 header、metadata 键值对、tensor 描述符，无外部依赖。
- **反量化** — Q4_K（4-bit）、Q5_K（5-bit）、Q6_K（6-bit）→ f32，按块反量化，含逐块缩放因子。
- **矩阵乘法** — 支持 GQA（分组查询注意力）的 KV 投影。
- **RoPE** — 从 GGUF 元数据自动检测 Qwen（NEOX 配对）与 Llama（相邻配对）的旋转位置编码风格。
- **分页 KV Cache** — 16KB 页面，写时复制淘汰策略，可选磁盘交换。
- **Prompt Cache** — 多轮对话中复用 KV Cache 前缀，后续回复仅需处理新 token。
- **采样策略** — temperature、top-k、top-p、重复惩罚、贪心采样。
- **流式输出** — 逐 token 输出，正确处理 UTF-8 边界。
- **OpenAI 兼容 HTTP API** — `/v1/chat/completions`（支持 SSE 流式）、`/v1/models`、CORS。
- **Qwen3 thinking mode** — 通过 `--no-think` 参数或 API 控制。

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/xissleah/Citlali-LLM-Engine.git
cd Citlali-LLM-Engine

# 2. 编译（推荐 release 模式）
cargo build --release -p citlali-cli
cargo build --release -p citlali-server

# 3. 下载一个 GGUF 模型（如 Qwen3-0.6B-Q4_K_M.gguf）放到 model/ 目录

# 4. 启动交互式 CLI
./target/release/citlali model/Qwen3-0.6B-Q4_K_M.gguf

# 5. 或启动 HTTP 服务
./target/release/citlali-server model/Qwen3-0.6B-Q4_K_M.gguf --port 8080
```

### CLI 用法

```
citlali <path-to-gguf> [选项]

选项：
  --mmap      内存映射模式（低内存，速度降低约 20-40%）
  --drop      KV Cache 满时直接丢弃（不写磁盘）
  --greedy    贪心采样（确定性输出）
  --no-think  禁用 Qwen3 thinking mode
```

### Server 用法

```bash
./target/release/citlali-server <path-to-gguf> [--port 8080] [--host 127.0.0.1]
```

```bash
# 用 curl 测试
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "citlali-qwen3",
    "messages": [{"role": "user", "content": "什么是 Rust？"}],
    "max_tokens": 256,
    "temperature": 0.7,
    "stream": true
  }'
```

### 引擎配置

| 预设 | 权重策略 | KV 淘汰 | 适用场景 |
|------|---------|--------|---------|
| `default_resident()` | 常驻内存 | 丢弃 | 通用，最快速度 |
| `low_memory()` | 内存映射 | 交换 | 低内存设备，长上下文 |
| `oneshot()` | 常驻内存 | 丢弃 | 单轮推理 |
| `minimal()` | 内存映射 | 丢弃 | 极限内存受限 |

### 内存占用（Qwen3-0.6B Q4_K）

| 模式 | 内存 | 速度 |
|------|------|------|
| 常驻（默认） | ~2.4 GB | 最快 |
| 常驻 + Drop | ~2.4 GB | 最快 |
| MMAP | ~50 MB | 降低约 20-40% |
| MMAP + Drop | ~50 MB | 最慢 |

### 项目结构

```
citlali-llm-engine/
├── Cargo.toml                      # 工作区
├── crates/
│   ├── citlali-gguf/               # GGUF 解析与反量化
│   │   └── src/
│   │       ├── header.rs           # Magic、版本号、metadata/tensor 计数
│   │       ├── metadata.rs         # 键值对元数据解析
│   │       ├── tensor.rs           # Tensor 描述符（名称、数据类型、维度、偏移）
│   │       └── dequant.rs          # Q4_K, Q5_K, Q6_K → f32
│   ├── citlali-compute/            # 数学运算与内存管理
│   │   └── src/
│   │       ├── ops.rs              # MatMul, RMSNorm, RoPE, Softmax, SwiGLU
│   │       ├── arena.rs            # 线性 bump 分配器
│   │       └── kv_cache.rs         # 分页 KV Cache，支持磁盘交换
│   ├── citlali-model/              # Transformer 前向计算与分词器
│   │   └── src/
│   │       ├── model.rs            # 常驻内存模型（Qwen3, Llama）
│   │       ├── mmap_model.rs       # 内存映射模型变体
│   │       ├── forward.rs          # Prefill + 单步 decode
│   │       ├── quantized_ops.rs    # 量化矩阵乘法
│   │       └── tokenizer.rs        # 从 GGUF 元数据提取的分词器
│   ├── citlali-runtime/            # 生成与采样
│   │   └── src/
│   │       ├── generate.rs         # Prefill/decode 循环、Prompt Cache
│   │       ├── sampler.rs          # Temperature, top-k, top-p, greedy
│   │       └── config.rs           # EngineConfig, WeightStrategy, KvPolicy
│   ├── citlali-cli/                # 交互式 CLI
│   │   └── src/main.rs
│   └── citlali-server/             # HTTP 服务
│       └── src/
│           ├── main.rs             # Axum 服务
│           ├── routes.rs           # /v1/chat/completions, /v1/models
│           └── types.rs            # OpenAI 兼容的请求/响应类型
├── examples/
│   ├── inspect_gguf.rs             # 打印 GGUF 文件头与元数据
│   ├── dequant_test.rs             # 验证反量化正确性
│   └── generate.rs                 # 最小推理示例
└── model/                          # 将 .gguf 文件放在此处
```

### 技术亮点

**自研 GGUF 解析器** — 从零实现 GGUF v3 二进制格式解析，包括 header（magic、版本、tensor/metadata 计数）、全部 metadata 值类型（字符串、整数、浮点、布尔、数组）和 tensor 描述符。

**手写反量化 kernel** — 逐块反量化：Q4_K（32 值块，16×4bit）、Q5_K（32 值块，16×5bit）、Q6_K（32 值块，16×6bit），每块独立缩放因子和最小值。

**架构自动检测** — 从 GGUF 元数据读取 `general.architecture`，自动选择正确的 RoPE 配对方式：NEOX（Qwen：`i` 与 `i+dim/2` 配对）与 NORM（Llama：`2i` 与 `2i+1` 配对）。

**分页 KV Cache** — 借鉴 vLLM 的 PagedAttention 思想。16KB 页面大小，页表映射逻辑位置到物理页面，写时复制淘汰，可选磁盘交换以支持超长上下文。

**Prompt Cache** — 多轮对话中复用前轮的 KV Cache 前缀。后续用户消息仅需对新 token 做 prefill，后续回复速度大幅提升。

**从 GGUF 提取分词器** — 直接从 GGUF 元数据中提取分词器配置（BPE 合并、特殊 token、对话模板），无需额外文件。支持 Qwen3 的 chat template 及 thinking mode 控制。

### 性能

在 Intel i7-12700H（笔记本 CPU）上测试，Qwen3-0.6B Q4_K：

| 模式 | Prefill | 生成 |
|------|---------|------|
| 常驻内存 | ~80 tok/s | ~12 tok/s |
| 内存映射 | ~50 tok/s | ~9 tok/s |

### 路线图

- [ ] SIMD 加速（SSE/AVX 用于矩阵乘法与反量化）
- [ ] Metal / CUDA 后端
- [ ] 推测解码（Speculative Decoding）
- [ ] LoRA 适配器支持
- [ ] 量化 KV Cache（int8）
- [ ] 批量推理 / 连续批处理
- [ ] Web UI 演示

### 贡献

欢迎贡献！这是一个面向学习的项目——如果你想深入理解 LLM 推理的底层原理，阅读和修改这份代码是最好的方式。

特别欢迎以下方向的贡献：SIMD/AVX kernel 优化、更多架构支持（Mistral、Phi、Gemma）、测试与基准、文档完善。

### 许可证

MIT © 2026 xissleah
