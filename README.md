# Citlali-LLM-Engine CUDA

一个从零手写的 **C++17 / CUDA** 大模型推理引擎，用于在本地 GPU 上跑 **Qwen3 稠密模型**的 GGUF 权重。

不依赖 llama.cpp、不依赖任何推理框架——GGUF 解析、BPE 分词、量化反量化、KV Cache、注意力、CUDA kernel 全部自己实现。目标是把「一次完整的前向传播」拆到能看懂的程度，再逐层优化。

当前版本：`v0.1` · 分支 `main`

---

## 目录

- [这个项目在做什么](#这个项目在做什么)
- [特性](#特性)
- [架构](#架构)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [编译](#编译)
- [快速开始](#快速开始)
- [两种权重加载策略](#两种权重加载策略)
- [异构远程推理](#异构远程推理)
- [工具与脚本](#工具与脚本)
- [已知限制](#已知限制)
- [关于模型文件](#关于模型文件)
- [路线图](#路线图)

---

## 这个项目在做什么

一件事：**把 Qwen3 的推理过程完整地自己写一遍**。

主流的做法是直接用 llama.cpp 或 PyTorch，但这个项目选择了另一条路——从 GGUF 二进制格式的第一个字节开始解析，自己实现 tokenizer、自己实现量化格式的反量化、自己写 CUDA kernel。所以它不是一个「能用的产品」，而是一个**可以逐行读懂推理全过程的实验平台**。

目前已经跑到：加载 Qwen3-0.6B 的 Q4_K_M 权重，单卡 GPU 上完成多轮对话生成；并支持把一部分层**卸载到 Android 平板的 GPU 上**做异构推理。

---

## 特性

### 模型与格式

- **GGUF v3 解析**：魔数校验、metadata 键值对（含数组类型）、tensor 目录、按需读取 tensor 数据（不全量载入内存）
- **Qwen3 配置提取**：从 GGUF metadata 直接读出层数、hidden、FFN、head 数、KV head 数（GQA）、head_dim、rope_theta、rms_norm_eps 等
- **Tokenizer**：直接从 GGUF metadata 里加载 Qwen 的 byte-level BPE 词表与合并规则，支持 chat template 与流式解码
- **量化格式**：反量化路径（转 FP16）实现了 `F32` / `F16` / `BF16` / `Q8_0` / `Q4_K` / `Q6_K`；量化常驻路径的 CUDA kernel 目前**只实现了 `Q4_K` 与 `Q6_K`**

### 计算

- **两套权重存取路径**（见[下文](#两种权重加载策略)）：
  - `DequantizeToFp16`（默认）——加载时全量反量化成 FP16 常驻显存
  - `KeepQuantized`（`--quantized`）——**量化块原样留在显存**，kernel 内动态反量化
- **手写 CUDA kernel**：
  - `matvec_fp16` / `matvec_fp16_to_float` —— FP16 稠密矩阵向量乘
  - `matvec_q4k` / `matvec_q6k` / `matvec_q6k_to_float` —— 量化权重直接参与计算
  - `embed_q6k` —— Q6_K 词嵌入表按 token id 取行并动态反量化
  - `rms_norm` / `head_rms_norm` / `rope` / `swiglu` / `attention` / `residual_add`
- **FP16 KV Cache**，按层独立分配，支持 GQA
- **单 token 解码循环**，CPU 侧贪心采样，逐 token 流式输出

### 执行

- **CLI 四种模式**：单次生成 / 交互式多轮对话 / 模型信息概览（不加载权重）/ prompt 分词调试
- **异构远程推理**：把指定层卸载到 Android 设备，通过 ADB（USB / Wi-Fi）隧道走 TCP 传输，对端用 Vulkan compute 执行（见[下文](#异构远程推理)）

---

## 架构

### 分层

```
┌──────────────────────────────────────────────────┐
│  citlali_cli                                     │  命令行入口
├──────────────────────────────────────────────────┤
│  runtime::InferenceSession                       │  会话 / 对话历史 / 生成循环
├──────────────────────────────────────────────────┤
│  model::QwenModel                                │  逐层前向：Norm→QKV→RoPE→Attn
│  model::WeightLoader   model::LinearBackend      │  →O→Norm→Gate/Up→SwiGLU→Down
├──────────────────────────────────────────────────┤
│  compute::(CUDA kernels, DeviceBuffer)           │  显存管理 + kernel 调度
│  remote::RemoteClient                            │  层卸载（可选）
├──────────────────────────────────────────────────┤
│  io::GgufFile     tokenizer::Tokenizer           │  GGUF 解析 / 分词
└──────────────────────────────────────────────────┘
```

### 关键设计边界：`WeightHandle`

模型层（`QwenModel`）**只看得到 `WeightHandle`**，看不到权重到底是什么格式：

```cpp
struct WeightHandle {
    std::string name;                              // 如 blk.0.attn_q.weight
    compute::GgufTensorType gguf_type;             // 原始量化类型 Q4_K / Q6_K / F16 ...
    compute::WeightStorageKind storage_kind;       // 显存里是量化块还是 FP16
    std::vector<uint64_t> dims;                    // [列, 行]
    compute::DeviceBufferPtr buffer;               // 显存缓冲区
};
```

这个边界是整个项目的核心：`QwenModel` 不需要知道权重是 FP16 还是 Q4_K，由 `Fp16LinearBackend::forward()` 按 `storage_kind` 分发到不同 kernel。**因此新增一种量化格式不需要改模型层一行代码。**

### 前向数据流

```
token_id
  → token_embedding_（Q6_K 或 FP16）
  → for each layer:
        attn_norm → [远程层？交给 RemoteClient] → attn_q/k/v(+bias)
        → q_norm / k_norm → rope → KV Cache 写入
        → attention(scores, value) → attn_o → residual
        → ffn_norm → gate / up → swiglu → down → residual
  → output_norm → output_weight → logits
  → CPU 贪心采样 → 下一个 token_id
```

被 `--offload` 指定的层，整层（`attn_norm` 到 `ffn_down` 加 residual）在远端执行，本地只负责 embedding、未卸载层、输出投影与采样。

---

## 目录结构

```
citlali-cuda/
├── CMakeLists.txt                  引擎 + CLI 构建
├── 编译指令.cmd                     一键编译脚本
├── run_cli.bat                     本地推理交互式启动器
├── run_remote.bat                  远程异构推理启动器（含 ADB 配置）
│
├── include/citlali/
│   ├── common.h                    错误处理 / 断言
│   ├── io/gguf_reader.h            GGUF 文件与 tensor 目录
│   ├── tokenizer/tokenizer.h       BPE 分词器
│   ├── model/
│   │   ├── qwen_config.h           超参数
│   │   ├── weight_loader.h         WeightHandle / 加载策略
│   │   ├── linear_backend.h        线性层后端抽象
│   │   └── qwen_model.h            模型与逐层权重
│   ├── compute/
│   │   ├── common/dtype.h          GGUF 类型枚举
│   │   └── cuda/{kernels,tensor,cuda_utils}.h
│   ├── remote/remote_client.h      远端层卸载客户端
│   └── runtime/session.h           会话与生成选项
│
├── src/                            上述头文件的实现
│   ├── cli/main.cpp                命令行参数解析与入口
│   └── remote/remote_client.cpp    TCP 客户端 + 帧协议
│
├── android/                        远端 Vulkan 推理服务（独立 CMake 工程）
│   ├── CMakeLists.txt              含 glslc 着色器编译规则
│   ├── include/citlali_remote/
│   │   ├── protocol.h              帧协议定义
│   │   ├── tcp_server.h
│   │   └── vulkan_context.h
│   ├── src/                        server 实现 + Vulkan 后端
│   │   ├── server_main.cpp
│   │   ├── tcp_server.cpp
│   │   ├── protocol.cpp
│   │   ├── vulkan_context.cpp
│   │   ├── vulkan_context_layer.cpp    整层前向调度
│   │   ├── vulkan_context_tensor.cpp   权重上传与缓冲区管理
│   │   └── main.cpp                Vulkan 能力探测工具（remote_probe）
│   └── shaders/*.comp              9 个 GLSL compute shader
│
├── bench/matvec_bench.cu           matvec 设计对比微基准（独立编译）
└── gguf-reader/gguf_inspect.py     GGUF 只读检查脚本（Python）
```

---

## 环境要求

| 组件 | 版本 / 说明 |
|---|---|
| CMake | ≥ 3.28（引擎） / ≥ 3.22（Android 端） |
| C++ | C++17 |
| CUDA Toolkit | 需包含 `cudart`；架构默认 `native`，本机实测用 `sm_120` |
| MSVC | Visual Studio 2022（Windows 路径需要，`/utf-8` 编译） |
| GPU | 支持 FP16 的 NVIDIA 显卡。跑 0.6B 建议 ≥ 4GB 显存 |
| Python | 仅 `gguf_inspect.py` 需要，3.8+ |
| Android（可选） | NDK r29、`glslc`、支持 Vulkan 1.1 compute 的设备 |

> **注意**：Windows 上 CLI 入口是 `wmain`（`/ENTRY:wmainCRTStartup`），为的是正确接收中文路径与中文 prompt。这是有意为之，不是配置错误。

---

## 编译

### Windows（Visual Studio 2022）

```powershell
cd C:\work\workspace\citlali-cuda

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j
```

产物：`build\Release\citlali_cli.exe`

`-DCMAKE_CUDA_ARCHITECTURES` 按自己的显卡改（`120` = Blackwell / RTX 50 系）。不指定时会用 `native` 自动探测。

等价的快捷方式：直接双击 `编译指令.cmd`。

### Windows（Ninja，编译更快）

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
```

产物：`build\citlali_cli.exe`（`run_cli.bat` 会自动找到这两种位置）

### Android 远端服务（可选）

需要 Android NDK r29。以下配置与 `android/build/CMakeCache.txt` 中实际使用的一致：

```powershell
cmake -S android -B android/build -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE=C:\Android\android-ndk-r29\build\cmake\android.toolchain.cmake `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-28 `
  -DCMAKE_BUILD_TYPE=Release

cmake --build android/build
```

产物在 `android/build/` 下：

- `citlali_remote_server` —— 远端推理服务
- `citlali_remote_probe` —— Vulkan 能力探测工具
- `shaders/*.spv` —— 由 `glslc` 从 `android/shaders/*.comp` 编译得到

`glslc` 默认取 `${ANDROID_NDK}/shader-tools/windows-x86_64/glslc.exe`，找不到会直接报错；可用 `-DGLSLC_EXECUTABLE=<path>` 覆盖。

---

## 快速开始

### 最简用法

```powershell
.\build\Release\citlali_cli.exe `
  --model .\Qwen3-0.6B-Q4_K_M.gguf `
  --prompt "你好，介绍一下你自己" `
  --max-new-tokens 64
```

输出是流式的，末尾会打印统计（格式示例，数字随硬件变化）：

```
[stats] tokens=64 time=2.145s speed=29.84 tok/s stop=max_new_tokens
```

`stop` 有三种取值：`eos`（遇到终止符）、`max_new_tokens`（达到长度上限）、`context_limit`（上下文塞满）。

### 交互式多轮对话

```powershell
.\build\Release\citlali_cli.exe --model .\Qwen3-0.6B-Q4_K_M.gguf --interactive
```

会话内命令：

- `/exit` 或 `/quit` —— 退出
- `/reset` —— 清空对话历史

### 调试：先看模型信息，再加载权重

排查问题时，这两个模式**不加载权重**，几秒就能出结果：

```powershell
# 打印架构、层数、head 配置、tensor 数量与各量化类型分布
citlali_cli.exe --model model.gguf --info

# 打印 prompt 的分词结果（token id、原始字节、解码文本，控制字符会转义）
citlali_cli.exe --model model.gguf --prompt "你好" --tokens
```

`--tokens` 特别适合排查「模型答非所问」——往往不是模型的问题，而是 chat template 套错了。加 `--no-chat-template` 可以对比原始文本的分词。

### 完整参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model <path>` | 无 | GGUF 模型路径，必填 |
| `--prompt <text>` | 无 | 提示词 / 用户消息 |
| `--max-new-tokens <N>` | `256` | 最多生成的 token 数 |
| `--ctx <N>` / `--context <N>` | `2048` | 最大上下文长度 |
| `--no-chat-template` | 关 | 把 `--prompt` 当纯文本，不套 chat 模板 |
| `--interactive` | 关 | 进入多轮对话模式 |
| `--info` | 关 | 只打印模型信息，不加载权重 |
| `--tokens` | 关 | 只打印分词结果，不加载权重 |
| `--quantized` | 关 | 权重保持量化格式常驻显存（见下节） |
| `--remote <host:port>` | 无 | 远端设备地址 |
| `--remote-transport <usb\|wifi6>` | `usb` | 传输方式 |
| `--remote-backend <gpu\|npu\|hybrid\|auto>` | `gpu` | 远端后端 |
| `--offload <layer:N[,N]>` | 无 | 卸载哪些层，如 `layer:0,1,2` |
| `--help` / `-h` | — | 打印用法 |

> `--remote` 与 `--offload` **必须成对出现**，只给一个会直接报错。
> 用 `usb` 且没写 `--remote` 时，端点默认 `127.0.0.1:27183`（配合 `adb forward`）。

---

## 两种权重加载策略

这是本项目性价比最高的优化点，值得单独说。

**默认（`DequantizeToFp16`）**：加载时把 Q4_K/Q6_K 全部反量化成 FP16 放进显存。

- 优点：kernel 简单，就是普通 FP16 矩阵乘
- 缺点：**显存膨胀约 3.5～7 倍**。Qwen3-4B 的 Q4_K 权重原本约 2.5GB，反量化成 FP16 后要 ~8GB，加上 KV cache 和中间缓冲，8GB 显卡直接爆

**量化常驻（`--quantized`，`KeepQuantized`）**：量化块**原样**拷进显存，在 kernel 内部边读边反量化。

```powershell
citlali_cli.exe --model .\Qwen3-8B-Q4_K_M.gguf --quantized --prompt "你好"
```

- 优点：显存占用回到量化文件大小，能跑更大的模型
- 代价：kernel 需要处理量化块布局（`Q4_K` 的 6-bit 尺度 + 4-bit 权重打包、`Q6_K` 的 8-bit 高 4 位 + 4-bit 低位），实现复杂度显著上升

对应实现在 `src/model/linear_backend.cpp` —— `Fp16LinearBackend::forward()` 按 `WeightHandle::storage_kind` 分发：

```cpp
if (storage_kind == QuantizedDevice) {
    switch (gguf_type) {
        case Q4_K: launch_matvec_q4k(...); return;
        case Q6_K: launch_matvec_q6k(...); return;
    }
} else if (storage_kind == Fp16Device) {
    launch_matvec_fp16(...);
}
```

> 类名仍叫 `Fp16LinearBackend`，是因为最初只做 FP16 路径；现在它同时服务两条路径，是后续重构的候选点。

⚠️ **类型支持的范围要分清**：`WeightLoader::check_quantized_type()` 放行了 `Q2_K`/`Q3_K`/`Q4_0`/`Q4_1`/`Q4_K`/`Q5_0`/`Q5_1`/`Q5_K`/`Q6_K`/`Q8_0`/`Q8_1`/`Q8_K` 共 12 种类型进入「量化常驻」状态，但 `Fp16LinearBackend` 里只写了 `Q4_K` 和 `Q6_K` 两个 kernel。**用 `--quantized` 加载其他量化类型（例如 Q5_K_M 模型），会在 kernel 分发处抛 `unsupported weight storage kind`。** 反量化路径则是 6 种类型（`F32`/`F16`/`BF16`/`Q8_0`/`Q4_K`/`Q6_K`），其余类型抛 `tensor type is not implemented by the FP16 loader yet`。

实际可用组合：

| 模型量化类型 | 默认（反量化 FP16） | `--quantized` |
|---|---|---|
| `Q4_K_M` | ✅ | ✅ |
| `Q6_K` | ✅ | ✅ |
| `Q8_0` | ✅ | ❌ 未实现 kernel |
| `F16` / `F32` / `BF16` | ✅ | ➖ 自动回落 FP16 路径，`--quantized` 等于没开 |
| 其他 `Q*_K` / `Q*_0` / `Q*_1` | ❌ | ❌ |

---

## 异构远程推理

把一部分 Transformer 层**丢给 Android 设备算**，本地 GPU 只算剩下的层。

### 为什么这么做

手机/平板的 SoC GPU（Adreno / Mali）通常有独立的显存和算力，和桌面独显互不抢占。把一部分层卸载过去，相当于**用另一块 GPU 白捡算力**，尤其适合显存吃紧的场景。

### 数据流

```
Windows PC (CUDA)                          Android 设备 (Vulkan)
─────────────────                          ────────────────────
启动时：
  RemoteClient::upload_layer()  ──TCP──▶   上传该层全部权重到显存
                                          （按 tensor 逐个 LoadTensor）

每次前向：
  hidden state (FP16)          ──TCP──▶   RunLayerRange
                                          → attn_norm → QKV → RoPE
                                          → attention → O → FFN
                              ◀──TCP──    LayerRangeResult (FP16)
  继续算本地层
```

权重**只在启动时传一次**，每次前向只传 hidden state（hidden 维度的 FP16 向量，几十 KB），所以 USB 隧道也够用。

### 帧协议

定义在 `android/include/citlali_remote/protocol.h`：

| 项 | 值 |
|---|---|
| 魔数 | `0x314C5443`（"CTL1"，小端） |
| 版本 | `1` |
| 帧头 | 24 字节 |
| 单帧上限 | 64 MiB |
| 消息类型 | `Ping`(1) / `Pong`(2) / `Hello`(3) / `Capabilities`(4) / `LoadTensor`(5) / `TensorLoaded`(6) / `RunLayerRange`(13) / `LayerRangeResult`(14) / `Error`(255) |

握手流程：`Hello`（版本协商 + 请求后端）→ `Capabilities`（回到 Vulkan API 版本、驱动版本、显存上限、subgroup size、是否支持 FP16 storage buffer 等）。传输的数据都带 64 位内容哈希，用于校验。

### 用法

**推荐直接跑 `run_remote.bat`**，它会自动完成：找 `adb` → 列出设备 → 选 USB 或 Wi-Fi → 推 server 到 `/data/local/tmp` → 启动 server → 建立端口转发 → 再拉起 CLI。

手动方式（USB）：

```powershell
# 1. 推 server 和 shader
adb push android/build/citlali_remote_server /data/local/tmp/
adb push android/build/shaders/q4k_matvec.spv /data/local/tmp/

# 2. 启动远端服务：<端口> <shader路径> <绑定地址>
adb shell "nohup /data/local/tmp/citlali_remote_server 27183 /data/local/tmp/q4k_matvec.spv 127.0.0.1 >/data/local/tmp/citlali_remote_server.log 2>&1 </dev/null &"

# 3. 端口转发
adb forward tcp:27183 tcp:27183

# 4. 跑推理，把第 0~13 层卸载过去
citlali_cli.exe --model .\Qwen3-0.6B-Q4_K_M.gguf `
  --prompt "你好" `
  --quantized `
  --remote 127.0.0.1:27183 `
  --remote-transport usb `
  --remote-backend gpu `
  --offload layer:0,1,2,3,4,5,6,7,8,9,10,11,12,13
```

Wi-Fi 6 模式额外走 `adb tcpip 5555` + `adb connect <平板IP>:5555`，本地转发端口用 `27184` 避开冲突。

> `--offload` 接受的是**整层卸载**，只能指定层号，不支持把单层的某个矩阵单独切出去。

---

## 工具与脚本

### `gguf-reader/gguf_inspect.py`

纯 Python 的 GGUF 只读检查器，不用编译就能看清一个模型文件里到底有什么。

```powershell
python gguf_inspect.py MODEL.gguf                # 列出全部 tensor
python gguf_inspect.py MODEL.gguf --type Q6_K    # 只看指定量化类型
python gguf_inspect.py MODEL.gguf --summary      # 汇总统计
```

排查「引擎读到的 tensor 名/形状和预期不一致」时非常有用。

### `bench/matvec_bench.cu`

独立的微基准，在 Qwen3-0.6B 的真实形状上对比四种 matvec 实现的差异：

- **A**：当前引擎设计（一行一个 block、256 线程、标量 2 字节读取、8 次 `__syncthreads` 的共享内存树形归约）
- **B**：只把归约换成 warp shuffle——用来把「归约开销」和「访存开销」两个因素**分离开**测量
- **C**：warp-per-row + `uint4`（一次 8 个 half）向量化加载 + shuffle 归约
- **D**：纯流式读取带宽参考——测出这块卡**理论上限**，好判断 A/B/C 距离天花板还有多远

不属于引擎 target，单独编译：

```powershell
nvcc -O3 -std=c++17 -arch=sm_120 -o matvec_bench.exe matvec_bench.cu
```

### 脚本

| 脚本 | 用途 |
|---|---|
| `编译指令.cmd` | 一键 CMake 配置 + Release 编译 |
| `run_cli.bat` | 中文交互式菜单：选模型 → 选模式（单次生成 / 多轮对话 / 模型信息 / 分词） |
| `run_remote.bat` | 全自动配好 ADB 与远端 server，再拉起远程推理 |

---

## 已知限制

这个项目目前是**能跑通、能读懂**的阶段，不是生产可用状态。明确没做的：

**计算层面**

- **只支持贪心采样**。没有 temperature / top-k / top-p / 重复惩罚
- **逐 token 前向**，没有批量 prefill、没有 continuous batching。长 prompt 会一个个 token 喂进去，首 token 延迟高
- **kernel 是朴素实现**。`matvec` 里每个线程标量读 2 字节，没有向量化加载（`__half2` / `float4`）、没有预取、没有 `cp.async`
- **FP16 累加**，数值精度不如 FP32 累加
- 仅支持 Qwen3 稠密架构，**没有 MoE**、没有量化 KV Cache

**工程层面**

- **Windows / MSVC 优先**，Linux 端未验证
- **无测试**。没有单测、没有数值对齐测试（比如和 llama.cpp 的输出逐 token 对比）
- **NPU 与 Hybrid 后端只有协议预留**。`--remote-backend npu` 能解析，远端没有对应实现
- **无性能剖析基建**。`nsys_reports/` 下是手工采集的 NSight 报告，未入库

**显存**

- 默认（反量化）路径下，**8GB 显卡跑不动 4B 模型**。Q4_K 的 4B 权重反量化成 FP16 后约 8GB，加上 KV cache 与中间缓冲会 OOM
- 想跑更大模型，用 `--quantized`，或减小 `--ctx`

---

## 关于模型文件

仓库**不包含**任何模型权重，`.gitignore` 已排除 `*.gguf`。

本机用到的测试模型（Qwen3 系列，Q4_K_M 量化）：

| 文件 | 大小 | 备注 |
|---|---|---|
| `Qwen3-0.6B-Q4_K_M.gguf` | 397 MB | 日常调试首选，显存压力小 |
| `Qwen3-4B-Q4_K_M.gguf` | 2.5 GB | 需要 `--quantized` 才能在 8GB 显卡上跑 |
| `Qwen3-8B-Q4_K_M.gguf` | 5.0 GB | 同上 |

模型需自行从 HuggingFace / ModelScope 下载，放在项目根目录或任意路径，用 `--model` 指定。

> 这也是仓库根目录必须有 `.gitignore` 的原因：一次误操作 `git add -A` 就会把 7.4GB 权重写进 git 历史，**且无法通过删除文件回退**。

---

## 路线图

- [x] GGUF 解析 + Qwen3 配置提取
- [x] BPE 分词器与 chat template
- [x] FP16 全链路 CUDA 推理
- [x] Q4_K / Q6_K 量化权重常驻显存
- [x] 整层卸载到 Android Vulkan 设备
- [ ] 数值对齐测试（与 llama.cpp 逐 token 比对）
- [ ] `matvec` 访存优化（向量化加载 + warp shuffle 归约）
- [ ] 采样策略（temperature / top-k / top-p）
- [ ] 批量 prefill
- [ ] FP32 累加
- [ ] Linux 端构建验证

---
