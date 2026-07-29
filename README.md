# Citlali-LLM-Engine CUDA

A small C++/CUDA inference skeleton for Qwen3 dense GGUF models.

This first CUDA version is intentionally simple:

- GGUF metadata and tensor directory parsing.
- Qwen3 dense config extraction.
- Qwen/Qwen3 byte-level BPE tokenizer loaded from GGUF metadata.
- Q4_K, Q6_K, Q8_0, F16 and F32 weights are loaded as FP16 device weights.
- Single-token decode loop with FP16 KV cache.
- Naive CUDA kernels for matvec, RMSNorm, RoPE, attention, SwiGLU and residual add.
- Greedy sampling on CPU after copying one logits vector back from GPU.

The important boundary for later optimization is already present: model code only sees WeightHandle. A later version can keep GGUF quantized blocks on device and add dynamic dequantization kernels without rewriting tokenizer, config, KV cache or Qwen layer flow.

Build with Visual Studio:

    cd C:\work\citlali\cuda
    cmake -S . -B build -G "Visual Studio 17 2022" -A x64
    cmake --build build --config Release

Build with Ninja:

    cd C:\work\citlali\cuda
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build

Run:

    .\build\Release\citlali_cli.exe --model C:\work\citlali\Qwen3-0.6B-Q4_K_M.gguf --prompt "你好，介绍一下你自己" --max-new-tokens 64

For this first version, use the 0.6B or 1.7B Q4_K model. The 4B model is recognized, but loading Q4_K as FP16 needs too much memory for an 8GB GPU once runtime buffers and KV cache are included.
