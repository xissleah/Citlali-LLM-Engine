// Standalone micro-benchmark: compares the engine's current matvec design against
// a vectorized warp-per-row design, on the exact shapes Qwen3-0.6B uses.
// Not part of the engine target; build with:
//   nvcc -O3 -std=c++17 -arch=sm_120 -o matvec_bench.exe matvec_bench.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); std::exit(1); } } while (0)

// ---------------------------------------------------------------------------
// A. Current engine design: one block per output row, 256 threads,
//    scalar 2-byte loads, shared-memory tree reduction (8 barriers).
//    Mirrors citlali::compute::matvec_fp16_kernel in src/compute/cuda/kernels.cu
// ---------------------------------------------------------------------------
__global__ void matvec_current(const uint16_t* __restrict__ w,
                               const uint16_t* __restrict__ x,
                               uint16_t* __restrict__ y, int K) {
    __shared__ float scratch[256];
    const int row = blockIdx.x;
    float sum = 0.0f;
    const uint16_t* row_weight = w + (size_t)row * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sum += __half2float(reinterpret_cast<const half*>(row_weight)[i])
             * __half2float(reinterpret_cast<const half*>(x)[i]);
    }
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) reinterpret_cast<half*>(y)[row] = __float2half(scratch[0]);
}

// ---------------------------------------------------------------------------
// B. Reduction-only change: still one block per row and still scalar 2-byte
//    loads, but the 8-barrier shared-memory tree is replaced by warp shuffles.
//    Isolates "reduction" from "memory access" so the two causes can be told apart.
// ---------------------------------------------------------------------------
__global__ void matvec_shfl(const uint16_t* __restrict__ w,
                            const uint16_t* __restrict__ x,
                            uint16_t* __restrict__ y, int K) {
    const int row = blockIdx.x;
    float sum = 0.0f;
    const uint16_t* row_weight = w + (size_t)row * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        sum += __half2float(reinterpret_cast<const half*>(row_weight)[i])
             * __half2float(reinterpret_cast<const half*>(x)[i]);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);

    __shared__ float wsum[32];
    const int warp = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) wsum[warp] = sum;
    __syncthreads();                       // single barrier, not eight
    if (warp == 0) {
        const int nwarp = blockDim.x >> 5;
        float v = (threadIdx.x < nwarp) ? wsum[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) reinterpret_cast<half*>(y)[row] = __float2half(v);
    }
}

// ---------------------------------------------------------------------------
// C. One warp per output row, uint4 (8 halves) loads, shuffle reduction,
//    x staged once per block in shared memory. 8 rows per 256-thread block.
// ---------------------------------------------------------------------------
__global__ void matvec_warp_vec(const uint16_t* __restrict__ w,
                                const uint16_t* __restrict__ x,
                                uint16_t* __restrict__ y, int K) {
    extern __shared__ uint4 xs[];               // K/8 uint4 = staged input vector
    const int nvec = K >> 3;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        xs[i] = reinterpret_cast<const uint4*>(x)[i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint4* rw = reinterpret_cast<const uint4*>(w + (size_t)row * K);

    float acc = 0.0f;
    for (int i = lane; i < nvec; i += 32) {
        const uint4 a = __ldg(rw + i);
        const uint4 b = xs[i];
        const half2* a2 = reinterpret_cast<const half2*>(&a);
        const half2* b2 = reinterpret_cast<const half2*>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __half22float2(a2[j]);
            const float2 fb = __half22float2(b2[j]);
            acc = fmaf(fa.x, fb.x, acc);
            acc = fmaf(fa.y, fb.y, acc);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    if (lane == 0) reinterpret_cast<half*>(y)[row] = __float2half(acc);
}

// ---------------------------------------------------------------------------
// D. Pure streaming-read bandwidth reference (what this GPU can actually do).
// ---------------------------------------------------------------------------
__global__ void stream_read(const uint4* __restrict__ p, float* __restrict__ sink, size_t nvec) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (; i < nvec; i += stride) {
        const uint4 v = __ldg(p + i);
        const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h[j]);
            acc += f.x + f.y;
        }
    }
    if (acc == 12345.678f) sink[0] = acc;   // never true; keeps the loop alive
}

struct Shape { int rows; int K; const char* name; };

int main() {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s  SMs=%d  clock=%.2f GHz  L2=%.1f MiB  mem=%.2f GB/s (spec)\n\n",
                prop.name, prop.multiProcessorCount, prop.clockRate / 1e6,
                prop.l2CacheSize / 1048576.0,
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    const Shape shapes[] = {
        {1024,   1024, "attn_k/v   (1024x1024)"},
        {2048,   1024, "attn_q     (2048x1024)"},
        {3072,   1024, "ffn_gate/up(3072x1024)"},
        {1024,   3072, "ffn_down   (1024x3072)"},
        {151936, 1024, "output proj(151936x1024)"},
    };

    // ---- streaming bandwidth ceiling ----
    {
        const size_t bytes = 512ull << 20;
        void* buf = nullptr;
        CK(cudaMalloc(&buf, bytes));
        CK(cudaMemset(buf, 1, bytes));
        float* sink = nullptr;
        CK(cudaMalloc(&sink, sizeof(float)));
        cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        const size_t nvec = bytes / sizeof(uint4);
        stream_read<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)buf, sink, nvec);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < 10; ++i) stream_read<<<prop.multiProcessorCount * 8, 256>>>((const uint4*)buf, sink, nvec);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
        std::printf("streaming-read ceiling: %.1f GB/s (%.1f MiB in %.2f ms)\n",
                    (double)bytes * 10.0 / (ms / 1000.0) / 1e9, bytes / 1048576.0, ms / 10);
        CK(cudaFree(buf)); CK(cudaFree(sink));
    }

    std::printf("\n%-24s %9s %9s %9s %9s %8s\n", "shape", "current", "shfl-only",
                "warp+vec", "vec gain", "cur GB/s");
    std::printf("%s\n", "--------------------------------------------------------------------------------------------");

    for (const Shape& s : shapes) {
        const size_t elems = (size_t)s.rows * s.K;
        const size_t bytes = elems * sizeof(uint16_t);
        uint16_t *w = nullptr, *x = nullptr, *y1 = nullptr, *y2 = nullptr, *y3 = nullptr;
        CK(cudaMalloc(&w, bytes));
        CK(cudaMalloc(&x, (size_t)s.K * sizeof(uint16_t)));
        CK(cudaMalloc(&y1, (size_t)s.rows * sizeof(uint16_t)));
        CK(cudaMalloc(&y2, (size_t)s.rows * sizeof(uint16_t)));
        CK(cudaMalloc(&y3, (size_t)s.rows * sizeof(uint16_t)));
        std::vector<uint16_t> host(elems);
        for (size_t i = 0; i < elems; ++i) host[i] = (uint16_t)((i * 2654435761u) >> 16);
        CK(cudaMemcpy(w, host.data(), bytes, cudaMemcpyHostToDevice));
        std::vector<uint16_t> hx((size_t)s.K);
        for (int i = 0; i < s.K; ++i) hx[i] = (uint16_t)((i * 40503u) >> 8);
        CK(cudaMemcpy(x, hx.data(), (size_t)s.K * 2, cudaMemcpyHostToDevice));

        const int iters = s.rows > 100000 ? 5 : 200;

        cudaEvent_t a, b; float ms1 = 0, ms2 = 0, ms3 = 0;
        CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
        matvec_current<<<s.rows, 256>>>(w, x, y1, s.K);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) matvec_current<<<s.rows, 256>>>(w, x, y1, s.K);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms1, a, b)); ms1 /= iters;

        matvec_shfl<<<s.rows, 256>>>(w, x, y3, s.K);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) matvec_shfl<<<s.rows, 256>>>(w, x, y3, s.K);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms3, a, b)); ms3 /= iters;

        const int smem = (s.K >> 3) * (int)sizeof(uint4);
        matvec_warp_vec<<<s.rows / 8, 256, smem>>>(w, x, y2, s.K);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(a));
        for (int i = 0; i < iters; ++i) matvec_warp_vec<<<s.rows / 8, 256, smem>>>(w, x, y2, s.K);
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms2, a, b)); ms2 /= iters;

        // correctness spot-check across all three variants
        std::vector<uint16_t> h1(s.rows), h2(s.rows), h3(s.rows);
        CK(cudaMemcpy(h1.data(), y1, (size_t)s.rows * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), y2, (size_t)s.rows * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h3.data(), y3, (size_t)s.rows * 2, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < s.rows; ++r) {
            const float a1 = __half2float(*(half*)&h1[r]);
            const float a2 = __half2float(*(half*)&h2[r]);
            const float a3 = __half2float(*(half*)&h3[r]);
            if (fabsf(a1 - a2) > 0.05f * (fabsf(a1) + 1.0f)) ++bad;
            if (fabsf(a1 - a3) > 0.05f * (fabsf(a1) + 1.0f)) ++bad;
        }

        std::printf("%-24s %7.1f us %7.1f us %7.1f us %8.2fx %8.1f   (bad rows=%d)\n",
                    s.name, ms1 * 1000, ms3 * 1000, ms2 * 1000, ms3 / ms2,
                    (double)bytes / (ms1 / 1000.0) / 1e9, bad);
        CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
        CK(cudaFree(w)); CK(cudaFree(x)); CK(cudaFree(y1)); CK(cudaFree(y2)); CK(cudaFree(y3));
    }
    CK(cudaDeviceReset());
    return 0;
}