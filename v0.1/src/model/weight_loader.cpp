#include "citlali/model/weight_loader.h"

#include "citlali/common.h"

#include <algorithm>
#include <cstring>
#include <iostream>


namespace citlali::model
{
    namespace
    {
        uint16_t read_u16(const uint8_t* p)
        {
            uint16_t v;
            std::memcpy(&v,p,sizeof(v));
            return v;
        }
        float read_f32(const uint8_t* p)
        {
            float v;
            std::memcpy(&v,p,sizeof(v));
            return v;
        }
        //从一段原始字节内存中，读取一个 uint16_t 或 float

        void get_q4k_scale_min(int j, const uint8_t* q,uint8_t& d, uint8_t& m)
        // j表示要读取第几个scale/min对
        {
            if (j < 4) {
                d = q[j] & 63;
                m = q[j + 4] & 63;
            } else {
                d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
                m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            }
        }
        // d = 解包后的 scale 编码
        // m = 解包后的 min 编码
        /*
        通常 Q4_K 块里有两类不同数据：
        scales 区域：保存各个子块的 scale 和 min
        qs 区域：保存真正的 4-bit 权重

        因为 Q4_K 为了节省空间，把 scale 和 min 压缩成 6 bit
        假设：
        8 个 scale：d0 ~ d7
        8 个 min：  m0 ~ m7
        总共：
        16 个值 × 6 bit = 96 bit = 12 字节
        所以它们被紧密地打包进 q[0] ~ q[11]

        数据布局大致是：
        q[0] ~ q[3]：
            保存 d0 ~ d3 的低6位
            同时保存 d4 ~ d7 的高2位

        q[4] ~ q[7]：
            保存 m0 ~ m3 的低6位
            同时保存 m4 ~ m7 的高2位

        q[8] ~ q[11]：
            保存 d4 ~ d7 的低4位
            保存 m4 ~ m7 的低4位

        提问：为什么不顺序存储？
        解答：这样不会跨字节，解码更简单
        */

        std::vector<uint16_t> dequant_f32(const std::vector<uint8_t>& bytes,uint64_t count)
        {
            std::vector<uint16_t> out(static_cast<size_t>(count));
            const float* source = reinterpret_cast<const float*>(bytes.data());
            for (uint64_t i =0 ;i < count; ++i)
            {
                out[static_cast<size_t>(i)] = compute::float_to_half_bits(source[i]);
            }
            return out;
        }
        // 把 bytes 中按 float32 存储的数，逐个转换成 IEEE 754 half/float16 的二进制位，并以 uint16_t 返回
        /*
        bytes
        ↓ 按 float32 解释
        float32
        ↓ float_to_half_bits()
        float16 的二进制编码
        ↓
        uint16_t
        */
        // 这样做通常是为了把 FP32 数据压缩成 FP16，减少内存和传输开销，并利用硬件的半精度计算能力; 当然，会掉精度，但是掉的精度微乎其微，
        // 可以忽略不计

        std::vector<uint16_t> dequant_f16(const std::vector<uint8_t>& bytes, uint64_t count)
        {
            std::vector<uint16_t> out(static_cast<size_t>(count));
            std::memcpy(out.data(), bytes.data(),static_cast<size_t>(count * sizeof(uint16_t)));
            return out;
        }
        // 读取已经存储为 FP16/half 的数据，并把它转换成程序内部使用的 uint16_t 数组
        // 也就是把两个bytes拼成一个fp16

        std::vector<uint16_t> dequant_bf16 (const std::vector<uint8_t>& bytes, uint64_t count)
        {
            std::vector<uint16_t> out(static_cast<size_t>(count));
            const uint16_t* source = reinterpret_cast<const uint16_t*>(bytes.data());
            // 和&vec[0]几乎等价，但data()更安全，vector为空的时候data()返回空指针，而&vec[0]会UB
            for (uint64_t i = 0; i < count; ++i)
            {
                out[static_cast<size_t>(i)] = compute::bfloat16_to_half_bits(source[i]);
            }
            return out;
        }
        // bf16->fp16 bf16可以表达比fp16更大的数，代价就是精度比fp16低

        std::vector<uint16_t> dequant_q8_0(const std::vector<uint8_t>& bytes, uint64_t count)
        {
            constexpr int qk = 32;
            // qk表示每个量化块包含多少个元素，Q8_0规定了32
            // constexpr 让变量、函数可以在编译期就算出结果，而不是等到程序运行时计算，结果编译时就确定，存入只读内存
            constexpr int block_bytes = 2+qk;
            std::vector<uint16_t> out(static_cast<size_t>(count));
            const uint64_t  blocks = (count + qk - 1) / qk;
            for (uint64_t b = 0; b < blocks; ++b)
            {
                const uint8_t* block = bytes.data() + b * block_bytes;
                // 计算block的偏移
                const float d = compute::half_bits_to_float(read_u16(block));
                // Q8_0规定前两个字节是scale
                const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
                // 跳过前两个字节
                // qs:量化后的数值数组
                for (int i = 0; i < qk; ++i)
                {
                    const uint64_t index = b*qk+i;
                    // 计算当前元素在out里的全局下标
                    if (index >= count) break;
                    // 防止最后一个block越界
                    out[static_cast<size_t>(index)] = compute::float_to_half_bits(d * static_cast<float>(qs[i]));
                    // Q8_0 反量化公式 x≈ d * q
                }
            }
            return out;
        }
        // 反量化 Q8_0 -> fp16

        std::vector<uint16_t> dequant_q4_k(const std::vector<uint8_t>& bytes, uint64_t count)
        {
            constexpr int qk = 256;
            constexpr int block_bytes = 2 + 2 + 12 + 128;
            // 每个block 144 字节
            /*
            2 字节：d_all
            2 字节：d_min_all
            12 字节：scales
            128 字节：4-bit 量化值 qs

            其中
            d_all      用于恢复局部 scale
            d_min_all  用于恢复局部 min
            */
            std::vector<uint16_t> out(static_cast<size_t>(count));
            const uint64_t blocks = (count+qk-1)/qk;
            for (uint64_t b = 0; b < blocks; ++b)
            {
                const uint8_t* block = bytes.data() + b * block_bytes;
                const float d_all = compute::half_bits_to_float(read_u16(block));
                const float d_min_all = compute::half_bits_to_float(read_u16(block+2));
                const uint8_t* scales = block + 4;
                const uint8_t* qs = block + 16;

                for (int super = 0 ; super < 4; ++super)
                // 一个 Q4_K block 中有 256 个元素，被分成 4 个大组
                {
                    uint8_t scale_low = 0;
                    uint8_t min_low = 0;
                    uint8_t scale_high = 0;
                    uint8_t min_high = 0;
                    const int scale_index = super * 2;
                    /*
                    每个 super 需要两个局部参数:
                    一个用于低 4 bit 数据
                    一个用于高 4 bit 数据

                    所以*2之后
                    super 0 → scale_index 0、1
                    super 1 → scale_index 2、3
                    super 2 → scale_index 4、5
                    super 3 → scale_index 6、7
                    */
                    get_q4k_scale_min(scale_index, scales, scale_low, min_low);
                    get_q4k_scale_min(scale_index + 1, scales, scale_high, min_high);
                    //  这2个辅助函数负责从 12 字节的压缩 scales 区域中解包出scale_low，min_low，scale_high，min_high

                    const float d_low = d_all * static_cast<float>(scale_low);
                    const float m_low = d_min_all * static_cast<float>(min_low);
                    const float d_high = d_all * static_cast<float>(scale_high);
                    const float m_high = d_min_all * static_cast<float>(min_high);
                    // 恢复实际的局部 scale 和 min
                    const uint8_t* q = qs + super*32;
                    // 定位当前 super 的量化数据

                    for (int i = 0; i < 32; ++i)
                    {
                        const uint64_t low_index = b*qk + super*64 + i;
                        /*
                        b * qk       当前 block 在整个数组中的起始位置
                        super * 64   当前 super-group 在 block 内的起始位置
                        i            当前字节在 super-group 内的偏移
                        */
                        const uint64_t high_index = low_index + 32;
                        // 一个字节拆出来两个权重，一个位置分配到low_index，一个在low_index+32
                        /*
                        这是量化格式的数据布局约定，主要是为了配合：
                            分组 scale
                            SIMD/GPU 并行处理
                            量化矩阵乘法
                            连续处理一整组 low 或 high 数据
                        */
                        if (low_index < count)
                        {
                            out[static_cast<size_t>(low_index)] = compute::float_to_half_bits(
                              d_low * static_cast<float>(q[i] & 0x0f) - m_low
                            );
                        }
                        // 设置两个if的原因是low_index没越界而high_index越界了
                        if (high_index < count)
                        {
                            out[static_cast<size_t>(high_index)] = compute::float_to_half_bits(
                            d_high * static_cast<float>(q[i] >> 4) - m_high
                            );
                        }
                    }
                }
            }
            return out;
        }
        // Q4_K 反量化

        std::vector<uint16_t> dequant_q6_k(const std::vector<uint8_t>& bytes,uint64_t count)
        {
            constexpr int qk = 256;
            constexpr int block_bytes = 128 + 64 + 16 + 2;
            std::vector<uint16_t> out(static_cast<size_t>(count));
            const uint64_t blocks = (count + qk - 1) / qk;

            for (uint64_t b = 0; b < blocks; ++b)
            {
                const uint8_t* block = bytes.data() + b * block_bytes;
                const uint8_t* ql = block;
                // ql 指向前 128 字节，保存量化值的低 4 bit
                const uint8_t* qh = block + 128;
                // qh 指向接下来的 64 字节，保存量化值的高 2 bit
                const int8_t* scales = reinterpret_cast<const int8_t*>(block + 128 + 64);
                // 跳过128 + 64 字节把后面的 16 字节解释为 16 个有符号 int8_t scale
                const float d = compute::half_bits_to_float(read_u16(block + 128 + 64 + 16));
                // 再跳过 16 字节 scale，读取最后 2 字节的 FP16 全局 scale
                /*
                Q6 使用 6 bit 表示一个量化值
                即6 bit = 4 bit 低位 + 2 bit 高位
                所以
                ql：128 字节
                qh：64 字节
                */
                for (int half = 0; half < 2; ++half)
                // 一个 block 有 256 个元素，被分成两个 half
                /*
                half = 0：元素 0～127
                half = 1：元素 128～255
                */
                {
                    const uint8_t* ql_half = ql + half * 64;
                    const uint8_t* qh_half = qh + half * 32;
                    const int8_t* sc = scales + half * 8;
                    // 当前 half 所使用的局部 scale 数组的起始地址
                    const int base = half * 128;
                    // 当前 half 在整个 block 输出数组中的起始偏移

                    for (int i = 0 ; i < 32 ; ++i)
                    // 每次循环解码出4个Q6量化值 q1 q2 q3 q4，循环32次
                    {
                        const int is = i / 16;
                        // i的范围是0~31 它的作用是用来选择scale
                        /*
                        当前 half 有 8 个 scale，四个量化值分别使用
                        q1 → sc[is + 0]
                        q2 → sc[is + 2]
                        q3 → sc[is + 4]
                        q4 → sc[is + 6]
                        */
                        const int q1 = static_cast<int>((ql_half[i] & 0x0f) | (((qh_half[i] >> 0) & 0x03) << 4)) - 32;
                        const int q2 = static_cast<int>((ql_half[i + 32] & 0x0f) | (((qh_half[i] >> 2) & 0x03) << 4)) - 32;
                        const int q3 = static_cast<int>((ql_half[i] >> 4) | (((qh_half[i] >> 4) & 0x03) << 4)) - 32;
                        const int q4 = static_cast<int>((ql_half[i + 32] >> 4) | (((qh_half[i] >> 6) & 0x03) << 4)) - 32;
                        // -32 是为了从无符号平移至有符号的范围，原先范围0~64，-32 后范围-32~32

                        const uint64_t idx1 = b * qk + base + i;
                        // idx1在out里的位置
                        const uint64_t idx2 = idx1 + 32;
                        const uint64_t idx3 = idx1 + 64;
                        const uint64_t idx4 = idx1 + 96;
                        if (idx1 < count)
                        {
                            out[static_cast<size_t>(idx1)] = compute::float_to_half_bits(d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1));
                        }
                        if (idx2 < count)
                        {
                            out[static_cast<size_t>(idx2)] = compute::float_to_half_bits(d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2));
                        }
                        if (idx3 < count)
                        {
                            out[static_cast<size_t>(idx3)] = compute::float_to_half_bits(d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3));
                        }
                        if (idx4 < count)
                        {
                            out[static_cast<size_t>(idx4)] = compute::float_to_half_bits(d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4));
                        }
                        // if检查四个idx是否越界
                    }
                }
            }
            return out;
        }
        // Q6_K 反量化
        /*
        反量化公式是： x ≈ d × scale × q
        d      整个 block 的 scale
        scale  局部 scale
        q      恢复出来的有符号 6-bit 量化值，范围大致是 -32～31

        ┌──────────┬─────────┬──────────┬────────┐
        │ ql 128B  │ qh 64B  │ scale16B │ d 2B   │
        └──────────┴─────────┴──────────┴────────┘
        */
    }

    WeightLoader::WeightLoader(
        const io::GgufFile& gguf,
        WeightLoadPolicy policy
        ) : gguf_(gguf), policy_(policy) {}
    // WeightLoader 的构造函数: 创建一个 WeightLoader 对象，并初始化它的两个成员变量gguf_和policy_

    WeightHandle WeightLoader::load_required(const std::string& name) const {
        const auto* tensor = gguf_.find_tensor(name);
        // 根据传入的名称查找 GGUF 文件中的张量
        require(tensor != nullptr, "missing required tensor: " + name);

        if (policy_ == WeightLoadPolicy::KeepQuantized) {
            throw Error("KeepQuantized policy is reserved for the later dynamic-dequant CUDA backend");
        }
        // 不允许保持量化格式

        std::vector<uint16_t> host = load_as_fp16(*tensor);
        // 将 tensor 转换为 FP16
        WeightHandle handle;
        // 这是一个权重句柄，用于保存权重的元数据和 GPU 缓冲区
        handle.name = name;
        handle.gguf_type = tensor->type;
        handle.storage_kind = compute::WeightStorageKind::Fp16Device;
        handle.dims = tensor->dims;
        handle.element_count = tensor->element_count();
        handle.buffer = compute::make_device_half_buffer(static_cast<size_t>(handle.element_count));
        handle.buffer->copy_from_host(host.data(), host.size() * sizeof(uint16_t));
        return handle;
    }
    // 根据张量名称加载一个必需的权重，并将其转换为 FP16 后拷贝到 GPU
    /*
    整体流程
    根据名称查找 GGUF tensor
            ↓
    确认 tensor 存在
            ↓
    拒绝 KeepQuantized 策略
            ↓
    将 tensor 转换为 FP16
            ↓
    在 GPU 上分配 FP16 buffer
            ↓
    将主机端 FP16 数据拷贝到 GPU
            ↓
    返回 WeightHandle
    */

    WeightHandle WeightLoader::load_optional(const std::string& name) const {
        const auto* tensor = gguf_.find_tensor(name);
        // 根据名称在 GGUF 文件中查找权重
        if (!tensor) return {};
        // 找不到时返回空句柄
        return load_required(name);
        // 找到时复用 load_required
    }
    // 用于加载一个可选权重

    std::vector<uint16_t> WeightLoader::load_as_fp16(const io::GgufTensorInfo& tensor) const {
        const std::vector<uint8_t> bytes = gguf_.read_tensor_bytes(tensor);
        // 从 GGUF 文件中读取该 tensor 的原始字节
        const uint64_t count = tensor.element_count();
        // 获取元素数量
        switch (tensor.type) {
            case compute::GgufTensorType::F32: return dequant_f32(bytes, count);
            case compute::GgufTensorType::F16: return dequant_f16(bytes, count);
            case compute::GgufTensorType::BF16: return dequant_bf16(bytes, count);
            case compute::GgufTensorType::Q8_0: return dequant_q8_0(bytes, count);
            case compute::GgufTensorType::Q4_K: return dequant_q4_k(bytes, count);
            case compute::GgufTensorType::Q6_K: return dequant_q6_k(bytes, count);
            default:
                throw Error("tensor type is not implemented by the FP16 loader yet: " + tensor.name + " type=" + compute::to_string(tensor.type));
        }
        // 根据原始类型选择转换函数
    }
    // 把 GGUF 中的任意支持类型统一转换成 FP16 格式
    /*
    具体流程
    读取 tensor 原始字节
            ↓
    获取元素数量
            ↓
    根据 tensor.type 判断原始格式
            ↓
    反量化/转换为 FP16
            ↓
    返回 FP16 原始位数据
    */
}
