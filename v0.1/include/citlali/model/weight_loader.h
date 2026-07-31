#pragma once

#include "citlali/compute/common/dtype.h"
#include "citlali/compute/cuda/tensor.h"
#include "citlali/io/gguf_reader.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace citlali::model
{
    enum class WeightLoadPolicy
    {
        DequantizeToFp16,   // 反量化
        KeepQuantized       // 保持原来
    };

    struct WeightHandle
    {
        std::string name;                  // 权重张量名，如 blk.0.attn_q.weight
        compute::GgufTensorType gguf_type = compute::GgufTensorType::Unknown; // 原始GGUF量化类型 Q4_K/F16/BF16
        compute::WeightStorageKind storage_kind = compute::WeightStorageKind::Fp16Device; // 显存存储模式：FP16/量化压缩
        std::vector<uint64_t> dims;        // 权重形状 [列,行]
        compute::DeviceBufferPtr buffer;   // GPU显存缓冲区指针，存真实权重数据
        uint64_t element_count = 0;        // 总元素数量（dims相乘）

        uint64_t cols() const { return dims.empty() ? 0 : dims[0]; }
        // 权重矩阵列数，矩阵乘：输出维度
        uint64_t rows() const { return dims.size() < 2 ? 1 : dims[1]; }
        // 权重矩阵行数，矩阵乘：输入维度

        // const只读对象调用，返回只读FP16显存指针
        const uint16_t* device_half_data() const { return buffer ? buffer->half_data() : nullptr; }
        // 普通对象调用，可读写显存FP16数据
        uint16_t* device_half_data() { return buffer ? buffer->half_data() : nullptr; }
    };

    class WeightLoader
    {
        public:
        explicit WeightLoader(const io::GgufFile& gguf, WeightLoadPolicy policy = WeightLoadPolicy::DequantizeToFp16);
        // 接收已经解析完成的 GGUF 文件只读引用以及加载策略
        WeightHandle load_required(const std::string& name) const;
        // 加载必须权重
        WeightHandle load_optional(const std::string& name) const;
        // 加载可选权重

        private:
        const io::GgufFile &gguf_;
        std::vector<uint16_t> load_as_fp16(const io::GgufTensorInfo& tensor) const;
        // tensor的信息
        WeightLoadPolicy policy_ = WeightLoadPolicy::DequantizeToFp16;
        //权重加载策略
    };
}