#pragma once

#include "citlali/compute/common/dtype.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace citlali::io
{
    enum class GgufMetadataType : uint32_t {
        Uint8 = 0,
        Int8 = 1,
        Uint16 = 2,
        Int16 = 3,
        Uint32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
        Uint64 = 10,
        Int64 = 11,
        Float64 = 12,
    };

    using GgufScalar = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
                                float, bool, std::string, uint64_t, int64_t, double>;
    // 标量的类型
    using GgufArray = std::variant<std::vector<uint8_t>, std::vector<int8_t>, std::vector<uint16_t>,
                                   std::vector<int16_t>, std::vector<uint32_t>, std::vector<int32_t>,
                                   std::vector<float>, std::vector<bool>, std::vector<std::string>,
                                   std::vector<uint64_t>, std::vector<int64_t>, std::vector<double>>;
    // 向量的类型
    using GgufValue = std::variant<GgufScalar, GgufArray>; // 值的类型是标量还是向量

    struct GgufTensorInfo {
        std::string name;
        std::vector<uint64_t> dims;
        compute::GgufTensorType type = compute::GgufTensorType::Unknown;
        uint64_t relative_offset = 0;
        uint64_t absolute_offset = 0;
        uint64_t nbytes = 0;

        uint64_t element_count() const;
    };

    class GgufFile
    {
        public:
            void load(const std::string& model_path);

            const std::string& path() const { return path_; }       // 模型路径
            uint32_t version() const { return version_; }           // 模型版本
            uint64_t alignment() const { return alignment_; }       // 文件内存对齐字节，量化权重块对齐要求（默认 32）
            const std::vector<GgufTensorInfo>& tensors() const { return tensors_; }     // 返回所有权重张量描述数组
            const std::unordered_map<std::string, GgufValue>& metadata() const { return metadata_; }   // 返回完整元数据(用于描述模型信息，不是权重)键值哈希表

            const GgufTensorInfo* find_tensor(const std::string& name) const;   // 张量查询，找到返回结构体指针，没找到返回nullptr
            bool has_key(const std::string& key) const;             // 判断元数据存在性

            // 读取key，如果key不存在，则返回fallback这个默认值
            std::string get_string(const std::string& key, const std::string& fallback = "") const;
            uint64_t get_u64(const std::string& key, uint64_t fallback = 0) const;
            int64_t get_i64(const std::string& key, int64_t fallback = 0) const;
            double get_f64(const std::string& key, double fallback = 0.0) const;
            float get_f32(const std::string& key, float fallback = 0.0f) const;
            bool get_bool(const std::string& key, bool fallback = false) const;


            // 读取数组类型元数据
            std::vector<std::string> get_string_array(const std::string& key) const;
            std::vector<float> get_f32_array(const std::string& key) const;
            std::vector<uint32_t> get_u32_array(const std::string& key) const;

            // 根据传入的张量信息（偏移、字节大小），从磁盘文件读取这段权重的原始二进制数据，返回字节数组，交给上层citlali::compute模块做量化解码、显存拷贝
            std::vector<uint8_t> read_tensor_bytes(const GgufTensorInfo& tensor) const;

        private:
            std::string path_;
            uint32_t version_ = 0;
            uint64_t alignment_ = 32;
            uint64_t data_offset_ = 0;
            std::vector<GgufTensorInfo> tensors_;
            std::unordered_map<std::string, GgufValue> metadata_;
            std::unordered_map<std::string, size_t> tensor_index_;
    };

    std::string metadata_value_to_string(const GgufValue& value);

}