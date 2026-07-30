#pragma once
// 防止头文件重复包含，等价于传统头文件守卫宏

#include <cstddef>    // 标准size_t类型
#include <cstdint>    // 定宽整数 uint32_t / uint16_t 等
#include <string>     // std::string 字符串，用于类型转文本输出

namespace citlali::compute {

/**
 * @brief GGUF模型文件支持的张量量化数据类型枚举
 * 对应GGUF标准里的tensor_type字段，用于区分权重存储精度/量化方案
 */
enum class GgufTensorType : uint32_t {
    F32 = 0,        // 32位单精度浮点数 float
    F16 = 1,        // 16位半精度浮点数 FP16
    Q4_0 = 2,       // 4bit量化，0阶量化方案
    Q4_1 = 3,       // 4bit量化，带偏移的1阶量化方案
    Q5_0 = 6,       // 5bit量化，0阶量化方案
    Q5_1 = 7,       // 5bit量化，带偏移的1阶量化方案
    Q8_0 = 8,       // 8bit量化，0阶量化方案
    Q8_1 = 9,       // 8bit量化，带偏移的1阶量化方案
    Q2_K = 10,      // K分组2bit量化
    Q3_K = 11,      // K分组3bit量化
    Q4_K = 12,      // K分组4bit量化（主流轻量化量化）
    Q5_K = 13,      // K分组5bit量化
    Q6_K = 14,      // K分组6bit量化
    Q8_K = 15,      // K分组8bit量化
    BF16 = 30,      // 16位脑浮点数 BF16
    Unknown = 0xffffffffu, // 未知/不识别的张量类型
};

/**
 * @brief 权重在计算设备上的存储形式分类
 */
enum class WeightStorageKind {
    Fp16Device,     // 设备显存中以FP16浮点存储（未量化）
    QuantizedDevice // 设备显存中以压缩量化块形式存储
};

/**
 * @brief 将GGUF张量类型枚举转为可读字符串
 * @param type GGUF张量数据类型
 * @return 类型名称文本，如"Q4_K"、"F16"
 */
std::string to_string(GgufTensorType type);

/**
 * @brief 获取该量化类型单个数据块包含多少个元素
 * @param type GGUF量化类型
 * @return 单个量化块内元素数量
 */
std::size_t gguf_type_block_size(GgufTensorType type);

/**
 * @brief 获取该量化类型单个数据块占用字节数
 * @param type GGUF量化类型
 * @return 单个量化块字节大小
 */
std::size_t gguf_type_block_bytes(GgufTensorType type);

/**
 * @brief 计算指定元素总数的张量占用总字节大小
 * @param type GGUF张量数据类型
 * @param elements 张量总元素个数
 * @return 张量整体字节数
 */
std::size_t gguf_tensor_nbytes(GgufTensorType type, std::size_t elements);

/**
 * @brief 32位float转换为FP16的二进制比特值(uint16_t)
 * @param value 输入单精度浮点数
 * @return FP16位存储的16位无符号整数
 */
uint16_t float_to_half_bits(float value);

/**
 * @brief FP16二进制比特值还原为32位float
 * @param bits FP16原始16位比特
 * @return 还原后的单精度浮点数
 */
float half_bits_to_float(uint16_t bits);

/**
 * @brief BF16 16位比特转标准FP16 16位比特
 * @param value BF16原始16位比特
 * @return 转换后的FP16比特数据
 */
uint16_t bfloat16_to_half_bits(uint16_t value);

} // namespace citlali::compute