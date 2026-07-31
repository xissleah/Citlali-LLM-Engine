#include "citlali/compute/common/dtype.h"

#include <cstring>

namespace citlali::compute
{
    std::string to_string(GgufTensorType type) {
        switch (type) {
        case GgufTensorType::F32: return "F32";
        case GgufTensorType::F16: return "F16";
        case GgufTensorType::Q4_0: return "Q4_0";
        case GgufTensorType::Q4_1: return "Q4_1";
        case GgufTensorType::Q5_0: return "Q5_0";
        case GgufTensorType::Q5_1: return "Q5_1";
        case GgufTensorType::Q8_0: return "Q8_0";
        case GgufTensorType::Q8_1: return "Q8_1";
        case GgufTensorType::Q2_K: return "Q2_K";
        case GgufTensorType::Q3_K: return "Q3_K";
        case GgufTensorType::Q4_K: return "Q4_K";
        case GgufTensorType::Q5_K: return "Q5_K";
        case GgufTensorType::Q6_K: return "Q6_K";
        case GgufTensorType::Q8_K: return "Q8_K";
        case GgufTensorType::BF16: return "BF16";
        default: return "Unknown";
        }
    }

    std::size_t gguf_type_block_size(GgufTensorType type) {
        switch (type) {
        case GgufTensorType::F32: return 1;
        case GgufTensorType::F16: return 1;
        case GgufTensorType::BF16: return 1;
        case GgufTensorType::Q8_0: return 32;
        case GgufTensorType::Q4_K: return 256;
        case GgufTensorType::Q6_K: return 256;
        default: return 0;
        }
    }
    //  返回指定 GGUF tensor 类型对应的量化 block 包含多少个元素
    //  F32、F16、BF16 都不是按量化 block 存储的，所以是1

    std::size_t gguf_type_block_bytes(GgufTensorType type) {
        switch (type) {
        case GgufTensorType::F32: return 4;
        case GgufTensorType::F16: return 2;
        case GgufTensorType::BF16: return 2;
        case GgufTensorType::Q8_0: return 2 + 32;
        case GgufTensorType::Q4_K: return 2 + 2 + 12 + 128;
        case GgufTensorType::Q6_K: return 128 + 64 + 16 + 2;
        default: return 0;
        }
    }
    // 返回每种 GGUF 类型一个存储 block 占用的字节数

    std::size_t gguf_tensor_nbytes(GgufTensorType type, std::size_t elements) {
        const std::size_t block = gguf_type_block_size(type);
        // 计算有多少个block
        const std::size_t bytes = gguf_type_block_bytes(type);
        // 计算一个block的大小
        if (block == 0 || bytes == 0) {
            return 0;
        }
        return ((elements + block - 1) / block) * bytes;
    }
    // 计算一个 GGUF tensor 在文件中需要占用多少字节

    uint16_t bfloat16_to_half_bits(uint16_t value) {
        uint32_t bits = (uint32_t)value << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return float_to_half_bits(f);
    }
    // bf16 转 fp16
    // 因为 BF16 正好等于 FP32 的高 16 位，所以把BF16转到32再传FP16

    float half_bits_to_float(uint16_t h)
    {
        const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;    // 符号位S (-1)^S
        uint32_t exponent = (h >> 10) & 0x1fu;                  // 指数E 2^(E-15)
        uint32_t mantissa = h & 0x03ffu;                        // 尾数M 1 + M/(x^10)     最后三个相乘
        // 这里使用0x的16进制数，是为了分别取符号位 5位指数和10位尾数，其实这里16进制完全可以写成2进制的，但是2进制要写很多个0
        /*
        关于一个数转FP16形式，举一个13.1415的例子
        首先正数的话sign位是1，反之为0
        然后13.1415介于2^3~2^4之间，这里取整数阶3
        所以12.1415 = 1.6426875 * 2^3
        因为真实指数为3，由上边的公式得指数位E是 3+15 = 18
        小数部分0.6426875，根据尾数位是10位得M = round(0.6426875 * 2^10) = 658
        最后三者换算成二进制拼接起来即可
        */
        uint32_t out;
        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                out = sign;
            }
            else
            {
                exponent = 1;
                // fp16非正规数是mantissa * 2^-24
                // 其中 -24 = (1 - 15) - 10，所以exponent初始化1
                while ((mantissa & 0x0400u) == 0)
                // FP16 非正规数没有隐藏的最高位1，需要把mantissa一直左移直到第11位是1完成规格化
                {
                    mantissa <<= 1;
                    // 左移相当于乘2
                    --exponent;
                    // 所以指数要-1
                }
                mantissa &= 0x03ffu;
                // 清除第11位的隐藏位1,0x03ffu就是01111111111
                exponent = exponent + (127 - 15);
                // FP16 偏置是 15，FP32 偏置是 127，所以要补偿
                out = sign | (exponent << 23) | (mantissa << 13);
                // 拼接，FP32 的尾数是23位，指数是8位
            }
        }
        else if (exponent == 31)
        {
            out = sign | 0x7f800000u | (mantissa << 13);
        }
        // 当exponent位31时，mantissa==0就是无穷，否则是NAN
        else
        // 常规数转换FP32
        {
            exponent = exponent + (127 - 15);
            out = sign | (exponent << 23) | (mantissa << 13);
        }
        float value;
        std::memcpy(&value, &out, sizeof(value));
        return value;
    }
    // 把 FP16 的 16 位 bit 表示转换成 float

    uint16_t float_to_half_bits(float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));

        const uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
        uint32_t mantissa = bits & 0x007fffffu;
        // 提取三要素

        if (exponent <= 0) {
            if (exponent < -10) {
                return (uint16_t)sign;
            }
            // 太小了直接为0
            mantissa |= 0x00800000u;
            // 给尾数补上隐藏最高位1
            const uint32_t shift = (uint32_t)(14 - exponent);
            // 根据指数确定要右移多少位
            uint32_t half_mantissa = mantissa >> shift;
            // 压缩
            if (shift > 0 && ((mantissa >> (shift - 1)) & 1u)) {
                ++half_mantissa;
            }
            // 如果被丢弃部分最高位是1就进1
            return (uint16_t)(sign | half_mantissa);
            // 非正规数没有单独的指数位
        }

        if (exponent >= 31) {
            return (uint16_t)(sign | 0x7c00u);
        }
        // 无穷大的指数位全是1

        uint32_t half = sign | ((uint32_t)exponent << 10) | (mantissa >> 13);
        // 常规数的转换
        if (mantissa & 0x00001000u) {
            ++half;
        }
        // 被舍弃部分最高位是1就进1
        return (uint16_t)half;
    }
    // 大体是上一个函数的逆运算
}