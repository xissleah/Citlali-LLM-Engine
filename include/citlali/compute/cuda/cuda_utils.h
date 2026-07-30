// cuda错误检查工具
#pragma once

#include "citlali/common.h"

#include <cuda_runtime.h>

#include <string>

namespace citlali::compute {
    /*
    status：CUDA 函数返回的错误码 cudaError_t
    expression：出错那行代码的字符串（比如 cudaMalloc）
    file：当前代码文件路径
    line：出错代码行号
     */
    inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
        if (status != cudaSuccess) {
            throw Error(std::string("CUDA error at ") + file + ":" + std::to_string(line) +
                        " for " + expression + ": " + cudaGetErrorString(status));
        }
    }

} // namespace citlali::compute

#define CITLALI_CUDA_CHECK(expr) ::citlali::compute::check_cuda((expr), #expr, __FILE__, __LINE__)
//  宏定义函数
//  (expr)：执行 CUDA 函数，拿到错误码
//  #expr：把传入的代码转成字符串字面量
//  __FILE__：编译器内置宏，当前文件名
//  __LINE__：编译器内置宏，当前代码行号
