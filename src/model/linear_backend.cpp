#include "citlali/model/linear_backend.h"

#include "citlali/common.h"
#include "citlali/compute/cuda/kernels.h"

#include <limits>

namespace citlali::model {
    namespace {

        int as_int(uint64_t value, const std::string& label) {
            require(value <= static_cast<uint64_t>(std::numeric_limits<int>::max()), label + " is too large for this minimal kernel path");
            return static_cast<int>(value);
        }
        // 安全地把u64转为int

        class Fp16LinearBackend final : public LinearBackend {
        public:
            void forward(const WeightHandle& weight,    // 权重对象
                         const compute::DeviceBufferPtr& input, // 输入向量
                         const compute::DeviceBufferPtr& output // 输出向量
                         ) const override
            // 线性层的前向计算接口
            {
                require(input && output, "linear backend received a null buffer");
                // 检查输入/输出缓冲区是否是空指针
                if (weight.storage_kind == compute::WeightStorageKind::QuantizedDevice)
                {
                    switch (weight.gguf_type)
                    {
                    case(compute::GgufTensorType::Q4_K):
                        compute::launch_matvec_q4k(weight.device_quantized_data(),
                                                    input->half_data(),
                                                    output->half_data(),
                                                    as_int(weight.cols(), weight.name + " cols"),
                                                    as_int(weight.rows(), weight.name + " rows"));
                        return;
                    case(compute::GgufTensorType::Q6_K):
                        compute::launch_matvec_q6k(weight.device_quantized_data(),
                                                    input->half_data(),
                                                    output->half_data(),
                                                    as_int(weight.cols(), weight.name + " cols"),
                                                    as_int(weight.rows(), weight.name + " rows"));
                        return;
                    default:
                        throw Error("unsupported weight storage kind for: " + weight.name);
                    }
                }
                else if (weight.storage_kind == compute::WeightStorageKind::Fp16Device){
                    compute::launch_matvec_fp16(weight.device_half_data(),
                                               input->half_data(),
                                               output->half_data(),
                                               as_int(weight.cols(), weight.name + " cols"),
                                               as_int(weight.rows(), weight.name + " rows"));
                }
                // 启动 FP16 矩阵-向量乘法
                else
                {
                   throw Error("unknown storage_kind");
                }
            }
        };

    }

    std::unique_ptr<LinearBackend> make_fp16_linear_backend() {
        return std::make_unique<Fp16LinearBackend>();
    }
    // 创建后端对象

}