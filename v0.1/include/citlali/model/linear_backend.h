#pragma once

#include "citlali/compute/cuda/tensor.h"
#include "citlali/model/weight_loader.h"

#include <memory>

namespace citlali::model {

    class LinearBackend {
    public:
        virtual ~LinearBackend() = default;
        // 虚析构，如果这个类会被继承，则需要加virtual

        virtual void forward(const WeightHandle& weight,
                             const compute::DeviceBufferPtr& input,
                             const compute::DeviceBufferPtr& output) const = 0;
        // 纯虚函数，不能被实例化（也就是不能new），子类得重写他
    };

    std::unique_ptr<LinearBackend> make_fp16_linear_backend();

} // namespace citlali::model
