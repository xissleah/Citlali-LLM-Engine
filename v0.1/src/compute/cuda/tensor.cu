#include "citlali/compute/cuda/tensor.h"

#include "citlali/compute/cuda/cuda_utils.h"

#include <cuda_runtime.h>

namespace citlali::compute {

    DeviceBuffer::DeviceBuffer(std::size_t bytes) {
        allocate(bytes);
    }
    // 创建一个 DeviceBuffer 对象时，传入需要分配的字节数，然后调用 allocate(bytes) 分配对应的设备内存

    DeviceBuffer::~DeviceBuffer() {
        reset();
    }
    // 析构函数，对象销毁时调用reset()

    DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), bytes_(other.bytes_) {
        other.data_ = nullptr;
        other.bytes_ = 0;
    }
    // 移动构造函数
    // noexcept 表示这个函数不会抛异常

    DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            // 这个if防止自己给自己移动
            reset();
            data_ = other.data_;
            bytes_ = other.bytes_;
            // 把other的所有权交给this
            other.data_ = nullptr;
            other.bytes_ = 0;
            // 清空other这个对象
        }
        return *this;
    }
    // 用于把一个已有对象的 GPU 内存所有权转移给另一个已有对象
    /*
    operator旁边跟了个= 相当于改变这个类型 赋值操作（=）的意思，使其原来赋值的功能得到改变，你也可以把=改成<，来重载<的作用使其不再是判断大小
    b = std::move(a);
    相当于把a的所有权转移给b
    b先销毁自己
    然后a转移给b
    a再销毁自己
    */

    void DeviceBuffer::allocate(std::size_t bytes) {
        reset();
        // 先释放旧资源
        if (bytes == 0) {
            return;
        }
        // 这个这玩意是空的就直接退出
        CITLALI_CUDA_CHECK(cudaMalloc(&data_, bytes));
        // 调用cuda分配设备内存
        bytes_ = bytes;
    }
    // 负责给 DeviceBuffer 分配 GPU 显存
    /*
    值得注意的是这个函数并不是把原来的数据迁移到新内存
    而是清空原来的内存，再重新开辟一块指定大小的新内存
    data_ 是存数据的指针
    */

    void DeviceBuffer::reset() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            bytes_ = 0;
        }
    }
    // 用于释放GPU内存

    void DeviceBuffer::copy_from_host(const void* source, std::size_t bytes) {
        CITLALI_CUDA_CHECK(cudaMemcpy(data_, source, bytes, cudaMemcpyHostToDevice));
    }
    // 把资源从CPU复制到GPU
    /*
    source指向CPU内存
    data_指向GPU内存
    把bytes 字节从CPU 复制到 GPU
    cudaMemcpyHostToDevice指复制方向
    CITLALI_CUDA_CHECK是错误检查宏
    */

    void DeviceBuffer::copy_to_host(void* target, std::size_t bytes) const {
        CITLALI_CUDA_CHECK(cudaMemcpy(target, data_, bytes, cudaMemcpyDeviceToHost));
    }
    // 从GPU 复制到 CPU
    // 加const是为了不改变DeviceBuffer本身
    // 上面那个函数是从source复制到data_，会改变DeviceBuffer的data_所指向的内容，所以上面没有const
    // 这里是从data_复制到source不会改变data_ 所以才加const

    DeviceBufferPtr make_device_buffer(std::size_t bytes) {
        return std::make_shared<DeviceBuffer>(bytes);
    }
    // 创建一个 DeviceBuffer，并返回一个共享指针

    DeviceBufferPtr make_device_half_buffer(std::size_t elements) {
        return make_device_buffer(elements * sizeof(uint16_t));
    }
    // 参数是元素数量，不是字节数，该函数是make_device_buffer函数的包装函数

}
