#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace citlali::compute
{
    class DeviceBuffer
    {
        public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(std::size_t bytes);   // 直接分配指定字节的gpu显存
        ~DeviceBuffer();                            //  自动释放gpu显存

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        // 禁用拷贝所有权（开销很大）

        DeviceBuffer(DeviceBuffer&& other) noexcept;
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
        // 允许移动所有权（开销小）

        void allocate(std::size_t bytes);   // 申请显存
        void reset();                       // 释放显存

        void copy_from_host(const void* source, std::size_t bytes);
        // 从CPU内存(source)拷贝数据到这块GPU显存
        void copy_to_host(void* target, std::size_t bytes) const;
        // 把GPU显存数据拷贝回CPU内存(target)，const只读，不修改显存

        // 返回对应信息
        void* data() { return data_; }
        const void* data() const { return data_; }
        std::size_t bytes() const { return bytes_; }

        // FP16 半精度专用指针，区分只读和可写
        uint16_t* half_data() { return static_cast<uint16_t*>(data_); }
        const uint16_t* half_data() const { return static_cast<const uint16_t*>(data_); }

        private:
        void* data_ = nullptr;
        std::size_t bytes_ = 0;

    };

    using DeviceBufferPtr = std::shared_ptr<DeviceBuffer>;
    // 使用智能指针管理这个对象
    DeviceBufferPtr make_device_buffer(std::size_t bytes);
    // 创建指定字节大小的GPU缓冲区，返回shared_ptr
    DeviceBufferPtr make_device_half_buffer(std::size_t elements);
    // 创建能存elements个FP16(float16)数字的显存缓冲区
    // 内部自动计算总字节：elements * sizeof(uint16_t)
}