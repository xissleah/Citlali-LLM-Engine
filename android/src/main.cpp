#include <vulkan/vulkan.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string version_string(std::uint32_t version) {
    return std::to_string(VK_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_VERSION_MINOR(version)) + "." +
           std::to_string(VK_VERSION_PATCH(version));
}
//  读取vulkan版本号

const char* device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated GPU";
        //  集显
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete GPU";
        //  独显
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual GPU";
        //  虚拟显卡
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "CPU";
        //  CPU
        default: return "other";
    }
}
//  枚举物理设备类型

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}
//  错误检查工具


}

int main() {
    VkInstance instance = VK_NULL_HANDLE;
    // 创建vulkan instance，先将输出句柄初始化为空，表示当前没有有效实例
    try {
        std::uint32_t loader_version = VK_API_VERSION_1_0;
        // 默认Vulkan版本 1.0
        const auto enumerate_instance_version =
        reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
        // 动态查找版本查询函数，传入 VK_NULL_HANDLE 表示此时还没有创建 VkInstance，查询的是 loader 层面支持的实例 API 版本

        if (enumerate_instance_version != nullptr) {
            check(enumerate_instance_version(&loader_version),
                  "vkEnumerateInstanceVersion");
            // 检查vulkan版本是否合法，并把 loader_version 改成对应的版本
        }
        // vkEnumerateInstanceVersion是1.1版本之后才有的

        VkApplicationInfo application_info{};
        // 配置信息结构体初始化
        application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        // Vulkan 的多数结构体都要求正确填写 sType
        application_info.pApplicationName = "Citlali Remote";
        // Remote功能名称
        application_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        // Remote功能版本
        application_info.pEngineName = "Citlali";
        // 引擎名字
        application_info.engineVersion = VK_MAKE_VERSION(0, 2, 0);
        // 引擎版本
        application_info.apiVersion = loader_version;
        // vulkan api版本

        VkInstanceCreateInfo instance_info{};
        // 初始化配置信息
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        // 结构体类型
        instance_info.pApplicationInfo = &application_info;
        // 关联应用信息
        /*
        当前等价于
        instance_info.flags = 0;
        instance_info.pNext = nullptr;
        instance_info.enabledLayerCount = 0;
        instance_info.ppEnabledLayerNames = nullptr;
        instance_info.enabledExtensionCount = 0;
        instance_info.ppEnabledExtensionNames = nullptr;
        */

        check(vkCreateInstance(&instance_info, nullptr, &instance),
              "vkCreateInstance");
        // 创建实例并检查错误
        /*
        &instance_info  	创建实例所需的配置
        nullptr             使用默认 Vulkan 内存分配器
        &instance           ulkan 创建成功后写入实例句柄
        */

        const auto get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2"));
        // 获取拓展版查询函数，vkGetPhysicalDeviceProperties2用于查询物理设备属性
        const auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2"));
        // vkGetPhysicalDeviceFeatures2用于查询 GPU 支持的 Vulkan feature

        std::uint32_t device_count = 0;
        // 初始化网络设备数量为0
        check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
              "vkEnumeratePhysicalDevices(count)");
        // 查询物理设备数量
        /*
        第一次调用时
        pPhysicalDevices == nullptr 表示只查询设备数量，不获取设备句柄
        调用成功后
        device_count 会保存系统中可用的vulkan物理设备数量
        */
        if (device_count == 0) {
            throw std::runtime_error("no Vulkan physical device found");
        }
        // 没找到就抛异常

        std::vector<VkPhysicalDevice> devices(device_count);
        // 分配物理设备数组，根据第一次查询得到的数量，分配一个数组，用于保存设备句柄VkPhysicalDevice
        // 需要注意的是：VkPhysicalDevice 不是应用创建出来的逻辑设备，它只是 Vulkan 返回的物理设备引用
        check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
              "vkEnumeratePhysicalDevices(devices)");
        // 获取所有物理设备
        /*
        第二次调用时传入device.data()
        vulkan会把找到的物理设备写入数组devices
        */

        std::cout << "Citlali Android Vulkan probe\n";
        std::cout << "Vulkan loader: " << version_string(loader_version) << "\n";
        std::cout << "Physical devices: " << device_count << "\n";
        // 打印探测信息

        for (std::uint32_t device_index = 0; device_index < device_count; ++device_index)
        {
            const VkPhysicalDevice device = devices[device_index];
            // 遍历物理设备

            VkPhysicalDeviceSubgroupProperties subgroup{};
            // 准备subgroup结构体，用于查询subgroup能力
            subgroup.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

            VkPhysicalDeviceProperties2 properties{};
            // 准备基础属性结构体
            properties.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

            if (get_properties2 != nullptr) {
                properties.pNext = &subgroup;
                get_properties2(device, &properties);
            }
            else {
                vkGetPhysicalDeviceProperties(device, &properties.properties);
            }
            // 使用扩展查询接口，如果找到了vkGetPhysicalDeviceProperties2，就通过pNext链同时查询 基础属性+subgroup属性
            // 如果没有就回退到旧的vulkan1.0 api的方式

            VkPhysicalDeviceShaderFloat16Int8Features float16{};
            float16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
            // 准备 Float16 / Int8 feature 方便查询设备是否支持shader中的fp16运算能力

            VkPhysicalDevice16BitStorageFeatures storage16{};
            storage16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
            storage16.pNext = &float16;
            // 准备 16-bit storage feature
            /*
            这里构造了如下 pNext 链
            VkPhysicalDeviceFeatures2
                pNext
                    ↓
            VkPhysicalDevice16BitStorageFeatures
                pNext
                    ↓
            VkPhysicalDeviceShaderFloat16Int8Features

            其中storage16.storageBuffer16BitAccess表示 shader 是否可以在 storage buffer 中使用 16-bit 类型
            storage16.uniformAndStorageBuffer16BitAccess表示 uniform buffer 和 storage buffer 是否都支持 16-bit 类型访问
            */

            VkPhysicalDeviceFeatures2 features{};
            features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            // 结果结构体
            if (get_features2 != nullptr) {
                features.pNext = &storage16;
                get_features2(device, &features);
            }
            // vkGetPhysicalDeviceFeatures2可用时，把额外的feature查询结构接到features后面
            else {
                vkGetPhysicalDeviceFeatures(device, &features.features);
                // 没有 vkGetPhysicalDeviceFeatures2 时的回退
            }

            std::uint32_t queue_count = 0;
            // 查询 queue family数量
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
            // 查询当前gpu有多少个queue family
            std::vector<VkQueueFamilyProperties> queues(queue_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, queues.data());
            // 获取每个 queue family的属性
            /*
            每个 VkQueueFamilyProperties 中包含：
            queueFlags
            queueCount
            timestampValidBits
            minImageTransferGranularity
            其中最关心的是：
            queues[queue_index].queueFlags
            */

            std::cout << "\n[device " << device_index << "]\n";
            std::cout << "name: " << properties.properties.deviceName << "\n";
            // 打印设备名称
            std::cout << "type: "
                      << device_type_name(properties.properties.deviceType)
                      << "\n";
            // 设备类型
            std::cout << "api: "
                      << version_string(properties.properties.apiVersion) << "\n";
            // 设备支持的vulkan api版本
            std::cout << "driver: "
                      << version_string(properties.properties.driverVersion)
                      << "\n";
            // 驱动版本
            std::cout << "vendor_id: 0x" << std::hex
                      << properties.properties.vendorID << std::dec << "\n";
            // Vendor id
            std::cout << "device_id: 0x" << std::hex
                      << properties.properties.deviceID << std::dec << "\n";
            // Device id
            std::cout << "max_storage_buffer_bytes: "
                      << properties.properties.limits.maxStorageBufferRange
                      << "\n";
            // 最大 storage buffer范围
            std::cout << "max_compute_workgroup_invocations: "
                      << properties.properties.limits.maxComputeWorkGroupInvocations
                      << "\n";
            // 最大compute workgroup invocation数，表示一个 compute workgroup 中最多允许的 invocation 数量
            std::cout << "subgroup_size: " << subgroup.subgroupSize << "\n";
            // 表示一个 Vulkan subgroup 中包含多少个 shader invocation
            std::cout << "shader_float16: "
                      << (float16.shaderFloat16 ? "yes" : "no") << "\n";
            // 表示 shader 是否支持使用 16 位浮点数
            std::cout << "storage_buffer_16bit: "
                      << (storage16.storageBuffer16BitAccess ? "yes" : "no")
                      << "\n";
            // 表示 shader 是否可以在 storage buffer 中访问 16 位类型
            std::cout << "uniform_and_storage_buffer_16bit: "
                      << (storage16.uniformAndStorageBuffer16BitAccess ? "yes" : "no")
                      << "\n";
            // 表示 shader 是否可以在uniform buffer/storage buffer中访问 16 位数据

            bool found_compute_queue = false;
            for (std::uint32_t queue_index = 0; queue_index < queue_count; ++queue_index) {
                if ((queues[queue_index].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
                    continue;
                }
                found_compute_queue = true;
                std::cout << "compute_queue_family: " << queue_index
                          << " queues=" << queues[queue_index].queueCount
                          << " timestamp_bits="
                          << queues[queue_index].timestampValidBits << "\n";
                /*
                输出的信息包括：
                queue family 索引
                该 family 中有多少条 queue
                timestamp 的有效 bit 数
                */
            }
            // 检查当前 queue family是否支持compute
            std::cout << "compute_queue_available: "
                      << (found_compute_queue ? "yes" : "no") << "\n";
            // 表示该物理设备是否至少有一个 compute queue family
        }
        // 逐个检查每个 VkPhysicalDevice，输出 GPU 的属性、feature、subgroup 信息以及 compute queue 信息
        /*
        遍历所有物理设备
            ↓
        查询设备属性
            ↓
        查询设备 feature
            ↓
        查询 queue family
            ↓
        打印 GPU 能力
        */

        vkDestroyInstance(instance, nullptr);
        return 0;
        /*
        前面的 Vulkan probe 全部执行成功
        销毁之前创建的 Vulkan 实例
        退出
        */
    }
    catch (const std::exception& error) {
        std::cerr << "probe failed: " << error.what() << "\n";
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }
        // 销毁之前创建的 Vulkan 实例并退出
        return 1;
    }
    // 异常路径
}
