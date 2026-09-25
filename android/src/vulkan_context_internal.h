#pragma once

#include "citlali_remote/vulkan_context.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace citlali::remote::detail {

inline void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

inline std::vector<std::uint32_t> read_spirv(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("unable to open SPIR-V shader: " + path);
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || (size % 4) != 0) {
        throw std::runtime_error("invalid SPIR-V shader size: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / 4U);
    if (!input.read(reinterpret_cast<char*>(words.data()), size)) {
        throw std::runtime_error("failed to read SPIR-V shader: " + path);
    }
    return words;
}

inline std::string sibling_shader_path(const std::string& path,
                                const std::string& filename) {
    const std::size_t separator = path.find_last_of("/\\");
    return separator == std::string::npos
               ? filename
               : path.substr(0, separator + 1) + filename;
}

inline VkShaderModule create_shader_module(VkDevice device,
                                    const std::string& path) {
    const std::vector<std::uint32_t> words = read_spirv(path);
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = words.size() * sizeof(std::uint32_t);
    info.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device, &info, nullptr, &module),
          "vkCreateShaderModule");
    return module;
}

inline VkPipeline create_compute_pipeline(VkDevice device, VkShaderModule shader,
                                   VkPipelineLayout layout) {
    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader;
    stage_info.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   nullptr, &pipeline),
          "vkCreateComputePipelines");
    return pipeline;
}

inline std::uint32_t find_host_memory_type(
    VkPhysicalDevice physical_device, std::uint32_t type_bits) {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t selected = std::numeric_limits<std::uint32_t>::max();
    for (std::uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
        if ((type_bits & (1U << i)) == 0 ||
            (properties.memoryTypes[i].propertyFlags & required) != required) {
            continue;
        }
        selected = i;
        if ((properties.memoryTypes[i].propertyFlags &
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            break;
        }
    }
    if (selected == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("no host-visible coherent Vulkan memory type");
    }
    return selected;
}

} // namespace citlali::remote::detail
