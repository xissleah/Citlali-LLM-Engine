cmake_minimum_required(VERSION 3.22)

project(CitlaliRemoteProbe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GLSLC_EXECUTABLE
    "${ANDROID_NDK}/shader-tools/windows-x86_64/glslc.exe"
    CACHE FILEPATH "Path to the NDK glslc executable"
)
if(NOT EXISTS "${GLSLC_EXECUTABLE}")
    message(FATAL_ERROR "glslc not found: ${GLSLC_EXECUTABLE}")
endif()

set(Q4K_MATVEC_SPV
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/q4k_matvec.spv"
)
set(Q6K_MATVEC_SPV
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/q6k_matvec.spv"
)
set(RMS_NORM_SPV
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/rms_norm.spv"
)
set(SWIGLU_SPV
    "${CMAKE_CURRENT_BINARY_DIR}/shaders/swiglu.spv"
)
set(HEAD_RMS_NORM_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders/head_rms_norm.spv")
set(ROPE_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders/rope.spv")
set(ATTENTION_SCORES_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders/attention_scores.spv")
set(ATTENTION_VALUE_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders/attention_value.spv")
set(RESIDUAL_ADD_SPV "${CMAKE_CURRENT_BINARY_DIR}/shaders/residual_add.spv")
add_custom_command(
    OUTPUT "${Q4K_MATVEC_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.1
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q4k_matvec.comp"
            -o "${Q4K_MATVEC_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q4k_matvec.comp"
    VERBATIM
)
add_custom_command(
    OUTPUT "${Q6K_MATVEC_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.1
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q6k_matvec.comp"
            -o "${Q6K_MATVEC_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/q6k_matvec.comp"
    VERBATIM
)
add_custom_command(
    OUTPUT "${RMS_NORM_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.1
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rms_norm.comp"
            -o "${RMS_NORM_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/rms_norm.comp"
    VERBATIM
)
add_custom_command(
    OUTPUT "${SWIGLU_SPV}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${CMAKE_CURRENT_BINARY_DIR}/shaders"
    COMMAND "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.1
            -O
            "${CMAKE_CURRENT_SOURCE_DIR}/shaders/swiglu.comp"
            -o "${SWIGLU_SPV}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/swiglu.comp"
    VERBATIM
)
foreach(SHADER_NAME IN ITEMS head_rms_norm rope attention_scores attention_value residual_add)
    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
        COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.1 -O
                "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.comp"
                -o "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}.spv"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}.comp"
        VERBATIM
    )
endforeach()
add_custom_target(citlali_remote_shaders DEPENDS
    "${Q4K_MATVEC_SPV}"
    "${Q6K_MATVEC_SPV}"
    "${RMS_NORM_SPV}"
    "${SWIGLU_SPV}"
    "${HEAD_RMS_NORM_SPV}"
    "${ROPE_SPV}"
    "${ATTENTION_SCORES_SPV}"
    "${ATTENTION_VALUE_SPV}"
    "${RESIDUAL_ADD_SPV}"
)

add_executable(citlali_remote_probe
    src/main.cpp
)

add_executable(citlali_remote_server
    src/server_main.cpp
    src/protocol.cpp
    src/tcp_server.cpp
    src/vulkan_context.cpp
    src/vulkan_context_layer.cpp
    src/vulkan_context_tensor.cpp
)
add_dependencies(citlali_remote_server citlali_remote_shaders)

find_library(VULKAN_LIBRARY vulkan REQUIRED)

target_link_libraries(citlali_remote_probe PRIVATE
    ${VULKAN_LIBRARY}
)

target_link_libraries(citlali_remote_server PRIVATE
    ${VULKAN_LIBRARY}
)

target_include_directories(citlali_remote_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(citlali_remote_probe PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_options(citlali_remote_server PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
