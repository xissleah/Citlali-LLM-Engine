cmake_minimum_required(VERSION 3.28)

project(CitlaliLLMEngine VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(citlali_engine STATIC
    src/compute/common/dtype.cpp
    src/io/gguf_reader.cpp
    src/tokenizer/tokenizer.cpp
    src/model/qwen_config.cpp
    src/model/weight_loader.cpp
    src/model/linear_backend.cpp
    src/model/qwen_model.cpp
    src/runtime/session.cpp
    src/compute/cuda/tensor.cu
    src/compute/cuda/kernels.cu
        include/citlali/compute/common/dtype.h
        include/citlali/compute/common/dtype.h
        include/citlali/compute/common/dtype.h
        include/citlali/compute/cuda/cuda_utils.h
)

target_include_directories(citlali_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(citlali_engine PUBLIC CUDA::cudart)

target_compile_definitions(citlali_engine PUBLIC NOMINMAX)

if(MSVC)
    target_compile_options(citlali_engine PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8 /EHsc>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -Xcompiler=/EHsc>
    )
else()
    target_compile_options(citlali_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(citlali_cli src/cli/main.cpp)
target_link_libraries(citlali_cli PRIVATE citlali_engine)
if(MSVC)
    target_link_options(citlali_cli PRIVATE /ENTRY:wmainCRTStartup)
endif()
