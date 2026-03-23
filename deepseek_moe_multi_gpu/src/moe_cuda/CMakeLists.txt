cmake_minimum_required(VERSION 3.18)
project(DeepSeekMoE LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(moe_nccl main.cu moe_kernels.cuh nccl_utils.cuh)

target_include_directories(moe_nccl PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(moe_nccl
    CUDA::cudart
    nccl
)

set_target_properties(moe_nccl PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89;90"
)