cmake_minimum_required(VERSION 3.18)
project(naive_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(naive_gemm
  src/main.cu
  src/gemm.cu
)

target_include_directories(naive_gemm PRIVATE include)

set_target_properties(naive_gemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Optional: set a specific GPU architecture (uncomment one)
# set(CMAKE_CUDA_ARCHITECTURES 75)  # T4
# set(CMAKE_CUDA_ARCHITECTURES 80)  # A100
# set(CMAKE_CUDA_ARCHITECTURES 90)  # H100
