cmake_minimum_required(VERSION 3.22)
project(cuda_mmm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set this to 90 or 90a for H100. If your cmake doesn't accept "90a", use 90.
# You can override at configure-time:
#   cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=90
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

add_executable(sgemm_bench
  src/main.cu
)

target_compile_options(sgemm_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(sgemm_bench PRIVATE CUDA::cublas)
