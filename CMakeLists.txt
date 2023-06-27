cmake_minimum_required(VERSION 3.16)
project(cuda_ntt LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(cuda_ntt main.cu helper.h ntt_30bits.cuh)

set_target_properties(cuda_ntt PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
