cmake_minimum_required(VERSION 3.24)
project(CUDA_Learning VERSION 1.0 LANGUAGES CXX CUDA)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_SOURCE_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_SOURCE_DIR}")

add_executable(vectorAdd vectorAdd.cu)
add_executable(shared_memo shared_memo.cu)
add_executable(hello hello.cu)
add_executable(matrixAdd matrixAdd.cu)
add_executable(copyMat copyMat.cu)
add_executable(matTranspose matTranspose.cu)
add_executable(matMul matMul.cu)
add_executable(Tiling Tiling.cu)
add_executable(naiveVsTile naiveVsTile.cu)

set_target_properties(vectorAdd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(shared_memo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(hello PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(matrixAdd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(copyMat PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(matTranspose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(matMul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(Tiling PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(naiveVsTile PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)