cmake_minimum_required(VERSION 3.17)
project(primer CUDA CXX)

option(WITHTRACY "Defines whether tracy is to be used")

set(CMAKE_CUDA_STANDARD 14)

include(FindOpenMP)

add_executable(primer main.cu "tracy/TracyClient.cpp")
set_source_files_properties(tracy/TracyClient.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(primer PUBLIC tracy)
target_link_libraries(primer PUBLIC pthread dl cuda OpenMP::OpenMP_CXX)
set_target_properties(
        primer
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET
        primer
        PROPERTY
        CUDA_ARCHITECTURES 72)

if (WITHTRACY MATCHES "ON")
    target_compile_definitions(primer PUBLIC TRACY_ENABLE)
    message("Building with Tracy")
endif()