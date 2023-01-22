cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
# if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.8)
#   cmake_policy(SET CMP0104 OLD)
# endif()


project(hip_ds LANGUAGES CXX HIP)

# Find hip
find_package(hip REQUIRED)

# C++ and HIP standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_HIP_STANDARD_REQUIRED TRUE)

set(CMAKE_HIP_FLAGS_DEBUG "-g -G")
set(CMAKE_HIP_FLAGS_RELEASE "-O3")


# Release mode is the default
if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()


set(CXX_FLAGS
        $<IF:$<CXX_COMPILER_ID:MSVC>,
        /std:c++20,
        --std=c++20
        >)

set(HIP_FLAGS
      -Rpass-analysis=kernel-resource-usage
   )


# Direct all output to /bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_library(hip_ds INTERFACE)

# options
option(build_benchmarks "Build benchmarks" OFF)
option(build_tests "Build tests" ON)


set(external_includes "${CMAKE_CURRENT_SOURCE_DIR}/external")
set(hip_ds_includes "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_compile_options(hip_ds INTERFACE
                        $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
                        $<$<COMPILE_LANGUAGE:HIP>:${HIP_FLAGS}>)

target_include_directories(hip_ds
                            INTERFACE ${external_includes})

target_link_libraries(hip_ds INTERFACE hip::device INTERFACE hip::host)

add_subdirectory(include)

if(build_tests)
   add_subdirectory(tests)
endif(build_tests)

if(build_benchmarks)
   add_subdirectory(benchmarks)
endif(build_benchmarks)
