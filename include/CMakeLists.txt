add_library(lib_src INTERFACE)
file(GLOB_RECURSE   lib_src
                    "*.h"
                    "*.hpp")
set(SOURCE_LIST ${lib_src})
target_sources(hip_ds INTERFACE ${lib_src})
target_include_directories(hip_ds INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")