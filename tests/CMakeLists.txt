project (test)

add_executable(test test.hip)
target_link_libraries(test PRIVATE hip_ds)
