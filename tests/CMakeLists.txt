project (test)

add_executable(test_lib test.hip)
target_link_libraries(test_lib PRIVATE template_lib)
