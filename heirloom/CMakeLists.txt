# Heirloom's C-ABI shim over llama.cpp (fork delta — not upstreamed). Builds heirloom_llama.dll
# next to llama.dll so .NET P/Invoke finds it + the ggml/cuda backends in one directory.
add_library(heirloom_llama SHARED heirloom_llama.cpp)
target_link_libraries(heirloom_llama PRIVATE llama mtmd)
target_include_directories(heirloom_llama PRIVATE "${CMAKE_SOURCE_DIR}/tools/mtmd")
set_target_properties(heirloom_llama PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if (LLAMA_BUILD_TESTS)
    enable_testing()
    add_executable(test-heirloom-sampler test-heirloom-sampler.cpp)
    target_link_libraries(test-heirloom-sampler PRIVATE llama)
    add_test(NAME test-heirloom-sampler COMMAND $<TARGET_FILE:test-heirloom-sampler>)
    set_property(TEST test-heirloom-sampler PROPERTY LABELS "heirloom")
endif()
