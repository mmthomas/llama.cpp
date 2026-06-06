# Heirloom's C-ABI shim over llama.cpp (fork delta — not upstreamed). Builds heirloom_llama.dll
# next to llama.dll so .NET P/Invoke finds it + the ggml/cuda backends in one directory.
add_library(heirloom_llama SHARED heirloom_llama.cpp)
target_link_libraries(heirloom_llama PRIVATE llama)
set_target_properties(heirloom_llama PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
