cmake_minimum_required(VERSION 3.14)
project(local-meeting-assistant)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(ExternalProject)

# llama.cpp is built as ExternalProject instead of add_subdirectory
# because whisper.cpp and llama.cpp both bundle their own ggml library,
# causing symbol conflicts when built in the same CMake tree.
ExternalProject_Add(llama_external
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/extern/llama.cpp
        CMAKE_ARGS
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=Release
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_TOOLS=OFF
        -DLLAMA_BUILD_COMMON=OFF
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/llama_install
        BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/llama_install/lib/libllama.a
)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/llama_install/include)
add_library(llama STATIC IMPORTED)
set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/llama_install/lib/libllama.a
        INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/llama_install/include
)

# Only llm_engine.cpp sees llama.h headers to avoid ggml redefinition errors
# between whisper.cpp's ggml and llama.cpp's ggml
set_source_files_properties(src/llm_engine.cpp PROPERTIES
        INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/llama_install/include
)
add_subdirectory(extern/whisper.cpp)
add_executable(local-meeting-assistant main.cpp
        src/ring_buffer.cpp
        src/audio_capture.cpp
        src/llm_engine.cpp
        src/meeting_assistant.cpp
        src/config.cpp
        src/speaker_diarizer.cpp)

add_dependencies(local-meeting-assistant llama_external)

set(ONNX_DIR ${CMAKE_SOURCE_DIR}/extern/onnxruntime)
target_include_directories(local-meeting-assistant PRIVATE ${ONNX_DIR}/include)

target_include_directories(local-meeting-assistant PRIVATE src)
target_link_libraries(local-meeting-assistant PRIVATE whisper
        "-framework CoreAudio"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Foundation"
        "-framework Accelerate"
        ${CMAKE_BINARY_DIR}/llama_install/lib/libllama.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-base.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-cpu.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-metal.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-blas.a
        ${ONNX_DIR}/lib/libonnxruntime.dylib
)