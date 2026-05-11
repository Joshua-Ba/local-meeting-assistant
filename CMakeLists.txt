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

set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)
include_directories(${CMAKE_SOURCE_DIR}/extern/kaldi-native-fbank)
if(DEFINED BUILD_SHARED_LIBS)
        set(_LOCAL_MEETING_ASSISTANT_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        set(_LOCAL_MEETING_ASSISTANT_BUILD_SHARED_LIBS_WAS_DEFINED TRUE)
endif()
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(extern/kaldi-native-fbank EXCLUDE_FROM_ALL)
if(_LOCAL_MEETING_ASSISTANT_BUILD_SHARED_LIBS_WAS_DEFINED)
        set(BUILD_SHARED_LIBS ${_LOCAL_MEETING_ASSISTANT_BUILD_SHARED_LIBS})
else()
        unset(BUILD_SHARED_LIBS)
endif()
unset(_LOCAL_MEETING_ASSISTANT_BUILD_SHARED_LIBS)
unset(_LOCAL_MEETING_ASSISTANT_BUILD_SHARED_LIBS_WAS_DEFINED)

set(ONNX_DIR ${CMAKE_SOURCE_DIR}/extern/onnxruntime)

set(LOCAL_MEETING_ASSISTANT_SOURCES
        src/ring_buffer.cpp
        src/audio_capture.cpp
        src/llm_engine.cpp
        src/meeting_assistant.cpp
        src/config.cpp
        src/wav_io.cpp
        src/speaker_diarizer.cpp
        src/MelFeatureExtractor.cpp
        src/ahc_pyannote.cpp
)

add_library(local-meeting-assistant-core STATIC ${LOCAL_MEETING_ASSISTANT_SOURCES})
add_dependencies(local-meeting-assistant-core llama_external)
target_include_directories(local-meeting-assistant-core PUBLIC src)
target_include_directories(local-meeting-assistant-core PUBLIC ${ONNX_DIR}/include)
target_include_directories(local-meeting-assistant-core PUBLIC extern/kaldi-native-fbank)
target_link_libraries(local-meeting-assistant-core PUBLIC whisper
        "-framework CoreAudio"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Foundation"
        kaldi-native-fbank-core
        ${CMAKE_BINARY_DIR}/llama_install/lib/libllama.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-base.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-cpu.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-metal.a
        ${CMAKE_BINARY_DIR}/llama_install/lib/libggml-blas.a
        ${ONNX_DIR}/lib/libonnxruntime.dylib
)

add_executable(local-meeting-assistant main.cpp)
target_link_libraries(local-meeting-assistant PRIVATE local-meeting-assistant-core)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(extern/googletest)

    add_executable(local-meeting-assistant-tests
            tests/ahc_pyannote_test.cpp
            tests/audio_capture_test.cpp
            tests/config_test.cpp
            tests/llm_engine_test.cpp
            tests/meeting_assistant_test.cpp
            tests/mel_feature_extractor_test.cpp
            tests/ring_buffer_test.cpp
            tests/speaker_diarizer_test.cpp
            tests/utils_test.cpp
            tests/wav_io_test.cpp
    )
    target_compile_definitions(local-meeting-assistant-tests PRIVATE UNIT_TESTS)
    target_link_libraries(local-meeting-assistant-tests PRIVATE
            local-meeting-assistant-core
            GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(local-meeting-assistant-tests
            PROPERTIES RESOURCE_LOCK local_meeting_assistant_generated_files
    )
endif()
