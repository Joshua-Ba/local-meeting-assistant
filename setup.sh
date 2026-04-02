#!/bin/bash
set -e

echo "Building..."
cmake -B build
cmake --build build

echo "Creating dist folder..."
mkdir -p dist/models

echo "Copying binary..."
cp build/local-meeting-assistant dist/

echo "Copying config..."
cp config.json dist/

echo "Copying shared libraries..."
mkdir -p dist/models dist/lib
cp build/extern/whisper.cpp/src/libwhisper.1.dylib dist/lib/
cp build/extern/whisper.cpp/src/libwhisper.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/libggml.0.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/libggml.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/libggml-base.0.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/libggml-base.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/libggml-cpu.0.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/libggml-cpu.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/ggml-metal/libggml-metal.0.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/ggml-metal/libggml-metal.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/ggml-blas/libggml-blas.0.dylib dist/lib/
cp build/extern/whisper.cpp/ggml/src/ggml-blas/libggml-blas.dylib dist/lib/

echo "Setting rpath..."
install_name_tool -add_rpath @executable_path/lib dist/local-meeting-assistant

echo "Downloading whisper model..."
./extern/whisper.cpp/models/download-ggml-model.sh base
cp extern/whisper.cpp/models/ggml-base.bin dist/models/

echo ""
echo "Done! Copy your LLM model into dist/models/:"
echo "  cp models/Qwen3-8B-Q4_K_M.gguf dist/models/"
echo ""
echo "Then run:"
echo "  cd dist && ./local-meeting-assistant config.json"