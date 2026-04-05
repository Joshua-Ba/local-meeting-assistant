#!/bin/bash
set -e

echo "Building..."
cmake -B build
cmake --build build

echo "Creating dist folder..."
mkdir -p dist/models
mkdir -p models/diarization

echo "Copying binary..."
cp build/local-meeting-assistant dist/

echo "Copying config..."
cp config.example.json dist/config.json

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
cp extern/onnxruntime/lib/libonnxruntime*.dylib dist/lib/

echo "Setting rpath..."
install_name_tool -add_rpath @executable_path/lib dist/local-meeting-assistant

echo "Downloading whisper model..."
./extern/whisper.cpp/models/download-ggml-model.sh base
cp extern/whisper.cpp/models/ggml-base.bin dist/models/

echo "Downloading speaker diarization models..."
mkdir -p dist/models/diarization

# Speaker Segmentation (Pyannote 3.0, ~5MB)
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" -o /tmp/segmentation.tar.bz2
tar xjf /tmp/segmentation.tar.bz2 -C dist/models/diarization/
rm /tmp/segmentation.tar.bz2

# Speaker Embedding (wespeaker ECAPA-TDNN, ~25MB)
curl -L "https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/resolve/main/voxceleb_ECAPA512_LM.onnx" -o dist/models/diarization/ecapa_tdnn512.onnx

echo ""
echo "Done! Copy your LLM model into dist/models/:"
echo "  cp models/Qwen3-8B-Q4_K_M.gguf dist/models/"
echo ""
echo "Then run:"
echo "  cd dist && ./local-meeting-assistant"