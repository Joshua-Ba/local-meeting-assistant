#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="${BUILD_DIR:-build}"
DIST_DIR="${DIST_DIR:-dist}"

if [[ "$BUILD_DIR" = /* ]]; then
  BUILD_DIR_ABS="$BUILD_DIR"
else
  BUILD_DIR_ABS="$ROOT_DIR/$BUILD_DIR"
fi

echo "Building..."
if [[ -d "$BUILD_DIR_ABS" ]]; then
  stale_cache=""
  while IFS= read -r cache_file; do
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache_file" | head -n 1)"
    cached_build="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache_file" | head -n 1)"
    if [[ -n "$cached_source" && "$cached_source" != "$ROOT_DIR" && "$cached_source" != "$ROOT_DIR"/* ]]; then
      stale_cache="$cache_file"
      break
    fi
    if [[ -n "$cached_build" && "$cached_build" != "$BUILD_DIR_ABS" && "$cached_build" != "$BUILD_DIR_ABS"/* ]]; then
      stale_cache="$cache_file"
      break
    fi
  done < <(find "$BUILD_DIR_ABS" -name CMakeCache.txt -print)

  if [[ -n "$stale_cache" ]]; then
    echo "Existing CMake cache belongs to another checkout: $stale_cache"
    echo "Removing $BUILD_DIR so CMake can configure this copy cleanly..."
    rm -rf "$BUILD_DIR_ABS"
  fi
fi
cmake -S "$ROOT_DIR" -B "$BUILD_DIR_ABS" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR_ABS" --target local-meeting-assistant

echo "Creating dist folder..."
mkdir -p "$DIST_DIR/models"
mkdir -p "$ROOT_DIR/models/diarization"

echo "Copying binary..."
cp "$BUILD_DIR_ABS/local-meeting-assistant" "$DIST_DIR/"

echo "Copying config..."
cp "$ROOT_DIR/config.example.json" "$DIST_DIR/config.json"

echo "Copying shared libraries..."
mkdir -p "$DIST_DIR/models" "$DIST_DIR/lib"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/src/libwhisper.1.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/src/libwhisper.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/libggml.0.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/libggml.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/libggml-base.0.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/libggml-base.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/libggml-cpu.0.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/libggml-cpu.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/ggml-metal/libggml-metal.0.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/ggml-metal/libggml-metal.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/ggml-blas/libggml-blas.0.dylib" "$DIST_DIR/lib/"
cp "$BUILD_DIR_ABS/extern/whisper.cpp/ggml/src/ggml-blas/libggml-blas.dylib" "$DIST_DIR/lib/"
cp "$ROOT_DIR"/extern/onnxruntime/lib/libonnxruntime*.dylib "$DIST_DIR/lib/"

echo "Setting rpath..."
if ! otool -l "$DIST_DIR/local-meeting-assistant" | grep -q "@executable_path/lib"; then
  install_name_tool -add_rpath @executable_path/lib "$DIST_DIR/local-meeting-assistant"
fi

echo "Downloading whisper model..."
"$ROOT_DIR/extern/whisper.cpp/models/download-ggml-model.sh" base
cp "$ROOT_DIR/extern/whisper.cpp/models/ggml-base.bin" "$DIST_DIR/models/"

echo "Downloading speaker diarization models..."
mkdir -p "$DIST_DIR/models/diarization"

# Speaker Segmentation (Pyannote 3.0, ~5MB)
curl -L "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2" -o /tmp/segmentation.tar.bz2
tar xjf /tmp/segmentation.tar.bz2 -C "$DIST_DIR/models/diarization/"
rm /tmp/segmentation.tar.bz2


# Speaker Embedding (wespeaker ResNet34, ~25MB)
curl -L "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx" -o "$DIST_DIR/models/diarization/wespeaker_en_voxceleb_resnet34_LM.onnx"

echo ""
echo "Done! Copy your LLM model into dist/models/:"
echo "  cp models/Qwen3-8B-Q4_K_M.gguf dist/models/"
echo ""
echo "Then run:"
echo "  cd dist && ./local-meeting-assistant"
