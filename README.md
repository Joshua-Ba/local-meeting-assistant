# local-meeting-assistant

A local, privacy-first meeting assistant that captures system audio, transcribes it in real-time using whisper.cpp, and generates live summaries and meeting notes using a local LLM via llama.cpp. Everything runs on your machine — no cloud, no API keys.

## Requirements

- macOS (Apple Silicon recommended)
- Xcode Command Line Tools (`xcode-select --install`)
- CMake (`brew install cmake`)
- Homebrew (`https://brew.sh`)

## Setup

### Clone

```bash
git clone --recursive https://github.com/Joshua-Ba/local-meeting-assistant.git
cd local-meeting-assistant
```

### Already cloned without submodules?

```bash
git submodule update --init --recursive
```

### BlackHole (Virtual Audio Driver)

BlackHole creates a virtual audio device that allows the program to capture system audio (e.g. from Zoom, Teams, YouTube).

```bash
brew install blackhole-2ch
```

After installation, restart your Mac. Then:

1. Open **Audio MIDI Setup** (search via Spotlight)
2. Click `+` → **Create Multi-Output Device**
3. Check both your headphones/speakers and **BlackHole 2ch**
4. Set the Multi-Output Device as your system output in **System Settings → Sound**

### Download Models

#### Whisper (Speech-to-Text)

```bash
./extern/whisper.cpp/models/download-ggml-model.sh base
```

For better transcription quality, you can use `small` or `medium` instead of `base`.

#### LLM (Summarization)

Download a Qwen3.5-9B GGUF model from Hugging Face:

```bash
mkdir -p models
# Download from https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
# Recommended: Qwen3.5-9B-Q4_K_M.gguf (~5.7 GB)
# Place it in the models/ directory
```

Other compatible models (Qwen3-8B, Qwen3.5-4B, etc.) can also be used. Adjust the model path in `config.json` accordingly.

## Build & Install

### Using the setup script (recommended)

```bash
chmod +x setup.sh
./setup.sh
```

This will build the project, copy all necessary files into a `dist/` folder, and download the Whisper model.

After the script finishes, copy your LLM model:

```bash
cp models/Qwen3.5-9B-Q4_K_M.gguf dist/models/
```

### Manual build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target local-meeting-assistant
```

## Tests

The project includes GoogleTest/CTest unit tests for the core logic:

- config loading and derived values
- ring buffer behavior
- utility path/session helpers
- CoreAudio callback resampling
- LLM and meeting-assistant unloaded-model behavior
- mel feature extraction pipeline
- speaker diarization powerset decoding, Pyannote input geometry, sliding-window activity averaging, binarization, parsing, cleanup, overlap handling, speaker lookup, and agglomerative centroid clustering

Build and run the tests with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target local-meeting-assistant-tests
ctest --test-dir build --output-on-failure
```

For a quicker loop after CMake has already been configured:

```bash
cmake --build build --target local-meeting-assistant-tests
ctest --test-dir build --output-on-failure
```

The tests do not require downloaded Whisper, LLM, or ONNX model files. Model-backed classes are tested through their safe unloaded/failure paths, while pure logic is tested directly.

To disable test targets during configuration:

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
```

## Configuration

All settings are in `config.json` (or `dist/config.json` when using the dist folder). You can adjust:

- `model_path` — path to the LLM model (GGUF)
- `whisper_model` — path to the Whisper model
- `audio_device` — name of the audio capture device (default: `BlackHole`)
- `context_size` — LLM context window size
- `temperature`, `top_p`, `top_k`, `presence_penalty` — LLM sampling parameters
- `segments_per_summary` — how many audio segments to collect before generating a summary
- `prompts` — the prompts used for snippet summaries, full summaries, and summary evaluation

## Run

### From dist folder

```bash
cd dist
./local-meeting-assistant config.json
```

Press **Enter** to stop the recording. After the stop, the captured audio is diarized offline, the final transcript is written with speaker labels, and RTTM is written to `output/diarization_<session-id>.rttm`.

### Diarization RTTM comparison

To generate a C++ RTTM file for comparison with a pyannote `write_rttm` output:

```bash
cd dist
./local-meeting-assistant --test path/to/audio.wav path/to/cpp-output.rttm
./local-meeting-assistant --test path/to/audio.wav path/to/cpp-output.rttm 2
```

If the output path is omitted, the program writes `<audio>.cpp.rttm` next to the input audio path. The optional final argument fixes the expected speaker count, for example `2`; when it is omitted, the diarizer uses `num_speakers_hint` from `config.json` or auto mode. RTTM lines use the standard pyannote-compatible shape:

```text
SPEAKER <uri> 1 <start> <duration> <NA> <NA> SPEAKER_00 <NA> <NA>
```

## How it works

1. System audio is captured via BlackHole and CoreAudio
2. Audio is resampled to 16kHz mono (configurable in `config.json`)
3. Audio chunks are transcribed by whisper.cpp at a configurable interval
4. The raw audio and Whisper segments are accumulated for final processing
5. When you stop the program, speaker diarization runs once over the complete audio
6. The final speaker-labeled transcript is summarized by the LLM

## Project Structure

```
local-meeting-assistant/
├── main.cpp                  # Entry point, audio loop thread, final summary
├── src/
│   ├── ring_buffer.h/.cpp    # Lock-free ring buffer for audio samples
│   ├── audio_capture.h/.cpp  # CoreAudio / BlackHole integration
│   ├── llm_engine.h/.cpp     # llama.cpp wrapper (isolated from whisper's ggml)
│   ├── meeting_assistant.h/.cpp  # Prompt management, summarization logic
│   ├── speaker_diarizer.h/.cpp   # ONNX speaker segmentation and clustering
│   ├── MelFeatureExtractor.h/.cpp # Mel feature extraction for embeddings
│   └── config.h/.cpp         # JSON config file parser
├── tests/                    # GoogleTest unit tests
├── extern/
│   ├── whisper.cpp/          # Speech-to-text (git submodule)
│   ├── llama.cpp/            # LLM inference (git submodule, built via ExternalProject)
│   ├── kaldi-native-fbank/   # Kaldi-compatible fbank features for embeddings
│   ├── googletest/           # Testing framework (git submodule)
│   └── json.hpp              # nlohmann/json (single header)
├── config.example.json               # Runtime configuration
├── setup.sh                  # Build + dist script
└── CMakeLists.txt
```
