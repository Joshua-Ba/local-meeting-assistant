# local-meeting-assistant

A local, privacy-first meeting assistant that captures system audio, transcribes it in real-time using `whisper.cpp`, diarizes speakers with an ONNX pyannote pipeline, and generates meeting summaries using a local LLM via `llama.cpp`. Everything runs on your machine — no cloud, no API keys.

## Requirements

- macOS (Apple Silicon recommended)
- Xcode Command Line Tools (`xcode-select --install`)
- CMake (`brew install cmake`)
- Homebrew (https://brew.sh)

## Setup

### Clone

```bash
git clone --recursive https://github.com/Joshua-Ba/local-meeting-assistant.git
cd local-meeting-assistant
```

Already cloned without submodules?

```bash
git submodule update --init --recursive
```

### BlackHole (Virtual Audio Driver)

BlackHole creates a virtual audio device that allows the program to capture system audio (e.g. from Zoom, Teams, YouTube). Only needed for `live` mode.

```bash
brew install blackhole-2ch
```

After installation, restart your Mac. Then:

1. Open *Audio MIDI Setup* (search via Spotlight)
2. Click `+` → *Create Multi-Output Device*
3. Check both your headphones/speakers and *BlackHole 2ch*
4. Set the Multi-Output Device as your system output in *System Settings → Sound*

### Models

The `setup.sh` script automatically downloads all model files except the LLM:

- Whisper speech-to-text model (`base` by default)
- Pyannote segmentation model (~5 MB)
- Wespeaker ResNet34 speaker embedding model (~25 MB)

The only model you need to download yourself is the LLM, because of its size (~5.7 GB).

#### LLM (manual)

Download a Qwen3.5-9B GGUF model from Hugging Face:

```bash
mkdir -p models
# Download from https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
# Recommended: Qwen3.5-9B-Q4_K_M.gguf
# Place it in the models/ directory
```

Other compatible models (Qwen3-8B, Qwen3.5-4B, etc.) can also be used. Adjust `model_path` in `config.json` accordingly.

#### Customizing Whisper or diarization models

If you want a different Whisper variant, edit `setup.sh` or download manually:

```bash
./extern/whisper.cpp/models/download-ggml-model.sh small   # or medium
```

The diarization models come from:

- Segmentation: [sherpa-onnx-pyannote-segmentation-3-0](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)
- Speaker embedding: [Wespeaker/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM)

Paths are referenced from `config.json` under `diarization.segmentation_model` and `diarization.embedding_model`.

## Build & Install

### Using the setup script (recommended)

```bash
chmod +x setup.sh
./setup.sh
```

This:

1. Builds the project
2. Copies all necessary files into a `dist/` folder
3. Downloads the Whisper model
4. Downloads the pyannote segmentation and Wespeaker embedding models

After the script finishes, copy your LLM model into the dist folder:

```bash
cp models/Qwen3.5-9B-Q4_K_M.gguf dist/models/
```

### Manual build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target local-meeting-assistant
```

Note that manual builds skip the model downloads — you'll need to place model files in `models/` yourself before running.

## Run

The application has two modes plus a help screen. Run from the `dist/` directory so relative model paths in `config.json` resolve correctly.

### Help

```bash
cd dist
./local-meeting-assistant --help
```

### Live mode (capture from BlackHole)

```bash
./local-meeting-assistant live
./local-meeting-assistant live -o my_outputs --speakers 3
./local-meeting-assistant live --no-summarize
```

Press `Enter` to stop recording. The captured audio is then diarized offline, a transcript with speaker labels is written, and (unless `--no-summarize`) the LLM produces a final summary.

### File mode (process an existing WAV)

```bash
./local-meeting-assistant file path/to/audio.wav
./local-meeting-assistant file path/to/audio.wav -o my_outputs --speakers 2
./local-meeting-assistant file path/to/audio.wav --no-summarize
```

The input WAV must be 16 kHz mono 16-bit PCM.

### Options

- `-o <dir>` — output directory (default: `./output`)
- `--speakers N` — hint the expected number of speakers (>= 1). Overrides `diarization.num_speakers_hint` from config.
- `--summarize` / `--no-summarize` — enable or disable LLM summarization (default: on)

### Outputs

Each run produces three files under the output directory, named by session ID (timestamp for `live`, filename stem for `file`):

- `transcript_<session>.txt` — speaker-labeled transcript
- `diarization_<session>.rttm` — pyannote-compatible RTTM
- `summary_<session>.txt` — LLM summary (skipped with `--no-summarize`)

RTTM lines use the standard pyannote-compatible shape:

```
SPEAKER <uri> 1 <start> <duration> <NA> <NA> SPEAKER_00 <NA> <NA>
```

## Configuration

All settings live in `config.json` (or `dist/config.json` when using the dist folder). See `config.example.json` for a complete example.

### LLM / Summarization

- `model_path` — path to the LLM model (GGUF)
- `context_size` — LLM context window size
- `batch_size`
- `temperature`, `top_p`, `top_k`, `presence_penalty` — LLM sampling parameters
- `segments_per_summary` — how many audio segments before generating a snippet summary
- `prompts.snippet_summary`, `prompts.full_summary`, `prompts.check_summary` — prompt templates

### Audio Capture

- `audio.blackhole_device_name` — substring match for the input device name (default: `BlackHole`)
- `audio.blackhole_sample_rate`
- `audio.blackhole_channels`

### Whisper

- `whisper_model` — path to the Whisper model binary
- `audio.whisper_sample_rate` — target sample rate used by Whisper and diarization
- `audio.chunk_seconds` — chunk length used for transcription in both `live` and `file` mode

### Diarization

- `diarization.segmentation_model` — ONNX segmentation model path
- `diarization.embedding_model` — ONNX speaker embedding model path
- `diarization.min_segment_duration`
- `diarization.merge_gap`
- `diarization.n_mels`
- `diarization.window_length_ms`
- `diarization.step_width_ms`
- `diarization.clustering_threshold`
- `diarization.num_speakers_hint` — default speaker-count hint used when `--speakers` is not provided

### Derived values

Computed automatically in `Config::load`, not stored in JSON:

- `resample_factor`
- `chunk_samples`
- `ringbuffer_size`

## Tests

The project ships with GoogleTest/CTest unit tests for the core logic.

### Quick test run

```bash
chmod +x run-tests.sh
./run-tests.sh
```

Filter to a subset:

```bash
./run-tests.sh -R 'SpeakerDiarizerTest|AhcPyannoteTest'
```

### Manual

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target local-meeting-assistant-tests
ctest --test-dir build --output-on-failure
```

To disable test targets during configuration:

```bash
cmake -S . -B build -DBUILD_TESTING=OFF
```

### Coverage

- Config loading and derived values
- Ring buffer behavior
- Utility helpers
- CoreAudio callback resampling
- WAV parsing
- LLM and meeting-assistant unloaded-model behavior
- Mel feature extraction
- Speaker diarization: powerset decoding, mask interpolation, reconstruction, segment extraction, cluster assignment
- Pyannote-style agglomerative hierarchical clustering in `ahc_pyannote`

The tests do not require downloaded Whisper, LLM, or ONNX model files. Model-backed classes are tested through their unloaded/failure paths, while pure logic is exercised directly.

## How it works

1. In `live` mode, CoreAudio captures samples from BlackHole into a ring buffer; in `file` mode, the WAV file is loaded into memory and chunked the same way.
2. Audio is resampled to Whisper's expected mono sample rate (decimation, every Nth sample).
3. The app processes audio in chunks of `audio.chunk_seconds`.
4. Each chunk is transcribed with `whisper.cpp`.
5. Whisper segments are stored with timestamps and appended to the in-memory meeting state.
6. Optional snippet summaries are generated every `segments_per_summary` chunks.
7. At the end of the session, the accumulated audio is diarized offline with the ONNX segmentation + embedding pipeline, followed by pyannote-style agglomerative clustering.
8. The transcript is rewritten with speaker labels and written to disk together with an RTTM file.
9. A final checked summary is generated with the local LLM unless `--no-summarize` was set.

## Project structure

```text
local-meeting-assistant/
├── main.cpp                       # CLI parsing, live/file orchestration, finalization
├── src/
│   ├── audio_capture.h/.cpp       # CoreAudio + BlackHole capture
│   ├── ring_buffer.h/.cpp         # SPSC ring buffer between callback and worker thread
│   ├── wav_io.h/.cpp              # Strict 16 kHz mono WAV reader for file mode
│   ├── llm_engine.h/.cpp          # llama.cpp wrapper
│   ├── meeting_assistant.h/.cpp   # Transcript/audio accumulation + summary flow
│   ├── MelFeatureExtractor.h/.cpp # Fbank feature extraction for speaker embeddings
│   ├── ahc_pyannote.h/.cpp        # Pyannote-style agglomerative clustering
│   ├── speaker_diarizer.h/.cpp    # Segmentation, embeddings, clustering, RTTM
│   ├── config.h/.cpp              # JSON config loading and derived fields
│   └── utils.h                    # Session id / path helpers
├── tests/
│   ├── ahc_pyannote_test.cpp
│   ├── speaker_diarizer_test.cpp
│   ├── mel_feature_extractor_test.cpp
│   ├── wav_io_test.cpp
│   └── ...                        # Remaining unit tests
├── extern/
│   ├── whisper.cpp/               # Speech-to-text (git submodule)
│   ├── llama.cpp/                 # LLM inference (git submodule)
│   ├── kaldi-native-fbank/        # Fbank implementation used by diarization
│   ├── googletest/                # Test framework (git submodule)
│   ├── onnxruntime/               # Local ONNX runtime binaries
│   └── json.hpp                   # nlohmann/json single header
├── config.example.json            # Example runtime configuration
├── models/                        # Runtime model directory expected by config.json
├── dist/                          # Generated runnable bundle created by setup.sh
├── setup.sh                       # Build + dist packaging script
├── run-tests.sh                   # Configure, build, and run CTest
└── CMakeLists.txt
```

## License & Credits

The diarization pipeline in `ahc_pyannote.cpp` is an independent C++ reimplementation modeled after:

- [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT License) — `AgglomerativeClustering` pipeline
- [scipy](https://github.com/scipy/scipy) (BSD-3-Clause License) — `linkage(method="centroid")` algorithm

No Python source was directly copied; the algorithms follow established hierarchical clustering literature (Lance-Williams update for centroid linkage).