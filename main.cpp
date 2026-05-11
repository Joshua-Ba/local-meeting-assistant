#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "extern/whisper.cpp/include/whisper.h"
#include "src/audio_capture.h"
#include "src/config.h"
#include "src/meeting_assistant.h"
#include "src/ring_buffer.h"
#include "src/speaker_diarizer.h"
#include "src/utils.h"
#include "src/wav_io.h"


// =====================================================================
// CLI parsing
// =====================================================================

struct CliOptions {
    enum class Mode { Help, Live, File };
    Mode mode = Mode::Help;
    std::optional<std::filesystem::path> input_wav;
    std::filesystem::path output_dir = "output";
    int num_speakers_hint = -1;
    bool summarize = true;
};


[[noreturn]] void usage_and_exit(int exit_code = 0) {
    std::cout <<
        "Usage:\n"
        "  local-meeting-assistant live   [-o <dir>] [--speakers N]\n"
        "  local-meeting-assistant file <input.wav> [-o <dir>] [--speakers N]\n"
        "  local-meeting-assistant --help\n"
        "\n"
        "Modes:\n"
        "  live   Capture audio via BlackHole, transcribe live, diarize\n"
        "         at the end. Press Enter to stop.\n"
        "  file   Process an existing 16kHz mono WAV file end-to-end.\n"
        "\n"
        "Options:\n"
        "  -o <dir>          Output directory (default: ./output)\n"
        "  --speakers N      Hint the expected number of speakers (>= 1)\n"
        "  --summarize       Generate LLM summary at the end (default: on)\n"
        "  --no-summarize    Skip LLM summary (faster, only transcript + RTTM)\n";
    std::exit(exit_code);
}


int parse_positive_int(const std::string& value) {
    size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Expected positive integer, got: " + value);
    }
    if (consumed != value.size() || parsed <= 0) {
        throw std::runtime_error("Expected positive integer, got: " + value);
    }
    return parsed;
}


CliOptions parse_cli(int argc, char* argv[]) {
    CliOptions opts;
    if (argc < 2) {
        return opts;  // Mode::Help
    }

    const std::string first = argv[1];
    if (first == "--help" || first == "-h") {
        return opts;  // Mode::Help
    }

    int i = 1;
    if (first == "live") {
        opts.mode = CliOptions::Mode::Live;
        i = 2;
    } else if (first == "file") {
        opts.mode = CliOptions::Mode::File;
        if (argc < 3 || argv[2][0] == '-') {
            throw std::runtime_error("'file' mode requires a WAV path");
        }
        opts.input_wav = std::filesystem::path(argv[2]);
        i = 3;
    } else {
        throw std::runtime_error("Unknown mode: " + first + ". Use --help.");
    }

    while (i < argc) {
        const std::string arg = argv[i];
        if (arg == "-o") {
            if (i + 1 >= argc) {
                throw std::runtime_error("-o requires a directory argument");
            }
            opts.output_dir = std::filesystem::path(argv[i + 1]);
            i += 2;
        } else if (arg == "--speakers") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--speakers requires an integer");
            }
            opts.num_speakers_hint = parse_positive_int(argv[i + 1]);
            i += 2;
        } else if (arg == "--summarize") {
            opts.summarize = true;
            i += 1;
        } else if (arg == "--no-summarize") {
            opts.summarize = false;
            i += 1;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return opts;
}


// =====================================================================
// Shared chunk processor (used by both live and file modes)
// =====================================================================

struct ChunkProcessorState {
    int summary_segment_counter = 0;
    float chunk_offset_seconds = 0.0f;
    std::string pending_text_for_summary;
    bool summarize = true;
};


// Process one audio chunk: feed Whisper, update assistant, optionally emit
// snippet summary. Returns nothing - state is updated in place.
void process_audio_chunk(std::span<const float> chunk,
                         whisper_context* ctx,
                         whisper_full_params& wparams,
                         MeetingAssistant& assistant,
                         const Config& config,
                         ChunkProcessorState& state) {
    assistant.add_audio(chunk);

    whisper_full(ctx, wparams, chunk.data(), static_cast<int>(chunk.size()));

    const int num_seg = whisper_full_n_segments(ctx);
    for (int i = 0; i < num_seg; ++i) {
        const auto* text = whisper_full_get_segment_text(ctx, i);
        const auto t0 = whisper_full_get_segment_t0(ctx, i);
        const auto t1 = whisper_full_get_segment_t1(ctx, i);
        const float start_sec = state.chunk_offset_seconds + static_cast<float>(t0) / 100.0f;
        const float end_sec = state.chunk_offset_seconds + static_cast<float>(t1) / 100.0f;

        assistant.add_transcript_segment(start_sec, end_sec, text);
        state.pending_text_for_summary.append(text);
    }

    ++state.summary_segment_counter;
    if (state.summary_segment_counter >= config.segments_per_summary) {
        state.summary_segment_counter = 0;
        assistant.add_segment(state.pending_text_for_summary);
        if (state.summarize) {
            auto summary = assistant.summarize_current_segment();
            if (summary) {
                std::cout << *summary << std::endl;
            }
        }
        state.pending_text_for_summary.clear();
    }

    const float chunk_duration = static_cast<float>(chunk.size()) /
                                 static_cast<float>(config.whisper_sample_rate);
    state.chunk_offset_seconds += chunk_duration;
}


void flush_pending_segment(MeetingAssistant& assistant,
                           ChunkProcessorState& state) {
    if (!state.pending_text_for_summary.empty()) {
        assistant.add_segment(state.pending_text_for_summary);
        state.pending_text_for_summary.clear();
    }
}


// =====================================================================
// Finalize: run diarization, write outputs, print result
// =====================================================================

std::string sanitize_for_filename(std::string s) {
    for (char& ch : s) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return s;
}


struct SessionPaths {
    std::filesystem::path transcript;
    std::filesystem::path rttm;
    std::filesystem::path summary;
    std::string rttm_uri;
};


SessionPaths make_session_paths(const std::filesystem::path& output_dir,
                                const std::string& session_id) {
    std::filesystem::create_directories(output_dir);
    SessionPaths paths;
    paths.transcript = output_dir / ("transcript_" + session_id + ".txt");
    paths.rttm = output_dir / ("diarization_" + session_id + ".rttm");
    paths.summary = output_dir / ("summary_" + session_id + ".txt");
    paths.rttm_uri = sanitize_for_filename(session_id);
    return paths;
}


void finalize_session(MeetingAssistant& assistant,
                      const SessionPaths& paths,
                      bool summarize) {
    assistant.finalize(paths.transcript, paths.rttm, paths.rttm_uri);

    std::cout << "\n=== Transcript ===\n";
    std::cout << assistant.get_full_text() << std::endl;

    if (!summarize) {
        std::cout << "\n(skipping summary, --no-summarize set)\n";
        std::cout << "\nOutputs:\n";
        std::cout << "  Transcript: " << paths.transcript << "\n";
        std::cout << "  RTTM:       " << paths.rttm << "\n";
        return;
    }

    auto summary = assistant.full_summary_checked();
    if (summary) {
        std::ofstream out(paths.summary);
        if (out.is_open()) {
            out << *summary;
        }
        std::cout << "\n=== Summary ===\n" << *summary << std::endl;
    } else {
        std::cout << "Fehler beim Erstellen der Zusammenfassung" << std::endl;
    }

    std::cout << "\nOutputs:\n";
    std::cout << "  Transcript: " << paths.transcript << "\n";
    std::cout << "  RTTM:       " << paths.rttm << "\n";
    if (summary) std::cout << "  Summary:    " << paths.summary << "\n";
}


// =====================================================================
// Live mode
// =====================================================================

void audio_loop_live(std::stop_token token,
                     RingBuffer& buffer,
                     whisper_context* ctx,
                     whisper_full_params& wparams,
                     MeetingAssistant& assistant,
                     const Config& config,
                     ChunkProcessorState& state) {
    std::vector<float> data;
    while (!token.stop_requested()) {
        if (buffer.size() >= config.chunk_samples) {
            buffer.batch_read(data, config.chunk_samples);
            process_audio_chunk(data, ctx, wparams, assistant, config, state);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    flush_pending_segment(assistant, state);
}


void run_live(const CliOptions& opts, Config& config) {
    const auto session_id = generate_session_id();
    const auto paths = make_session_paths(opts.output_dir, session_id);
    std::cout << "Session ID: " << session_id << std::endl;

    RingBuffer ring(config.ringbuffer_size);
    AudioCapture capture(&ring, config.audio_device, config.resample_factor);

    auto ctx_params = whisper_context_default_params();
    auto* ctx = whisper_init_from_file_with_params(config.whisper_model.c_str(), ctx_params);
    if (!ctx) {
        throw std::runtime_error("Failed to load Whisper model");
    }
    auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "auto";

    MeetingAssistant assistant(config);
    if (!assistant.is_loaded()) {
        whisper_free(ctx);
        throw std::runtime_error("Assistant failed to load");
    }
    std::cout << "Assistant loaded. Press Enter to stop recording." << std::endl;

    capture.start();
    ChunkProcessorState state;
    state.summarize = opts.summarize;

    std::jthread worker(audio_loop_live,
                        std::ref(ring),
                        ctx,
                        std::ref(wparams),
                        std::ref(assistant),
                        std::cref(config),
                        std::ref(state));

    std::cin.get();
    worker.request_stop();
    if (worker.joinable()) worker.join();
    capture.stop();

    finalize_session(assistant, paths, opts.summarize);
    whisper_free(ctx);
}


// =====================================================================
// File mode
// =====================================================================

void run_file(const CliOptions& opts, Config& config) {
    if (!opts.input_wav) {
        throw std::runtime_error("file mode requires input WAV");
    }
    if (!std::filesystem::exists(*opts.input_wav)) {
        throw std::runtime_error("Input WAV not found: " + opts.input_wav->string());
    }
    if (opts.num_speakers_hint > 0) {
        config.num_speakers_hint = opts.num_speakers_hint;
    }

    const auto session_id = opts.input_wav->stem().string();
    const auto paths = make_session_paths(opts.output_dir, session_id);
    std::cout << "Session ID: " << session_id << std::endl;
    std::cout << "Loading audio: " << *opts.input_wav << std::endl;

    auto audio = load_wav_16k_mono(opts.input_wav->string());
    if (audio.empty()) {
        throw std::runtime_error("Audio empty after loading");
    }
    const float duration = static_cast<float>(audio.size()) /
                           static_cast<float>(config.whisper_sample_rate);
    std::cout << "Audio duration: " << duration << "s" << std::endl;

    auto ctx_params = whisper_context_default_params();
    auto* ctx = whisper_init_from_file_with_params(config.whisper_model.c_str(), ctx_params);
    if (!ctx) {
        throw std::runtime_error("Failed to load Whisper model");
    }
    auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "auto";

    MeetingAssistant assistant(config);
    if (!assistant.is_loaded()) {
        whisper_free(ctx);
        throw std::runtime_error("Assistant failed to load");
    }

    ChunkProcessorState state;
    state.summarize = opts.summarize;
    const size_t chunk_size = static_cast<size_t>(config.chunk_samples);
    const size_t total_samples = audio.size();
    const size_t total_chunks = (total_samples + chunk_size - 1) / chunk_size;
    std::cout << "Processing " << total_chunks << " chunks..." << std::endl;

    for (size_t offset = 0; offset < total_samples; offset += chunk_size) {
        const size_t remaining = total_samples - offset;
        const size_t this_chunk = std::min(chunk_size, remaining);
        const std::span<const float> chunk(audio.data() + offset, this_chunk);

        const size_t chunk_idx = offset / chunk_size + 1;
        std::cout << "  chunk " << chunk_idx << "/" << total_chunks << "\r" << std::flush;

        process_audio_chunk(chunk, ctx, wparams, assistant, config, state);
    }
    std::cout << std::endl;
    flush_pending_segment(assistant, state);

    finalize_session(assistant, paths, opts.summarize);
    whisper_free(ctx);
}


// =====================================================================
// main
// =====================================================================

int main(int argc, char* argv[]) {
    // Binary cd's into its own directory so relative paths in config.json
    // (e.g. models/...) resolve correctly.
    std::filesystem::current_path(std::filesystem::path(argv[0]).parent_path());

    CliOptions opts;
    try {
        opts = parse_cli(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n\n";
        usage_and_exit(2);
    }

    if (opts.mode == CliOptions::Mode::Help) {
        usage_and_exit(0);
    }

    auto config = Config::load("config.json");

    try {
        if (opts.mode == CliOptions::Mode::Live) {
            run_live(opts, config);
        } else {
            run_file(opts, config);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}