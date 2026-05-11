// config.h
//
// Application configuration loaded from JSON. Fields are grouped by
// concern (LLM, audio capture, whisper, diarization) and a handful of
// derived values are filled in by Config::load.

#pragma once

#include <fstream>
#include <string>


struct Config {
    // LLM / summarization
    std::string model_path;
    int context_size;
    int batch_size;
    float temperature;
    float top_p;
    int top_k;
    float presence_penalty;
    int segments_per_summary;
    std::string snippet_prompt;
    std::string full_summary_prompt;
    std::string check_summary_prompt;

    // Audio capture (BlackHole / CoreAudio)
    std::string audio_device;
    int blackhole_sample_rate;
    int blackhole_channels;

    // Whisper transcription
    std::string whisper_model;
    int whisper_sample_rate;
    int chunk_size_seconds;

    // Diarization (segmentation + embedding + clustering)
    std::string segmentation_model;
    std::string embedding_model;
    float min_segment_duration;
    float merge_gap;
    int n_mels;
    int window_length_ms;
    int step_width_ms;
    float clustering_threshold;
    int num_speakers_hint;

    // Derived values (not loaded from JSON; computed in load()).
    int resample_factor;
    int chunk_samples;
    int ringbuffer_size;

    static Config load(const std::string& path);
};