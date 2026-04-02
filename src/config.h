#pragma once
#include <string>
#include <fstream>
struct Config {
    std::string model_path;
    std::string whisper_model;
    std::string audio_device;
    int context_size;
    int batch_size;
    float temperature;
    float top_p;
    int top_k;
    float presence_penalty;
    int segments_per_summary;
    int blackhole_sample_rate;
    int blackhole_channels;
    int whisper_sample_rate;
    int chunk_size_seconds;
    std::string snippet_prompt;
    std::string full_summary_prompt;
    std::string check_summary_prompt;
    int resample_factor;
    int chunk_samples;
    int ringbuffer_size;

    static Config load(const std::string& path);
};