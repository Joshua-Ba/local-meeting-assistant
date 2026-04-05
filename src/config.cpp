#include "config.h"
#include "../extern/json.hpp"

Config Config::load(const std::string& path) {
    std::ifstream file(path);
    auto j = nlohmann::json::parse(file);
    Config c;
    c.model_path = j["model_path"];
    c.whisper_model = j["whisper_model"];
    c.context_size = j["context_size"];
    c.batch_size = j["batch_size"];
    c.temperature = j["temperature"];
    c.top_p = j["top_p"];
    c.top_k = j["top_k"];
    c.presence_penalty = j["presence_penalty"];
    c.segments_per_summary = j["segments_per_summary"];
    c.snippet_prompt = j["prompts"]["snippet_summary"];
    c.full_summary_prompt = j["prompts"]["full_summary"];
    c.check_summary_prompt = j["prompts"]["check_summary"];
    c.blackhole_sample_rate= j["audio"]["blackhole_sample_rate"];
    c.blackhole_channels = j["audio"]["blackhole_channels"];
    c.whisper_sample_rate = j["audio"]["whisper_sample_rate"];
    c.chunk_size_seconds = j["audio"]["chunk_seconds"];
    c.audio_device = j["audio"]["blackhole_device_name"];

    c.segmentation_model = j["diarization"]["segmentation_model"];
    c.embedding_model = j["diarization"]["embedding_model"];
    c.speaker_threshold = j["diarization"]["speaker_threshold"];
    c.min_segment_duration = j["diarization"]["min_segment_duration"];
    c.merge_gap = j["diarization"]["merge_gap"];


    // Derived values calculated from config, not stored in JSON
    c.resample_factor = (c.blackhole_sample_rate / c.whisper_sample_rate) * c.blackhole_channels;
    c.chunk_samples = c.whisper_sample_rate * c.chunk_size_seconds;
    c.ringbuffer_size = c.whisper_sample_rate * 10;
    return c;
}