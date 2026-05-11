#pragma once

#include "config.h"

#include <cmath>
#include <vector>

inline Config make_test_config() {
    Config c{};
    c.model_path = "__missing_model__.gguf";
    c.whisper_model = "__missing_whisper_model__.bin";
    c.audio_device = "__missing_audio_device__";
    c.context_size = 64;
    c.batch_size = 8;
    c.temperature = 0.7f;
    c.top_p = 0.8f;
    c.top_k = 20;
    c.presence_penalty = 1.5f;
    c.segments_per_summary = 2;
    c.blackhole_sample_rate = 48000;
    c.blackhole_channels = 2;
    c.whisper_sample_rate = 16000;
    c.chunk_size_seconds = 10;
    c.snippet_prompt = "snippet";
    c.full_summary_prompt = "full";
    c.check_summary_prompt = "check";
    c.resample_factor = 6;
    c.chunk_samples = c.whisper_sample_rate * c.chunk_size_seconds;
    c.ringbuffer_size = c.chunk_samples * 3;
    c.segmentation_model = "__missing_segmentation__.onnx";
    c.embedding_model = "__missing_embedding__.onnx";
    c.min_segment_duration = 0.5f;
    c.merge_gap = 0.25f;
    c.n_mels = 80;
    c.window_length_ms = 25;
    c.step_width_ms = 10;
    c.clustering_threshold = 0.05f;
    c.num_speakers_hint = -1;
    return c;
}

inline std::vector<float> make_test_audio(int samples, int sample_rate = 16000) {
    std::vector<float> audio(samples);
    for (int i = 0; i < samples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        const float envelope = 0.6f + 0.3f * std::sin(2.0f * 3.14159265f * 3.0f * t);
        audio[i] = envelope * std::sin(2.0f * 3.14159265f * 440.0f * t);
    }
    return audio;
}
