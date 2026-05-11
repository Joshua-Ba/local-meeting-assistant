#include "config.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

TEST(ConfigTest, LoadsAllFieldsAndDerivedValues) {
    const auto path = std::filesystem::current_path() / "config_test.json";
    std::ofstream file(path, std::ios::trunc);
    file << R"json({
  "model_path": "model.gguf",
  "whisper_model": "whisper.bin",
  "context_size": 32768,
  "batch_size": 8192,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.5,
  "segments_per_summary": 8,
  "prompts": {
    "snippet_summary": "snippet prompt",
    "full_summary": "full prompt",
    "check_summary": "check prompt"
  },
  "audio": {
    "blackhole_device_name": "BlackHole",
    "blackhole_sample_rate": 48000,
    "blackhole_channels": 2,
    "whisper_sample_rate": 16000,
    "chunk_seconds": 10
  },
  "diarization": {
    "segmentation_model": "segmentation.onnx",
    "embedding_model": "embedding.onnx",
    "min_segment_duration": 0.5,
    "merge_gap": 0.4,
    "n_mels": 80,
    "window_length_ms": 25,
    "step_width_ms": 10,
    "clustering_threshold": 0.45,
    "num_speakers_hint": 2
  }
})json";
    file.close();

    const Config config = Config::load(path.string());

    EXPECT_EQ(config.model_path, "model.gguf");
    EXPECT_EQ(config.whisper_model, "whisper.bin");
    EXPECT_EQ(config.audio_device, "BlackHole");
    EXPECT_EQ(config.context_size, 32768);
    EXPECT_EQ(config.batch_size, 8192);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_FLOAT_EQ(config.top_p, 0.8f);
    EXPECT_EQ(config.top_k, 20);
    EXPECT_FLOAT_EQ(config.presence_penalty, 1.5f);
    EXPECT_EQ(config.segments_per_summary, 8);
    EXPECT_EQ(config.snippet_prompt, "snippet prompt");
    EXPECT_EQ(config.full_summary_prompt, "full prompt");
    EXPECT_EQ(config.check_summary_prompt, "check prompt");
    EXPECT_EQ(config.blackhole_sample_rate, 48000);
    EXPECT_EQ(config.blackhole_channels, 2);
    EXPECT_EQ(config.whisper_sample_rate, 16000);
    EXPECT_EQ(config.chunk_size_seconds, 10);
    EXPECT_EQ(config.segmentation_model, "segmentation.onnx");
    EXPECT_EQ(config.embedding_model, "embedding.onnx");
    EXPECT_FLOAT_EQ(config.min_segment_duration, 0.5f);
    EXPECT_FLOAT_EQ(config.merge_gap, 0.4f);
    EXPECT_EQ(config.n_mels, 80);
    EXPECT_EQ(config.window_length_ms, 25);
    EXPECT_EQ(config.step_width_ms, 10);
    EXPECT_FLOAT_EQ(config.clustering_threshold, 0.45f);
    EXPECT_EQ(config.num_speakers_hint, 2);

    EXPECT_EQ(config.resample_factor, 6);
    EXPECT_EQ(config.chunk_samples, 160000);
    EXPECT_EQ(config.ringbuffer_size, 480000);
}

TEST(ConfigTest, MissingFileThrowsHelpfulError) {
    EXPECT_THROW(Config::load("__does_not_exist__/missing.json"), std::runtime_error);
}
