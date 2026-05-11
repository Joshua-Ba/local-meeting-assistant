// MelFeatureExtractor.h
//
// Mel filter bank feature extraction using kaldi-native-fbank with
// torchaudio-compatible options, plus per-frame mean subtraction to
// match pyannote's preprocessing pipeline.

#pragma once

#include <span>
#include <vector>


class MelFeatureExtractor {
public:
    MelFeatureExtractor(int n_mels,
                        int sample_rate,
                        int window_length_ms,
                        int step_width_ms);

    MelFeatureExtractor(MelFeatureExtractor&&) noexcept = default;
    MelFeatureExtractor& operator=(MelFeatureExtractor&&) noexcept = delete;
    MelFeatureExtractor(const MelFeatureExtractor&) = delete;
    MelFeatureExtractor& operator=(const MelFeatureExtractor&) = delete;
    ~MelFeatureExtractor() = default;

    // Extract mel features with per-frame mean normalization (pyannote
    // preprocessing). Returns a flat row-major vector of shape
    // (num_frames, n_mels).
    [[nodiscard]] std::vector<float> extract(std::span<const float> audio) const;

    // Same as extract() but without the frame-wise mean subtraction
    // (raw Kaldi fbank output). Useful for debugging.
    [[nodiscard]] std::vector<float> extract_raw(std::span<const float> audio) const;

private:
    [[nodiscard]] std::vector<float> extract_impl(
        std::span<const float> audio,
        bool apply_frame_mean_normalization) const;

    const int n_mels_;
    const int sample_rate_;
    const int window_length_ms_;
    const int step_width_ms_;
};