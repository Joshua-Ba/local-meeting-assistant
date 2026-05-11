// MelFeatureExtractor.cpp
//
// Mel filter bank feature extraction matching pyannote's preprocessing
// pipeline: kaldi-native-fbank with torchaudio-compatible options, then
// per-frame mean subtraction.

#include "MelFeatureExtractor.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"


namespace {

constexpr float kKaldiInt16Scale = 32768.0f;
constexpr float kBlackmanCoeff = 0.42f;
constexpr float kDither = 0.0f;
constexpr float kEnergyFloor = 1.0f;
constexpr float kLowFreqHz = 20.0f;
constexpr float kHighFreqHzOffset = 0.0f;
constexpr float kPreemphasisCoeff = 0.97f;
constexpr float kVtlnLowHz = 100.0f;
constexpr float kVtlnHighHzOffset = -500.0f;


// Subtract the per-frame mean across all mel bins. This matches the
// extra normalization step that pyannote applies on top of its Kaldi
// fbank features.
std::vector<float> frame_mean_normalize(const std::vector<float>& features,
                                        const int n_mels) {
    std::vector<float> normalized;
    normalized.reserve(features.size());

    const size_t num_frames = features.size() / static_cast<size_t>(n_mels);
    for (size_t frame = 0; frame < num_frames; ++frame) {
        const size_t frame_offset = frame * static_cast<size_t>(n_mels);
        float mean = 0.0f;
        for (int mel = 0; mel < n_mels; ++mel) {
            mean += features[frame_offset + static_cast<size_t>(mel)];
        }
        mean /= static_cast<float>(n_mels);

        for (int mel = 0; mel < n_mels; ++mel) {
            normalized.push_back(features[frame_offset + static_cast<size_t>(mel)] - mean);
        }
    }

    return normalized;
}


// Build the Kaldi fbank options to mirror torchaudio.compliance.kaldi.fbank
// defaults that pyannote uses upstream.
knf::FbankOptions make_fbank_options(const int n_mels,
                                     const int sample_rate,
                                     const int window_length_ms,
                                     const int step_width_ms) {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = static_cast<float>(sample_rate);
    opts.frame_opts.frame_length_ms = static_cast<float>(window_length_ms);
    opts.frame_opts.frame_shift_ms = static_cast<float>(step_width_ms);
    opts.frame_opts.dither = kDither;
    opts.frame_opts.preemph_coeff = kPreemphasisCoeff;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.round_to_power_of_two = true;
    opts.frame_opts.blackman_coeff = kBlackmanCoeff;
    opts.frame_opts.snip_edges = true;

    opts.mel_opts.num_bins = n_mels;
    opts.mel_opts.low_freq = kLowFreqHz;
    opts.mel_opts.high_freq = kHighFreqHzOffset;
    opts.mel_opts.vtln_low = kVtlnLowHz;
    opts.mel_opts.vtln_high = kVtlnHighHzOffset;
    opts.mel_opts.debug_mel = false;
    opts.mel_opts.htk_mode = false;
    opts.mel_opts.is_librosa = false;
    opts.mel_opts.norm = "slaney";
    opts.mel_opts.use_slaney_mel_scale = true;
    opts.mel_opts.floor_to_int_bin = false;

    opts.use_energy = false;
    opts.energy_floor = kEnergyFloor;
    opts.raw_energy = true;
    opts.htk_compat = false;
    opts.use_log_fbank = true;
    opts.use_power = true;
    return opts;
}

}  // namespace


// =====================================================================
// Construction
// =====================================================================

MelFeatureExtractor::MelFeatureExtractor(const int n_mels,
                                         const int sample_rate,
                                         const int window_length_ms,
                                         const int step_width_ms)
    : n_mels_(n_mels),
      sample_rate_(sample_rate),
      window_length_ms_(window_length_ms),
      step_width_ms_(step_width_ms) {
    if (n_mels_ <= 0) {
        throw std::invalid_argument("n_mels must be positive");
    }
    if (sample_rate_ <= 0) {
        throw std::invalid_argument("sample_rate must be positive");
    }
    if (window_length_ms_ <= 0) {
        throw std::invalid_argument("window_length_ms must be positive");
    }
    if (step_width_ms_ <= 0) {
        throw std::invalid_argument("step_width_ms must be positive");
    }
}


// =====================================================================
// Public extract entry points
// =====================================================================

std::vector<float> MelFeatureExtractor::extract(std::span<const float> audio) const {
    return extract_impl(audio, true);
}


std::vector<float> MelFeatureExtractor::extract_raw(std::span<const float> audio) const {
    return extract_impl(audio, false);
}


// =====================================================================
// Implementation
// =====================================================================

std::vector<float> MelFeatureExtractor::extract_impl(
    std::span<const float> audio,
    const bool apply_frame_mean_normalization) const {
    if (audio.empty()) {
        return {};
    }

    const auto opts = make_fbank_options(n_mels_, sample_rate_,
                                         window_length_ms_, step_width_ms_);
    knf::OnlineFbank fbank(opts);

    // Kaldi expects int16-scaled floats; the rest of the pipeline works
    // with [-1, 1] float audio, so scale up here.
    std::vector<float> kaldi_scaled_audio;
    kaldi_scaled_audio.reserve(audio.size());
    for (const float sample : audio) {
        kaldi_scaled_audio.push_back(sample * kKaldiInt16Scale);
    }

    fbank.AcceptWaveform(static_cast<float>(sample_rate_),
                         kaldi_scaled_audio.data(),
                         static_cast<int32_t>(kaldi_scaled_audio.size()));
    fbank.InputFinished();

    const int32_t num_frames = fbank.NumFramesReady();
    if (num_frames <= 0) {
        return {};
    }

    std::vector<float> features;
    features.reserve(static_cast<size_t>(num_frames) *
                     static_cast<size_t>(n_mels_));
    for (int32_t frame = 0; frame < num_frames; ++frame) {
        const float* frame_data = fbank.GetFrame(frame);
        for (int mel = 0; mel < n_mels_; ++mel) {
            features.push_back(frame_data[mel]);
        }
    }

    if (!apply_frame_mean_normalization) {
        return features;
    }
    return frame_mean_normalize(features, n_mels_);
}