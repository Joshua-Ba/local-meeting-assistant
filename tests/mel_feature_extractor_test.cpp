#include "MelFeatureExtractor.h"

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/online-feature.h"
#include "test_helpers.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kMels = 80;
constexpr int kSampleRate = 16000;
constexpr int kWindowLengthMs = 25;
constexpr int kStepWidthMs = 10;
constexpr float kKaldiInt16Scale = 32768.0f;
constexpr float kBlackmanCoeff = 0.42f;
constexpr float kDither = 0.0f;
constexpr float kEnergyFloor = 1.0f;
constexpr float kLowFreqHz = 20.0f;
constexpr float kHighFreqHzOffset = 0.0f;
constexpr float kPreemphasisCoeff = 0.97f;
constexpr float kVtlnLowHz = 100.0f;
constexpr float kVtlnHighHzOffset = -500.0f;

std::vector<float> frame_mean_normalize(const std::vector<float>& features, const int n_mels) {
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

std::vector<float> extract_reference_raw_features(const std::vector<float>& audio) {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = static_cast<float>(kSampleRate);
    opts.frame_opts.frame_length_ms = static_cast<float>(kWindowLengthMs);
    opts.frame_opts.frame_shift_ms = static_cast<float>(kStepWidthMs);
    opts.frame_opts.dither = kDither;
    opts.frame_opts.preemph_coeff = kPreemphasisCoeff;
    opts.frame_opts.remove_dc_offset = true;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.round_to_power_of_two = true;
    opts.frame_opts.blackman_coeff = kBlackmanCoeff;
    opts.frame_opts.snip_edges = true;

    opts.mel_opts.num_bins = kMels;
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

    std::vector<float> scaled_audio;
    scaled_audio.reserve(audio.size());
    for (const float sample : audio) {
        scaled_audio.push_back(sample * kKaldiInt16Scale);
    }

    knf::OnlineFbank fbank(opts);
    fbank.AcceptWaveform(static_cast<float>(kSampleRate),
                         scaled_audio.data(),
                         static_cast<int32_t>(scaled_audio.size()));
    fbank.InputFinished();

    const int32_t num_frames = fbank.NumFramesReady();
    std::vector<float> features;
    features.reserve(static_cast<size_t>(num_frames) * kMels);
    for (int32_t frame = 0; frame < num_frames; ++frame) {
        const float* frame_data = fbank.GetFrame(frame);
        for (int mel = 0; mel < kMels; ++mel) {
            features.push_back(frame_data[mel]);
        }
    }

    return features;
}

std::vector<float> extract_reference_features(const std::vector<float>& audio) {
    return frame_mean_normalize(extract_reference_raw_features(audio), kMels);
}
}

TEST(MelFeatureExtractorTest, ConstructorRejectsInvalidOptions) {
    EXPECT_THROW(MelFeatureExtractor(0, 16000, 25, 10), std::invalid_argument);
    EXPECT_THROW(MelFeatureExtractor(80, 0, 25, 10), std::invalid_argument);
    EXPECT_THROW(MelFeatureExtractor(80, 16000, 0, 10), std::invalid_argument);
    EXPECT_THROW(MelFeatureExtractor(80, 16000, 25, 0), std::invalid_argument);
}

TEST(MelFeatureExtractorTest, ExtractReturnsKaldiNativeFbankFeatures) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(kSampleRate);

    const auto features = extractor.extract(audio);

    ASSERT_FALSE(features.empty());
    ASSERT_EQ(features.size() % kMels, 0u);
    const size_t num_frames = features.size() / kMels;
    EXPECT_EQ(num_frames, 98u);
    for (float value : features) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(MelFeatureExtractorTest, MatchesPyannoteCompatibleKaldiNativeReference) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(kSampleRate);

    const auto features = extractor.extract(audio);
    const auto reference = extract_reference_features(audio);

    ASSERT_EQ(features.size(), reference.size());
    for (size_t i = 0; i < features.size(); ++i) {
        EXPECT_NEAR(features[i], reference[i], 1e-5f);
    }
}

TEST(MelFeatureExtractorTest, ExtractRawMatchesKaldiReferenceBeforePyannotePostNormalization) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(kSampleRate);

    const auto raw_features = extractor.extract_raw(audio);
    const auto reference = extract_reference_raw_features(audio);

    ASSERT_EQ(raw_features.size(), reference.size());
    for (size_t i = 0; i < raw_features.size(); ++i) {
        EXPECT_NEAR(raw_features[i], reference[i], 1e-5f);
    }
}

TEST(MelFeatureExtractorTest, ExtractIsFrameMeanNormalizedVersionOfRawFeatures) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(kSampleRate);

    const auto raw_features = extractor.extract_raw(audio);
    const auto normalized_features = extractor.extract(audio);

    ASSERT_EQ(normalized_features.size(), raw_features.size());
    const auto expected = frame_mean_normalize(raw_features, kMels);
    ASSERT_EQ(normalized_features.size(), expected.size());
    for (size_t i = 0; i < normalized_features.size(); ++i) {
        EXPECT_NEAR(normalized_features[i], expected[i], 1e-6f);
    }
}

TEST(MelFeatureExtractorTest, AudioShorterThanOneWindowReturnsNoFeatures) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(399, kSampleRate);

    EXPECT_TRUE(extractor.extract(audio).empty());
}

TEST(MelFeatureExtractorTest, ExactWindowLengthProducesOneFrame) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(400, kSampleRate);

    const auto features = extractor.extract(audio);

    ASSERT_EQ(features.size(), static_cast<size_t>(kMels));
}

TEST(MelFeatureExtractorTest, OneFrameShiftPastWindowProducesTwoFrames) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(560, kSampleRate);

    const auto features = extractor.extract(audio);

    ASSERT_EQ(features.size(), static_cast<size_t>(2 * kMels));
}

TEST(MelFeatureExtractorTest, EachFrameIsMeanNormalizedAcrossMelBins) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(kSampleRate);

    const auto features = extractor.extract(audio);

    ASSERT_FALSE(features.empty());
    ASSERT_EQ(features.size() % kMels, 0u);
    const size_t num_frames = features.size() / kMels;
    for (size_t frame = 0; frame < num_frames; ++frame) {
        float mean = 0.0f;
        for (int mel = 0; mel < kMels; ++mel) {
            mean += features[frame * kMels + static_cast<size_t>(mel)];
        }
        mean /= static_cast<float>(kMels);
        EXPECT_NEAR(mean, 0.0f, 5e-6f);
    }
}

TEST(MelFeatureExtractorTest, ExtractIsDeterministicBecauseDitherIsDisabled) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const auto audio = make_test_audio(kSampleRate);

    const auto first = extractor.extract(audio);
    const auto second = extractor.extract(audio);

    ASSERT_EQ(first.size(), second.size());
    for (size_t i = 0; i < first.size(); ++i) {
        EXPECT_FLOAT_EQ(first[i], second[i]);
    }
}

TEST(MelFeatureExtractorTest, EmptyAudioReturnsNoFeatures) {
    MelFeatureExtractor extractor(kMels, kSampleRate, kWindowLengthMs, kStepWidthMs);
    const std::vector<float> empty_audio;

    EXPECT_TRUE(extractor.extract(empty_audio).empty());
}
