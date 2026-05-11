#include "speaker_diarizer.h"

#include "test_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

class SpeakerDiarizerTestAccess {
   public:

    static float cosine_sim(const std::vector<float>& lhs, const std::vector<float>& rhs) {
        return SpeakerDiarizer::cosine_sim(lhs, rhs);
    }

    static SpeakerDiarizer::ActivityFrame powerset_to_speaker_activity(const std::vector<float>& logits) {
        return SpeakerDiarizer::powerset_to_speaker_activity(logits.data());
    }

    static SpeakerDiarizer::ActivityMatrix decode_scores(const std::vector<float>& output,
                                                         int frames) {
        return SpeakerDiarizer::decode_powerset_scores(output.data(), frames);
    }

    static SpeakerDiarizer::BinaryActivityMatrix binarize_activity(
        const SpeakerDiarizer::ActivityMatrix& activity) {
        return SpeakerDiarizer::binarize_activity(activity);
    }

    static std::vector<bool> nearest_neighbor_interpolate_mask(
        const SpeakerDiarizer::BinaryActivityMatrix& activity,
        int slot_id,
        int output_frames) {
        return SpeakerDiarizer::nearest_neighbor_interpolate_mask(activity,
                                                                  slot_id,
                                                                  output_frames);
    }

    static std::vector<bool> embedding_mask_for_slot(
        const SpeakerDiarizer::BinaryActivityMatrix& activity,
        int slot_id,
        int output_frames,
        int min_frames,
        bool exclude_overlap) {
        return SpeakerDiarizer::embedding_mask_for_slot(activity,
                                                        slot_id,
                                                        output_frames,
                                                        min_frames,
                                                        exclude_overlap);
    }

    static bool embedding_is_valid(const std::vector<float>& embedding) {
        return SpeakerDiarizer::embedding_is_valid(embedding);
    }

    static SpeakerDiarizer::ReconstructedActivityMatrix reconstruct_global_activity(
        SpeakerDiarizer& diarizer,
        const SpeakerDiarizer::SlidingSegmentationResult& segmentation,
        const SpeakerDiarizer::HardClusterMatrix& hard_clusters,
        int num_clusters) {
        return diarizer.reconstruct_global_activity(segmentation, hard_clusters, num_clusters);
    }

    static std::vector<int> compute_speaker_count(
        SpeakerDiarizer& diarizer,
        const SpeakerDiarizer::SlidingSegmentationResult& segmentation) {
        return diarizer.compute_speaker_count(segmentation);
    }

    static SpeakerDiarizer::ReconstructedActivityMatrix to_diarization(
        SpeakerDiarizer& diarizer,
        const SpeakerDiarizer::ReconstructedActivityMatrix& clustered_activity,
        const std::vector<int>& count) {
        return diarizer.to_diarization(clustered_activity, count);
    }

    static std::vector<SpeakerSegment> segments_from_binary_diarization(
        SpeakerDiarizer& diarizer,
        const SpeakerDiarizer::ReconstructedActivityMatrix& binary_diarization,
        float frame_duration_seconds,
        float audio_seconds) {
        return diarizer.segments_from_binary_diarization(binary_diarization,
                                                         frame_duration_seconds,
                                                         audio_seconds);
    }

    static SpeakerDiarizer::HardClusterMatrix assign_chunk_slot_clusters(
        SpeakerDiarizer& diarizer,
        const std::vector<SpeakerSegment>& chunk_slot_embeddings,
        const SpeakerDiarizer::SlidingSegmentationResult& segmentation,
        int num_speakers_hint = -1,
        float clustering_threshold = 0.05f) {
        return diarizer.assign_chunk_slot_clusters(chunk_slot_embeddings,
                                                   segmentation,
                                                   num_speakers_hint,
                                                   clustering_threshold);
    }

    static int cluster_count_from_hard_clusters(
        const SpeakerDiarizer::HardClusterMatrix& hard_clusters) {
        return SpeakerDiarizer::cluster_count_from_hard_clusters(hard_clusters);
    }

    static std::vector<SpeakerSegment> segments_from_reconstructed_activity(
        SpeakerDiarizer& diarizer,
        const SpeakerDiarizer::ReconstructedActivityMatrix& activity,
        float frame_duration_seconds,
        float audio_seconds) {
        return diarizer.segments_from_reconstructed_activity(activity,
                                                            frame_duration_seconds,
                                                            audio_seconds);
    }

    static std::vector<SpeakerSegment> segments_from_reconstructed_activity(
        SpeakerDiarizer& diarizer,
        const SpeakerDiarizer::ReconstructedActivityMatrix& activity,
        float frame_duration_seconds,
        float audio_seconds,
        const SpeakerDiarizer::BinarizationConfig& binarization_config) {
        return diarizer.segments_from_reconstructed_activity(activity,
                                                            frame_duration_seconds,
                                                            audio_seconds,
                                                            binarization_config);
    }

    static void normalize_l2(std::vector<float>& embedding) {
        SpeakerDiarizer::normalize_l2(embedding);
    }

};

namespace {
std::vector<float> make_logits_for_classes(const std::vector<int>& classes) {
    std::vector<float> logits(classes.size() * 7, -8.0f);
    for (size_t frame = 0; frame < classes.size(); ++frame) {
        logits[frame * 7 + classes[frame]] = 8.0f;
    }
    return logits;
}
}

class SpeakerDiarizerTest : public testing::Test {
};

TEST_F(SpeakerDiarizerTest, MissingModelsLeaveDiarizerUnloaded) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);

    EXPECT_FALSE(diarizer.is_loaded());
}

TEST_F(SpeakerDiarizerTest, PyannoteInputGeometryUsesTenSecondWindowsAndOneSecondSteps) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);

    EXPECT_FLOAT_EQ(SpeakerDiarizer::kSegmentationWindowSeconds, 10.0f);
    EXPECT_FLOAT_EQ(SpeakerDiarizer::kSegmentationStepRatio, 0.1f);
    EXPECT_GT(SpeakerDiarizer::kExpectedSegmentationFramesPerWindow, 0);
    EXPECT_EQ(diarizer.segmentation_window_samples(), 160000);
    EXPECT_EQ(diarizer.segmentation_step_samples(), 16000);
    EXPECT_EQ(diarizer.segmentation_frames_per_window(),
              SpeakerDiarizer::kExpectedSegmentationFramesPerWindow);
}

TEST_F(SpeakerDiarizerTest, MathHelpersReturnExpectedValues) {

    EXPECT_NEAR(SpeakerDiarizerTestAccess::cosine_sim({1.0f, 0.0f}, {1.0f, 0.0f}), 1.0f, 1e-6f);
    EXPECT_NEAR(SpeakerDiarizerTestAccess::cosine_sim({1.0f, 0.0f}, {0.0f, 1.0f}), 0.0f, 1e-6f);
    EXPECT_NEAR(SpeakerDiarizerTestAccess::cosine_sim({0.0f, 0.0f}, {1.0f, 0.0f}), 0.0f, 1e-6f);
    EXPECT_THROW(SpeakerDiarizerTestAccess::cosine_sim({1.0f}, {1.0f, 2.0f}), std::invalid_argument);

    std::vector<float> embedding = {3.0f, 4.0f};
    SpeakerDiarizerTestAccess::normalize_l2(embedding);
    EXPECT_NEAR(embedding[0], 0.6f, 1e-6f);
    EXPECT_NEAR(embedding[1], 0.8f, 1e-6f);

}

TEST_F(SpeakerDiarizerTest, PowersetToSpeakerActivityUsesArgmaxClassMapping) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);
    const std::vector<float> equal_logits(7, 0.0f);

    const auto scores = SpeakerDiarizerTestAccess::powerset_to_speaker_activity(equal_logits);

    EXPECT_EQ(scores, (SpeakerDiarizer::ActivityFrame{0.0f, 0.0f, 0.0f}));

    const auto overlap_logits = make_logits_for_classes({4});
    const auto overlap_scores = SpeakerDiarizerTestAccess::powerset_to_speaker_activity(overlap_logits);

    EXPECT_GT(overlap_scores[0], 0.99f);
    EXPECT_GT(overlap_scores[1], 0.99f);
    EXPECT_LT(overlap_scores[2], 0.01f);
}

TEST_F(SpeakerDiarizerTest, DecodePowersetScoresReturnsOneActivityFramePerModelFrame) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);
    const auto logits = make_logits_for_classes({1, 6});

    const auto scores = SpeakerDiarizerTestAccess::decode_scores(logits, 2);

    ASSERT_EQ(scores.size(), 2u);
    EXPECT_GT(scores[0][0], 0.99f);
    EXPECT_LT(scores[0][1], 0.01f);
    EXPECT_LT(scores[0][2], 0.01f);
    EXPECT_LT(scores[1][0], 0.01f);
    EXPECT_GT(scores[1][1], 0.99f);
    EXPECT_GT(scores[1][2], 0.99f);
}

TEST_F(SpeakerDiarizerTest, BinarizeActivityUsesHalfThresholdPerSpeakerSlot) {
    const SpeakerDiarizer::ActivityMatrix activity = {
        SpeakerDiarizer::ActivityFrame{0.49f, 0.50f, 0.51f},
        SpeakerDiarizer::ActivityFrame{0.0f, 1.0f, 0.25f},
    };

    const auto binary = SpeakerDiarizerTestAccess::binarize_activity(activity);

    ASSERT_EQ(binary.size(), 2u);
    EXPECT_EQ(binary[0], (SpeakerDiarizer::BinaryActivityFrame{false, true, true}));
    EXPECT_EQ(binary[1], (SpeakerDiarizer::BinaryActivityFrame{false, true, false}));
}

TEST_F(SpeakerDiarizerTest, NearestNeighborInterpolationMapsPyannoteMasksToFbankFrames) {
    const SpeakerDiarizer::BinaryActivityMatrix activity = {
        SpeakerDiarizer::BinaryActivityFrame{true, false, false},
        SpeakerDiarizer::BinaryActivityFrame{false, true, false},
        SpeakerDiarizer::BinaryActivityFrame{true, false, true},
    };

    const auto slot_0_mask = SpeakerDiarizerTestAccess::nearest_neighbor_interpolate_mask(activity, 0, 5);
    const auto slot_1_mask = SpeakerDiarizerTestAccess::nearest_neighbor_interpolate_mask(activity, 1, 5);

    EXPECT_EQ(slot_0_mask, (std::vector<bool>{true, true, false, true, true}));
    EXPECT_EQ(slot_1_mask, (std::vector<bool>{false, false, true, false, false}));
    EXPECT_TRUE(SpeakerDiarizerTestAccess::nearest_neighbor_interpolate_mask(activity, 3, 5).empty());
    EXPECT_TRUE(SpeakerDiarizerTestAccess::nearest_neighbor_interpolate_mask(activity, 0, 0).empty());
}

TEST_F(SpeakerDiarizerTest, EmbeddingMaskExcludesOverlapWhenEnoughCleanFramesRemain) {
    const SpeakerDiarizer::BinaryActivityMatrix activity = {
        SpeakerDiarizer::BinaryActivityFrame{true, false, false},
        SpeakerDiarizer::BinaryActivityFrame{true, true, false},
        SpeakerDiarizer::BinaryActivityFrame{true, false, false},
        SpeakerDiarizer::BinaryActivityFrame{false, true, false},
    };

    const auto clean_mask = SpeakerDiarizerTestAccess::embedding_mask_for_slot(activity, 0, 4, 2, true);
    const auto fallback_mask = SpeakerDiarizerTestAccess::embedding_mask_for_slot(activity, 0, 4, 3, true);
    const auto full_mask = SpeakerDiarizerTestAccess::embedding_mask_for_slot(activity, 0, 4, 2, false);

    EXPECT_EQ(clean_mask, (std::vector<bool>{true, false, true, false}));
    EXPECT_EQ(fallback_mask, (std::vector<bool>{true, true, true, false}));
    EXPECT_EQ(full_mask, (std::vector<bool>{true, true, true, false}));
}

TEST_F(SpeakerDiarizerTest, ReconstructGlobalActivitySumsOverlappingChunkContributions) {
    Config config = make_test_config();
    config.min_segment_duration = 0.1f;
    SpeakerDiarizer diarizer(config);
    SpeakerDiarizer::SlidingSegmentationResult segmentation;
    segmentation.audio_seconds = 3.0f;
    segmentation.global_frame_duration_seconds = 1.0f;
    segmentation.global_activity = {
        SpeakerDiarizer::ActivityFrame{},
        SpeakerDiarizer::ActivityFrame{},
        SpeakerDiarizer::ActivityFrame{},
    };
    segmentation.chunks = {
        {
            0,
            16000,
            0.0f,
            1.0f,
            {
                SpeakerDiarizer::ActivityFrame{0.90f, 0.10f, 0.00f},
                SpeakerDiarizer::ActivityFrame{0.80f, 0.70f, 0.00f},
            },
            {
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{true, true, false},
            },
        },
        {
            16000,
            16000,
            1.0f,
            1.0f,
            {
                SpeakerDiarizer::ActivityFrame{0.60f, 0.20f, 0.95f},
                SpeakerDiarizer::ActivityFrame{0.40f, 0.90f, 0.10f},
            },
            {
                SpeakerDiarizer::BinaryActivityFrame{true, false, true},
                SpeakerDiarizer::BinaryActivityFrame{false, true, false},
            },
        },
    };
    const SpeakerDiarizer::HardClusterMatrix hard_clusters = {
        SpeakerDiarizer::HardClusterFrame{0, 1, SpeakerDiarizer::kDiscardClusterLabel},
        SpeakerDiarizer::HardClusterFrame{0, 1, 0},
    };

    const auto reconstructed = SpeakerDiarizerTestAccess::reconstruct_global_activity(diarizer,
                                                                                      segmentation,
                                                                                      hard_clusters,
                                                                                      2);

    ASSERT_EQ(reconstructed.size(), 3u);
    ASSERT_EQ(reconstructed[0].size(), 2u);
    EXPECT_NEAR(reconstructed[0][0], 0.90f, 1e-6f);
    EXPECT_NEAR(reconstructed[0][1], 0.10f, 1e-6f);
    EXPECT_NEAR(reconstructed[1][0], 1.75f, 1e-6f);
    EXPECT_NEAR(reconstructed[1][1], 0.90f, 1e-6f);
    EXPECT_NEAR(reconstructed[2][0], 0.40f, 1e-6f);
    EXPECT_NEAR(reconstructed[2][1], 0.90f, 1e-6f);

    const auto segments = SpeakerDiarizerTestAccess::segments_from_reconstructed_activity(diarizer,
                                                                                         reconstructed,
                                                                                         1.0f,
                                                                                         3.0f);
    ASSERT_EQ(segments.size(), 2u);
    EXPECT_EQ(segments[0].speaker_id, 0);
    EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
    EXPECT_FLOAT_EQ(segments[0].end_time, 2.0f);
    EXPECT_EQ(segments[1].speaker_id, 1);
    EXPECT_FLOAT_EQ(segments[1].start_time, 1.0f);
    EXPECT_FLOAT_EQ(segments[1].end_time, 3.0f);
}

TEST_F(SpeakerDiarizerTest, ComputeSpeakerCountRoundsMeanAcrossOverlappingChunks) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);
    SpeakerDiarizer::SlidingSegmentationResult segmentation;
    segmentation.audio_seconds = 3.0f;
    segmentation.global_frame_duration_seconds = 1.0f;
    segmentation.global_activity = {
        SpeakerDiarizer::ActivityFrame{},
        SpeakerDiarizer::ActivityFrame{},
        SpeakerDiarizer::ActivityFrame{},
    };
    segmentation.chunks = {
        {
            0,
            16000,
            0.0f,
            1.0f,
            {},
            {
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
            },
        },
        {
            16000,
            16000,
            1.0f,
            1.0f,
            {},
            {
                SpeakerDiarizer::BinaryActivityFrame{true, true, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
            },
        },
    };

    const auto count = SpeakerDiarizerTestAccess::compute_speaker_count(diarizer, segmentation);

    EXPECT_EQ(count, (std::vector<int>{1, 2, 0}));
}

TEST_F(SpeakerDiarizerTest, ToDiarizationSelectsTopKClustersPerFrame) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);
    const SpeakerDiarizer::ReconstructedActivityMatrix clustered_activity = {
        {0.8f, 0.2f, 0.1f},
        {0.1f, 0.9f, 0.8f},
        {0.5f, 0.4f, 0.3f},
        {0.7f, 0.95f, 0.6f},
    };
    const std::vector<int> count = {1, 2, 0, 1};

    const auto diarization = SpeakerDiarizerTestAccess::to_diarization(diarizer,
                                                                       clustered_activity,
                                                                       count);

    EXPECT_EQ(diarization,
              (SpeakerDiarizer::ReconstructedActivityMatrix{
                  {1.0f, 0.0f, 0.0f},
                  {0.0f, 1.0f, 1.0f},
                  {0.0f, 0.0f, 0.0f},
                  {0.0f, 1.0f, 0.0f},
              }));
}

TEST_F(SpeakerDiarizerTest, SegmentsFromBinaryDiarizationBuildsRunsPerCluster) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);
    const SpeakerDiarizer::ReconstructedActivityMatrix binary_diarization = {
        {1.0f, 0.0f},
        {1.0f, 1.0f},
        {0.0f, 1.0f},
        {0.0f, 0.0f},
    };

    const auto segments = SpeakerDiarizerTestAccess::segments_from_binary_diarization(diarizer,
                                                                                       binary_diarization,
                                                                                       0.5f,
                                                                                       2.0f);

    ASSERT_EQ(segments.size(), 2u);
    EXPECT_EQ(segments[0].speaker_id, 0);
    EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
    EXPECT_FLOAT_EQ(segments[0].end_time, 1.0f);
    EXPECT_EQ(segments[1].speaker_id, 1);
    EXPECT_FLOAT_EQ(segments[1].start_time, 0.5f);
    EXPECT_FLOAT_EQ(segments[1].end_time, 1.5f);
}

TEST_F(SpeakerDiarizerTest, SegmentsFromReconstructedActivityUsesHysteresisThresholds) {
    Config config = make_test_config();
    config.min_segment_duration = 0.0f;
    SpeakerDiarizer diarizer(config);
    const SpeakerDiarizer::ReconstructedActivityMatrix activity = {
        {0.50f},
        {0.54f},
        {0.56f},
        {0.50f},
        {0.46f},
        {0.44f},
    };
    SpeakerDiarizer::BinarizationConfig binarization_config;
    binarization_config.onset = 0.55f;
    binarization_config.offset = 0.45f;
    binarization_config.min_duration_on = 0.0f;

    const auto segments = SpeakerDiarizerTestAccess::segments_from_reconstructed_activity(
        diarizer,
        activity,
        0.1f,
        0.6f,
        binarization_config);

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(segments[0].speaker_id, 0);
    EXPECT_FLOAT_EQ(segments[0].start_time, 0.2f);
    EXPECT_FLOAT_EQ(segments[0].end_time, 0.5f);
}

TEST_F(SpeakerDiarizerTest, SegmentsFromReconstructedActivityDropsShortIslands) {
    Config config = make_test_config();
    config.min_segment_duration = 0.0f;
    SpeakerDiarizer diarizer(config);
    const SpeakerDiarizer::ReconstructedActivityMatrix activity = {
        {0.00f},
        {0.60f},
        {0.40f},
        {0.00f},
    };
    SpeakerDiarizer::BinarizationConfig binarization_config;
    binarization_config.min_duration_on = 0.10f;

    const auto segments = SpeakerDiarizerTestAccess::segments_from_reconstructed_activity(
        diarizer,
        activity,
        0.05f,
        0.2f,
        binarization_config);

    EXPECT_TRUE(segments.empty());
}

TEST_F(SpeakerDiarizerTest, SegmentsFromReconstructedActivityClosesShortGaps) {
    Config config = make_test_config();
    config.min_segment_duration = 0.0f;
    SpeakerDiarizer diarizer(config);
    const SpeakerDiarizer::ReconstructedActivityMatrix activity = {
        {0.60f},
        {0.60f},
        {0.40f},
        {0.60f},
        {0.60f},
    };
    SpeakerDiarizer::BinarizationConfig binarization_config;
    binarization_config.min_duration_on = 0.0f;
    binarization_config.min_duration_off = 0.06f;

    const auto segments = SpeakerDiarizerTestAccess::segments_from_reconstructed_activity(
        diarizer,
        activity,
        0.05f,
        0.25f,
        binarization_config);

    ASSERT_EQ(segments.size(), 1u);
    EXPECT_EQ(segments[0].speaker_id, 0);
    EXPECT_FLOAT_EQ(segments[0].start_time, 0.0f);
    EXPECT_FLOAT_EQ(segments[0].end_time, 0.25f);
}

TEST_F(SpeakerDiarizerTest, WriteRttmUsesPyannoteCompatibleSpeakerLines) {
    std::vector<SpeakerSegment> segments = {
        {0, 1.25f, 3.0f, {}},
        {12, 4.0f, 4.5f, {}},
        {-1, 5.0f, 6.0f, {}},
        {2, 7.0f, 7.0f, {}},
    };
    std::ostringstream out;

    SpeakerDiarizer::write_rttm(out, "sample", segments, 10.0f);

    EXPECT_EQ(out.str(),
              "SPEAKER sample 1 11.250 1.750 <NA> <NA> SPEAKER_00 <NA> <NA>\n"
              "SPEAKER sample 1 14.000 0.500 <NA> <NA> SPEAKER_12 <NA> <NA>\n");
}


TEST_F(SpeakerDiarizerTest, AssignChunkSlotClustersKeepsDistinctCentroidsSeparateAndDiscardsInactiveSlots) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    SpeakerDiarizer::SlidingSegmentationResult segmentation;
    segmentation.chunks = {
        {
            0,
            160000,
            0.0f,
            0.0f,
            {},
            {
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, true, false},
                SpeakerDiarizer::BinaryActivityFrame{false, true, false},
                SpeakerDiarizer::BinaryActivityFrame{false, true, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
            },
        },
        {
            160000,
            160000,
            10.0f,
            0.0f,
            {},
            {
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{true, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, true, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, true},
                SpeakerDiarizer::BinaryActivityFrame{false, false, true},
                SpeakerDiarizer::BinaryActivityFrame{false, false, true},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
                SpeakerDiarizer::BinaryActivityFrame{false, false, false},
            },
        },
    };
    const std::vector<SpeakerSegment> chunk_slot_embeddings = {
        {-1, 0.0f, 10.0f, {1.0f, 0.0f}},
        {-1, 0.0f, 10.0f, {0.0f, 1.0f}},
        {-1, 0.0f, 10.0f, {}},
        {-1, 1.0f, 11.0f, {0.99f, 0.01f}},
        {-1, 1.0f, 11.0f, {nan, nan}},
        {-1, 1.0f, 11.0f, {0.01f, 0.99f}},
    };

    const auto hard_clusters = SpeakerDiarizerTestAccess::assign_chunk_slot_clusters(diarizer,
                                                                                     chunk_slot_embeddings,
                                                                                     segmentation,
                                                                                     2,
                                                                                     0.0f);

    ASSERT_EQ(hard_clusters.size(), 2u);
    EXPECT_EQ(hard_clusters[0], (SpeakerDiarizer::HardClusterFrame{
                                    0,
                                    1,
                                    SpeakerDiarizer::kDiscardClusterLabel,
                                }));
    EXPECT_EQ(hard_clusters[1], (SpeakerDiarizer::HardClusterFrame{
                                    0,
                                    SpeakerDiarizer::kDiscardClusterLabel,
                                    1,
                                }));
    EXPECT_EQ(SpeakerDiarizerTestAccess::cluster_count_from_hard_clusters(hard_clusters), 2);
    EXPECT_TRUE(SpeakerDiarizerTestAccess::embedding_is_valid({1.0f, 0.0f}));
    EXPECT_FALSE(SpeakerDiarizerTestAccess::embedding_is_valid({nan, nan}));
    EXPECT_FALSE(SpeakerDiarizerTestAccess::embedding_is_valid({}));
}

TEST_F(SpeakerDiarizerTest, GetSpeakerAtReturnsSegmentWithLargestOverlap) {
    const std::vector<SpeakerSegment> segments = {
        {0, 0.0f, 1.0f, {}},
        {1, 1.0f, 3.0f, {}},
    };

    EXPECT_EQ(get_speaker_at(segments, 0.25f, 0.75f), 0);
    EXPECT_EQ(get_speaker_at(segments, 1.1f, 2.0f), 1);
    EXPECT_EQ(get_speaker_at(segments, 3.1f, 4.0f), -1);
}

TEST_F(SpeakerDiarizerTest, DiarizeReturnsEmptyWhenModelsAreNotLoaded) {
    Config config = make_test_config();
    SpeakerDiarizer diarizer(config);

    EXPECT_TRUE(diarizer.diarize({}).empty());
}
