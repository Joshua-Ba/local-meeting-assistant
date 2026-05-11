// speaker_diarizer.h
//
// Pyannote-style speaker diarization pipeline:
//   1. Sliding-window segmentation (powerset model -> per-frame speaker
//      slot activations) over the input audio.
//   2. Per-chunk speaker embedding extraction (mel features -> embedding
//      model) for each of the up-to-three speaker slots in each chunk.
//   3. Agglomerative hierarchical clustering of the embeddings using
//      ahc_pyannote, with optional num_speakers hint.
//   4. Reconstruction of a global per-cluster activity matrix from the
//      windowed segmentation + cluster labels, then conversion to
//      per-speaker time segments using a top-K speaker_count rule.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "MelFeatureExtractor.h"
#include "config.h"


// One diarized speaker segment with optional embedding (used internally
// when building chunk-slot speaker hypotheses).
struct SpeakerSegment {
    int speaker_id;
    float start_time;
    float end_time;
    std::vector<float> embedding;
};


#ifdef UNIT_TESTS
class SpeakerDiarizerTestAccess;
#endif


class SpeakerDiarizer {
public:
    static constexpr int kSpeakerSlotCount = 3;

    using ActivityFrame = std::array<float, kSpeakerSlotCount>;
    using ActivityMatrix = std::vector<ActivityFrame>;
    using BinaryActivityFrame = std::array<bool, kSpeakerSlotCount>;
    using BinaryActivityMatrix = std::vector<BinaryActivityFrame>;
    using HardClusterFrame = std::array<int, kSpeakerSlotCount>;
    using HardClusterMatrix = std::vector<HardClusterFrame>;
    using ReconstructedActivityMatrix = std::vector<std::vector<float>>;

    // Per-window segmentation output: powerset-decoded activity and
    // its binarization, plus where the window sits in the global audio.
    struct SlidingWindowActivity {
        size_t start_sample = 0;
        size_t copied_samples = 0;
        float start_time = 0.0f;
        float frame_duration_seconds = 0.0f;
        ActivityMatrix activity;
        BinaryActivityMatrix binary_activity;
    };

    // Result of running the segmentation model over the whole audio
    // with a sliding window.
    struct SlidingSegmentationResult {
        float audio_seconds = 0.0f;
        float global_frame_duration_seconds = 0.0f;
        ActivityMatrix global_activity;
        std::vector<SlidingWindowActivity> chunks;
    };

#ifdef UNIT_TESTS
    friend class SpeakerDiarizerTestAccess;
#endif

private:
    Ort::Env env;
    std::unique_ptr<Ort::Session> segmentation_session;
    std::unique_ptr<Ort::Session> embedding_session;
    float min_segment_duration;
    float merge_gap;
    float clustering_threshold_;
    int num_speakers_hint_;
    int sample_rate_;
    int segmentation_window_samples_;
    int segmentation_step_samples_;
    int segmentation_frames_per_window_;
    int embedding_feature_dim_;
    int min_embedding_frames_;
    bool embedding_exclude_overlap_;

    MelFeatureExtractor melExtractor;

    // Upsample a binary activity mask for one speaker slot from
    // segmentation frame rate to embedding fbank frame rate using
    // PyTorch-equivalent nearest-neighbor interpolation.
    static std::vector<bool> nearest_neighbor_interpolate_mask(
        const BinaryActivityMatrix& activity,
        int slot_id,
        int output_frames);

    static int count_active_mask_frames(const std::vector<bool>& mask);

    // Build the fbank-frame mask used to compute the embedding for one
    // slot. If `exclude_overlap` and the resulting mask still has
    // enough frames, frames where multiple slots are active are dropped.
    static std::vector<bool> embedding_mask_for_slot(
        const BinaryActivityMatrix& activity,
        int slot_id,
        int output_frames,
        int min_frames,
        bool exclude_overlap);

    static bool embedding_is_valid(const std::vector<float>& embedding);

    int verify_segmentation_frame_geometry();

    [[nodiscard]] float segmentation_frame_duration_seconds(
        int num_frames, float window_seconds) const;

    [[nodiscard]] std::vector<SpeakerSegment> compute_chunk_slot_embeddings(
        std::span<const float> audio,
        const SlidingSegmentationResult& segmentation);

    // pyannote speaker_count: per global frame, the rounded mean of
    // "number of active slots" across all overlapping chunks.
    [[nodiscard]] std::vector<int> compute_speaker_count(
        const SlidingSegmentationResult& segmentation) const;

    // pyannote to_diarization: per global frame, pick the top count[t]
    // clusters by activation and set them active.
    [[nodiscard]] ReconstructedActivityMatrix to_diarization(
        const ReconstructedActivityMatrix& clustered_activity,
        const std::vector<int>& count) const;

    [[nodiscard]] std::vector<SpeakerSegment> segments_from_binary_diarization(
        const ReconstructedActivityMatrix& binary_diarization,
        float frame_duration_seconds,
        float audio_seconds) const;

    [[nodiscard]] HardClusterMatrix assign_chunk_slot_clusters(
        const std::vector<SpeakerSegment>& chunk_slot_embeddings,
        const SlidingSegmentationResult& segmentation,
        int num_speakers_hint,
        float clustering_threshold) const;

    static void normalize_l2(std::vector<float>& v);

    static float cosine_sim(const std::vector<float>& v1,
                            const std::vector<float>& v2);

    std::vector<Ort::Value> run_inference(const std::vector<int64_t>& input_shape,
                                          std::span<const float> data,
                                          Ort::Session& session);

public:
    static constexpr float kSegmentationWindowSeconds = 10.0f;
    static constexpr float kSegmentationStepRatio = 0.1f;
    static constexpr int kExpectedSegmentationFramesPerWindow = 589;
    static constexpr int kDiscardClusterLabel = -2;

    // Binarization parameters used by the legacy reconstruction-based
    // segment extractor (still used by unit tests). pyannote defaults
    // do no smoothing.
    struct BinarizationConfig {
        float onset = 0.50f;
        float offset = 0.50f;
        float min_duration_on = 0.00f;
        float min_duration_off = 0.00f;
    };

    SpeakerDiarizer(Config& config);

    bool is_loaded() const;

    [[nodiscard]] int segmentation_window_samples() const;
    [[nodiscard]] int segmentation_step_samples() const;
    [[nodiscard]] int segmentation_frames_per_window() const;

    // Powerset decoding: pick the argmax class from the 7-way softmax
    // and convert it to hard 0/1 per-slot activity.
    static ActivityFrame powerset_to_speaker_activity(const float* logits);

    static ActivityMatrix decode_powerset_scores(const float* output_data,
                                                 int num_frames);

    static BinaryActivityMatrix binarize_activity(const ActivityMatrix& activity);

    // Run the segmentation ONNX model on a single audio window.
    std::vector<float> run_segmentation(std::span<const float> audio,
                                        int& num_frames);

    [[nodiscard]] std::vector<float> compute_embedding(
        std::span<const float> chunk_audio,
        const BinaryActivityMatrix& activity,
        int slot_id);

    SlidingSegmentationResult run_sliding_segmentation(std::span<const float> audio);

    [[nodiscard]] ReconstructedActivityMatrix reconstruct_global_activity(
        const SlidingSegmentationResult& segmentation,
        const HardClusterMatrix& hard_clusters,
        int num_clusters) const;

    std::vector<SpeakerSegment> segments_from_reconstructed_activity(
        const ReconstructedActivityMatrix& activity,
        float frame_duration_seconds,
        float audio_seconds) const;

    std::vector<SpeakerSegment> segments_from_reconstructed_activity(
        const ReconstructedActivityMatrix& activity,
        float frame_duration_seconds,
        float audio_seconds,
        const BinarizationConfig& binarization_config) const;

    static int cluster_count_from_hard_clusters(const HardClusterMatrix& hard_clusters);

    // Top-level entry point. Returns diarized speaker segments sorted
    // by start time. `num_speakers_hint > 0` forces the cluster count;
    // otherwise the value from Config is used.
    std::vector<SpeakerSegment> diarize(std::span<const float> audio,
                                        int num_speakers_hint = -1);

    static void write_rttm(std::ostream& out,
                           std::string_view uri,
                           const std::vector<SpeakerSegment>& segments,
                           float time_offset_seconds = 0.0f);
};


// Find the speaker id with the largest overlap with the time range
// [start_time, end_time). Returns -1 if there is no overlap.
int get_speaker_at(const std::vector<SpeakerSegment>& segments,
                   float start_time,
                   float end_time);