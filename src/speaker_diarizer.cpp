// speaker_diarizer.cpp
//
// Implementation of the pyannote-style diarization pipeline. The
// high-level flow lives in diarize(); the helpers below implement each
// stage (segmentation, embedding, clustering, reconstruction, segment
// extraction).

#include "speaker_diarizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "ahc_pyannote.h"


// =====================================================================
// Internal helpers (file-local)
// =====================================================================

namespace {

constexpr int kPowersetClassCount = 7;
constexpr int kSpeakerSlotCount = SpeakerDiarizer::kSpeakerSlotCount;
constexpr float kSpeakerActivityThreshold = 0.5f;
constexpr int kDefaultMinEmbeddingFrames = 200;

using PowersetProbabilities = std::array<float, kPowersetClassCount>;

// Powerset class to per-slot activity mapping.
// Class order: silence, A, B, C, A+B, A+C, B+C.
constexpr std::array<std::array<bool, kSpeakerSlotCount>, kPowersetClassCount>
kPowersetClassToSpeakers{{
    {false, false, false},
    {true, false, false},
    {false, true, false},
    {false, false, true},
    {true, true, false},
    {true, false, true},
    {false, true, true},
}};


// Numerically-stable softmax over the 7 powerset logits.
PowersetProbabilities softmax_probabilities(const float* logits) {
    PowersetProbabilities probabilities{};
    const float max_logit = *std::max_element(logits, logits + kPowersetClassCount);
    float denominator = 0.0f;
    for (int i = 0; i < kPowersetClassCount; ++i) {
        denominator += std::exp(logits[i] - max_logit);
    }
    for (int i = 0; i < kPowersetClassCount; ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit) / denominator;
    }
    return probabilities;
}


std::string format_rttm_speaker_label(const int speaker_id) {
    std::ostringstream label;
    label << "SPEAKER_" << std::setw(2) << std::setfill('0') << speaker_id;
    return label.str();
}

}  // namespace


// =====================================================================
// Construction
// =====================================================================

SpeakerDiarizer::SpeakerDiarizer(Config& config)
    : env(ORT_LOGGING_LEVEL_WARNING, "diarizer"),
      min_segment_duration(config.min_segment_duration),
      merge_gap(config.merge_gap),
      clustering_threshold_(config.clustering_threshold),
      num_speakers_hint_(config.num_speakers_hint),
      sample_rate_(config.whisper_sample_rate),
      segmentation_window_samples_(static_cast<int>(
          kSegmentationWindowSeconds * static_cast<float>(config.whisper_sample_rate))),
      segmentation_step_samples_(static_cast<int>(
          kSegmentationWindowSeconds * kSegmentationStepRatio *
          static_cast<float>(config.whisper_sample_rate))),
      segmentation_frames_per_window_(kExpectedSegmentationFramesPerWindow),
      embedding_feature_dim_(config.n_mels),
      min_embedding_frames_(kDefaultMinEmbeddingFrames),
      embedding_exclude_overlap_(true),
      melExtractor(config.n_mels,
                   config.whisper_sample_rate,
                   config.window_length_ms,
                   config.step_width_ms) {
    try {
        Ort::SessionOptions options;
        segmentation_session = std::make_unique<Ort::Session>(
            env, config.segmentation_model.c_str(), options);
        embedding_session = std::make_unique<Ort::Session>(
            env, config.embedding_model.c_str(), options);

        segmentation_frames_per_window_ = verify_segmentation_frame_geometry();
        if (segmentation_frames_per_window_ != kExpectedSegmentationFramesPerWindow) {
            std::cerr << "Warning: segmentation model produced "
                      << segmentation_frames_per_window_
                      << " frames for a " << kSegmentationWindowSeconds
                      << "s window; expected "
                      << kExpectedSegmentationFramesPerWindow << std::endl;
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load diarization models: " << e.what() << std::endl;
        segmentation_session.reset();
        embedding_session.reset();
    }
}


bool SpeakerDiarizer::is_loaded() const {
    return segmentation_session && embedding_session;
}


int SpeakerDiarizer::segmentation_window_samples() const {
    return segmentation_window_samples_;
}


int SpeakerDiarizer::segmentation_step_samples() const {
    return segmentation_step_samples_;
}


int SpeakerDiarizer::segmentation_frames_per_window() const {
    return segmentation_frames_per_window_;
}


// =====================================================================
// Top-level pipeline
// =====================================================================

std::vector<SpeakerSegment> SpeakerDiarizer::diarize(std::span<const float> audio,
                                                    int num_speakers_hint) {
    if (!is_loaded()) {
        return {};
    }

    const float audio_seconds = static_cast<float>(audio.size()) /
                                static_cast<float>(sample_rate_);
    if (audio_seconds <= 0.0f) {
        return {};
    }

    const auto segmentation = run_sliding_segmentation(audio);
    auto chunk_slot_speakers = compute_chunk_slot_embeddings(audio, segmentation);

    const int effective_num_speakers_hint = num_speakers_hint > 0
        ? num_speakers_hint
        : num_speakers_hint_;
    const auto hard_clusters = assign_chunk_slot_clusters(
        chunk_slot_speakers,
        segmentation,
        effective_num_speakers_hint,
        clustering_threshold_);

    const auto reconstructed_activity = reconstruct_global_activity(
        segmentation,
        hard_clusters,
        cluster_count_from_hard_clusters(hard_clusters));

    const auto count = compute_speaker_count(segmentation);
    const auto binary_diarization = to_diarization(reconstructed_activity, count);
    return segments_from_binary_diarization(binary_diarization,
                                            segmentation.global_frame_duration_seconds,
                                            segmentation.audio_seconds);
}


// =====================================================================
// Segmentation: powerset decoding and sliding-window inference
// =====================================================================

SpeakerDiarizer::ActivityFrame SpeakerDiarizer::powerset_to_speaker_activity(
    const float* logits) {
    ActivityFrame frame_scores{};
    if (!logits) {
        return frame_scores;
    }

    const auto probabilities = softmax_probabilities(logits);

    // Pyannote-style: argmax over powerset classes, then hard conversion
    // to speaker indicators (0 or 1). NOT a sum of marginal probabilities.
    int best_class = 0;
    float best_prob = probabilities[0];
    for (int c = 1; c < kPowersetClassCount; ++c) {
        if (probabilities[c] > best_prob) {
            best_prob = probabilities[c];
            best_class = c;
        }
    }

    for (int speaker = 0; speaker < kSpeakerSlotCount; ++speaker) {
        if (kPowersetClassToSpeakers[best_class][speaker]) {
            frame_scores[speaker] = 1.0f;
        }
    }
    return frame_scores;
}


SpeakerDiarizer::ActivityMatrix SpeakerDiarizer::decode_powerset_scores(
    const float* output_data,
    const int num_frames) {
    ActivityMatrix scores;
    if (!output_data || num_frames <= 0) {
        return scores;
    }

    scores.reserve(static_cast<size_t>(num_frames));
    for (int frame = 0; frame < num_frames; ++frame) {
        scores.push_back(powerset_to_speaker_activity(
            output_data + frame * kPowersetClassCount));
    }
    return scores;
}


SpeakerDiarizer::BinaryActivityMatrix SpeakerDiarizer::binarize_activity(
    const ActivityMatrix& activity) {
    BinaryActivityMatrix binary_activity;
    binary_activity.reserve(activity.size());

    for (const auto& frame : activity) {
        BinaryActivityFrame binary_frame{};
        for (int speaker = 0; speaker < kSpeakerSlotCount; ++speaker) {
            binary_frame[speaker] = frame[speaker] >= kSpeakerActivityThreshold;
        }
        binary_activity.push_back(binary_frame);
    }
    return binary_activity;
}


std::vector<float> SpeakerDiarizer::run_segmentation(std::span<const float> audio,
                                                    int& num_frames) {
    std::vector<int64_t> input_shape = {1, 1, static_cast<int64_t>(audio.size())};
    auto output = run_inference(input_shape, audio, *segmentation_session);
    float* output_data = output[0].GetTensorMutableData<float>();
    auto output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    num_frames = output_shape[1];
    return std::vector<float>(output_data, output_data + num_frames * 7);
}


SpeakerDiarizer::SlidingSegmentationResult SpeakerDiarizer::run_sliding_segmentation(
    std::span<const float> audio) {
    SlidingSegmentationResult result;
    if (audio.empty() || sample_rate_ <= 0 || segmentation_window_samples_ <= 0 ||
        segmentation_step_samples_ <= 0 || segmentation_frames_per_window_ <= 0) {
        return result;
    }

    result.audio_seconds = static_cast<float>(audio.size()) /
                           static_cast<float>(sample_rate_);
    if (result.audio_seconds <= 0.0f) {
        return result;
    }

    result.global_frame_duration_seconds = segmentation_frame_duration_seconds(
        segmentation_frames_per_window_, kSegmentationWindowSeconds);
    if (result.global_frame_duration_seconds <= 0.0f) {
        return result;
    }

    const auto num_global_frames = static_cast<size_t>(
        std::ceil(result.audio_seconds / result.global_frame_duration_seconds));
    ActivityMatrix score_sums(num_global_frames, ActivityFrame{});
    std::vector<int> frame_counts(num_global_frames, 0);
    std::vector<float> window(static_cast<size_t>(segmentation_window_samples_), 0.0f);

    // Slide the segmentation window across the audio at fixed step size.
    for (size_t window_start_sample = 0; window_start_sample < audio.size();
         window_start_sample += static_cast<size_t>(segmentation_step_samples_)) {
        std::fill(window.begin(), window.end(), 0.0f);

        const size_t remaining_samples = audio.size() - window_start_sample;
        const size_t samples_to_copy = std::min(window.size(), remaining_samples);
        std::copy_n(audio.data() + window_start_sample, samples_to_copy, window.begin());

        int num_frames = 0;
        const auto output_data = run_segmentation(window, num_frames);
        const auto window_activity = decode_powerset_scores(output_data.data(), num_frames);
        const auto window_binary_activity = binarize_activity(window_activity);
        const float local_frame_duration = segmentation_frame_duration_seconds(
            num_frames, kSegmentationWindowSeconds);
        const float window_start_seconds = static_cast<float>(window_start_sample) /
                                           static_cast<float>(sample_rate_);

        result.chunks.push_back({
            window_start_sample,
            samples_to_copy,
            window_start_seconds,
            local_frame_duration,
            window_activity,
            window_binary_activity,
        });

        // Accumulate per-window scores into the global frame grid.
        for (int frame = 0; frame < num_frames; ++frame) {
            const float global_time = window_start_seconds +
                                      static_cast<float>(frame) * local_frame_duration;
            if (global_time >= result.audio_seconds) {
                continue;
            }

            const auto global_frame = static_cast<size_t>(
                std::floor(global_time / result.global_frame_duration_seconds));
            if (global_frame >= score_sums.size()) {
                continue;
            }

            for (int speaker = 0; speaker < kSpeakerSlotCount; ++speaker) {
                score_sums[global_frame][speaker] +=
                    window_activity[static_cast<size_t>(frame)][speaker];
            }
            frame_counts[global_frame] += 1;
        }
    }

    // Average overlapping windows into the global activity matrix.
    result.global_activity.assign(num_global_frames, ActivityFrame{});
    for (size_t frame = 0; frame < result.global_activity.size(); ++frame) {
        if (frame_counts[frame] == 0) {
            continue;
        }
        for (int speaker = 0; speaker < kSpeakerSlotCount; ++speaker) {
            result.global_activity[frame][speaker] = score_sums[frame][speaker] /
                                                     static_cast<float>(frame_counts[frame]);
        }
    }

    return result;
}


int SpeakerDiarizer::verify_segmentation_frame_geometry() {
    int num_frames = 0;
    const std::vector<float> dummy_audio(
        static_cast<size_t>(segmentation_window_samples_), 0.0f);
    (void)run_segmentation(dummy_audio, num_frames);
    return num_frames > 0 ? num_frames : kExpectedSegmentationFramesPerWindow;
}


float SpeakerDiarizer::segmentation_frame_duration_seconds(
    const int num_frames,
    const float window_seconds) const {
    if (num_frames <= 0) {
        return 0.0f;
    }
    if (num_frames == segmentation_frames_per_window_ &&
        segmentation_frames_per_window_ > 0) {
        return kSegmentationWindowSeconds /
               static_cast<float>(segmentation_frames_per_window_);
    }
    return window_seconds / static_cast<float>(num_frames);
}


// =====================================================================
// Embedding extraction
// =====================================================================

std::vector<bool> SpeakerDiarizer::nearest_neighbor_interpolate_mask(
    const BinaryActivityMatrix& activity,
    const int slot_id,
    const int output_frames) {
    if (activity.empty() || output_frames <= 0 ||
        slot_id < 0 || slot_id >= kSpeakerSlotCount) {
        return {};
    }

    std::vector<bool> interpolated(static_cast<size_t>(output_frames), false);
    const auto input_size = static_cast<float>(activity.size());
    const auto output_size = static_cast<float>(output_frames);

    // PyTorch F.interpolate(mode="nearest") formula:
    //   src_idx = floor((dst_idx + 0.5) * input_size / output_size)
    for (int frame = 0; frame < output_frames; ++frame) {
        const float src_float = (static_cast<float>(frame) + 0.5f) *
                                input_size / output_size;
        size_t src = static_cast<size_t>(std::floor(src_float));
        if (src >= activity.size()) src = activity.size() - 1;

        interpolated[static_cast<size_t>(frame)] =
            activity[src][static_cast<size_t>(slot_id)];
    }
    return interpolated;
}


int SpeakerDiarizer::count_active_mask_frames(const std::vector<bool>& mask) {
    return static_cast<int>(std::ranges::count(mask, true));
}


std::vector<bool> SpeakerDiarizer::embedding_mask_for_slot(
    const BinaryActivityMatrix& activity,
    const int slot_id,
    const int output_frames,
    const int min_frames,
    const bool exclude_overlap) {
    const auto full_mask = nearest_neighbor_interpolate_mask(
        activity, slot_id, output_frames);
    if (!exclude_overlap || full_mask.empty()) {
        return full_mask;
    }

    // Build a "clean" activity matrix where frames with multiple active
    // slots are zeroed out, then check whether the resulting upsampled
    // mask still has enough frames.
    BinaryActivityMatrix clean_activity;
    clean_activity.reserve(activity.size());
    for (const auto& frame : activity) {
        const int active_slots = static_cast<int>(std::ranges::count(frame, true));
        clean_activity.push_back(active_slots < 2 ? frame : BinaryActivityFrame{});
    }

    const auto clean_mask = nearest_neighbor_interpolate_mask(
        clean_activity, slot_id, output_frames);
    if (count_active_mask_frames(clean_mask) >= min_frames) {
        return clean_mask;
    }
    return full_mask;
}


bool SpeakerDiarizer::embedding_is_valid(const std::vector<float>& embedding) {
    return !embedding.empty() &&
           std::ranges::all_of(embedding, [](const float value) {
               return std::isfinite(value);
           });
}


std::vector<float> SpeakerDiarizer::compute_embedding(
    std::span<const float> chunk_audio,
    const BinaryActivityMatrix& activity,
    const int slot_id) {
    if (!embedding_session || chunk_audio.empty() || activity.empty() ||
        slot_id < 0 || slot_id >= kSpeakerSlotCount ||
        embedding_feature_dim_ <= 0) {
        return {};
    }

    const auto mel_features = melExtractor.extract(chunk_audio);
    if (mel_features.empty() ||
        mel_features.size() % static_cast<size_t>(embedding_feature_dim_) != 0) {
        return {};
    }

    const int num_fbank_frames = static_cast<int>(
        mel_features.size() / static_cast<size_t>(embedding_feature_dim_));
    const auto upsampled_mask = embedding_mask_for_slot(
        activity,
        slot_id,
        num_fbank_frames,
        min_embedding_frames_,
        embedding_exclude_overlap_);
    if (upsampled_mask.empty()) {
        return {};
    }

    // Pack the active fbank frames into a contiguous buffer for the model.
    std::vector<float> masked_fbank;
    masked_fbank.reserve(mel_features.size());
    int active_frame_count = 0;
    for (int frame = 0; frame < num_fbank_frames; ++frame) {
        if (!upsampled_mask[static_cast<size_t>(frame)]) {
            continue;
        }
        const auto frame_offset = static_cast<size_t>(frame) *
                                  static_cast<size_t>(embedding_feature_dim_);
        masked_fbank.insert(
            masked_fbank.end(),
            mel_features.begin() + static_cast<std::ptrdiff_t>(frame_offset),
            mel_features.begin() + static_cast<std::ptrdiff_t>(
                frame_offset + static_cast<size_t>(embedding_feature_dim_)));
        active_frame_count += 1;
    }

    if (active_frame_count < min_embedding_frames_) {
        return {};
    }

    const std::vector<int64_t> input_shape = {
        1,
        active_frame_count,
        embedding_feature_dim_,
    };
    auto output = run_inference(input_shape, masked_fbank, *embedding_session);
    auto output_info = output[0].GetTensorTypeAndShapeInfo();
    const auto embedding_size = output_info.GetElementCount();
    if (embedding_size == 0) {
        return {};
    }

    const float* embedding_data = output[0].GetTensorData<float>();
    return std::vector<float>(embedding_data, embedding_data + embedding_size);
}


std::vector<SpeakerSegment> SpeakerDiarizer::compute_chunk_slot_embeddings(
    std::span<const float> audio,
    const SlidingSegmentationResult& segmentation) {
    std::vector<SpeakerSegment> chunk_slot_speakers;
    chunk_slot_speakers.reserve(segmentation.chunks.size() * kSpeakerSlotCount);
    if (audio.empty() || segmentation_window_samples_ <= 0) {
        return chunk_slot_speakers;
    }

    std::vector<float> chunk_audio(
        static_cast<size_t>(segmentation_window_samples_), 0.0f);
    for (const auto& chunk : segmentation.chunks) {
        std::fill(chunk_audio.begin(), chunk_audio.end(), 0.0f);
        if (chunk.start_sample < audio.size()) {
            const size_t samples_to_copy = std::min(
                chunk_audio.size(), audio.size() - chunk.start_sample);
            std::copy_n(audio.data() + chunk.start_sample,
                        samples_to_copy,
                        chunk_audio.begin());
        }

        const float chunk_end_time = std::min(
            chunk.start_time + kSegmentationWindowSeconds,
            segmentation.audio_seconds);
        for (int slot = 0; slot < kSpeakerSlotCount; ++slot) {
            auto emb = compute_embedding(chunk_audio, chunk.binary_activity, slot);
            chunk_slot_speakers.push_back({
                -1,
                chunk.start_time,
                chunk_end_time,
                std::move(emb),
            });
        }
    }

    return chunk_slot_speakers;
}


// =====================================================================
// Clustering: assign chunk-slot embeddings to speaker clusters
// =====================================================================

SpeakerDiarizer::HardClusterMatrix SpeakerDiarizer::assign_chunk_slot_clusters(
    const std::vector<SpeakerSegment>& chunk_slot_embeddings,
    const SlidingSegmentationResult& segmentation,
    const int num_speakers_hint,
    const float clustering_threshold) const {
    (void)clustering_threshold;

    const size_t num_chunks = segmentation.chunks.size();

    HardClusterMatrix hard_clusters(num_chunks);
    for (auto& chunk_clusters : hard_clusters) {
        chunk_clusters.fill(kDiscardClusterLabel);
    }

    if (num_chunks == 0 || chunk_slot_embeddings.empty()) {
        return hard_clusters;
    }

    // 1. Collect all slot embeddings and their activity status.
    constexpr float kMinSoloRatio = 0.2f;
    const int num_pyannote_frames = segmentation.chunks.empty()
        ? 0
        : static_cast<int>(segmentation.chunks.front().binary_activity.size());
    const int min_solo_frames = static_cast<int>(
        kMinSoloRatio * static_cast<float>(num_pyannote_frames));

    std::vector<std::vector<float>> all_embeddings;
    std::vector<int> all_solo_frames;
    std::vector<bool> all_slot_has_activity;
    all_embeddings.reserve(num_chunks * kSpeakerSlotCount);
    all_solo_frames.reserve(num_chunks * kSpeakerSlotCount);
    all_slot_has_activity.reserve(num_chunks * kSpeakerSlotCount);

    size_t expected_dim = 0;
    for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
        const auto& chunk = segmentation.chunks[chunk_index];
        for (int slot = 0; slot < kSpeakerSlotCount; ++slot) {
            const size_t flat_index = chunk_index * kSpeakerSlotCount +
                                      static_cast<size_t>(slot);
            const auto& embedding = chunk_slot_embeddings[flat_index].embedding;

            if (expected_dim == 0 && embedding_is_valid(embedding)) {
                expected_dim = embedding.size();
            }

            all_embeddings.push_back(embedding);

            int solo_count = 0;
            int active_count = 0;
            for (const auto& frame : chunk.binary_activity) {
                int active_slots = 0;
                for (int s = 0; s < kSpeakerSlotCount; ++s) {
                    if (frame[static_cast<size_t>(s)]) ++active_slots;
                }
                if (frame[static_cast<size_t>(slot)]) {
                    ++active_count;
                    if (active_slots == 1) ++solo_count;
                }
            }

            all_solo_frames.push_back(solo_count);
            all_slot_has_activity.push_back(active_count > 0);
        }
    }

    // 2. Filter training embeddings (pyannote: solo_frames >= 0.2*N + valid).
    std::vector<std::vector<float>> train_embeddings_raw;
    std::vector<size_t> train_flat_indices;
    for (size_t flat_index = 0; flat_index < all_embeddings.size(); ++flat_index) {
        const auto& embedding = all_embeddings[flat_index];
        if (!all_slot_has_activity[flat_index]) continue;
        if (!embedding_is_valid(embedding)) continue;
        if (embedding.size() != expected_dim) continue;
        if (all_solo_frames[flat_index] < min_solo_frames) continue;

        train_embeddings_raw.push_back(embedding);
        train_flat_indices.push_back(flat_index);
    }

    const int training_count = static_cast<int>(train_embeddings_raw.size());
    if (training_count == 0) {
        return hard_clusters;
    }

    // 3. L2-normalize for cosine-via-euclidean (pyannote convention).
    std::vector<std::vector<float>> train_normed = train_embeddings_raw;
    for (auto& emb : train_normed) {
        normalize_l2(emb);
    }

    // 4. Run the pyannote-conformant AHC.
    std::vector<int> ahc_labels;
    if (training_count == 1) {
        ahc_labels = {0};
    } else {
        constexpr float kThreshold = 0.7045f;
        constexpr int kMinClusterSize = 12;
        const int target_clusters = num_speakers_hint > 0
            ? std::min(num_speakers_hint, training_count)
            : -1;
        // pyannote defaults: min_clusters=1, max_clusters=num_embeddings.
        const int min_clusters = target_clusters > 0 ? target_clusters : 1;
        const int max_clusters = target_clusters > 0 ? target_clusters : training_count;

        ahc_labels = ahc_pyannote::agglomerative_cluster_pyannote(
            train_normed,
            kThreshold,
            kMinClusterSize,
            min_clusters,
            max_clusters,
            target_clusters);
    }

    int num_final_clusters = 0;
    for (const int lbl : ahc_labels) {
        num_final_clusters = std::max(num_final_clusters, lbl + 1);
    }
    if (num_final_clusters == 0) num_final_clusters = 1;

    // 5. Compute centroids over the raw (NOT normalized) training
    //    embeddings, then reassign ALL slot embeddings by cosine
    //    distance to the closest centroid.
    std::vector<std::vector<float>> centroids(
        num_final_clusters, std::vector<float>(expected_dim, 0.0f));
    std::vector<int> centroid_counts(num_final_clusters, 0);
    for (int i = 0; i < training_count; ++i) {
        const int lbl = ahc_labels[i];
        for (size_t d = 0; d < expected_dim; ++d) {
            centroids[lbl][d] += train_embeddings_raw[i][d];
        }
        ++centroid_counts[lbl];
    }
    for (int c = 0; c < num_final_clusters; ++c) {
        if (centroid_counts[c] == 0) continue;
        const float inv = 1.0f / static_cast<float>(centroid_counts[c]);
        for (float& v : centroids[c]) v *= inv;
    }

    for (size_t flat_index = 0; flat_index < all_embeddings.size(); ++flat_index) {
        const auto& emb = all_embeddings[flat_index];
        const size_t chunk_index = flat_index / kSpeakerSlotCount;
        const size_t slot_index = flat_index % kSpeakerSlotCount;

        if (!embedding_is_valid(emb) || emb.size() != expected_dim) {
            hard_clusters[chunk_index][slot_index] = kDiscardClusterLabel;
            continue;
        }

        int best_label = 0;
        float best_dist = std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_final_clusters; ++c) {
            const float dist = 1.0f - cosine_sim(emb, centroids[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_label = c;
            }
        }
        hard_clusters[chunk_index][slot_index] = best_label;
    }

    // 6. Mark inactive slots as discarded (pyannote: sum(binary) == 0).
    for (size_t flat_index = 0; flat_index < all_slot_has_activity.size(); ++flat_index) {
        if (!all_slot_has_activity[flat_index]) {
            const size_t chunk_index = flat_index / kSpeakerSlotCount;
            const size_t slot_index = flat_index % kSpeakerSlotCount;
            hard_clusters[chunk_index][slot_index] = kDiscardClusterLabel;
        }
    }

    return hard_clusters;
}


int SpeakerDiarizer::cluster_count_from_hard_clusters(
    const HardClusterMatrix& hard_clusters) {
    int cluster_count = 0;
    for (const auto& chunk_clusters : hard_clusters) {
        for (const int cluster : chunk_clusters) {
            if (cluster >= 0) {
                cluster_count = std::max(cluster_count, cluster + 1);
            }
        }
    }
    return cluster_count;
}


// =====================================================================
// Global activity reconstruction and speaker count
// =====================================================================

SpeakerDiarizer::ReconstructedActivityMatrix SpeakerDiarizer::reconstruct_global_activity(
    const SlidingSegmentationResult& segmentation,
    const HardClusterMatrix& hard_clusters,
    const int num_clusters) const {
    if (num_clusters <= 0 || segmentation.global_activity.empty() ||
        segmentation.global_frame_duration_seconds <= 0.0f ||
        segmentation.audio_seconds <= 0.0f) {
        return {};
    }

    ReconstructedActivityMatrix score_sums(
        segmentation.global_activity.size(),
        std::vector<float>(static_cast<size_t>(num_clusters), 0.0f));
    std::vector<std::vector<int>> score_counts(
        segmentation.global_activity.size(),
        std::vector<int>(static_cast<size_t>(num_clusters), 0));

    for (size_t chunk_index = 0; chunk_index < segmentation.chunks.size(); ++chunk_index) {
        if (chunk_index >= hard_clusters.size()) {
            continue;
        }

        const auto& chunk = segmentation.chunks[chunk_index];
        const size_t frame_count = std::min(chunk.activity.size(),
                                            chunk.binary_activity.size());
        const float nan = std::numeric_limits<float>::quiet_NaN();
        ReconstructedActivityMatrix clustered_chunk(
            frame_count,
            std::vector<float>(static_cast<size_t>(num_clusters), nan));

        // For each cluster present in this chunk, take the per-frame max
        // activity over all slots assigned to that cluster.
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            bool cluster_is_present = false;
            for (int slot = 0; slot < kSpeakerSlotCount; ++slot) {
                if (hard_clusters[chunk_index][static_cast<size_t>(slot)] == cluster) {
                    cluster_is_present = true;
                    break;
                }
            }
            if (!cluster_is_present) {
                continue;
            }

            for (size_t frame = 0; frame < frame_count; ++frame) {
                float max_score = -std::numeric_limits<float>::infinity();
                for (int slot = 0; slot < kSpeakerSlotCount; ++slot) {
                    if (hard_clusters[chunk_index][static_cast<size_t>(slot)] != cluster) {
                        continue;
                    }
                    max_score = std::max(max_score,
                                         chunk.activity[frame][static_cast<size_t>(slot)]);
                }
                if (std::isfinite(max_score)) {
                    clustered_chunk[frame][static_cast<size_t>(cluster)] = max_score;
                }
            }
        }

        // Aggregate this chunk into the global frame grid.
        for (size_t frame = 0; frame < frame_count; ++frame) {
            const float global_time = chunk.start_time +
                                      static_cast<float>(frame) * chunk.frame_duration_seconds;
            if (global_time >= segmentation.audio_seconds) {
                continue;
            }
            const auto global_frame = static_cast<size_t>(
                std::floor(global_time / segmentation.global_frame_duration_seconds));
            if (global_frame >= score_sums.size()) {
                continue;
            }
            for (int cluster = 0; cluster < num_clusters; ++cluster) {
                const float value = clustered_chunk[frame][static_cast<size_t>(cluster)];
                if (std::isnan(value)) {
                    continue;
                }
                score_sums[global_frame][static_cast<size_t>(cluster)] += value;
                score_counts[global_frame][static_cast<size_t>(cluster)] += 1;
            }
        }
    }

    // pyannote aggregate(skip_average=True) -> Sum, no division.
    // Missing chunk contributions count as zero, which keeps the
    // aggregation comparable across all clusters.
    ReconstructedActivityMatrix reconstructed(
        segmentation.global_activity.size(),
        std::vector<float>(static_cast<size_t>(num_clusters), 0.0f));
    for (size_t frame = 0; frame < reconstructed.size(); ++frame) {
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            reconstructed[frame][static_cast<size_t>(cluster)] =
                score_sums[frame][static_cast<size_t>(cluster)];
        }
    }

    return reconstructed;
}


std::vector<int> SpeakerDiarizer::compute_speaker_count(
    const SlidingSegmentationResult& segmentation) const {
    // pyannote speaker_count: per global frame, the rounded mean of
    // "number of active slots" across all overlapping chunks.
    const size_t num_global_frames = segmentation.global_activity.size();
    std::vector<int> count(num_global_frames, 0);
    if (num_global_frames == 0 ||
        segmentation.global_frame_duration_seconds <= 0.0f) {
        return count;
    }

    std::vector<float> sum_active(num_global_frames, 0.0f);
    std::vector<int> num_overlap(num_global_frames, 0);

    for (const auto& chunk : segmentation.chunks) {
        const size_t frame_count = chunk.binary_activity.size();
        for (size_t frame = 0; frame < frame_count; ++frame) {
            const float global_time = chunk.start_time +
                                      static_cast<float>(frame) * chunk.frame_duration_seconds;
            if (global_time >= segmentation.audio_seconds) {
                continue;
            }
            const auto global_frame = static_cast<size_t>(
                std::floor(global_time / segmentation.global_frame_duration_seconds));
            if (global_frame >= num_global_frames) {
                continue;
            }

            int active_slots = 0;
            for (int slot = 0; slot < kSpeakerSlotCount; ++slot) {
                if (chunk.binary_activity[frame][static_cast<size_t>(slot)]) {
                    ++active_slots;
                }
            }
            sum_active[global_frame] += static_cast<float>(active_slots);
            num_overlap[global_frame] += 1;
        }
    }

    for (size_t frame = 0; frame < num_global_frames; ++frame) {
        if (num_overlap[frame] == 0) {
            count[frame] = 0;
            continue;
        }
        const float mean = sum_active[frame] / static_cast<float>(num_overlap[frame]);
        count[frame] = static_cast<int>(std::lround(mean));
    }

    return count;
}


SpeakerDiarizer::ReconstructedActivityMatrix SpeakerDiarizer::to_diarization(
    const ReconstructedActivityMatrix& clustered_activity,
    const std::vector<int>& count) const {
    // pyannote to_diarization: per global frame, pick the top count[t]
    // clusters by activation (argsort desc) and set them to 1.
    ReconstructedActivityMatrix binary;
    if (clustered_activity.empty()) {
        return binary;
    }
    const size_t num_clusters = clustered_activity.front().size();
    if (num_clusters == 0) {
        return binary;
    }

    binary.assign(clustered_activity.size(),
                  std::vector<float>(num_clusters, 0.0f));

    const size_t n = std::min(clustered_activity.size(), count.size());
    for (size_t frame = 0; frame < n; ++frame) {
        const int c = std::clamp(count[frame], 0, static_cast<int>(num_clusters));
        if (c == 0) {
            continue;
        }

        // Argsort descending by activation.
        std::vector<int> indices(num_clusters);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(),
            indices.begin() + c,
            indices.end(),
            [&clustered_activity, frame](int a, int b) {
                return clustered_activity[frame][static_cast<size_t>(a)] >
                       clustered_activity[frame][static_cast<size_t>(b)];
            });

        for (int i = 0; i < c; ++i) {
            binary[frame][static_cast<size_t>(indices[static_cast<size_t>(i)])] = 1.0f;
        }
    }

    return binary;
}


// =====================================================================
// Segment extraction from activity matrices
// =====================================================================

std::vector<SpeakerSegment> SpeakerDiarizer::segments_from_reconstructed_activity(
    const ReconstructedActivityMatrix& activity,
    const float frame_duration_seconds,
    const float audio_seconds) const {
    return segments_from_reconstructed_activity(
        activity,
        frame_duration_seconds,
        audio_seconds,
        BinarizationConfig{});
}


std::vector<SpeakerSegment> SpeakerDiarizer::segments_from_reconstructed_activity(
    const ReconstructedActivityMatrix& activity,
    const float frame_duration_seconds,
    const float audio_seconds,
    const BinarizationConfig& binarization_config) const {
    std::vector<SpeakerSegment> segments;
    if (activity.empty() || frame_duration_seconds <= 0.0f || audio_seconds <= 0.0f) {
        return segments;
    }

    const size_t speaker_count = activity.front().size();
    if (speaker_count == 0) {
        return segments;
    }

    const float end_time = std::min(
        static_cast<float>(activity.size()) * frame_duration_seconds,
        audio_seconds);

    for (size_t speaker = 0; speaker < speaker_count; ++speaker) {
        std::vector<SpeakerSegment> speaker_segments;
        bool active = false;
        float start_time = 0.0f;

        // Onset/offset state machine over the per-speaker activation track.
        for (size_t frame = 0; frame < activity.size(); ++frame) {
            const float current_time = static_cast<float>(frame) * frame_duration_seconds;
            if (current_time >= audio_seconds) {
                break;
            }

            const float score = activity[frame][speaker];
            if (!active) {
                if (score >= binarization_config.onset) {
                    active = true;
                    start_time = current_time;
                }
            } else if (score < binarization_config.offset) {
                speaker_segments.push_back({
                    static_cast<int>(speaker),
                    start_time,
                    current_time,
                    {},
                });
                active = false;
            }
        }

        if (active) {
            speaker_segments.push_back({
                static_cast<int>(speaker),
                start_time,
                end_time,
                {},
            });
        }

        // Drop segments shorter than min_duration_on.
        std::erase_if(speaker_segments,
                      [&binarization_config](const SpeakerSegment& segment) {
                          return (segment.end_time - segment.start_time) <
                                 binarization_config.min_duration_on;
                      });

        // Merge segments separated by less than min_duration_off.
        if (binarization_config.min_duration_off > 0.0f && speaker_segments.size() > 1) {
            std::vector<SpeakerSegment> merged_segments;
            merged_segments.reserve(speaker_segments.size());
            merged_segments.push_back(speaker_segments.front());

            for (size_t i = 1; i < speaker_segments.size(); ++i) {
                auto& previous = merged_segments.back();
                const auto& current = speaker_segments[i];
                if ((current.start_time - previous.end_time) <
                    binarization_config.min_duration_off) {
                    previous.end_time = current.end_time;
                } else {
                    merged_segments.push_back(current);
                }
            }
            speaker_segments = std::move(merged_segments);
        }

        segments.insert(segments.end(),
                        speaker_segments.begin(), speaker_segments.end());
    }

    std::ranges::sort(segments, [](const SpeakerSegment& a, const SpeakerSegment& b) {
        if (a.start_time == b.start_time) return a.speaker_id < b.speaker_id;
        return a.start_time < b.start_time;
    });
    return segments;
}


std::vector<SpeakerSegment> SpeakerDiarizer::segments_from_binary_diarization(
    const ReconstructedActivityMatrix& binary_diarization,
    const float frame_duration_seconds,
    const float audio_seconds) const {
    // Direct run-length encoder on {0, 1} per cluster.
    // pyannote default: min_duration_on=0, min_duration_off=0, so no smoothing.
    std::vector<SpeakerSegment> segments;
    if (binary_diarization.empty() || frame_duration_seconds <= 0.0f ||
        audio_seconds <= 0.0f) {
        return segments;
    }

    const size_t num_clusters = binary_diarization.front().size();
    const float end_time = std::min(
        static_cast<float>(binary_diarization.size()) * frame_duration_seconds,
        audio_seconds);

    for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
        bool active = false;
        float start_time = 0.0f;

        for (size_t frame = 0; frame < binary_diarization.size(); ++frame) {
            const float current_time = static_cast<float>(frame) * frame_duration_seconds;
            if (current_time >= audio_seconds) {
                break;
            }
            const bool is_active = binary_diarization[frame][cluster] >= 0.5f;

            if (!active && is_active) {
                active = true;
                start_time = current_time;
            } else if (active && !is_active) {
                segments.push_back({
                    static_cast<int>(cluster), start_time, current_time, {},
                });
                active = false;
            }
        }
        if (active) {
            segments.push_back({
                static_cast<int>(cluster), start_time, end_time, {},
            });
        }
    }

    std::ranges::sort(segments, [](const SpeakerSegment& a, const SpeakerSegment& b) {
        if (a.start_time == b.start_time) return a.speaker_id < b.speaker_id;
        return a.start_time < b.start_time;
    });
    return segments;
}


// =====================================================================
// Math / inference helpers
// =====================================================================

void SpeakerDiarizer::normalize_l2(std::vector<float>& v) {
    float squared_norm = 0.0f;
    for (const float value : v) {
        squared_norm += value * value;
    }
    if (squared_norm <= 0.0f) {
        return;
    }
    const float inv_norm = 1.0f / std::sqrt(squared_norm);
    for (float& value : v) {
        value *= inv_norm;
    }
}


float SpeakerDiarizer::cosine_sim(const std::vector<float>& v1,
                                  const std::vector<float>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same length");
    }
    float v12_dot = 0.0f;
    float v1_length_mag = 0.0f;
    float v2_length_mag = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        const float v1_val = v1[i];
        const float v2_val = v2[i];
        v12_dot += v1_val * v2_val;
        v1_length_mag += v1_val * v1_val;
        v2_length_mag += v2_val * v2_val;
    }
    v1_length_mag = std::sqrt(v1_length_mag);
    v2_length_mag = std::sqrt(v2_length_mag);
    return v12_dot / (v1_length_mag * v2_length_mag + 1e-10f);
}


std::vector<Ort::Value> SpeakerDiarizer::run_inference(
    const std::vector<int64_t>& input_shape,
    std::span<const float> data,
    Ort::Session& session) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(data.data()),
        data.size(),
        input_shape.data(),
        input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    return session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1);
}


// =====================================================================
// RTTM output and free-function helpers
// =====================================================================

void SpeakerDiarizer::write_rttm(std::ostream& out,
                                 std::string_view uri,
                                 const std::vector<SpeakerSegment>& segments,
                                 const float time_offset_seconds) {
    const auto old_flags = out.flags();
    const auto old_precision = out.precision();
    const auto old_fill = out.fill();

    out << std::fixed << std::setprecision(3);
    for (const auto& segment : segments) {
        const float duration = segment.end_time - segment.start_time;
        if (segment.speaker_id < 0 || duration <= 0.0f) {
            continue;
        }
        out << "SPEAKER " << uri << " 1 "
            << (segment.start_time + time_offset_seconds) << " "
            << duration << " <NA> <NA> "
            << format_rttm_speaker_label(segment.speaker_id)
            << " <NA> <NA>\n";
    }

    out.flags(old_flags);
    out.precision(old_precision);
    out.fill(old_fill);
}


int get_speaker_at(const std::vector<SpeakerSegment>& segments,
                   float start_time,
                   float end_time) {
    int best_id = -1;
    float best_overlap = 0.0f;

    for (const auto& seg : segments) {
        if (seg.end_time <= start_time) continue;
        if (seg.start_time >= end_time) break;

        const float overlap = std::min(seg.end_time, end_time) -
                              std::max(seg.start_time, start_time);
        if (overlap > best_overlap) {
            best_overlap = overlap;
            best_id = seg.speaker_id;
        }
    }
    return best_id;
}