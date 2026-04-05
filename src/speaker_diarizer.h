#pragma once
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "config.h"

struct SpeakerSegment {
    int speaker_id;
    float start_time;
    float end_time;
};

class SpeakerDiarizer {
    Ort::Env env;
    std::unique_ptr<Ort::Session> segmentation_session;
    std::unique_ptr<Ort::Session> embedding_session;
    float speaker_threshold;
    float min_segment_duration;
    float merge_gap;

    std::vector<SpeakerSegment> clean_up_speaker_segments(const std::vector<SpeakerSegment> &segments);

    static float get_sigmoid(float value);

public:
    SpeakerDiarizer(Config &config);

    bool is_loaded() const;

    std::vector<float> run_segmentation(const std::vector<float> &audio, int &num_frames);

    std::vector<SpeakerSegment> parse_frames(const float *output_data, int num_frames, float chunk_seconds);

    std::vector<SpeakerSegment> segment(const std::vector<float> &audio, float chunk_seconds);

    static float frame_to_time(int f, float time_per_frame);

};




