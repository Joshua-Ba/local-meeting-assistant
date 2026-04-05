#include "speaker_diarizer.h"



SpeakerDiarizer::SpeakerDiarizer(Config &config)
    : env(ORT_LOGGING_LEVEL_WARNING, "diarizer"),
      speaker_threshold(config.speaker_threshold),
      min_segment_duration(config.min_segment_duration),
      merge_gap(config.merge_gap) {
    try {
        Ort::SessionOptions options;
        segmentation_session = std::make_unique<Ort::Session>(env, config.segmentation_model.c_str(), options);
        embedding_session = std::make_unique<Ort::Session>(env, config.embedding_model.c_str(), options);
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load diarization models: " << e.what() << std::endl;
    }
}


bool SpeakerDiarizer::is_loaded() const{
    return segmentation_session && embedding_session;
}


std::vector<float> SpeakerDiarizer::run_segmentation(const std::vector<float>& audio, int& num_frames) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 1, static_cast<int64_t>(audio.size())};
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(audio.data()),
        audio.size(),
        input_shape.data(),
        input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = segmentation_session->GetInputNameAllocated(0, allocator);
    auto output_name = segmentation_session->GetOutputNameAllocated(0, allocator);
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};

    auto output = segmentation_session->Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* output_data = output[0].GetTensorMutableData<float>();
    auto output_shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    num_frames = output_shape[1];

    return std::vector<float>(output_data, output_data + num_frames * 7);
}


std::vector<SpeakerSegment> SpeakerDiarizer::parse_frames(const float* output_data, int num_frames, float chunk_seconds) {
    std::vector<SpeakerSegment> segments;
    std::map<int, float> speaker;
    float frame_time = chunk_seconds / num_frames;

    for (int f = 0; f < num_frames; f++) {
        for (int s = 0; s < 7; s++) {
            if (speaker.contains(s)) {
                if (!(get_sigmoid(output_data[f * 7 + s]) >= speaker_threshold)) {
                    segments.push_back({s, speaker[s], frame_to_time(f, frame_time)});
                    speaker.erase(s);
                }
            }else {
                if ((get_sigmoid(output_data[f * 7 + s]) >= speaker_threshold)) {
                    speaker[s] = frame_to_time(f, frame_time);
                }
            }
        }
    }
    float end_time = frame_to_time(num_frames, frame_time);
    for (auto const& [key, val] : speaker)
    {
        segments.push_back({key,val, end_time});
    }
    return segments;
}


std::vector<SpeakerSegment> SpeakerDiarizer::segment(const std::vector<float>& audio, float chunk_seconds) {
    int num_frames = 0;
    auto output_data = run_segmentation(audio, num_frames);
    auto segments = parse_frames(output_data.data(), num_frames, chunk_seconds);
    return clean_up_speaker_segments(segments);
}


float SpeakerDiarizer::frame_to_time(int f, float time_per_frame) {
    return static_cast<float>(f) * time_per_frame;
}


std::vector<SpeakerSegment> SpeakerDiarizer::clean_up_speaker_segments(const std::vector<SpeakerSegment>& input) {
    auto segments = input;
    std::sort(segments.begin(), segments.end(), [](const SpeakerSegment& a, const SpeakerSegment& b) {
        if (a.speaker_id == b.speaker_id) return a.start_time < b.start_time;
        return a.speaker_id < b.speaker_id;
    });

    std::vector<SpeakerSegment> merged;
    for (const auto& seg : segments) {
        if (!merged.empty() && merged.back().speaker_id == seg.speaker_id && (seg.start_time - merged.back().end_time) <= merge_gap) {
            merged.back().end_time = seg.end_time;
        } else {
            merged.push_back(seg);
        }
    }
    segments = merged;

    std::erase_if(segments, [this](const SpeakerSegment& seg) {
        return (seg.end_time - seg.start_time) < min_segment_duration;
    });

    std::sort(segments.begin(), segments.end(), [](const SpeakerSegment& a, const SpeakerSegment& b) {
        return a.start_time < b.start_time;
    });

    return segments;
}



float SpeakerDiarizer::get_sigmoid(const float value) {
    const float e = exp(-value);
    const float result = (1.0/(1.0+e));
    return result;
}