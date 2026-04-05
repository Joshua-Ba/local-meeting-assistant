#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <filesystem>

#include "src/speaker_diarizer.h"
#include "src/utils.h"
#include "src/ring_buffer.h"
#include "src/audio_capture.h"
#include "extern/whisper.cpp/include/whisper.h"
#include "src/meeting_assistant.h"
#include "src/config.h"


void audio_loop(std::stop_token token, RingBuffer& buffer, MeetingAssistant& assistant, whisper_context* ctx, whisper_full_params& wparams, Config& config, const std::string &session_id) {
    int counter = 0;
    const int num_segments_per_summary = config.segments_per_summary;
    std::ofstream transcript_file(get_filepath("output", "transcript", session_id), std::ios::app);
    std::string new_text_seg;

    SpeakerDiarizer diarizer(config);

    float chunk_duration = static_cast<float>(config.chunk_samples) / static_cast<float>(config.whisper_sample_rate);

    while (!token.stop_requested()) {
        std::vector<float> data;
        if (buffer.size() >= config.chunk_samples) {
            buffer.batch_read(data, config.chunk_samples);
            whisper_full(ctx, wparams, data.data(), config.chunk_samples);
            diarizer.segment(data, chunk_duration);
            int num_seg = whisper_full_n_segments(ctx);
            for (int i = 0; i < num_seg; i++) {
                auto text = whisper_full_get_segment_text(ctx, i);
                new_text_seg.append(text);
                transcript_file << text << std::flush;
            }
            counter += 1;
            if (counter >= num_segments_per_summary) {
                counter = 0;
                assistant.add_segment(new_text_seg);
                auto summary = assistant.summarize_current_segment();
                if (summary) {
                    std::cout << *summary << std::endl;
                }
                new_text_seg.clear();
            }
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    if (!new_text_seg.empty()) {
        assistant.add_segment(new_text_seg);
    }
}


int main(int argc, char* argv[]) {


    std::filesystem::current_path(std::filesystem::path(argv[0]).parent_path());
    std::string config_path = (argc > 1) ? argv[1] : "config.json";
    std::filesystem::create_directories("output");
    auto config = Config::load(config_path);
    auto session_id = generate_session_id();

    RingBuffer ringBuffer(config.ringbuffer_size);
    AudioCapture audioCapture(&ringBuffer, config.audio_device, config.resample_factor);


    auto ctx_params = whisper_context_default_params();
    auto ctx = whisper_init_from_file_with_params(config.whisper_model.c_str(), ctx_params);
    auto wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.language = "auto";

    MeetingAssistant assistant(config);
    if (!assistant.is_loaded()) {
        std::cerr << "Assistant failed to load" << std::endl;
        return 1;
    }
    std::cout << "Assistant loaded successfully" << std::endl;

    audioCapture.start();

    std::jthread worker(audio_loop, std::ref(ringBuffer), std::ref(assistant), ctx, std::ref(wparams), std::ref(config), std::ref(session_id));
    std::cin.get();
    worker.request_stop();

    audioCapture.stop();

    std::cout << assistant.get_full_text() << std::endl;
    auto result = assistant.full_summary_checked();
    if (result) {
        std::ofstream file(get_filepath("output", "summary", session_id));
        file << *result;
        std::cout << *result << std::endl;
    }
    else {
        std::cout << "Fehler beim erstellen der Zusammenfassung" << std::endl;
    }

    whisper_free(ctx);
    return 0;
}
