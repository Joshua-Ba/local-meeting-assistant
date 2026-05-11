// meeting_assistant.cpp
//
// Implementation of MeetingAssistant. The class accumulates audio and
// Whisper-produced text segments during a session, then in finalize()
// runs diarization, tags the transcript by speaker, writes outputs to
// disk, and exposes the result for summarization.

#include "meeting_assistant.h"

#include <fstream>


// =====================================================================
// Construction / accessors
// =====================================================================

MeetingAssistant::MeetingAssistant(const Config& config)
    : engin(config), config(config) {}


bool MeetingAssistant::is_loaded() const {
    return engin.is_loaded();
}


std::string MeetingAssistant::get_full_text() {
    return full_text;
}


const std::vector<float>& MeetingAssistant::get_full_audio() const {
    return full_audio_buffer_;
}


void MeetingAssistant::clear_text() {
    full_text.clear();
    current_segment.clear();
    full_audio_buffer_.clear();
    transcript_segments_.clear();
}


// =====================================================================
// Append API
// =====================================================================

void MeetingAssistant::add_segment(const std::string& text) {
    current_segment = text;
    full_text.append(current_segment);
}


void MeetingAssistant::add_audio(std::span<const float> audio) {
    full_audio_buffer_.insert(full_audio_buffer_.end(),
                              audio.begin(), audio.end());
}


void MeetingAssistant::add_transcript_segment(float start_time,
                                              float end_time,
                                              const std::string& text) {
    transcript_segments_.push_back({start_time, end_time, text});
}


// =====================================================================
// Summarization
// =====================================================================

std::optional<std::string> MeetingAssistant::summarize_current_segment() {
    if (current_segment.empty()) return std::nullopt;
    auto result = engin.generate(current_segment, 1000, config.snippet_prompt);
    if (result) {
        return *result;
    }
    return std::nullopt;
}


std::optional<std::string> MeetingAssistant::full_summary() {
    if (full_text.empty()) return std::nullopt;
    auto result = engin.generate(full_text, 5000, config.full_summary_prompt);
    if (result) {
        return *result;
    }
    return std::nullopt;
}


// Two-pass summary: first run full_summary(), then run a check pass that
// verifies the summary against the full transcript.
std::optional<std::string> MeetingAssistant::full_summary_checked() {
    if (full_text.empty()) return std::nullopt;

    std::string prompt_check_summary = config.check_summary_prompt;
    prompt_check_summary.append(full_text);

    auto summary = full_summary();
    if (!summary) return std::nullopt;

    auto result = engin.generate(*summary, 5000, prompt_check_summary);
    if (result) return *result;
    return std::nullopt;
}


// =====================================================================
// Finalize: diarize, tag transcript by speaker, write outputs
// =====================================================================

std::vector<SpeakerSegment> MeetingAssistant::finalize(
    const std::filesystem::path& transcript_path,
    const std::filesystem::path& rttm_path,
    std::string_view rttm_uri) {
    SpeakerDiarizer diarizer(config);
    const auto diarized_segments = diarizer.diarize(full_audio_buffer_);

    // Build a speaker-tagged transcript. Whenever the dominant speaker
    // changes between Whisper segments, emit a `[Speaker N]` header.
    std::string final_transcript;
    int current_speaker = -2;
    for (const auto& segment : transcript_segments_) {
        const int speaker = get_speaker_at(diarized_segments,
                                           segment.start_time,
                                           segment.end_time);
        if (speaker != current_speaker) {
            final_transcript += "\n[Speaker " + std::to_string(speaker) + "]\n";
            current_speaker = speaker;
        }
        final_transcript += segment.text;
    }

    full_text = final_transcript;
    current_segment.clear();

    // Write transcript.
    if (!transcript_path.parent_path().empty()) {
        std::filesystem::create_directories(transcript_path.parent_path());
    }
    std::ofstream transcript_file(transcript_path);
    if (transcript_file.is_open()) {
        transcript_file << final_transcript;
    }

    // Write RTTM.
    if (!rttm_path.parent_path().empty()) {
        std::filesystem::create_directories(rttm_path.parent_path());
    }
    std::ofstream rttm_file(rttm_path);
    if (rttm_file.is_open()) {
        SpeakerDiarizer::write_rttm(rttm_file, rttm_uri, diarized_segments);
    }

    return diarized_segments;
}