// meeting_assistant.h
//
// High-level orchestrator that ties together transcription input
// (from Whisper), the audio buffer, the diarizer, and the LLM-based
// summarization. Used by both live and file modes of the application.

#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "config.h"
#include "llm_engine.h"
#include "speaker_diarizer.h"


// One Whisper-produced transcript snippet with its start/end times in
// seconds (relative to the start of the session).
struct TranscriptSegment {
    float start_time;
    float end_time;
    std::string text;
};


class MeetingAssistant {
private:
    LlmEngine engin;
    std::string current_segment;
    std::string full_text;
    std::vector<float> full_audio_buffer_;
    std::vector<TranscriptSegment> transcript_segments_;
    Config config;

public:
    MeetingAssistant(const Config& config);

    bool is_loaded() const;

    // Append a finished text segment and flush it into the running
    // full_text buffer for later summarization.
    void add_segment(const std::string& text);

    // Append PCM audio to the internal buffer. This is the audio that
    // will be diarized in finalize().
    void add_audio(std::span<const float> audio);

    // Record one Whisper-produced segment with its time range.
    void add_transcript_segment(float start_time, float end_time,
                                const std::string& text);

    // Generate an LLM summary of the current (most recently appended)
    // segment. Returns std::nullopt if there is nothing to summarize or
    // generation failed.
    std::optional<std::string> summarize_current_segment();

    // Run diarization on the accumulated audio, write the speaker-tagged
    // transcript and RTTM to disk, and return the diarized segments.
    std::vector<SpeakerSegment> finalize(const std::filesystem::path& transcript_path,
                                         const std::filesystem::path& rttm_path,
                                         std::string_view rttm_uri);

    // Generate a summary over the full collected transcript.
    std::optional<std::string> full_summary();

    // full_summary() with a follow-up consistency check pass.
    std::optional<std::string> full_summary_checked();

    void clear_text();

    std::string get_full_text();

    const std::vector<float>& get_full_audio() const;
};