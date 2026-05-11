#include "meeting_assistant.h"

#include "test_helpers.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <vector>

TEST(MeetingAssistantTest, TextBufferCanAppendAndClearWithoutLoadedModel) {
    Config config = make_test_config();
    MeetingAssistant assistant(config);

    EXPECT_FALSE(assistant.is_loaded());
    EXPECT_EQ(assistant.get_full_text(), "");
    EXPECT_EQ(assistant.summarize_current_segment(), std::nullopt);
    EXPECT_EQ(assistant.full_summary(), std::nullopt);
    EXPECT_EQ(assistant.full_summary_checked(), std::nullopt);

    assistant.add_segment("Segment eins. ");
    assistant.add_segment("Segment zwei.");
    const std::vector<float> audio = {0.1f, 0.2f, 0.3f};
    assistant.add_audio(audio);
    EXPECT_EQ(assistant.get_full_text(), "Segment eins. Segment zwei.");
    EXPECT_EQ(assistant.get_full_audio(), audio);

    EXPECT_EQ(assistant.summarize_current_segment(), std::nullopt);
    EXPECT_EQ(assistant.full_summary(), std::nullopt);
    EXPECT_EQ(assistant.full_summary_checked(), std::nullopt);

    assistant.clear_text();
    EXPECT_EQ(assistant.get_full_text(), "");
    EXPECT_TRUE(assistant.get_full_audio().empty());
    EXPECT_EQ(assistant.summarize_current_segment(), std::nullopt);
}

TEST(MeetingAssistantTest, FinalizeWritesSpeakerLabeledTranscriptAndRttm) {
    Config config = make_test_config();
    MeetingAssistant assistant(config);
    const auto transcript_path = std::filesystem::current_path() / "meeting_finalize_transcript.txt";
    const auto rttm_path = std::filesystem::current_path() / "meeting_finalize.rttm";

    const std::vector<float> audio(1600, 0.0f);
    assistant.add_audio(audio);
    assistant.add_transcript_segment(0.0f, 0.5f, "Hallo ");
    assistant.add_transcript_segment(0.5f, 1.0f, "Welt.");

    const auto segments = assistant.finalize(transcript_path, rttm_path, "meeting");

    EXPECT_TRUE(segments.empty());
    EXPECT_EQ(assistant.get_full_text(), "\n[Speaker -1]\nHallo Welt.");

    std::ifstream transcript_file(transcript_path);
    std::string transcript_content((std::istreambuf_iterator<char>(transcript_file)),
                                   std::istreambuf_iterator<char>());
    EXPECT_EQ(transcript_content, assistant.get_full_text());

    std::ifstream rttm_file(rttm_path);
    std::string rttm_content((std::istreambuf_iterator<char>(rttm_file)),
                             std::istreambuf_iterator<char>());
    EXPECT_TRUE(rttm_content.empty());
}
