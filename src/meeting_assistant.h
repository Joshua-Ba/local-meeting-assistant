#pragma once
#include "llm_engine.h"
#include "config.h"
#include <optional>

class MeetingAssistant
{
private:
    LlmEngine engin;
    std::string current_segment;
    std::string full_text;
    Config config;

public:
    MeetingAssistant(const Config &config);

    bool is_loaded() const;

    void add_segment(const std::string& text);

    std::optional<std::string> summarize_current_segment();

    std::optional<std::string> full_summary();

    std::optional<std::string> full_summary_checked();

    void clear_text();

    std::string get_full_text();
};