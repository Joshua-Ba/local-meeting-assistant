#include "meeting_assistant.h"



MeetingAssistant::MeetingAssistant(const Config &config) :engin(config), config(config){
}


bool MeetingAssistant::is_loaded() const {
    return engin.is_loaded();
}

void MeetingAssistant::add_segment(const std::string& text) {
    current_segment = text;
    full_text.append(current_segment);
}

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

void MeetingAssistant::clear_text() {
    full_text.clear();
    current_segment.clear();
}

std::string MeetingAssistant::get_full_text() {
    return full_text;
}