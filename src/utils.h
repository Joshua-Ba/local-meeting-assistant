#pragma once

#include <chrono>
#include <format>
#include <string>


inline std::string generate_session_id() {
    const auto now = std::chrono::system_clock::now();
    return std::format("{:%Y_%m_%d_%H_%M}", now);
}

inline std::string get_filepath(std::string_view path, std::string_view name, std::string_view session_id) {
    return std::format("{}/{}_{}.txt", path, name, session_id);
}