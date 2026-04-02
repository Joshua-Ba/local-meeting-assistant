#pragma once
#include <string>
#include <optional>
#include "config.h"

class LlmEngine {
public:
    LlmEngine(const Config &config);
    ~LlmEngine();
    std::optional<std::string> generate(const std::string& input, int max_tokens, const std::string& prompt);
    bool is_loaded() const;

private:
    Config config;
    void* model_;
    void* ctx_;
    void* sampler_;
    void* vocab_;
};