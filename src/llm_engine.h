// llm_engine.h
//
// Thin wrapper around llama.cpp. Loads a GGUF model, builds a sampler
// chain, and exposes a single generate() call that runs a system+user
// chat prompt and streams tokens into a string.

#pragma once

#include <optional>
#include <string>

#include "config.h"


class LlmEngine {
public:
    LlmEngine(const Config& config);
    ~LlmEngine();

    // Generate up to `max_tokens` tokens given a system `prompt` and a
    // user `input`. Returns std::nullopt if the engine failed to load.
    std::optional<std::string> generate(const std::string& input,
                                        int max_tokens,
                                        const std::string& prompt);

    bool is_loaded() const;

private:
    // Opaque pointers to avoid pulling llama.h into the header.
    Config config;
    void* model_;
    void* ctx_;
    void* sampler_;
    void* vocab_;
};