#include "llm_engine.h"
#include "llama.h"
#include <vector>
#include <iostream>


LlmEngine::LlmEngine(const Config &config) : model_(nullptr), ctx_(nullptr), sampler_(nullptr), vocab_(nullptr) {
    llama_model_params model_params = llama_model_default_params();
    model_ = llama_model_load_from_file(config.model_path.c_str(), model_params);
    if (!model_) {
        std::cerr << "Failed to load model: " << config.model_path << std::endl;
        return;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.context_size;
    ctx_params.n_batch = config.batch_size;
    ctx_ = llama_init_from_model(static_cast<llama_model*>(model_), ctx_params);
    if (!ctx_) {
        std::cerr << "Failed to create context" << std::endl;
        return;
    }

    vocab_ = const_cast<llama_vocab*>(llama_model_get_vocab(static_cast<llama_model*>(model_)));

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler_), llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler_), llama_sampler_init_top_k(config.top_k));
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler_), llama_sampler_init_top_p(config.top_p, 1));
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler_), llama_sampler_init_min_p(0.0f, 1));
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler_), llama_sampler_init_penalties(128, 1.0f, 0.0f, config.presence_penalty));
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler_), llama_sampler_init_dist(0));
}

LlmEngine::~LlmEngine() {
    if (sampler_) llama_sampler_free(static_cast<llama_sampler*>(sampler_));
    if (ctx_) llama_free(static_cast<llama_context*>(ctx_));
    if (model_) llama_model_free(static_cast<llama_model*>(model_));
}

bool LlmEngine::is_loaded() const {
    return model_ && ctx_ && sampler_;
}

std::optional<std::string> LlmEngine::generate(const std::string& input, int max_tokens, const std::string& prompt) {
    if (!is_loaded()) return std::nullopt;

    auto* vocab = static_cast<const llama_vocab*>(vocab_);
    auto* ctx = static_cast<llama_context*>(ctx_);
    auto* smpl = static_cast<llama_sampler*>(sampler_);
    auto* model = static_cast<llama_model*>(model_);

    // chat template anwenden
    llama_chat_message messages[] = {
        {"system", prompt.c_str()},
        {"user", input.c_str()},
    };

    // Manual chat template instead of llama_chat_apply_template because
    // Qwen3.5 ignores enable_thinking=false via the API (known llama.cpp bug).
    // The empty think block <think>\n\n</think> signals non-thinking mode.
    std::string formatted_prompt = "<|im_start|>system\n" + std::string(messages[0].content) + "<|im_end|>\n<|im_start|>user\n" + std::string(messages[1].content) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    // tokenize
    int n_prompt = -llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);

    // decode prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_decode(ctx, batch);

    // generate
    std::string result;
    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }

        batch = llama_batch_get_one(&new_token, 1);
        llama_decode(ctx, batch);
    }

    return result;
}