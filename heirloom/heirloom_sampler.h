#pragma once

#include "llama.h"

#include <cmath>
#include <memory>

using hl_sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

inline bool hl_valid_temperature(float temp) {
    return std::isfinite(temp) && temp >= 0.0f;
}

// Greedy ignores all filtering. Sampled generation applies penalties before truncation and temperature.
inline hl_sampler_ptr hl_make_sampler(int n_vocab, float temp, float top_p, int top_k, float repeat_penalty) {
    if (!hl_valid_temperature(temp)) return { nullptr, llama_sampler_free };

    hl_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()), llama_sampler_free);
    if (temp == 0.0f) {
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
    } else {
        if (repeat_penalty > 1.0f) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(n_vocab, 64, repeat_penalty, 0.0f, 0.0f));
        }
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(top_k > 0 ? top_k : 40));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(top_p > 0.0f ? top_p : 0.95f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    return sampler;
}
