#include "heirloom_sampler.h"

#include <cstring>
#include <limits>
#include <vector>

static void check_chain(llama_sampler* sampler, const std::vector<const char*>& names) {
    GGML_ASSERT(sampler);
    GGML_ASSERT(llama_sampler_chain_n(sampler) == (int)names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        GGML_ASSERT(strcmp(llama_sampler_name(llama_sampler_chain_get(sampler, (int)i)), names[i]) == 0);
    }
}

int main() {
    for (float temp : { -0.1f, -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        GGML_ASSERT(!hl_valid_temperature(temp));
        GGML_ASSERT(!hl_make_sampler(100, temp, 0.95f, 40, 1.0f));
    }

    // A repeat of the highest-logit token must still win in greedy mode, even with aggressive filters.
    auto greedy = hl_make_sampler(100, 0.0f, 0.01f, 1, 100.0f);
    check_chain(greedy.get(), { "greedy" });
    llama_sampler_accept(greedy.get(), 1);
    llama_token_data greedy_data[] = { { 0, 1.0f, 0.0f }, { 1, 2.0f, 0.0f } };
    llama_token_data_array greedy_candidates = { greedy_data, 2, -1, false };
    llama_sampler_apply(greedy.get(), &greedy_candidates);
    GGML_ASSERT(greedy_candidates.data[greedy_candidates.selected].id == 1);
    GGML_ASSERT(greedy_data[1].logit == 2.0f);

    auto sampled = hl_make_sampler(100, 0.3f, 0.8f, 7, 1.1f);
    check_chain(sampled.get(), { "penalties", "top-k", "top-p", "temp", "dist" });
    llama_sampler_accept(sampled.get(), 1);
    llama_token_data penalty_data[] = { { 0, 1.0f, 0.0f }, { 1, 2.2f, 0.0f } };
    llama_token_data_array penalties = { penalty_data, 2, -1, false };
    llama_sampler_apply(llama_sampler_chain_get(sampled.get(), 0), &penalties);
    GGML_ASSERT(std::fabs(penalty_data[1].logit - 2.0f) < 0.00001f);
    llama_sampler_apply(llama_sampler_chain_get(sampled.get(), 3), &penalties);
    GGML_ASSERT(std::fabs(penalty_data[1].logit - 2.0f / 0.3f) < 0.00001f);

    // Check our resolved defaults and explicit cutoffs on the constructed chain, without a model or backend.
    for (bool defaults : { false, true }) {
        auto sampler = hl_make_sampler(100, 0.7f, defaults ? 0.0f : 0.8f, defaults ? 0 : 7, 1.0f);
        check_chain(sampler.get(), { "top-k", "top-p", "temp", "dist" });
        std::vector<llama_token_data> data;
        for (int i = 0; i < 100; ++i) data.push_back({ i, 0.0f, 0.0f });
        llama_token_data_array candidates = { data.data(), data.size(), -1, false };
        llama_sampler_apply(llama_sampler_chain_get(sampler.get(), 0), &candidates);
        GGML_ASSERT(candidates.size == (defaults ? 40u : 7u));
        candidates.size = data.size();
        candidates.sorted = false;
        llama_sampler_apply(llama_sampler_chain_get(sampler.get(), 1), &candidates);
        // Float accumulation can put the boundary one token beyond the mathematical cutoff.
        GGML_ASSERT(candidates.size >= (defaults ? 95u : 80u));
        GGML_ASSERT(candidates.size <= (defaults ? 96u : 81u));
    }
    return 0;
}
