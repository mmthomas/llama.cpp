// heirloom_llama — implementation. See heirloom_llama.h. Compiled against llama.h (structs handled here).
#include "heirloom_llama.h"
#include "llama.h"

#include <cstring>
#include <string>
#include <vector>

struct hl_model { llama_model* model; };

extern "C" {

void hl_backend_init(void) { llama_backend_init(); }
void hl_backend_free(void) { llama_backend_free(); }

hl_model* hl_load(const char* model_path, int n_gpu_layers) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) return nullptr;
    return new hl_model{ model };
}

void hl_free(hl_model* m) {
    if (!m) return;
    if (m->model) llama_model_free(m->model);
    delete m;
}

int64_t hl_n_params(hl_model* m) {
    return (m && m->model) ? (int64_t)llama_model_n_params(m->model) : -1;
}

int hl_desc(hl_model* m, char* out, int out_cap) {
    if (!m || !m->model || !out || out_cap <= 0) return -1;
    int n = llama_model_desc(m->model, out, (size_t)out_cap);
    return n;
}

int hl_complete(hl_model* m, const char* prompt, int n_predict,
                int n_ctx, int n_ubatch, int flash_attn,
                char* out, int out_cap) {
    if (!m || !m->model || !prompt || !out || out_cap <= 0) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx          = (uint32_t)(n_ctx   > 0 ? n_ctx   : 4096);
    cp.n_ubatch       = (uint32_t)(n_ubatch > 0 ? n_ubatch : 512);
    cp.n_batch        = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch; // logical >= physical, room for prompt
    cp.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context* ctx = llama_init_from_model(m->model, cp);
    if (!ctx) return -1;

    // Tokenize (two-pass: query length, then fill).
    const int32_t len = (int32_t)strlen(prompt);
    int32_t n_prompt = -llama_tokenize(vocab, prompt, len, nullptr, 0, true, true);
    if (n_prompt <= 0) { llama_free(ctx); return -1; }
    std::vector<llama_token> toks((size_t)n_prompt);
    if (llama_tokenize(vocab, prompt, len, toks.data(), n_prompt, true, true) < 0) { llama_free(ctx); return -1; }

    // Sampler chain: top-k / top-p / temp / dist (mild, for natural text in the smoke).
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string result;
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t)toks.size());
    llama_token next = 0; // stable address across iterations for the single-token batch
    char piece[512];
    for (int i = 0; i < n_predict; ++i) {
        if (llama_decode(ctx, batch) != 0) break;
        next = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int np = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, true);
        if (np > 0) result.append(piece, (size_t)np);
        batch = llama_batch_get_one(&next, 1);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);

    int n = (int)result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    memcpy(out, result.data(), (size_t)n);
    out[n] = '\0';
    return n;
}

} // extern "C"
