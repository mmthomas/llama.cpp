// heirloom_llama - implementation. See heirloom_llama.h. Compiled against llama.h (structs handled here).
#include "heirloom_llama.h"
#include "heirloom_sampler.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

struct hl_model { llama_model* model; };

using hl_context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

// CONT lines inherit the previous message's level, so warning continuations stay intact.
static void hl_log_filter(ggml_log_level level, const char* text, void* /*user*/) {
    static ggml_log_level last = GGML_LOG_LEVEL_INFO;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (last == GGML_LOG_LEVEL_WARN || last == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

static void hl_install_log_filter(void) {
    llama_log_set(hl_log_filter, nullptr);
    ggml_log_set(hl_log_filter, nullptr);
    mtmd_log_set(hl_log_filter, nullptr);
}

// The caller owns prefill and the context. This loop owns the sampler and output for both modalities.
static int hl_generate(llama_context* ctx, const llama_vocab* vocab, int n_predict,
                       float temp, float top_p, int top_k, float repeat_penalty,
                       const int* cancel, char* out, int out_cap, int& n_gen) {
    auto sampler = hl_make_sampler(llama_vocab_n_tokens(vocab), temp, top_p, top_k, repeat_penalty);
    if (!sampler) return -1;

    std::string result;
    std::vector<char> piece(512);
    for (int i = 0; i < n_predict; ++i) {
        if (cancel && *(const volatile int*)cancel) break;
        llama_token next = llama_sampler_sample(sampler.get(), ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int np = llama_token_to_piece(vocab, next, piece.data(), (int32_t)piece.size(), 0, true);
        if (np < 0) {
            piece.resize((size_t)-np);
            np = llama_token_to_piece(vocab, next, piece.data(), (int32_t)piece.size(), 0, true);
        }
        if (np < 0) return -1;
        result.append(piece.data(), (size_t)np);
        ++n_gen;
        llama_batch batch = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, batch) != 0) return -1;
    }

    int n = (int)result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    memcpy(out, result.data(), (size_t)n);
    out[n] = '\0';
    return n;
}

extern "C" {

void hl_backend_init(void) { hl_install_log_filter(); llama_backend_init(); }
void hl_backend_free(void) { llama_backend_free(); }

hl_model* hl_load(const char* model_path, int n_gpu_layers) {
    hl_install_log_filter();   // idempotent; covers callers that skip hl_backend_init
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
                float temp, float top_p, int top_k, float repeat_penalty,
                const int* cancel,
                char* out, int out_cap) {
    if (!m || !m->model || !prompt || !out || out_cap <= 0 || !hl_valid_temperature(temp)) return -1;
    out[0] = '\0';
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx          = (uint32_t)(n_ctx   > 0 ? n_ctx   : 4096);
    cp.n_ubatch       = (uint32_t)(n_ubatch > 0 ? n_ubatch : 512);
    cp.n_batch        = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch;
    cp.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    hl_context_ptr ctx(llama_init_from_model(m->model, cp), llama_free);
    if (!ctx) return -1;

    const int32_t len = (int32_t)strlen(prompt);
    int32_t n_prompt = -llama_tokenize(vocab, prompt, len, nullptr, 0, true, true);
    if (n_prompt <= 0) return -1;
    std::vector<llama_token> toks((size_t)n_prompt);
    if (llama_tokenize(vocab, prompt, len, toks.data(), n_prompt, true, true) < 0) return -1;

    const auto t_start = std::chrono::steady_clock::now();
    // Logical batches bound each decode call, including long caller-formatted text prompts.
    for (int32_t i = 0; i < n_prompt; i += (int32_t)cp.n_batch) {
        int32_t n = (int32_t)cp.n_batch;
        if (i + n > n_prompt) n = n_prompt - i;
        llama_batch chunk = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(ctx.get(), chunk) != 0) return -1;
    }
    const auto t_prefilled = std::chrono::steady_clock::now();

    int n_gen = 0;
    int rc = hl_generate(ctx.get(), vocab, n_predict, temp, top_p, top_k, repeat_penalty, cancel, out, out_cap, n_gen);
    const auto t_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefilled - t_start).count();
    double gen_ms = std::chrono::duration<double, std::milli>(t_end - t_prefilled).count();
    fprintf(stderr, "[hl_complete] ctx=%u fa=%d prompt=%d tok prefill=%.0fms (%.0f t/s) | gen=%d tok %.0fms (%.1f t/s) rc=%d\n",
            cp.n_ctx, flash_attn, n_prompt, prefill_ms, prefill_ms > 0 ? n_prompt * 1000.0 / prefill_ms : 0.0,
            n_gen, gen_ms, gen_ms > 0 ? n_gen * 1000.0 / gen_ms : 0.0, rc);
    fflush(stderr);
    return rc;
}

int hl_generate_image(hl_model* m, const char* mmproj_path,
                      const unsigned char* image_buf, int image_len,
                      const char* prompt, int prompt_kind, int n_predict,
                      int n_ctx, int n_ubatch, int image_min_tokens,
                      float temp, float top_p, int top_k, float repeat_penalty,
                      const int* cancel,
                      char* out, int out_cap) {
    if (!m || !m->model || !mmproj_path || !image_buf || image_len <= 0 || !prompt || !out || out_cap <= 0 ||
        !hl_valid_temperature(temp) || (prompt_kind != HL_PROMPT_WIRE && prompt_kind != HL_PROMPT_MODEL_TEMPLATE)) return -1;
    out[0] = '\0';
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    // Force vision FA with warmup disabled.
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = true;
    mp.print_timings    = false;
    mp.warmup           = false;
    mp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (image_min_tokens > 0) mp.image_min_tokens = image_min_tokens;
    mtmd::context_ptr mctx(mtmd_init_from_file(mmproj_path, m->model, mp));
    if (!mctx) return -1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)(n_ctx   > 0 ? n_ctx   : 8192);
    cp.n_ubatch        = (uint32_t)(n_ubatch > 0 ? n_ubatch : 2048);
    cp.n_batch         = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    hl_context_ptr ctx(llama_init_from_model(m->model, cp), llama_free);
    if (!ctx) return -1;

    std::string full = prompt;
    if (prompt_kind == HL_PROMPT_MODEL_TEMPLATE) {
        std::string content = std::string(mtmd_default_marker()) + "\n" + prompt;
        llama_chat_message msg{ "user", content.c_str() };
        const char* tmpl = llama_model_chat_template(m->model, nullptr);
        std::vector<char> fbuf(content.size() * 2 + 1024);
        int fn = llama_chat_apply_template(tmpl, &msg, 1, true, fbuf.data(), (int32_t)fbuf.size());
        if (fn > (int)fbuf.size()) {
            fbuf.resize((size_t)fn + 1);
            fn = llama_chat_apply_template(tmpl, &msg, 1, true, fbuf.data(), (int32_t)fbuf.size());
        }
        if (fn <= 0 || fn > (int)fbuf.size()) return -1;
        full.assign(fbuf.data(), (size_t)fn);
    }

    auto media = mtmd_helper_bitmap_init_from_buf(mctx.get(), image_buf, (size_t)image_len, false, mtmd_helper_init_opt_default());
    mtmd::bitmap_ptr bmp(media.bitmap);
    mtmd_helper::video_ptr video(media.video_ctx);
    if (!bmp) return HL_ERROR_IMAGE_DECODE;

    mtmd_input_text it{ full.c_str(), full.size(), true, true };
    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    const mtmd_bitmap* bmps[1] = { bmp.get() };
    int tok = mtmd_tokenize(mctx.get(), chunks.get(), &it, bmps, 1);
    bmp.reset();
    video.reset();
    if (tok != 0) return -1;

    const auto t_start = std::chrono::steady_clock::now();
    llama_pos new_n_past = 0;
    int ev = mtmd_helper_eval_chunks(mctx.get(), ctx.get(), chunks.get(), /*n_past*/0, /*seq_id*/0,
                                     /*n_batch*/(int32_t)cp.n_batch, /*logits_last*/true, &new_n_past);
    chunks.reset();
    if (ev != 0) return -1;
    const auto t_prefilled = std::chrono::steady_clock::now();

    int n_gen = 0;
    int rc = hl_generate(ctx.get(), vocab, n_predict, temp, top_p, top_k, repeat_penalty, cancel, out, out_cap, n_gen);
    const auto t_end = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefilled - t_start).count();
    double gen_ms = std::chrono::duration<double, std::milli>(t_end - t_prefilled).count();
    fprintf(stderr, "[hl_generate_image] ctx=%u img=%dB n_past=%d prefill=%.0fms | gen=%d tok %.0fms (%.1f t/s) rc=%d\n",
            cp.n_ctx, image_len, (int)new_n_past, prefill_ms,
            n_gen, gen_ms, gen_ms > 0 ? n_gen * 1000.0 / gen_ms : 0.0, rc);
    fflush(stderr);
    return rc;
}

// Shared: build an embeddings context (embeddings on, last-token pooling — Qwen3-VL-Embedding).
static llama_context* hl_embed_ctx(hl_model* m, int n_ctx, int n_ubatch) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx            = (uint32_t)(n_ctx   > 0 ? n_ctx   : 4096);
    cp.n_ubatch         = (uint32_t)(n_ubatch > 0 ? n_ubatch : 2048);
    cp.n_batch          = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch;
    cp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.embeddings       = true;
    cp.pooling_type     = LLAMA_POOLING_TYPE_LAST;
    return llama_init_from_model(m->model, cp);
}

static int hl_embed_finish(llama_context* ctx, hl_model* m, float* out, int out_cap) {
    const float* emb = llama_get_embeddings_seq(ctx, 0);
    if (!emb) emb = llama_get_embeddings(ctx);
    if (!emb) return -1;
    int n_embd = llama_model_n_embd(m->model);
    int n = n_embd < out_cap ? n_embd : out_cap;
    memcpy(out, emb, (size_t)n * sizeof(float));
    return n_embd;
}

int hl_embed_image(hl_model* m, const char* mmproj_path,
                   const unsigned char* image_buf, int image_len,
                   int n_ctx, int n_ubatch, int image_min_tokens,
                   float* out, int out_cap) {
    if (!m || !m->model || !mmproj_path || !image_buf || image_len <= 0 || !out || out_cap <= 0) return -1;

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu = true; mp.print_timings = false; mp.warmup = false;
    mp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (image_min_tokens > 0) mp.image_min_tokens = image_min_tokens;
    mtmd_context* mctx = mtmd_init_from_file(mmproj_path, m->model, mp);
    if (!mctx) return -1;

    llama_context* ctx = hl_embed_ctx(m, n_ctx, n_ubatch);
    if (!ctx) { mtmd_free(mctx); return -1; }

    auto media = mtmd_helper_bitmap_init_from_buf(mctx, image_buf, (size_t)image_len, false, mtmd_helper_init_opt_default());
    mtmd::bitmap_ptr bmp(media.bitmap);
    mtmd_helper::video_ptr video(media.video_ctx);
    if (!bmp) { llama_free(ctx); mtmd_free(mctx); return -1; }
    std::string content = std::string(mtmd_default_marker());   // document = the image
    mtmd_input_text it{ content.c_str(), content.size(), true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = { bmp.get() };
    int tok = mtmd_tokenize(mctx, chunks, &it, bmps, 1);
    bmp.reset();
    video.reset();
    if (tok != 0) { mtmd_input_chunks_free(chunks); llama_free(ctx); mtmd_free(mctx); return -1; }
    llama_pos n_past = 0;
    int ev = mtmd_helper_eval_chunks(mctx, ctx, chunks, /*n_past*/0, /*seq_id*/0, /*n_batch*/(int32_t)2048, /*logits_last*/true, &n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) { llama_free(ctx); mtmd_free(mctx); return -1; }

    int rc = hl_embed_finish(ctx, m, out, out_cap);
    llama_free(ctx); mtmd_free(mctx);
    return rc;
}

int hl_embed_text(hl_model* m, const char* prompt, int n_ctx, int n_ubatch, float* out, int out_cap) {
    if (!m || !m->model || !prompt || !out || out_cap <= 0) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    llama_context* ctx = hl_embed_ctx(m, n_ctx, n_ubatch);
    if (!ctx) return -1;

    const int32_t len = (int32_t)strlen(prompt);
    int32_t n_prompt = -llama_tokenize(vocab, prompt, len, nullptr, 0, true, true);
    if (n_prompt <= 0) { llama_free(ctx); return -1; }
    std::vector<llama_token> toks((size_t)n_prompt);
    if (llama_tokenize(vocab, prompt, len, toks.data(), n_prompt, true, true) < 0) { llama_free(ctx); return -1; }
    llama_batch b = llama_batch_get_one(toks.data(), n_prompt);
    if (llama_decode(ctx, b) != 0) { llama_free(ctx); return -1; }

    int rc = hl_embed_finish(ctx, m, out, out_cap);
    llama_free(ctx);
    return rc;
}

} // extern "C"
