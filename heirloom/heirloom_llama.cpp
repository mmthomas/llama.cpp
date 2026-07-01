// heirloom_llama — implementation. See heirloom_llama.h. Compiled against llama.h (structs handled here).
#include "heirloom_llama.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cstdio>
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
                float temp, float top_p, int top_k, float repeat_penalty,
                const int* cancel,
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

    // Sampler chain: caller-tuned (optional repetition penalty -> top-k -> top-p -> temp -> dist); non-positive
    // values fall back to mild defaults. Penalty first so it adjusts logits before truncation/temperature.
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    if (repeat_penalty > 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k > 0 ? top_k : 40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p > 0.0f ? top_p : 0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp > 0.0f ? temp : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string result;
    char piece[512];

    const auto t_start = std::chrono::steady_clock::now();

    // Prefill the prompt in n_batch-sized chunks: a single llama_decode is capped at n_batch (logical), so a
    // long prompt (e.g. the agent's grammar system prompt) must be fed in pieces rather than one batch.
    bool prefilled = true;
    for (int32_t i = 0; i < n_prompt; i += (int32_t)cp.n_batch) {
        int32_t n = (int32_t)cp.n_batch;
        if (i + n > n_prompt) n = n_prompt - i;
        llama_batch chunk = llama_batch_get_one(toks.data() + i, n);
        if (llama_decode(ctx, chunk) != 0) { prefilled = false; break; }
    }

    const auto t_prefilled = std::chrono::steady_clock::now();

    // Generate from the last prompt position (logits of the final prefill token).
    int n_gen = 0;
    llama_token next = 0; // stable address across iterations for the single-token batch
    for (int i = 0; prefilled && i < n_predict; ++i) {
        if (cancel && *(const volatile int*)cancel) break;   // cooperative cancel (user Stop / GPU reclaim)
        next = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int np = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, true);
        if (np > 0) result.append(piece, (size_t)np);
        ++n_gen;
        llama_batch gen = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, gen) != 0) break;
    }

    // Per-turn timing breakdown (stderr → daemon.err.log): isolates prefill vs generation so we optimize the
    // real bottleneck (generation t/s vs re-prefill cost) rather than guess.
    {
        const auto t_end = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefilled - t_start).count();
        double gen_ms     = std::chrono::duration<double, std::milli>(t_end - t_prefilled).count();
        fprintf(stderr, "[hl_complete] ctx=%u fa=%d prompt=%d tok prefill=%.0fms (%.0f t/s) | gen=%d tok %.0fms (%.1f t/s)\n",
                cp.n_ctx, flash_attn, n_prompt, prefill_ms, prefill_ms > 0 ? n_prompt * 1000.0 / prefill_ms : 0.0,
                n_gen, gen_ms, gen_ms > 0 ? n_gen * 1000.0 / gen_ms : 0.0);
        fflush(stderr);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);

    int n = (int)result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    memcpy(out, result.data(), (size_t)n);
    out[n] = '\0';
    return n;
}

int hl_caption(hl_model* m, const char* mmproj_path, const char* image_path, const char* user_prompt,
               int n_predict, int n_ctx, int n_ubatch, int image_min_tokens,
               char* out, int out_cap) {
    if (!m || !m->model || !mmproj_path || !image_path || !out || out_cap <= 0) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    // multimodal context (loads the mmproj/clip + audio projectors).
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = true;
    mp.print_timings    = false;
    mp.warmup           = false;
    mp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (image_min_tokens > 0) mp.image_min_tokens = image_min_tokens;
    mtmd_context* mctx = mtmd_init_from_file(mmproj_path, m->model, mp);
    if (!mctx) return -1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)(n_ctx   > 0 ? n_ctx   : 8192);
    cp.n_ubatch        = (uint32_t)(n_ubatch > 0 ? n_ubatch : 2048);
    cp.n_batch         = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch; // image tokens must fit one ubatch
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(m->model, cp);
    if (!ctx) { mtmd_free(mctx); return -1; }

    // Templated prompt: user content = media-marker + the request. mtmd_tokenize() swaps the marker for the image.
    std::string content = std::string(mtmd_default_marker()) + "\n" + (user_prompt ? user_prompt : "Describe this image.");
    llama_chat_message msg{ "user", content.c_str() };
    const char* tmpl = llama_model_chat_template(m->model, nullptr);
    std::vector<char> fbuf(content.size() * 2 + 1024);
    int fn = llama_chat_apply_template(tmpl, &msg, 1, true, fbuf.data(), (int32_t)fbuf.size());
    if (fn > (int)fbuf.size()) { fbuf.resize((size_t)fn + 1); fn = llama_chat_apply_template(tmpl, &msg, 1, true, fbuf.data(), (int32_t)fbuf.size()); }
    if (fn <= 0) { llama_free(ctx); mtmd_free(mctx); return -1; }
    std::string full(fbuf.data(), (size_t)fn);

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(mctx, image_path, false);
    if (!bmp) { llama_free(ctx); mtmd_free(mctx); return -1; }

    mtmd_input_text it{ full.c_str(), true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = { bmp };
    int tok = mtmd_tokenize(mctx, chunks, &it, bmps, 1);
    mtmd_bitmap_free(bmp);
    if (tok != 0) { mtmd_input_chunks_free(chunks); llama_free(ctx); mtmd_free(mctx); return -1; }

    llama_pos new_n_past = 0;
    int ev = mtmd_helper_eval_chunks(mctx, ctx, chunks, /*n_past*/0, /*seq_id*/0,
                                     /*n_batch*/(int32_t)cp.n_batch, /*logits_last*/true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) { llama_free(ctx); mtmd_free(mctx); return -1; }

    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // eval_chunks left last-token logits ready (logits_last=true): sample, then decode the chosen token.
    std::string result;
    llama_token next = 0;
    char piece[512];
    for (int i = 0; i < n_predict; ++i) {
        next = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int np = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, true);
        if (np > 0) result.append(piece, (size_t)np);
        llama_batch b = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, b) != 0) break;
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    mtmd_free(mctx);

    int n = (int)result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    memcpy(out, result.data(), (size_t)n);
    out[n] = '\0';
    return n;
}

int hl_complete_image(hl_model* m, const char* mmproj_path,
                      const unsigned char* image_buf, int image_len,
                      const char* prompt, int n_predict,
                      int n_ctx, int n_ubatch, int image_min_tokens,
                      float temp, float top_p, int top_k, float repeat_penalty,
                      const int* cancel,
                      char* out, int out_cap) {
    if (!m || !m->model || !mmproj_path || !image_buf || image_len <= 0 || !prompt || !out || out_cap <= 0) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    // multimodal context (loads the mmproj/clip). flash_attn off on sm_120a (#21159), as in hl_caption.
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = true;
    mp.print_timings    = false;
    mp.warmup           = false;
    mp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (image_min_tokens > 0) mp.image_min_tokens = image_min_tokens;
    mtmd_context* mctx = mtmd_init_from_file(mmproj_path, m->model, mp);
    if (!mctx) return -1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)(n_ctx   > 0 ? n_ctx   : 8192);
    cp.n_ubatch        = (uint32_t)(n_ubatch > 0 ? n_ubatch : 2048);
    cp.n_batch         = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch; // image tokens must fit one ubatch
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(m->model, cp);
    if (!ctx) { mtmd_free(mctx); return -1; }

    // The prompt is ALREADY fully templated (the agent's <|im_start|> system + <tools> + turns) with the media
    // marker embedded in the last user turn; mtmd_tokenize swaps the marker for the in-memory image. add_special
    // / parse_special = true, matching hl_complete's verbatim tokenize.
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(mctx, image_buf, (size_t)image_len, false);
    if (!bmp) { llama_free(ctx); mtmd_free(mctx); return -1; }

    mtmd_input_text it{ prompt, true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = { bmp };
    int tok = mtmd_tokenize(mctx, chunks, &it, bmps, 1);
    mtmd_bitmap_free(bmp);
    if (tok != 0) { mtmd_input_chunks_free(chunks); llama_free(ctx); mtmd_free(mctx); return -1; }

    const auto t_start = std::chrono::steady_clock::now();
    llama_pos new_n_past = 0;
    int ev = mtmd_helper_eval_chunks(mctx, ctx, chunks, /*n_past*/0, /*seq_id*/0,
                                     /*n_batch*/(int32_t)cp.n_batch, /*logits_last*/true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) { llama_free(ctx); mtmd_free(mctx); return -1; }
    const auto t_prefilled = std::chrono::steady_clock::now();

    // Sampler chain: caller-tuned, identical to hl_complete (penalty -> top-k -> top-p -> temp -> dist) so the
    // tool-calling temperature matches the text path.
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    if (repeat_penalty > 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k > 0 ? top_k : 40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p > 0.0f ? top_p : 0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp > 0.0f ? temp : 0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string result;
    llama_token next = 0;
    char piece[512];
    int n_gen = 0;
    for (int i = 0; i < n_predict; ++i) {
        if (cancel && *(const volatile int*)cancel) break;   // cooperative cancel (user Stop / GPU reclaim)
        next = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int np = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, true);
        if (np > 0) result.append(piece, (size_t)np);
        ++n_gen;
        llama_batch b = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, b) != 0) break;
    }

    {
        const auto t_end = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefilled - t_start).count();
        double gen_ms     = std::chrono::duration<double, std::milli>(t_end - t_prefilled).count();
        fprintf(stderr, "[hl_complete_image] ctx=%u img=%dB n_past=%d prefill=%.0fms | gen=%d tok %.0fms (%.1f t/s)\n",
                cp.n_ctx, image_len, (int)new_n_past, prefill_ms,
                n_gen, gen_ms, gen_ms > 0 ? n_gen * 1000.0 / gen_ms : 0.0);
        fflush(stderr);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    mtmd_free(mctx);

    int n = (int)result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    memcpy(out, result.data(), (size_t)n);
    out[n] = '\0';
    return n;
}

int hl_ocr(hl_model* m, const char* mmproj_path,
           const unsigned char* image_buf, int image_len,
           const char* user_prompt, int n_predict,
           int n_ctx, int n_ubatch, int image_min_tokens,
           float temp, float top_p, int top_k, float repeat_penalty,
           const int* cancel,
           char* out, int out_cap) {
    if (!m || !m->model || !mmproj_path || !image_buf || image_len <= 0 || !user_prompt || !out || out_cap <= 0) return -1;
    const llama_vocab* vocab = llama_model_get_vocab(m->model);

    // multimodal context (loads the mmproj/clip). The GLM4V vision encoder runs full O(n^2) attention over the
    // patch grid (HF parity: Glm4vVisionModel has no windowing); at OCR resolution a 8+ MP image is a ~42k-patch
    // grid whose fp32 KQ score matrix reserves ~85 GiB and OOMs a 32 GiB card. Flash attention (ggml_flash_attn_ext,
    // matching HF's SDPA path) never materializes that matrix and bounds the vision buffer to a few GiB. warmup is
    // off here, so AUTO would never resolve to ENABLED (the resolve runs only under mtmd warmup) — force ENABLED.
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu          = true;
    mp.print_timings    = false;
    mp.warmup           = false;
    mp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (image_min_tokens > 0) mp.image_min_tokens = image_min_tokens;
    mtmd_context* mctx = mtmd_init_from_file(mmproj_path, m->model, mp);
    if (!mctx) return -1;

    // The text decode context keeps flash_attn off (the separate sm_120a decode-FA concern); only the vision
    // encoder above needs it, and that is where the O(n^2) blow-up lives.
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)(n_ctx   > 0 ? n_ctx   : 8192);
    cp.n_ubatch        = (uint32_t)(n_ubatch > 0 ? n_ubatch : 2048);
    cp.n_batch         = cp.n_ubatch < 2048 ? 2048 : cp.n_ubatch; // image tokens must fit one ubatch
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context* ctx = llama_init_from_model(m->model, cp);
    if (!ctx) { mtmd_free(mctx); return -1; }

    // Model-templated prompt: user content = media-marker + the OCR instruction; the model's own chat template
    // (llama_chat_apply_template) wraps it, exactly as hl_caption does. mtmd_tokenize swaps the marker for the
    // in-memory image bytes (hl_complete_image's path — no temp file).
    std::string content = std::string(mtmd_default_marker()) + "\n" + user_prompt;
    llama_chat_message msg{ "user", content.c_str() };
    const char* tmpl = llama_model_chat_template(m->model, nullptr);
    std::vector<char> fbuf(content.size() * 2 + 1024);
    int fn = llama_chat_apply_template(tmpl, &msg, 1, true, fbuf.data(), (int32_t)fbuf.size());
    if (fn > (int)fbuf.size()) { fbuf.resize((size_t)fn + 1); fn = llama_chat_apply_template(tmpl, &msg, 1, true, fbuf.data(), (int32_t)fbuf.size()); }
    if (fn <= 0) { llama_free(ctx); mtmd_free(mctx); return -1; }
    std::string full(fbuf.data(), (size_t)fn);

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(mctx, image_buf, (size_t)image_len, false);
    if (!bmp) { llama_free(ctx); mtmd_free(mctx); return -1; }

    mtmd_input_text it{ full.c_str(), true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = { bmp };
    int tok = mtmd_tokenize(mctx, chunks, &it, bmps, 1);
    mtmd_bitmap_free(bmp);
    if (tok != 0) { mtmd_input_chunks_free(chunks); llama_free(ctx); mtmd_free(mctx); return -1; }

    const auto t_start = std::chrono::steady_clock::now();
    llama_pos new_n_past = 0;
    int ev = mtmd_helper_eval_chunks(mctx, ctx, chunks, /*n_past*/0, /*seq_id*/0,
                                     /*n_batch*/(int32_t)cp.n_batch, /*logits_last*/true, &new_n_past);
    mtmd_input_chunks_free(chunks);
    if (ev != 0) { llama_free(ctx); mtmd_free(mctx); return -1; }
    const auto t_prefilled = std::chrono::steady_clock::now();

    // Sampler chain. OCR default is GREEDY (argmax): GLM-OCR ships do_sample=false, and greedy is the deterministic
    // transcription the qualification study ran; the entropy post-filter backstops the dense-grid loop greedy risks
    // (Holtzman). temp<=0 selects greedy; temp>0 samples (penalty -> top-k -> top-p -> temp -> dist) for A/B tests.
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sp);
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (repeat_penalty > 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, repeat_penalty, 0.0f, 0.0f));
        }
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k > 0 ? top_k : 40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p > 0.0f ? top_p : 0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    std::string result;
    llama_token next = 0;
    char piece[512];
    int n_gen = 0;
    for (int i = 0; i < n_predict; ++i) {
        if (cancel && *(const volatile int*)cancel) break;   // cooperative cancel (user Stop / GPU reclaim)
        next = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int np = llama_token_to_piece(vocab, next, piece, (int32_t)sizeof(piece), 0, true);
        if (np > 0) result.append(piece, (size_t)np);
        ++n_gen;
        llama_batch b = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, b) != 0) break;
    }

    {
        const auto t_end = std::chrono::steady_clock::now();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefilled - t_start).count();
        double gen_ms     = std::chrono::duration<double, std::milli>(t_end - t_prefilled).count();
        fprintf(stderr, "[hl_ocr] ctx=%u img=%dB n_past=%d prefill=%.0fms | gen=%d tok %.0fms (%.1f t/s)\n",
                cp.n_ctx, image_len, (int)new_n_past, prefill_ms,
                n_gen, gen_ms, gen_ms > 0 ? n_gen * 1000.0 / gen_ms : 0.0);
        fflush(stderr);
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    mtmd_free(mctx);

    int n = (int)result.size();
    if (n > out_cap - 1) n = out_cap - 1;
    memcpy(out, result.data(), (size_t)n);
    out[n] = '\0';
    return n;
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

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(mctx, image_buf, (size_t)image_len, false);
    if (!bmp) { llama_free(ctx); mtmd_free(mctx); return -1; }
    std::string content = std::string(mtmd_default_marker());   // document = the image
    mtmd_input_text it{ content.c_str(), true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bmps[1] = { bmp };
    int tok = mtmd_tokenize(mctx, chunks, &it, bmps, 1);
    mtmd_bitmap_free(bmp);
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
