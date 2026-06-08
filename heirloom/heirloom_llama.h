// heirloom_llama — a thin, stable C ABI over llama.cpp for P/Invoke from .NET (Heirloom).
// Rationale: llama.h passes config as C structs *by value*, which is fragile to marshal from C#.
// This shim keeps the structs inside C++ and exposes flat int/char*/void* functions that P/Invoke
// cleanly. It is Heirloom's permanent fork delta (NOT upstreamed); keep it additive + minimal.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #define HL_API __declspec(dllexport)
#else
  #define HL_API __attribute__((visibility("default")))
#endif

typedef struct hl_model hl_model; // opaque handle (owns a llama_model)

// Process-wide backend init/teardown (call once each).
HL_API void hl_backend_init(void);
HL_API void hl_backend_free(void);

// Load a model. n_gpu_layers: 0 = CPU, large (e.g. 999) = offload all. Returns null on failure.
HL_API hl_model* hl_load(const char* model_path, int n_gpu_layers);
HL_API void      hl_free(hl_model* m);

// Model metadata (smoke/diagnostics).
HL_API int64_t hl_n_params(hl_model* m);
HL_API int     hl_desc(hl_model* m, char* out, int out_cap); // bytes written (excl. null), or -1

// One-shot raw completion (no chat template — caller owns templating for now).
//   flash_attn: 0 = disabled, 1 = enabled.
//   temp/top_p/top_k/repeat_penalty: sampler knobs (<=0 / <=1.0 fall back to mild defaults; repeat_penalty<=1 = off).
//   cancel: optional pointer to an int polled once per generated token — nonzero stops generation (cooperative
//           cancel for user-Stop / GPU reclaim). Pass null to disable.
// Writes a null-terminated UTF-8 string into out (truncated to out_cap-1). Returns bytes written, or -1.
HL_API int hl_complete(hl_model* m, const char* prompt, int n_predict,
                       int n_ctx, int n_ubatch, int flash_attn,
                       float temp, float top_p, int top_k, float repeat_penalty,
                       const int* cancel,
                       char* out, int out_cap);

// Vision: caption/answer about an image via an mmproj (multimodal projector). Applies the model's chat
// template around (media-marker + user_prompt), runs the mtmd encode+decode, then generates. flash_attn
// forced off (sm_120a). image_min_tokens<=0 = model default (Qwen-VL wants >=1024 for grounding).
// Writes a null-terminated UTF-8 string into out. Returns bytes written, or -1.
HL_API int hl_caption(hl_model* m, const char* mmproj_path, const char* image_path, const char* user_prompt,
                      int n_predict, int n_ctx, int n_ubatch, int image_min_tokens,
                      char* out, int out_cap);

// Multimodal completion over a PRE-TEMPLATED prompt (the agent's tool-calling path). Unlike hl_caption, the
// caller owns the full chat template (system + <tools> + history turns) and embeds mtmd's image marker
// ("<__media__>", see mtmd_default_marker()) in the last user turn; mtmd swaps the marker for the image
// (passed as in-memory JPEG/PNG bytes — no temp file). Then it generates with the same caller-tuned sampler +
// cooperative cancel as hl_complete, so the model can answer in plain text OR emit a <tool_call>. flash_attn
// forced off (sm_120a). image_min_tokens<=0 = model default. Returns bytes written, or -1.
HL_API int hl_complete_image(hl_model* m, const char* mmproj_path,
                             const unsigned char* image_buf, int image_len,
                             const char* prompt, int n_predict,
                             int n_ctx, int n_ubatch, int image_min_tokens,
                             float temp, float top_p, int top_k, float repeat_penalty,
                             const int* cancel,
                             char* out, int out_cap);

#ifdef __cplusplus
}
#endif
