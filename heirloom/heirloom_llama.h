// heirloom_llama — a thin, stable C ABI over llama.cpp for P/Invoke from .NET (Heirloom).
// Rationale: llama.h passes config as C structs *by value*, which is fragile to marshal from C#.
// This shim keeps the structs inside C++ and exposes flat int/char*/void* functions that P/Invoke
// cleanly. It is Heirloom's permanent fork delta (NOT upstreamed); keep it minimal.
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
//   temp: finite and >=0; 0 = greedy (no penalties/filtering), positive = sampled.
//   Sampled order: penalty -> top-k -> top-p -> temp -> dist (LLAMA_DEFAULT_SEED).
//   top_p<=0 = 0.95, top_k<=0 = 40, repeat_penalty<=1 = off (otherwise a 64-token window).
//   cancel: optional pointer to an int polled once per generated token — nonzero stops generation (cooperative
//           cancel for user-Stop / GPU reclaim). Pass null to disable.
// Writes a null-terminated UTF-8 string into out (truncated to out_cap-1). Returns bytes written, or -1 on
// invalid arguments, prefill or generation failure. Cancellation returns successful partial output (>=0).
HL_API int hl_complete(hl_model* m, const char* prompt, int n_predict,
                       int n_ctx, int n_ubatch, int flash_attn,
                       float temp, float top_p, int top_k, float repeat_penalty,
                       const int* cancel,
                       char* out, int out_cap);

typedef enum hl_prompt_kind {
    HL_PROMPT_WIRE = 0,           // Full wire text containing mtmd's "<__media__>" marker; used verbatim.
    HL_PROMPT_MODEL_TEMPLATE = 1, // User instruction; model chat template wraps (media-marker + "\n" + prompt).
} hl_prompt_kind;

enum { HL_ERROR_IMAGE_DECODE = -2 };

// Image generation from in-memory bytes. prompt_kind is an hl_prompt_kind value, passed as a flat int.
// Same sampler, output and cancellation contract as hl_complete. -2 means image bytes could not be decoded;
// all other failures return -1. Image generation always enables vision and decoder FA.
// n_ctx<=0 = 8192; n_ubatch<=0 = 2048; image_min_tokens<=0 = model default.
HL_API int hl_generate_image(hl_model* m, const char* mmproj_path,
                             const unsigned char* image_buf, int image_len,
                             const char* prompt, int prompt_kind, int n_predict,
                             int n_ctx, int n_ubatch, int image_min_tokens,
                             float temp, float top_p, int top_k, float repeat_penalty,
                             const int* cancel,
                             char* out, int out_cap);

// Multimodal EMBEDDING: encode an image (in-memory bytes) via the mmproj + LM into a single pooled vector
// (Qwen3-VL-Embedding, last-token pooling). Writes up to out_cap floats; returns the embedding dim (n_embd),
// or -1. image_min_tokens<=0 = model default.
HL_API int hl_embed_image(hl_model* m, const char* mmproj_path,
                          const unsigned char* image_buf, int image_len,
                          int n_ctx, int n_ubatch, int image_min_tokens,
                          float* out, int out_cap);

// Text EMBEDDING into the same joint space (queries). `prompt` is the raw text (caller adds any instruction).
// Writes up to out_cap floats; returns n_embd, or -1.
HL_API int hl_embed_text(hl_model* m, const char* prompt, int n_ctx, int n_ubatch, float* out, int out_cap);

#ifdef __cplusplus
}
#endif
