# Heirloom native runtime

The maintained branch is `heirloom` in `https://github.com/mmthomas/llama.cpp.git`. Its native source is pinned to reviewed upstream commit `73ab7599b553c03f6f5d2db24a18ad76f2eb36a3`, not to a moving `master`. Qualified source commit `1fc214646504b2f0bf6236ac80ae003bf4fcc8e0` is integrated by merge `cb51dfa2efed58fa22fb1f5420b25f771d686982`, whose tree is identical to the qualified commit.

**The unified-generation vision/decoder-FA variant is deployed as an immutable bundle.** The daemon's current state is available through `daemon-ctl status`; a source build does not mutate deployed binaries. Build from a checkout matching the selected ABI. The temporary qualification worktrees and their build outputs are disposable; their source commits remain in Git.

## Unified generation

The `generation-consistency` branch and matching Heirloom inference changes supply the deployed application/native ABI. The immutable runtime manifest preserves the pre-commit build's base revision and source hashes. Its ten exports are `hl_backend_init`, `hl_backend_free`, `hl_load`, `hl_free`, `hl_n_params`, `hl_desc`, `hl_complete`, `hl_generate_image`, `hl_embed_image`, and `hl_embed_text`. There are no compatibility image entry points.

`hl_generate_image` takes in-memory bytes and a flat `hl_prompt_kind` integer: `HL_PROMPT_WIRE` uses full caller wire text verbatim; `HL_PROMPT_MODEL_TEMPLATE` applies the model's native chat template to its media marker plus a user instruction. Text and image generation share sampler construction and the token decode loop. Finite temperature zero selects greedy without penalties/filtering, positive temperature preserves sampled penalty/top-k/top-p/temp/dist order, and negative/nonfinite temperature fails. Prefill/decode failures return -1; image-byte decoding failure returns -2. Cancellation can return successful partial output.

Image generation enables vision and decoder FA, with no caption configuration toggle. The managed code-level text-completion decoder option is explicitly text-only. Qwen caption settings and wire, GLM template and greedy settings, embedding attention/pooling, weights, resolution, and resident-model lifecycle are preserved. No template framework or native context caching is introduced.

The matching C# application requires `hl_generate_image` before initializing a backend or loading a model. The selected bundle is `C:\dev\heirloom-run\runtimes\llama-cuda\73ab7599b-5fa2f93ea24e`, core fingerprint `5fa2f93ea24e3f0e52e2b3e4f0bd8c2a549200e36fce45cd522978b1e8b0ac54`. The paired indexer executable has SHA-256 `f2a838030e682713d23093539c20ecd91c0dc5565aae2c0dd4351ec2e95a5b7c`. The September 8 rollout preserved the daemon's paused state and removed the obsolete caption-attention setting. The September 7 bundle is the preceding ABI and remains separate.

Build the focused targets with `cmake --build .\build --config Release --target heirloom_llama test-heirloom-sampler -j8` in the configured VS environment. Run `ctest --test-dir .\build -C Release -R "^test-heirloom-sampler$" --output-on-failure --no-tests=error`; this policy test uses no model or GPU. Model-level caption/OCR qualification and deployment are separate gates.

The September 8 decoder-off unified-generation probe records core fingerprint `40911dde034371e1b25290736f0b165f7253a3a887fa8a9ea85836f99b836094`. Nine reference/candidate cases match all recorded inputs/settings and output bytes; Qwen uses top-k=1 diagnostic sampling and GLM uses its production greedy settings. Candidate cases also establish zero-temperature equivalence, invalid-temperature/prompt-kind rejection and pre-cancellation behavior. The managed Qwen and raw OCR probes complete against this ABI, with raw OCR matching the reference. This is not a full-corpus stochastic-quality study or an 80-image lifecycle-memory qualification. The selected deployment uses the subsequently accepted decoder-FA-on variant.

An isolated decoder-FA study changes only the image decoder attention flag. Warm GLM complete-call medians are 1029.5 ms off versus 940.5 ms on with unchanged outputs on three inputs. Sampled process peaks fall from 20694.11 to 20334.02 MiB dedicated and 544 to 320 MiB shared memory. Qwen generation is effectively equal in the first matched pair; its pooled 68.87 versus 70.96 tok/s difference is within run-to-run variation. All six distinct Qwen captions change, including depicted details, so complete-caption latency is confounded by output length. Both modes are repeatable across three runs each, but no numerical factual-quality equivalence is claimed. The operator accepted the six-photo comparison on September 8; image decoder FA is the fixed source default. The tested decoder-on study fingerprint is `0b72d46398e374d5a8fd1cff77a958f4209537aeba142de88a5daa90500b4241`. Detailed evidence and the offline comparison are under the session's `files\decoder-fa-study`.

## Patch boundary

The starting fork delta is the net change from upstream tag `v0.3.0` to `refresh-v0.3.0` (`7c417ff12eb55bad737df51676e9cf093f8513fe`), not a merge of old fork history or the uncommitted `shim-consistency` refactor.

The permanent application adapter consists of `heirloom\heirloom_llama.cpp`, `heirloom\heirloom_llama.h`, the private `heirloom_sampler.h`, `heirloom\CMakeLists.txt`, the root `CMakeLists.txt` inclusion, and the two `heirloom-*-cuda.bat` scripts. `test-heirloom-sampler.cpp` exercises its sampler policy using the existing CTest runner. Model metadata and embedding entry points remain part of the ABI.

The bounded upstream adaptation passes `mtmd_helper_init_opt_default()` to image-loading helpers. The unified generation contract is specified above; the recorded September 7 qualification below applies to the preceding ABI and deployed fingerprint. Embedding FA settings remain unchanged. No MTP, DFlash2, resolution, or model/weight change accompanies this work.

Only three general GLM-OCR patches remain outside the adapter:

- `conversion\qwen3vl.py`: write GLM vision pixel bounds from `min_pixels`/`max_pixels` or `size.shortest_edge`/`size.longest_edge`; fail explicitly when missing.
- `tools\mtmd\clip.cpp`: retain bicubic resize, use `PAD_NONE`, default to 16..12288 image tokens, then read optional model pixel bounds.
- `tools\mtmd\models\glm4v.cpp`: skip post-convolution normalization only when `norm_embd_w` is absent.

These small core patches are removable when upstream supplies equivalent behavior. The pinned upstream already supplies Qwen preprocessing/position interpolation and CUDA reduction fixes; no duplicate local hunks are replayed. Its GDN normalization and CUDA FA barrier fixes motivate this candidate, but do not establish behavioral qualification.

There is no local ggml/CUDA backend policy delta. The build requests `CMAKE_CUDA_ARCHITECTURES=120`; unmodified upstream resolves it to `compute_120a`/`sm_120a` for architecture-specific Blackwell instructions. This matches the actual generated CUDA flags in both production and the older v0.3.0 candidate, despite their caches also recording `120`. Native Blackwell MMA support and dispatch remain upstream defaults.

## Windows build

Use the existing VS 2026 BuildTools environment at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat`, portable CMake at `C:\dev\tools\cmake\bin`, Ninja at `C:\dev\tools\ninja`, and CUDA 13.3 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`.

From PowerShell in the checkout root:

```powershell
New-Item -ItemType Directory -Force .\build | Out-Null
& $env:ComSpec /d /c 'call ".\heirloom-configure-cuda.bat" > ".\build\configure.log" 2>&1'
if ($LASTEXITCODE -ne 0) { throw "Candidate configuration failed; see build\configure.log" }
& $env:ComSpec /d /c 'call ".\heirloom-build-cuda.bat" > ".\build\build.log" 2>&1'
if ($LASTEXITCODE -ne 0) { throw "Candidate build failed; see build\build.log" }
```

Each script initializes `vcvars64.bat` and runs CMake in the same `cmd.exe` process. Configuration uses Ninja, Release, `BUILD_SHARED_LIBS=ON`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120` (resolved by upstream to `120a`), and explicit CUDA 13.3 compiler/toolkit paths. `LLAMA_BUILD_COMMON`, `LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_SERVER`, `LLAMA_SUBPROCESS`, and `MTMD_VIDEO` are explicitly ON. Video support is preserved like-for-like; upstream requires `ffmpeg`/`ffprobe` on PATH for relevant media inputs.

`LLAMA_CURL=OFF` is retained as requested, although this upstream revision deprecates that option and uses cpp-httplib instead. It is not a network-disable switch.

The build command is `cmake --build <candidate>\build --config Release --target heirloom_llama llama-cli llama-mtmd-cli llama-server test-backend-ops -j8`. It builds the shim, llama, mtmd, ggml-base, ggml, CPU and CUDA DLLs plus the four requested executables in `build\bin`. The current CLIs also require `llama-common.dll`, `llama-cli-impl.dll`, and `llama-server-impl.dll` from that directory. It does not run those executables or install/deploy anything. Keep the candidate's DLL set together; do not mix it with production DLLs.

### Recorded build limitations

Files named `*-sm120-custom-mma.*` from qualification are diagnostic provenance from the superseded custom-policy build, not current qualification evidence. Accepted logs and artifact hashes use the unsuffixed filenames.

This host selects the AVX512 CPU backend. OpenSSL was not found, so the CLIs/server have no HTTPS support. Optional ccache/NCCL were absent; VS environment setup reported missing `vswhere.exe` but initialized MSVC 19.51 successfully. Upstream compiler warnings remain in the build log; they were not suppressed or patched as part of this refresh.

Upstream's UI provisioning could not download the `b10840` bundle and fell back to `latest`, whose embedded `build.json` identifies `b10839`. The qualification download had SHA-256 `f2bff086773a9de542987e75b95210e631a959876daf60adfa78797cb45f9099`. Native source remains pinned, but that UI fallback is not source-pinned. The deployed immutable bundle retains the qualified CLI binaries; the temporary UI download is not retained.

## Vision-FA qualification and activation: 2026-09-07

The qualified core fingerprint is `96ab893cb765c64954a3664973732d67e840c5ce0708bb26db44506f4ae363fa`. The selected bundle is `C:\dev\heirloom-run\runtimes\llama-cuda\73ab7599b-96ab893cb765`; its manifest records native and CUDA dependency hashes. The bundle loads its own cuBLAS dependencies without relying on toolkit PATH entries.

An 80-image native lifecycle replay used the approved album prompt and crossed the model release/reacquire boundary at item 75. Across 1,893 per-process Windows samples, dedicated GPU memory peaked at 20,694.1 MiB. Shared usage stayed within a fixed 240 MiB allocation without the control's multi-GiB growth. All 80 captions completed; the first 30 retained the exact approved image/prompt/grounding inputs and passed the mechanical format gates.

The user reviewed the 30-photo vision-FA comparison and approved caption quality and the paused rollout. No numerical factual-error score or byte-identical output claim is made. A top-k=1 diagnostic on a portrait and two square scans still produced wording differences; image encoding on the squares fell from about 34 seconds to 1.2 seconds. Production keeps its original sampling settings.

The indexer uses `caption@grounded-6` and initially launched with `--paused`, without acquiring a GPU job. A separate default-runtime probe from the deployed application resolved this bundle and its exact fingerprint. Settings and deployment provenance remain under `C:\dev\heirloom-run\rollback\caption-20260907`; old indexer executables and temporary native build trees are not retained.

## No-vision-FA control provenance

The parent's later production-lifecycle replay of the no-vision-FA candidate observed about 7.5 GiB of WDDM shared GPU memory on 2048-square scans before the 75-item recycle. This prompted the separately approved vision-FA comparison; it is not evidence that the new variant resolves the memory behavior.

The historical control's `artifact-sha256.csv` and `source-residual.patch` are retained in `C:\Users\media\.copilot\session-state\037914b8-45f0-4b10-8148-6916f1895bc0\files\native-merge-cleanup-20260908\no-vision-fa`. The manifest retains its original paths as provenance, not as locations of retained binaries. This control did not pass the deployment memory gate.

The accepted build's artifact hashes, configure/build logs, CUDA architecture report and shim export report are retained in the parent `native-merge-cleanup-20260908` directory. The deployed binaries are in the explicit immutable bundle above; future builds require their own qualification and bundle selection.

## Historical no-vision-FA control evidence: 2026-09-07

The preserved no-vision-FA candidate has runtime fingerprint `eefb02fd1e4d8fe61a161e577b529043e1a9fc7f3033331fa92845f69a13dcd8`. The production control has fingerprint `2b8abb44e7d0e375216e1fa5fe2c510be929386ef58972575e1d24c692b1d1c8`. Each historical application probe loaded its complete seven-module core runtime from the explicitly selected directory. These results do not qualify the new vision-FA variant.

| Surface | Observation |
|---|---|
| CUDA backend correctness | 6137/6137 cases passed for `NORM,RMS_NORM,GROUP_NORM,SOFT_MAX,MUL_MAT,MUL_MAT_ID,ROPE,FLASH_ATTN_EXT,CPY`. |
| Existing .NET callers | The text probe returned `READY`; the public landscape probe described the river, forest, and mountains. |
| GLM-OCR | The existing raw OCR path reproduced all three lines of a generated known-text fixture exactly, matching the control. |
| Additional ABI coverage | All 12 exports remain present; pre-cancelled generation returned zero bytes; both embedding exports returned finite 5120-dimensional pooled states. Three public text/image pairs ranked their matching image first on both runtimes. |
| Frozen caption corpus | Both runs completed 30/30. All paired hashes, prompts, grounding keys, and detection counts matched. Both had 30/30 heading-compliant captions and zero failures, scaffolding leaks, repetition, grounding echoes, or forbidden absence claims. |
| Observed caption latency | Median 5673.5 ms for the candidate versus 5423.5 ms for the control, a ratio of 1.046, within the registered 1.25 limit. |

These were single caption passes, not a controlled speedup study. The control overlapped a CPU build, and baseline device memory differed. Whole-device peaks were 31983 MiB for the candidate and 31767 MiB for the control; the harness does not reproduce the daemon's memory lifecycle, so these readings do not establish a deployment memory gate.

**Factual caption equivalence remains unestablished.** Mechanical output checks are not a substitute for reviewing invented details, relationships, and name assignments against the fitted images. No new factual-error score or production promotion is claimed.

Private caption rows and images remain outside the repository at `C:\Users\media\.copilot\session-state\037914b8-45f0-4b10-8148-6916f1895bc0\files\llama-refresh-20260907-qualification`. The control and candidate labels are `fidelity-control-20260907` and `upstream-20260907`. That directory also holds native/public probe logs and the generated OCR fixture; the auxiliary ABI probe is in its parent directory.

## Qualification boundary

Future activation requires factual caption review, a same-lifecycle memory observation, and a separate deployment decision. Preserve decoder/OCR/embedding policies, sampling, resolution, weights, and lifecycle profile unless explicitly qualifying another change. Keep prior bundles intact for rollback and preserve the daemon's operator-selected pause state.

For a worktree build probe, put that checkout's `build\bin` and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64` before other runtime directories on PATH. CUDA 13.3's vendor DLLs, including `cublas64_13.dll` and `cublasLt64_13.dll`, live in `bin\x64`, not the toolkit's top-level `bin`. The deployed immutable bundle includes those cuBLAS DLLs; the installed MSVC/OpenMP runtime remains a host prerequisite.

Historical no-FA results do not qualify future runtime variants. The activation above is specific to the recorded fingerprint, approved prompt, and measured lifecycle.
