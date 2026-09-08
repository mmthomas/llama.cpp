# Heirloom native runtime refresh candidate

This private-fork candidate is pinned to reviewed upstream commit `73ab7599b553c03f6f5d2db24a18ad76f2eb36a3` from `https://github.com/ggml-org/llama.cpp.git`, not to a moving `master`.

**The vision-FA variant is deployed as an immutable bundle; the daemon is paused.** The source worktree is `C:\dev\llama-refresh-20260907`, branch `refresh-20260907`. Building this worktree does not mutate deployed binaries.

## Patch boundary

The starting fork delta is the net change from upstream tag `v0.3.0` to `refresh-v0.3.0` (`7c417ff12eb55bad737df51676e9cf093f8513fe`), not a merge of old fork history or the uncommitted `shim-consistency` refactor.

The permanent application adapter consists of `heirloom\heirloom_llama.cpp`, `heirloom\heirloom_llama.h`, `heirloom\CMakeLists.txt`, the root `CMakeLists.txt` inclusion, and the two `heirloom-*-cuda.bat` scripts. Its stable C ABI retains `hl_backend_init`, `hl_backend_free`, `hl_load`, `hl_free`, `hl_n_params`, `hl_desc`, `hl_complete`, `hl_caption`, `hl_complete_image`, `hl_ocr`, `hl_embed_image`, and `hl_embed_text`, including exports used outside .NET.

The bounded API adaptation passes `mtmd_helper_init_opt_default()` to upstream's image-loading helpers. Sampling, cooperative cancellation, log filtering, prompt prefill chunking, and embedding behavior are carried forward without redesign. The approved vision-FA variant enables only the mtmd vision encoder in `hl_complete_image` and `hl_caption`; their decoder flash attention remains OFF. OCR keeps vision flash attention ON, decoder OFF, and greedy decoding for non-positive temperature. Embedding FA settings remain unchanged. No MTP, DFlash2, sampling, resolution, or model/weight change accompanies this variant.

Only three general GLM-OCR patches remain outside the adapter:

- `conversion\qwen3vl.py`: write GLM vision pixel bounds from `min_pixels`/`max_pixels` or `size.shortest_edge`/`size.longest_edge`; fail explicitly when missing.
- `tools\mtmd\clip.cpp`: retain bicubic resize, use `PAD_NONE`, default to 16..12288 image tokens, then read optional model pixel bounds.
- `tools\mtmd\models\glm4v.cpp`: skip post-convolution normalization only when `norm_embd_w` is absent.

These small core patches are removable when upstream supplies equivalent behavior. The pinned upstream already supplies Qwen preprocessing/position interpolation and CUDA reduction fixes; no duplicate local hunks are replayed. Its GDN normalization and CUDA FA barrier fixes motivate this candidate, but do not establish behavioral qualification.

There is no local ggml/CUDA backend policy delta. The build requests `CMAKE_CUDA_ARCHITECTURES=120`; unmodified upstream resolves it to `compute_120a`/`sm_120a` for architecture-specific Blackwell instructions. This matches the actual generated CUDA flags in both production and the older v0.3.0 candidate, despite their caches also recording `120`. Native Blackwell MMA support and dispatch remain upstream defaults.

## Windows build

Use the existing VS 2026 BuildTools environment at `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat`, portable CMake at `C:\dev\tools\cmake\bin`, Ninja at `C:\dev\tools\ninja`, and CUDA 13.3 at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`.

From PowerShell:

```powershell
New-Item -ItemType Directory -Force C:\dev\llama-refresh-20260907\build | Out-Null
& $env:ComSpec /d /c 'call "C:\dev\llama-refresh-20260907\heirloom-configure-cuda.bat" > "C:\dev\llama-refresh-20260907\build\configure.log" 2>&1'
if ($LASTEXITCODE -ne 0) { throw "Candidate configuration failed; see build\configure.log" }
& $env:ComSpec /d /c 'call "C:\dev\llama-refresh-20260907\heirloom-build-cuda.bat" > "C:\dev\llama-refresh-20260907\build\build.log" 2>&1'
if ($LASTEXITCODE -ne 0) { throw "Candidate build failed; see build\build.log" }
```

Each script initializes `vcvars64.bat` and runs CMake in the same `cmd.exe` process. Configuration uses Ninja, Release, `BUILD_SHARED_LIBS=ON`, `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120` (resolved by upstream to `120a`), and explicit CUDA 13.3 compiler/toolkit paths. `LLAMA_BUILD_COMMON`, `LLAMA_BUILD_TOOLS`, `LLAMA_BUILD_TESTS`, `LLAMA_BUILD_SERVER`, `LLAMA_SUBPROCESS`, and `MTMD_VIDEO` are explicitly ON. Video support is preserved like-for-like; upstream requires `ffmpeg`/`ffprobe` on PATH for relevant media inputs.

`LLAMA_CURL=OFF` is retained as requested, although this upstream revision deprecates that option and uses cpp-httplib instead. It is not a network-disable switch.

The build command is `cmake --build <candidate>\build --config Release --target heirloom_llama llama-cli llama-mtmd-cli llama-server test-backend-ops -j8`. It builds the shim, llama, mtmd, ggml-base, ggml, CPU and CUDA DLLs plus the four requested executables in `build\bin`. The current CLIs also require `llama-common.dll`, `llama-cli-impl.dll`, and `llama-server-impl.dll` from that directory. It does not run those executables or install/deploy anything. Keep the candidate's DLL set together; do not mix it with production DLLs.

### Recorded build limitations

Files named `*-sm120-custom-mma.*` under `build` are diagnostic provenance from the superseded custom-policy build, not current qualification evidence. Current logs and artifact hashes use the unsuffixed filenames.

This host selects the AVX512 CPU backend. OpenSSL was not found, so the CLIs/server have no HTTPS support. Optional ccache/NCCL were absent; VS environment setup reported missing `vswhere.exe` but initialized MSVC 19.51 successfully. Upstream compiler warnings remain in the build log; they were not suppressed or patched as part of this refresh.

Upstream's UI provisioning could not download the `b10840` bundle and fell back to `latest`, whose embedded `build.json` identifies `b10839`. The downloaded `build\tools\ui\dist.tar.gz` has SHA-256 `f2bff086773a9de542987e75b95210e631a959876daf60adfa78797cb45f9099`. Native source remains pinned, but that UI fallback is not source-pinned. Preserve the downloaded bundle and `build\artifact-sha256.csv` for this candidate's provenance.

## Vision-FA qualification and activation: 2026-09-07

The qualified core fingerprint is `96ab893cb765c64954a3664973732d67e840c5ce0708bb26db44506f4ae363fa`. The selected bundle is `C:\dev\heirloom-run\runtimes\llama-cuda\73ab7599b-96ab893cb765`; its manifest records native and CUDA dependency hashes. The bundle loads its own cuBLAS dependencies without relying on toolkit PATH entries.

An 80-image native lifecycle replay used the approved album prompt and crossed the model release/reacquire boundary at item 75. Across 1,893 per-process Windows samples, dedicated GPU memory peaked at 20,694.1 MiB. Shared usage stayed within a fixed 240 MiB allocation without the control's multi-GiB growth. All 80 captions completed; the first 30 retained the exact approved image/prompt/grounding inputs and passed the mechanical format gates.

The user reviewed the 30-photo vision-FA comparison and approved caption quality and the paused rollout. No numerical factual-error score or byte-identical output claim is made. A top-k=1 diagnostic on a portrait and two square scans still produced wording differences; image encoding on the squares fell from about 34 seconds to 1.2 seconds. Production keeps its original sampling settings.

The indexer uses `caption@grounded-6` and launched with `--paused`, without acquiring a GPU job. A separate default-runtime probe from the deployed application resolved this bundle and its exact fingerprint. The old executable/settings snapshot is under `C:\dev\heirloom-run\rollback\caption-20260907`; the old runtime at `C:\dev\llama-production\build\bin` remains intact.

## Retained no-vision-FA control

The parent's later production-lifecycle replay of the no-vision-FA candidate observed about 7.5 GiB of WDDM shared GPU memory on 2048-square scans before the 75-item recycle. This prompted the separately approved vision-FA comparison; it is not evidence that the new variant resolves the memory behavior.

Before changing the two encoder settings, all candidate DLLs/EXEs and their original `artifact-sha256.csv` were preserved byte-for-byte in `build\qualified-no-vision-fa`. That directory also contains `source-residual.patch` for the historical control. The copied manifest retains its original paths; use each filename under the snapshot directory when checking its saved binaries. The directory name does not imply that the control passed the deployment memory or factual-caption gates.

The source worktree's binaries and hashes are in `build\bin` and `build\artifact-sha256.csv`. The deployed copy is the explicit immutable bundle above; future builds require their own qualification and bundle selection.

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

For the parent probe process, put this candidate's `build\bin` and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64` before other runtime directories on PATH. CUDA 13.3's vendor DLLs, including `cublas64_13.dll` and `cublasLt64_13.dll`, live in `bin\x64`, not the toolkit's top-level `bin`. They and the installed MSVC/OpenMP runtime are external prerequisites, not copied deployment files.

Historical no-FA results do not qualify future runtime variants. The activation above is specific to the recorded fingerprint, approved prompt, and measured lifecycle.
