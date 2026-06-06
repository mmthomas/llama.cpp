@echo off
REM Heirloom build recipe for our llama.cpp fork — pinned, reproducible (→ ADR amending 021).
REM Toolchain: VS 2026 BuildTools (MSVC 14.51) + CUDA 13.3 (accepts MSVC <1960) + Ninja + portable CMake.
REM sm_120 = Blackwell (RTX 5090). NOTE: must use CUDA 13.3 (not the PATH-default 13.0, which rejects VS2026).
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=C:\dev\tools\cmake\bin;C:\dev\tools\ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
cmake -S "%~dp0." -B "%~dp0build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120 ^
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin/nvcc.exe" ^
  -DLLAMA_CURL=OFF
