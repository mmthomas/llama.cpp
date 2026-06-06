@echo off
REM Heirloom build step for our llama.cpp fork (run heirloom-configure-cuda.bat first).
REM Same pinned toolchain env (VS2026 + CUDA 13.3 + Ninja). Builds the CLIs needed for the bring-up spike;
REM libllama is built as their dependency. (P/Invoke shared-lib build comes later.)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=C:\dev\tools\cmake\bin;C:\dev\tools\ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
cmake --build "%~dp0build" --target llama-cli llama-mtmd-cli -j
