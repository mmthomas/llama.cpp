@echo off
REM Run heirloom-configure-cuda.bat first. Build only; do not run qualification here.
REM Dependencies include the llama, mtmd, ggml, CPU, and CUDA shared libraries.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=C:\dev\tools\cmake\bin;C:\dev\tools\ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
cmake --build "%~dp0build" --config Release --target heirloom_llama llama-cli llama-mtmd-cli llama-server test-backend-ops -j8
exit /b %errorlevel%
