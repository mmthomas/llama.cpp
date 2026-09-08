@echo off
REM Isolated candidate: VS 2026 BuildTools + CUDA 13.3 + Ninja + portable CMake.
REM Request 120; upstream resolves this to 120a, matching existing production builds.
REM Preserve shared libraries and mtmd video support.
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=C:\dev\tools\cmake\bin;C:\dev\tools\ninja;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%"
cmake -S "%~dp0." -B "%~dp0build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120 ^
  "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" ^
  "-DCUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3" ^
  -DLLAMA_BUILD_COMMON=ON ^
  -DLLAMA_BUILD_TOOLS=ON ^
  -DLLAMA_BUILD_TESTS=ON ^
  -DLLAMA_BUILD_SERVER=ON ^
  -DLLAMA_SUBPROCESS=ON ^
  -DMTMD_VIDEO=ON ^
  -DLLAMA_CURL=OFF
exit /b %errorlevel%
