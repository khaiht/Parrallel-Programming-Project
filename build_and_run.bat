@echo off
setlocal

echo ========================================
echo Setting up Visual Studio Environment
echo ========================================

REM Try to find and run vcvars64.bat
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else (
    echo Warning: Could not find Visual Studio. Trying without vcvars...
)

echo.
echo ========================================
echo Building kernel_SDL.cu
echo ========================================

nvcc kernel_SDL.cu -o kernel_sdl.exe -O3 -arch=sm_52 -Wno-deprecated-gpu-targets ^
    -I"C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\include" ^
    -L"C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\VisualC\x64\Release" ^
    -lSDL3

if %errorlevel% neq 0 (
    echo.
    echo BUILD FAILED!
    echo.
    echo Make sure you have:
    echo 1. NVIDIA CUDA Toolkit installed
    echo 2. Visual Studio with C++ compiler
    echo 3. SDL3 library installed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build successful! Running simulation...
echo ========================================
echo.

kernel_sdl.exe

pause
