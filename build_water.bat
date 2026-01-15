@echo off
setlocal

echo ========================================
echo    CUDA WATER SIMULATION - Build Script
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
echo Building water_simulation.cu...
echo.

nvcc water_simulation.cu -o water_simulation.exe -O3 -arch=sm_52 -Wno-deprecated-gpu-targets ^
    -I"C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\include" ^
    -L"C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\VisualC\x64\Release" ^
    -lSDL3

if %errorlevel% neq 0 (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Make sure you have:
    echo 1. NVIDIA CUDA Toolkit installed
    echo 2. Visual Studio with C++ compiler
    echo 3. SDL3 library installed at:
    echo    C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\
    echo.
    echo If SDL3 is in a different location, edit this script.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build successful! 
echo ========================================
echo.
echo Starting Water Simulation...
echo.
echo Controls:
echo   ESC       - Exit
echo   SPACE     - Pause/Resume
echo   R         - Reset simulation
echo   1/2/3     - Switch scenario
echo   +/-       - Zoom in/out
echo   Arrows    - Pan camera
echo   Left Click - Add water particles
echo.

water_simulation.exe

pause
