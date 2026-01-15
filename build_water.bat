@echo off
setlocal

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

nvcc water_simulation.cu -o water_simulation.exe -O3 -arch=sm_52 -Wno-deprecated-gpu-targets ^
    -I"C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\include" ^
    -L"C:\SDL2\SDL3-3.2.26\SDL3-3.2.26\VisualC\x64\Release" ^
    -lSDL3

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

water_simulation.exe
