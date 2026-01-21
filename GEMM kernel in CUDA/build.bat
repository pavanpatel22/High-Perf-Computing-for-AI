@echo off
echo Building CUDA GEMM project...
echo.

:: Check if CUDA is installed
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found in PATH!
    echo Please install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

:: Build the project
echo Compiling with nvcc...
nvcc -arch=sm_70 -O2 -o gemm_test.exe main.cu gemm.cu

if %errorlevel% equ 0 (
    echo.
    echo Build successful! Run gemm_test.exe
) else (
    echo.
    echo Build failed!
)

pause