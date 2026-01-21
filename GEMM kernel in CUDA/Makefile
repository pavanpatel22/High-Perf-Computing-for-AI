# Windows CUDA Compilation
NVCC = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/nvcc.exe"
TARGET = gemm_test.exe
SRC = main.cu gemm.cu
HEADERS = gemm.h

# Default target
all: $(TARGET)

# Main executable
$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) -arch=sm_70 -O2 -o $(TARGET) $(SRC)

# Clean
clean:
	del $(TARGET) *.o 2>nul || exit 0

# Run
run: $(TARGET)
	.\$(TARGET)

# Debug build
debug:
	$(NVCC) -arch=sm_70 -G -g -o $(TARGET)_debug.exe $(SRC)

.PHONY: all clean run debug