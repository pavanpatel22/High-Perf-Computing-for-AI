CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

CFLAGS    := -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter
NVCCFLAGS := -O2 -std=c++17

INCLUDES  := -Isrc

all: flashattn

flashattn: src/main.cu src/flashattn_cuda.cu src/flashattn_cpu.c src/naive_attention.c
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
	  src/main.cu src/flashattn_cuda.cu src/flashattn_cpu.c src/naive_attention.c \
	  -o $@

clean:
	rm -f flashattn
