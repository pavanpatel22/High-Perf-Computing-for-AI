#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
    } else {
        printf("Found %d CUDA device(s):\n", deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("Device %d: %s\n", i, prop.name);
        }
    }
    
    return 0;
}