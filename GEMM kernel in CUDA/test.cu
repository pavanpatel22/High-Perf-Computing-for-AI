#include <cuda_runtime.h>
__global__ void kernel() {}
int main() { kernel<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }