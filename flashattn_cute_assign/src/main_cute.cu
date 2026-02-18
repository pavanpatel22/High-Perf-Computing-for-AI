#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include "flashattn_cuda_cute.h"

#define CUDA_CHECK(call) do {                               \
  cudaError_t err = call;                                   \
  if (err != cudaSuccess) {                                 \
    printf("CUDA error: %s\n", cudaGetErrorString(err));     \
    std::exit(1);                                           \
  }                                                         \
} while (0)

static int parse_dtype(const char* s) {
  if (strcmp(s,"f32")==0) return 0;
  if (strcmp(s,"f16")==0) return 1;
  if (strcmp(s,"bf16")==0) return 2;
  return -1;
}

int main(int argc, char** argv) {
  int N=256, D=64, Br=64, Bc=64, B=1, H=1;
  int dtype=1; // f16
  bool causal=false;

  for (int i=1;i<argc;i++) {
    if (!strcmp(argv[i],"--N")) N=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--D")) D=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--Br")) Br=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--Bc")) Bc=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--B")) B=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--H")) H=atoi(argv[++i]);
    else if (!strcmp(argv[i],"--dtype")) dtype=parse_dtype(argv[++i]);
    else if (!strcmp(argv[i],"--causal")) causal=true;
  }

  printf("CuTe FlashAttn: B=%d H=%d N=%d D=%d Br=%d Bc=%d causal=%d dtype=%d\n",
         B,H,N,D,Br,Bc,(int)causal,dtype);

  int BH = B*H;
  size_t elems = (size_t)BH*N*D;

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);

  std::vector<float> Qf(elems), Kf(elems), Vf(elems);
  for (size_t i=0;i<elems;i++) { Qf[i]=dist(rng); Kf[i]=dist(rng); Vf[i]=dist(rng); }

  void *dQ=nullptr,*dK=nullptr,*dV=nullptr;
  float *dO=nullptr,*dL=nullptr;

  size_t in_bytes = (dtype==0)? elems*sizeof(float) : elems*sizeof(__half); // bf16 omitted for brevity here
  CUDA_CHECK(cudaMalloc(&dQ,in_bytes));
  CUDA_CHECK(cudaMalloc(&dK,in_bytes));
  CUDA_CHECK(cudaMalloc(&dV,in_bytes));
  CUDA_CHECK(cudaMalloc(&dO, elems*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dL, (size_t)BH*N*sizeof(float)));

  if (dtype==0) {
    CUDA_CHECK(cudaMemcpy(dQ,Qf.data(),in_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,Kf.data(),in_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,Vf.data(),in_bytes,cudaMemcpyHostToDevice));
  } else {
    std::vector<__half> Qh(elems),Kh(elems),Vh(elems);
    for (size_t i=0;i<elems;i++) { Qh[i]=__float2half(Qf[i]); Kh[i]=__float2half(Kf[i]); Vh[i]=__float2half(Vf[i]); }
    CUDA_CHECK(cudaMemcpy(dQ,Qh.data(),in_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK,Kh.data(),in_bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,Vh.data(),in_bytes,cudaMemcpyHostToDevice));
  }

  flashattn_forward_cute(dQ,dK,dV,dO,dL,B,H,N,D,Br,Bc,causal,dtype);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(dQ));
  CUDA_CHECK(cudaFree(dK));
  CUDA_CHECK(cudaFree(dV));
  CUDA_CHECK(cudaFree(dO));
  CUDA_CHECK(cudaFree(dL));

  printf("Done.\n");
  return 0;
}
