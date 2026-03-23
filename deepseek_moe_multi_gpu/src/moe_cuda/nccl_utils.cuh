#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

// NCCL check macro
#define NCCL_CHECK(call) do { \
    ncclResult_t r = (call); \
    if (r != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d -> %s\n", \
                __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d -> %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// Holds per-rank NCCL state
struct NcclState {
    ncclComm_t comm;
    int        rank;
    int        world_size;
    cudaStream_t stream;
};

inline NcclState nccl_init_single_node(int rank, int world_size) {
    NcclState s;
    s.rank       = rank;
    s.world_size = world_size;
    CUDA_CHECK(cudaSetDevice(rank));
    CUDA_CHECK(cudaStreamCreate(&s.stream));

    // For single-node we use ncclCommInitAll which is simpler
    // (no MPI needed for single machine)
    ncclComm_t* comms = (ncclComm_t*)malloc(world_size * sizeof(ncclComm_t));
    int* devs = (int*)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++) devs[i] = i;
    NCCL_CHECK(ncclCommInitAll(comms, world_size, devs));
    s.comm = comms[rank];
    free(devs);
    free(comms);
    return s;
}

// Broadcast a float buffer from rank 0 to all ranks
inline void nccl_broadcast_weights(float* d_buf, size_t count, NcclState& s) {
    NCCL_CHECK(ncclBcast(d_buf, count, ncclFloat, 0, s.comm, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

// All-gather: each rank contributes send_count floats,
// recv_buf must be [world_size * send_count]
inline void nccl_all_gather(
    const float* d_send, float* d_recv,
    size_t send_count, NcclState& s)
{
    NCCL_CHECK(ncclAllGather(d_send, d_recv, send_count, ncclFloat, s.comm, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

// All-reduce sum
inline void nccl_all_reduce_sum(float* d_buf, size_t count, NcclState& s) {
    NCCL_CHECK(ncclAllReduce(d_buf, d_buf, count, ncclFloat, ncclSum, s.comm, s.stream));
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
}