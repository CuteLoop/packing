#include "gpu_data.h"
#include <stdio.h>

__global__ void k_init_rng(curandState* state, unsigned long long seed, int n_chains) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n_chains) return;
    curand_init(seed, id, 0, &state[id]);
}

extern "C" void gpu_init_rng(DeviceSoA* soa, int n_chains, unsigned long long seed) {
    int blockSize = 128;
    int numBlocks = (n_chains + blockSize - 1) / blockSize;

    printf("Initializing RNG for %d chains (Seed: %llu)...\n", n_chains, seed);
    k_init_rng<<<numBlocks, blockSize>>>(soa->rng, seed, n_chains);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("RNG Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
