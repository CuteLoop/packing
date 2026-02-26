#include "chimera_system.h"
#include <stdio.h>
#include <cuda_runtime.h>

int chimera_init(ChimeraContext* ctx, int active_chains, int active_polys) {
    if (!ctx) return 0;
    if (active_chains < 1 || active_chains > MAX_CHAINS) {
        fprintf(stderr, "FATAL: active_chains=%d exceeds MAX_CHAINS=%d\n", active_chains, MAX_CHAINS);
        return 0;
    }
    if (active_polys < 1 || active_polys > MAX_POLYS) {
        fprintf(stderr, "FATAL: active_polys=%d exceeds MAX_POLYS=%d\n", active_polys, MAX_POLYS);
        return 0;
    }

    ctx->active_chains = active_chains;
    ctx->active_polys = active_polys;

    // Arena allocation: allocate maximum capacity once.
    gpu_alloc_soa(&ctx->soa, MAX_CHAINS, MAX_POLYS);

    size_t big_size = (size_t)MAX_CHAINS * MAX_POLYS * sizeof(float);
    size_t meta_size = (size_t)MAX_CHAINS * sizeof(float);
    size_t rng_size = (size_t)MAX_CHAINS * sizeof(curandState);
    size_t accept_size = (size_t)MAX_CHAINS * sizeof(int);
    ctx->total_vram_bytes = big_size * 3 + meta_size * 3 + rng_size + accept_size;

    printf("[Chimera] Arena init: chains=%d/%d polys=%d/%d (reserved %.2f MB VRAM)\n",
           active_chains, MAX_CHAINS, active_polys, MAX_POLYS,
           (double)ctx->total_vram_bytes / (1024.0 * 1024.0));

    return 1;
}

void chimera_free(ChimeraContext* ctx) {
    if (!ctx) return;
    gpu_free_soa(&ctx->soa);
    ctx->active_chains = 0;
    ctx->active_polys = 0;
    ctx->total_vram_bytes = 0;
}
