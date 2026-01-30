#ifndef GPU_DATA_H
#define GPU_DATA_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdbool.h>
#include "gpu_interface.h"

#define MAX_CHAINS 5120
#define MAX_POLYS  200

typedef struct DeviceSoA {
    float* pos_x;
    float* pos_y;
    float* angle;

    float* energy;
    float* chain_energies;
    float* temperature;
    int* accept_count;

    curandState* rng;
} DeviceSoA;

#define SOA_IDX(poly_idx, chain_idx, num_chains) ((poly_idx) * (num_chains) + (chain_idx))

// C linkage for host-side callers
#ifdef __cplusplus
extern "C" {
#endif

void gpu_alloc_soa(DeviceSoA* soa, int n_chains, int n_polys);
void gpu_free_soa(DeviceSoA* soa);

void gpu_upload_state(DeviceSoA* dev_soa,
                      const float* host_x,
                      const float* host_y,
                      const float* host_ang,
                      int n_chains,
                      int n_polys);

void gpu_download_state(DeviceSoA* dev_soa,
                        float* host_x,
                        float* host_y,
                        float* host_ang,
                        float* host_energy,
                        int n_chains,
                        int n_polys);

void gpu_sync_metadata(
    DeviceSoA* dev_soa,
    float* host_temp,
    float* host_energy,
    int n_chains
);

void gpu_init_rng(DeviceSoA* soa, int n_chains, unsigned long long seed);

void gpu_sync_rng(DeviceSoA* dev_soa, void* host_rng, int n_chains, bool to_gpu);
void gpu_sync_accept(DeviceSoA* dev_soa, int* host_accept, int n_chains, bool to_gpu);
void gpu_download_stats(DeviceSoA* dev_soa, float* host_energy, int* host_accept, int n_chains);
void gpu_download_energies(DeviceSoA* dev_soa, float* host_dst, int n_chains);
void gpu_download_chain_geometry(DeviceSoA* dev_soa, int chain_idx,
                                 float* h_x, float* h_y, float* h_ang, int n_polys, int n_chains);
void gpu_upload_chain_geometry(DeviceSoA* dev_soa, int chain_idx,
                               const float* h_x, const float* h_y, const float* h_ang, int n_polys, int n_chains);

#ifdef __cplusplus
}
#endif

#endif
