#include "gpu_data.h"
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

void gpu_alloc_soa(DeviceSoA* soa, int n_chains, int n_polys) {
    size_t big_size = (size_t)n_chains * n_polys * sizeof(float);
    size_t meta_size = (size_t)n_chains * sizeof(float);

    printf("Allocating GPU Memory: %.2f MB per state array.\n", big_size / (1024.0*1024.0));

    CUDA_CHECK(cudaMalloc((void**)&soa->pos_x, big_size));
    CUDA_CHECK(cudaMalloc((void**)&soa->pos_y, big_size));
    CUDA_CHECK(cudaMalloc((void**)&soa->angle, big_size));

    CUDA_CHECK(cudaMalloc((void**)&soa->energy, meta_size));
    CUDA_CHECK(cudaMalloc((void**)&soa->chain_energies, meta_size));
    CUDA_CHECK(cudaMalloc((void**)&soa->temperature, meta_size));
    CUDA_CHECK(cudaMalloc((void**)&soa->accept_count, n_chains * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&soa->rng, n_chains * sizeof(curandState)));

    CUDA_CHECK(cudaMemset(soa->accept_count, 0, n_chains * sizeof(int)));
}

void gpu_free_soa(DeviceSoA* soa) {
    cudaFree(soa->pos_x);
    cudaFree(soa->pos_y);
    cudaFree(soa->angle);
    cudaFree(soa->energy);
    cudaFree(soa->chain_energies);
    cudaFree(soa->temperature);
    cudaFree(soa->accept_count);
    cudaFree(soa->rng);
}

void gpu_upload_state(DeviceSoA* dev_soa,
                      const float* h_x, const float* h_y, const float* h_ang,
                      int n_chains, int n_polys) {
    size_t total_floats = (size_t)n_chains * n_polys;
    float* temp_buf = (float*)malloc(total_floats * sizeof(float));
    if (!temp_buf) {
        printf("Host alloc failed\n");
        exit(1);
    }

    for(int c=0; c<n_chains; c++) {
        for(int p=0; p<n_polys; p++) {
            int in_idx  = c * n_polys + p;
            int out_idx = SOA_IDX(p, c, n_chains);
            temp_buf[out_idx] = h_x[in_idx];
        }
    }
    CUDA_CHECK(cudaMemcpy(dev_soa->pos_x, temp_buf, total_floats*sizeof(float), cudaMemcpyHostToDevice));

    for(int c=0; c<n_chains; c++) {
        for(int p=0; p<n_polys; p++) {
            int in_idx  = c * n_polys + p;
            int out_idx = SOA_IDX(p, c, n_chains);
            temp_buf[out_idx] = h_y[in_idx];
        }
    }
    CUDA_CHECK(cudaMemcpy(dev_soa->pos_y, temp_buf, total_floats*sizeof(float), cudaMemcpyHostToDevice));

    for(int c=0; c<n_chains; c++) {
        for(int p=0; p<n_polys; p++) {
            int in_idx  = c * n_polys + p;
            int out_idx = SOA_IDX(p, c, n_chains);
            temp_buf[out_idx] = h_ang[in_idx];
        }
    }
    CUDA_CHECK(cudaMemcpy(dev_soa->angle, temp_buf, total_floats*sizeof(float), cudaMemcpyHostToDevice));

    free(temp_buf);
}

void gpu_download_state(DeviceSoA* dev_soa,
                        float* h_x, float* h_y, float* h_ang, float* h_e,
                        int n_chains, int n_polys) {
    size_t total_floats = (size_t)n_chains * n_polys;
    float* temp_buf = (float*)malloc(total_floats * sizeof(float));
    if (!temp_buf) {
        printf("Host alloc failed\n");
        exit(1);
    }

    CUDA_CHECK(cudaMemcpy(temp_buf, dev_soa->pos_x, total_floats*sizeof(float), cudaMemcpyDeviceToHost));
    for(int c=0; c<n_chains; c++) {
        for(int p=0; p<n_polys; p++) {
            h_x[c*n_polys + p] = temp_buf[SOA_IDX(p, c, n_chains)];
        }
    }

    CUDA_CHECK(cudaMemcpy(temp_buf, dev_soa->pos_y, total_floats*sizeof(float), cudaMemcpyDeviceToHost));
    for(int c=0; c<n_chains; c++) {
        for(int p=0; p<n_polys; p++) {
            h_y[c*n_polys + p] = temp_buf[SOA_IDX(p, c, n_chains)];
        }
    }

    CUDA_CHECK(cudaMemcpy(temp_buf, dev_soa->angle, total_floats*sizeof(float), cudaMemcpyDeviceToHost));
    for(int c=0; c<n_chains; c++) {
        for(int p=0; p<n_polys; p++) {
            h_ang[c*n_polys + p] = temp_buf[SOA_IDX(p, c, n_chains)];
        }
    }

    CUDA_CHECK(cudaMemcpy(h_e, dev_soa->energy, n_chains*sizeof(float), cudaMemcpyDeviceToHost));

    free(temp_buf);
}

extern "C" void gpu_sync_metadata(
    DeviceSoA* dev_soa,
    float* host_energy,
    float* host_temp,
    bool to_gpu
) {
    if (to_gpu) {
        CUDA_CHECK(cudaMemcpy(dev_soa->temperature, host_temp, MAX_CHAINS * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(host_energy, dev_soa->energy, MAX_CHAINS * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_temp, dev_soa->temperature, MAX_CHAINS * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

extern "C" void gpu_sync_rng(DeviceSoA* dev_soa, void* host_rng, int n_chains, bool to_gpu)
{
    size_t bytes = (size_t)n_chains * sizeof(curandState);
    if (to_gpu) {
        CUDA_CHECK(cudaMemcpy(dev_soa->rng, host_rng, bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(host_rng, dev_soa->rng, bytes, cudaMemcpyDeviceToHost));
    }
}

extern "C" void gpu_sync_accept(DeviceSoA* dev_soa, int* host_accept, int n_chains, bool to_gpu)
{
    if (to_gpu) {
        CUDA_CHECK(cudaMemcpy(dev_soa->accept_count, host_accept, n_chains * sizeof(int), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(host_accept, dev_soa->accept_count, n_chains * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

extern "C" void gpu_download_stats(DeviceSoA* dev_soa, float* host_energy, int* host_accept, int n_chains)
{
    CUDA_CHECK(cudaMemcpy(host_energy, dev_soa->energy, n_chains * sizeof(float), cudaMemcpyDeviceToHost));
    if (host_accept) {
        CUDA_CHECK(cudaMemcpy(host_accept, dev_soa->accept_count, n_chains * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

extern "C" void gpu_download_energies(DeviceSoA* dev_soa, float* host_dst, int n_chains)
{
    CUDA_CHECK(cudaMemcpy(host_dst, dev_soa->chain_energies, n_chains * sizeof(float), cudaMemcpyDeviceToHost));
}
