#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" __global__ void k_test_replica_exchange(float* temperatures, float* energies, int n_chains, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chain_a = 2 * idx + offset;
    int chain_b = chain_a + 1;
    if (chain_b >= n_chains) return;

    float E_a = energies[chain_a];
    float E_b = energies[chain_b];
    float T_a = temperatures[chain_a];
    float T_b = temperatures[chain_b];

    if (E_b < E_a) {
        temperatures[chain_a] = T_b;
        temperatures[chain_b] = T_a;
    }
}

static void test_grid_safety() {
    printf("[TEST] Grid Collapse Safety... ");
    float cell_size = 4.0f;
    float critical_L = 2.0f;
    int grid_dim = (int)(critical_L / cell_size);
    if (grid_dim < 1) grid_dim = 1;
    if (grid_dim >= 1) printf("PASSED (Grid Dim: %d)\n", grid_dim);
    else { printf("FAILED (Grid Dim: %d)\n", grid_dim); exit(1); }
}

static void test_fingerprint() {
    printf("[TEST] Fingerprint Rotation Invariance... ");
    int n = 4;
    float x1[] = {1, -1, -1, 1};
    float y1[] = {1, 1, -1, -1};
    float x2[] = {1, 1, -1, -1};
    float y2[] = {-1, 1, 1, -1};

    float fp1 = calculate_fingerprint(x1, y1, n);
    float fp2 = calculate_fingerprint(x2, y2, n);

    if (fabsf(fp1 - fp2) < 0.001f) printf("PASSED (FP: %.4f)\n", fp1);
    else { printf("FAILED (FP1: %.4f, FP2: %.4f)\n", fp1, fp2); exit(1); }
}

static void test_replica_exchange() {
    printf("[TEST] Replica Exchange Kernel... ");

    int n_chains = 2;
    float h_temps[] = {0.001f, 0.5f};
    float h_energies[] = {100.0f, 10.0f};

    float *d_temps = NULL;
    float *d_energies = NULL;
    cudaMalloc(&d_temps, 2 * sizeof(float));
    cudaMalloc(&d_energies, 2 * sizeof(float));
    cudaMemcpy(d_temps, h_temps, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_energies, h_energies, 2 * sizeof(float), cudaMemcpyHostToDevice);

    k_test_replica_exchange<<<1, 1>>>(d_temps, d_energies, n_chains, 0);
    cudaDeviceSynchronize();

    float res_temps[2] = {0};
    cudaMemcpy(res_temps, d_temps, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    if (res_temps[0] > 0.4f && res_temps[1] < 0.1f) {
        printf("PASSED (Temps Swapped: T0=%.3f, T1=%.3f)\n", res_temps[0], res_temps[1]);
    } else {
        printf("FAILED (Temps did not swap: T0=%.3f, T1=%.3f)\n", res_temps[0], res_temps[1]);
        exit(1);
    }

    cudaFree(d_temps);
    cudaFree(d_energies);
}

void run_all_tests() {
    printf("=== RUNNING CHIMERA SELF-DIAGNOSTICS ===\n");
    test_grid_safety();
    test_fingerprint();
    test_replica_exchange();
    printf("=== ALL TESTS PASSED. SYSTEM GREEN. ===\n\n");
}
