#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

#include "gpu_data.h"
#include "gpu_interface.h"
#include "geometry_bake.h"
#include "convex_decomp.h"
#include "shape_tree.h"
#include "triangulate_earclip.h"

#define N_CHAINS 5120
#define N_POLYS  200
#define SWAP_INTERVAL 50
#define STEPS_PER_LAUNCH 1000
#define CHECKPOINT_SEC 600

static volatile int keep_running = 1;

static void handle_sig(int sig) {
    (void)sig;
    keep_running = 0;
    printf("\n[Host] Interrupt received. Finishing current step...\n");
}

static void save_checkpoint(const char* filename, float box_size, int step, DeviceSoA* soa) {
    char tmp_name[256];
    snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);

    FILE* f = fopen(tmp_name, "wb");
    if (!f) return;

    fwrite(&box_size, sizeof(float), 1, f);
    fwrite(&step, sizeof(int), 1, f);

    float *h_x = (float*)malloc(N_CHAINS * N_POLYS * sizeof(float));
    float *h_y = (float*)malloc(N_CHAINS * N_POLYS * sizeof(float));
    float *h_a = (float*)malloc(N_CHAINS * N_POLYS * sizeof(float));
    float *h_e = (float*)malloc(N_CHAINS * sizeof(float));

    gpu_download_state(soa, h_x, h_y, h_a, h_e, N_CHAINS, N_POLYS);

    fwrite(h_x, sizeof(float), N_CHAINS * N_POLYS, f);
    fwrite(h_y, sizeof(float), N_CHAINS * N_POLYS, f);
    fwrite(h_a, sizeof(float), N_CHAINS * N_POLYS, f);

    free(h_x); free(h_y); free(h_a); free(h_e);
    fclose(f);

    rename(tmp_name, filename);
    printf("[Checkpoint] Saved state at Step %d, L=%.4f\n", step, box_size);
}

static void perform_swaps(float* energies, float* temps) {
    for (int i = 0; i < N_CHAINS - 1; i++) {
        float E1 = energies[i];
        float E2 = energies[i+1];
        float T1 = temps[i];
        float T2 = temps[i+1];

        float beta1 = 1.0f / (T1 + 1e-9f);
        float beta2 = 1.0f / (T2 + 1e-9f);
        float delta = (beta2 - beta1) * (E1 - E2);

        float r = (float)rand() / (float)RAND_MAX;
        if (delta > 0.0f || r < expf(delta)) {
            temps[i] = T2;
            temps[i+1] = T1;
        }
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    signal(SIGINT, handle_sig);
    srand((unsigned)time(NULL));

    printf("=== CHIMERA N=%d SOLVER STARTING ===\n", N_POLYS);

    Poly tree = make_tree_poly_local();
    Triangulation T = triangulate_earclip(&tree);
    ConvexDecomp D = convex_decomp_merge_tris(&tree, &T);

    BakedGeometry baked;
    if (!baked_geometry_build(&D, &baked)) {
        fprintf(stderr, "Failed to bake geometry.\n");
        return 1;
    }

    gpu_upload_geometry_from_baked(&baked);
    printf("[Host] Geometry baked and uploaded.\n");

    DeviceSoA soa;
    gpu_alloc_soa(&soa, N_CHAINS, N_POLYS);
    gpu_init_rng(&soa, N_CHAINS, 12345ULL);

    float current_box = 100.0f;
    int start_step = 0;

    float* h_x = (float*)malloc(N_CHAINS * N_POLYS * sizeof(float));
    float* h_y = (float*)malloc(N_CHAINS * N_POLYS * sizeof(float));
    float* h_a = (float*)malloc(N_CHAINS * N_POLYS * sizeof(float));
    float* h_t = (float*)malloc(N_CHAINS * sizeof(float));

    float T_min = 1e-5f;
    float T_max = 0.5f;
    for(int i=0; i<N_CHAINS; i++) {
        float ratio = (float)i / (float)(N_CHAINS - 1);
        h_t[i] = T_min * powf(T_max / T_min, ratio);

        for(int p=0; p<N_POLYS; p++) {
            h_x[i*N_POLYS + p] = ((float)rand()/RAND_MAX - 0.5f) * current_box;
            h_y[i*N_POLYS + p] = ((float)rand()/RAND_MAX - 0.5f) * current_box;
            h_a[i*N_POLYS + p] = ((float)rand()/RAND_MAX) * 6.28f;
        }
    }

    gpu_upload_state(&soa, h_x, h_y, h_a, N_CHAINS, N_POLYS);
    gpu_sync_metadata(&soa, NULL, h_t, true);

    free(h_x); free(h_y); free(h_a);

    float* host_energies = (float*)malloc(N_CHAINS * sizeof(float));
    time_t last_ckpt = time(NULL);

    int total_launches = start_step;
    while(keep_running) {
        gpu_launch_anneal(&soa, N_CHAINS, N_POLYS, current_box, STEPS_PER_LAUNCH);
        total_launches++;

        if (total_launches % SWAP_INTERVAL == 0) {
            gpu_sync_metadata(&soa, host_energies, h_t, false);

            float min_E = 1e9f;
            for(int i=0; i<N_CHAINS; i++) if(host_energies[i] < min_E) min_E = host_energies[i];

            perform_swaps(host_energies, h_t);
            gpu_sync_metadata(&soa, NULL, h_t, true);

            if (min_E < 1e-4f) {
                current_box *= 0.999f;
                printf("[Time %ds] Success! Shrinking L -> %.4f\n",
                       (int)(total_launches * STEPS_PER_LAUNCH / 10000), current_box);
            } else if (total_launches % 100 == 0) {
                printf("[Status] Stuck at L=%.4f. Min E=%.4f\n", current_box, min_E);
            }
        }

        if (time(NULL) - last_ckpt > CHECKPOINT_SEC) {
            save_checkpoint("run_checkpoint.bin", current_box, total_launches, &soa);
            last_ckpt = time(NULL);
        }
    }

    gpu_free_soa(&soa);
    gpu_free_geometry();
    baked_geometry_free(&baked);
    free_convex_decomp(&D);
    free_triangulation(&T);
    free(tree.v);

    free(host_energies);
    free(h_t);
    printf("[Host] Shutdown complete.\n");
    return 0;
}
