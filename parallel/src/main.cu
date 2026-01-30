#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cuda_runtime.h>
#include <signal.h>

#include "gpu_data.h"
#include "gpu_interface.h"

extern "C" {
#include "geometry_bake.h"
#include "convex_decomp.h"
#include "shape_tree.h"
#include "triangulate_earclip.h"
}

#define N_CHAINS 5120
#define N_POLYS  200
#define SWAP_INTERVAL 50
#define STEPS_PER_LAUNCH 1000
#define CHECKPOINT_SEC 600
#define LOG_SEC 5

static volatile int keep_running = 1;

static void handle_sig(int sig) {
    (void)sig;
    keep_running = 0;
    printf("\n[Host] Interrupt received. Finishing current step...\n");
}

typedef struct {
    float box_size;
    int step;
    int n_chains;
    int n_polys;
    int steps_per_launch;
    float step_dx;
    float step_dy;
    float step_da;
} CheckpointHeader;

static int fsync_file(FILE* f) {
    int fd = fileno(f);
    if (fd < 0) return -1;
    return fsync(fd);
}

static void save_checkpoint(const char* filename, float box_size, int step, DeviceSoA* soa,
                            const float* temps, const int* accept, const void* rng_state,
                            int n_chains, int n_polys, int steps_per_launch) {
    char tmp_name[256];
    snprintf(tmp_name, sizeof(tmp_name), "%s.tmp", filename);

    FILE* f = fopen(tmp_name, "wb");
    if (!f) return;

    CheckpointHeader hdr = {
        .box_size = box_size,
        .step = step,
        .n_chains = n_chains,
        .n_polys = n_polys,
        .steps_per_launch = steps_per_launch,
        .step_dx = 1.0f,
        .step_dy = 1.0f,
        .step_da = 0.5f
    };
    fwrite(&hdr, sizeof(hdr), 1, f);

    float *h_x = (float*)malloc((size_t)n_chains * n_polys * sizeof(float));
    float *h_y = (float*)malloc((size_t)n_chains * n_polys * sizeof(float));
    float *h_a = (float*)malloc((size_t)n_chains * n_polys * sizeof(float));
    float *h_e = (float*)malloc((size_t)n_chains * sizeof(float));

    gpu_download_state(soa, h_x, h_y, h_a, h_e, n_chains, n_polys);

    fwrite(h_x, sizeof(float), (size_t)n_chains * n_polys, f);
    fwrite(h_y, sizeof(float), (size_t)n_chains * n_polys, f);
    fwrite(h_a, sizeof(float), (size_t)n_chains * n_polys, f);
    fwrite(temps, sizeof(float), (size_t)n_chains, f);
    fwrite(accept, sizeof(int), (size_t)n_chains, f);
    fwrite(rng_state, sizeof(curandState), (size_t)n_chains, f);

    free(h_x); free(h_y); free(h_a); free(h_e);
    fflush(f);
    fsync_file(f);
    fclose(f);

    rename(tmp_name, filename);
    printf("[Checkpoint] Saved state at Step %d, L=%.4f\n", step, box_size);
}

static int load_checkpoint(const char* filename, float* box_size, int* step,
                           float* h_x, float* h_y, float* h_a, float* h_t,
                           int* h_accept, void* h_rng, int n_chains, int n_polys) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;

    CheckpointHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return 0; }
    if (hdr.n_chains != n_chains || hdr.n_polys != n_polys) { fclose(f); return 0; }

    *box_size = hdr.box_size;
    *step = hdr.step;

    size_t nstate = (size_t)n_chains * n_polys;
    if (fread(h_x, sizeof(float), nstate, f) != nstate) { fclose(f); return 0; }
    if (fread(h_y, sizeof(float), nstate, f) != nstate) { fclose(f); return 0; }
    if (fread(h_a, sizeof(float), nstate, f) != nstate) { fclose(f); return 0; }
    if (fread(h_t, sizeof(float), (size_t)n_chains, f) != (size_t)n_chains) { fclose(f); return 0; }
    if (fread(h_accept, sizeof(int), (size_t)n_chains, f) != (size_t)n_chains) { fclose(f); return 0; }
    if (fread(h_rng, sizeof(curandState), (size_t)n_chains, f) != (size_t)n_chains) { fclose(f); return 0; }

    fclose(f);
    return 1;
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

static float calculate_fingerprint(const float* x, const float* y, int n) {
    float cx = 0.0f, cy = 0.0f;
    for (int i = 0; i < n; i++) {
        cx += x[i];
        cy += y[i];
    }
    cx /= (float)n;
    cy /= (float)n;
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        float dx = x[i] - cx;
        float dy = y[i] - cy;
        sum_sq += dx * dx + dy * dy;
    }
    return sum_sq;
}

void run_all_tests() {
    printf("[TEST] Tests disabled in this build.\n");
}

static void run_single_simulation(int n_chains, int n_polys, int n_epochs, double poly_area_base, FILE* csv_log) {
    if (n_chains < 1) n_chains = 1;
    if (n_chains > N_CHAINS) n_chains = N_CHAINS;
    if (n_polys < 1) n_polys = 1;
    if (n_polys > N_POLYS) n_polys = N_POLYS;

    float total_area = (float)(poly_area_base * (double)n_polys);
    float min_L_density = (total_area > 0.0f) ? sqrtf(total_area) : 0.0f;
    float min_L_geom = 4.0f;
    float min_L = min_L_density > min_L_geom ? min_L_density : min_L_geom;
    min_L *= 1.01f;
    float current_box = min_L * 1.5f;

    printf(">> [Setup N=%3d] Min L: %.2f | Start L: %.2f\n", n_polys, min_L, current_box);

    DeviceSoA soa;
    gpu_alloc_soa(&soa, n_chains, n_polys);
    gpu_audit_memory(&soa, n_chains, n_polys);
    gpu_init_rng(&soa, n_chains, (unsigned long long)time(NULL));

    float* h_x = (float*)malloc((size_t)n_chains * n_polys * sizeof(float));
    float* h_y = (float*)malloc((size_t)n_chains * n_polys * sizeof(float));
    float* h_a = (float*)malloc((size_t)n_chains * n_polys * sizeof(float));
    float* h_t = (float*)malloc((size_t)n_chains * sizeof(float));
    int* h_accept = (int*)calloc((size_t)n_chains, sizeof(int));

    float T_min = 1e-5f;
    float T_max = 0.1f;
    for (int i = 0; i < n_chains; i++) {
        float ratio = (n_chains > 1) ? (float)i / (float)(n_chains - 1) : 0.0f;
        h_t[i] = T_min * powf(T_max / T_min, ratio);
        for (int p = 0; p < n_polys; p++) {
            h_x[i * n_polys + p] = ((float)rand() / RAND_MAX - 0.5f) * current_box;
            h_y[i * n_polys + p] = ((float)rand() / RAND_MAX - 0.5f) * current_box;
            h_a[i * n_polys + p] = ((float)rand() / RAND_MAX) * 6.28f;
        }
    }

    gpu_upload_state(&soa, h_x, h_y, h_a, n_chains, n_polys);
    gpu_sync_metadata(&soa, h_t, NULL, n_chains);

    float* host_energies = (float*)malloc((size_t)n_chains * sizeof(float));
    int HOF_SIZE = 5;
    float* hof_energies = (float*)malloc((size_t)HOF_SIZE * sizeof(float));
    float* hof_fingerprints = (float*)malloc((size_t)HOF_SIZE * sizeof(float));
    float* snap_x = (float*)malloc((size_t)n_polys * sizeof(float));
    float* snap_y = (float*)malloc((size_t)n_polys * sizeof(float));
    float* snap_a = (float*)malloc((size_t)n_polys * sizeof(float));
    float* hof_x = (float*)malloc((size_t)HOF_SIZE * n_polys * sizeof(float));
    float* hof_y = (float*)malloc((size_t)HOF_SIZE * n_polys * sizeof(float));
    float* hof_a = (float*)malloc((size_t)HOF_SIZE * n_polys * sizeof(float));

    for (int k = 0; k < HOF_SIZE; k++) {
        hof_energies[k] = 1e9f;
        hof_fingerprints[k] = -1.0f;
    }

    int epoch_steps = 1000;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        gpu_launch_anneal(&soa, n_chains, n_polys, current_box, epoch_steps);
        gpu_download_stats(&soa, host_energies, h_accept, n_chains);

        int best_idx = -1;
        float best_E = 1e9f;
        for (int i = 0; i < n_chains; i++) {
            if (host_energies[i] < best_E) {
                best_E = host_energies[i];
                best_idx = i;
            }
        }

        if (epoch % 10 == 0) {
            perform_swaps(host_energies, h_t);
            gpu_sync_metadata(&soa, h_t, NULL, n_chains);
        }

        if (epoch % 50 == 0 && best_idx >= 0) {
            gpu_download_chain_geometry(&soa, best_idx, snap_x, snap_y, snap_a, n_polys, n_chains);
            float fp = calculate_fingerprint(snap_x, snap_y, n_polys);
            int worst_hof = 0;
            for (int k = 1; k < HOF_SIZE; k++) {
                if (hof_energies[k] > hof_energies[worst_hof]) worst_hof = k;
            }
            bool unique = true;
            for (int k = 0; k < HOF_SIZE; k++) {
                if (hof_energies[k] < 1e8f && fabsf(fp - hof_fingerprints[k]) < 0.5f) {
                    unique = false;
                    break;
                }
            }
            if (unique && best_E < hof_energies[worst_hof]) {
                hof_energies[worst_hof] = best_E;
                hof_fingerprints[worst_hof] = fp;
                memcpy(&hof_x[(size_t)worst_hof * n_polys], snap_x, (size_t)n_polys * sizeof(float));
                memcpy(&hof_y[(size_t)worst_hof * n_polys], snap_y, (size_t)n_polys * sizeof(float));
                memcpy(&hof_a[(size_t)worst_hof * n_polys], snap_a, (size_t)n_polys * sizeof(float));
            }
        }

        // HOF injection disabled for sweep stability/performance.

        if (epoch % 100 == 0) {
            float hof_best = hof_energies[0];
            for (int k = 1; k < HOF_SIZE; k++) if (hof_energies[k] < hof_best) hof_best = hof_energies[k];
            if (hof_best < 0.1f) {
                float next_L = current_box * 0.99f;
                if (next_L > min_L) {
                    float scale_factor = next_L / current_box;
                    current_box = next_L;
                    gpu_rescale_world(&soa, n_chains, n_polys, scale_factor);
                }
            }
        }
    }

    float density = (total_area > 0.0f) ? (total_area / (current_box * current_box)) : 0.0f;
    if (csv_log) {
        fprintf(csv_log, "%d,%.4f,%.4f\n", n_polys, current_box, density);
        fflush(csv_log);
    }

    gpu_free_soa(&soa);

    free(h_x);
    free(h_y);
    free(h_a);
    free(h_t);
    free(h_accept);
    free(host_energies);
    free(hof_energies);
    free(hof_fingerprints);
    free(snap_x);
    free(snap_y);
    free(snap_a);
    free(hof_x);
    free(hof_y);
    free(hof_a);
}

void run_simulation(int n_chains, int n_polys, int n_epochs) {
    run_single_simulation(n_chains, n_polys, n_epochs, 0.0, NULL);
}

void run_sweep_step(int n_polys, int n_epochs, void* csv_file_ptr) {
    double base_area = 0.0;
    Poly tmp_tree = make_tree_poly_local();
    base_area = poly_area(&tmp_tree);
    if (base_area < 0.0) base_area = -base_area;
    free(tmp_tree.v);
    run_single_simulation(200, n_polys, n_epochs, base_area, (FILE*)csv_file_ptr);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    signal(SIGINT, handle_sig);

    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        run_all_tests();
        return 0;
    }

    if (argc == 1 || (argc > 1 && strcmp(argv[1], "--sweep") == 0) || argc >= 3) {
        int arg_offset = (argc > 1 && strcmp(argv[1], "--sweep") == 0) ? 1 : 0;
        int start_n = 1;
        int end_n = 200;
        char output_filename[256] = "sweep_results.csv";

        if (argc >= 3 + arg_offset) {
            start_n = atoi(argv[1 + arg_offset]);
            end_n = atoi(argv[2 + arg_offset]);
        }
        if (argc >= 4 + arg_offset) {
            strncpy(output_filename, argv[3 + arg_offset], sizeof(output_filename) - 1);
            output_filename[sizeof(output_filename) - 1] = '\0';
        }
        if (start_n < 1) start_n = 1;
        if (end_n < start_n) end_n = start_n;

        printf(">> JOB STARTED: N=%d to %d. Output: %s\n", start_n, end_n, output_filename);

        FILE* csv = fopen(output_filename, "w");
        if (!csv) {
            fprintf(stderr, "Error opening file %s\n", output_filename);
            return 1;
        }
        fprintf(csv, "N,Final_L,Density\n");
        fflush(csv);

        for (int n = start_n; n <= end_n; n++) {
            run_sweep_step(n, 5000, csv);
        }
        fclose(csv);
        printf(">> JOB COMPLETE: %s\n", output_filename);
        return 0;
    }

    unsigned seed = (unsigned)time(NULL);
    int checkpoint_every = CHECKPOINT_SEC;
    int log_every = LOG_SEC;
    int device_id = 0;
    int resume = 0;
    int n_polys = N_POLYS;
    int n_chains = N_CHAINS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--n") == 0) && i + 1 < argc) n_polys = atoi(argv[++i]);
        else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--chains") == 0) && i + 1 < argc) n_chains = atoi(argv[++i]);
        else if (strcmp(argv[i], "--checkpoint-every-sec") == 0 && i + 1 < argc) checkpoint_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--log-every-sec") == 0 && i + 1 < argc) log_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) device_id = atoi(argv[++i]);
        else if (strcmp(argv[i], "--resume") == 0) resume = 1;
    }

    if (n_polys < 1) n_polys = 1;
    if (n_polys > N_POLYS) n_polys = N_POLYS;
    if (n_chains < 1) n_chains = 1;
    if (n_chains > N_CHAINS) n_chains = N_CHAINS;

    srand(seed);
    cudaSetDevice(device_id);

    printf("=== CHIMERA N=%d SOLVER STARTING ===\n", n_polys);

    Poly tree = make_tree_poly_local();
    Triangulation T = triangulate_earclip(&tree);
    ConvexDecomp D = convex_decomp_merge_tris(&tree, &T);
    float min_L_theoretical = 0.0f;
    {
        double area = poly_area(&tree);
        if (area < 0.0) area = -area;
        if (area > 0.0) {
            min_L_theoretical = (float)sqrt(area * (double)n_polys);
        }
    }
    printf("[Host] Theoretical min box size ~ %.4f\n", min_L_theoretical);

    BakedGeometry baked;
    if (!baked_geometry_build(&D, &baked)) {
        fprintf(stderr, "Failed to bake geometry.\n");
        return 1;
    }

    gpu_upload_geometry_from_baked(&baked);
    printf("[Host] Geometry baked and uploaded.\n");

    DeviceSoA soa;
    gpu_alloc_soa(&soa, n_chains, n_polys);
    gpu_audit_memory(&soa, n_chains, n_polys);
    gpu_init_rng(&soa, n_chains, 12345ULL);

    float current_box = 100.0f;
    int start_step = 0;

    float* h_x = (float*)malloc(n_chains * n_polys * sizeof(float));
    float* h_y = (float*)malloc(n_chains * n_polys * sizeof(float));
    float* h_a = (float*)malloc(n_chains * n_polys * sizeof(float));
    float* h_t = (float*)malloc(n_chains * sizeof(float));
    int* h_accept = (int*)calloc(n_chains, sizeof(int));
    void* h_rng = malloc(n_chains * sizeof(curandState));

    if (resume && load_checkpoint("run_checkpoint.bin", &current_box, &start_step, h_x, h_y, h_a, h_t, h_accept, h_rng, n_chains, n_polys)) {
        gpu_upload_state(&soa, h_x, h_y, h_a, n_chains, n_polys);
        gpu_sync_metadata(&soa, h_t, NULL, n_chains);
        gpu_sync_rng(&soa, h_rng, n_chains, true);
        gpu_sync_accept(&soa, h_accept, n_chains, true);
        printf("[Host] Resumed from checkpoint at step %d, L=%.4f\n", start_step, current_box);
    } else {
        float T_min = 1e-5f;
        float T_max = 0.5f;
        for(int i=0; i<n_chains; i++) {
            float ratio = (n_chains > 1) ? (float)i / (float)(n_chains - 1) : 0.0f;
            h_t[i] = T_min * powf(T_max / T_min, ratio);

            for(int p=0; p<n_polys; p++) {
                h_x[i*n_polys + p] = ((float)rand()/RAND_MAX - 0.5f) * current_box;
                h_y[i*n_polys + p] = ((float)rand()/RAND_MAX - 0.5f) * current_box;
                h_a[i*n_polys + p] = ((float)rand()/RAND_MAX) * 6.28f;
            }
        }
        gpu_upload_state(&soa, h_x, h_y, h_a, n_chains, n_polys);
        gpu_sync_metadata(&soa, h_t, NULL, n_chains);
    }

    free(h_x); free(h_y); free(h_a);

    float* host_energies = (float*)malloc(n_chains * sizeof(float));
    time_t last_ckpt = time(NULL);
    time_t last_log = time(NULL);

    mkdir("frames", 0777);
    FILE* stats_csv = fopen("evolution.csv", "w");
    if (stats_csv) {
        fprintf(stats_csv, "Epoch,BestEnergy,BoxSizeL\n");
        fflush(stats_csv);
    }

    float *snap_x = (float*)malloc(n_polys * sizeof(float));
    float *snap_y = (float*)malloc(n_polys * sizeof(float));
    float *snap_a = (float*)malloc(n_polys * sizeof(float));

    int HOF_SIZE = 5;
    float* hof_x = (float*)malloc((size_t)HOF_SIZE * n_polys * sizeof(float));
    float* hof_y = (float*)malloc((size_t)HOF_SIZE * n_polys * sizeof(float));
    float* hof_a = (float*)malloc((size_t)HOF_SIZE * n_polys * sizeof(float));
    float* hof_energies = (float*)malloc((size_t)HOF_SIZE * sizeof(float));
    float* hof_fingerprints = (float*)malloc((size_t)HOF_SIZE * sizeof(float));
    for (int k = 0; k < HOF_SIZE; k++) {
        hof_energies[k] = 1e9f;
        hof_fingerprints[k] = -1.0f;
    }

    FILE* logf = fopen("results.csv", "w");
    if (logf) {
        fprintf(logf, "t_sec,step,L,minE,medianE,p10E,p90E,acceptRateHot,acceptRateCold,swapsAccepted,broadphaseRejectRate\n");
        fflush(logf);
    }

    long total_steps = 1000000;
    int epoch_steps = 5000;
    int n_epochs = (int)(total_steps / epoch_steps);

    printf("Starting Collaborative Annealing: %d Epochs of %d steps...\n", n_epochs, epoch_steps);

    int total_launches = start_step;
    for (int e = 0; e < n_epochs && keep_running; e++) {
        gpu_launch_anneal(&soa, n_chains, n_polys, current_box, epoch_steps);
        total_launches += epoch_steps;

        gpu_download_stats(&soa, host_energies, h_accept, n_chains);

        int best_idx = -1;
        float best_E = 1e9f;
        for (int i = 0; i < n_chains; i++) {
            if (host_energies[i] < best_E) {
                best_E = host_energies[i];
                best_idx = i;
            }
        }

        if (e % 100 == 0 && best_idx >= 0) {
            printf("[Epoch %d] Best Energy: %.5f (Chain %d)\n", e, best_E, best_idx);
        }

        if (stats_csv && e % 100 == 0) {
            fprintf(stats_csv, "%d,%.6f,%.6f\n", e, best_E, current_box);
            fflush(stats_csv);
        }

        if (best_idx >= 0 && e > 0 && (e % 100 == 0)) {
            printf("[Collab] Performing genetic exchange at epoch %d...\n", e);
            int n_replace = 25;
            if (n_replace > n_chains) n_replace = n_chains - 1;
            for (int i = 0; i < n_replace; i++) {
                if (i == best_idx) continue;
                gpu_overwrite_chain(&soa, best_idx, i, n_polys, n_chains);
            }
        }

        if (e > 0 && (e % 100 == 0)) {
            perform_swaps(host_energies, h_t);
            gpu_sync_metadata(&soa, h_t, NULL, n_chains);
        }

        if (e > 0 && (e % 100 == 0) && best_idx >= 0) {
            gpu_download_chain_geometry(&soa, best_idx, snap_x, snap_y, snap_a, n_polys, n_chains);
            float fp = calculate_fingerprint(snap_x, snap_y, n_polys);

            int worst_hof = 0;
            for (int k = 1; k < HOF_SIZE; k++) {
                if (hof_energies[k] > hof_energies[worst_hof]) worst_hof = k;
            }

            bool unique = true;
            for (int k = 0; k < HOF_SIZE; k++) {
                if (hof_energies[k] < 1e8f && fabsf(fp - hof_fingerprints[k]) < 0.5f) {
                    unique = false;
                    break;
                }
            }

            if (unique && best_E < hof_energies[worst_hof]) {
                hof_energies[worst_hof] = best_E;
                hof_fingerprints[worst_hof] = fp;
                memcpy(&hof_x[(size_t)worst_hof * n_polys], snap_x, (size_t)n_polys * sizeof(float));
                memcpy(&hof_y[(size_t)worst_hof * n_polys], snap_y, (size_t)n_polys * sizeof(float));
                memcpy(&hof_a[(size_t)worst_hof * n_polys], snap_a, (size_t)n_polys * sizeof(float));
                printf(">> [HOF] New Elite in Slot %d (E: %.4f)\n", worst_hof, best_E);
            }
        }

        // HOF injection disabled for stability/performance.

        if (e % 100 == 0 && best_E < 1e-4f) {
            float next_L = current_box * 0.99f;
            if (min_L_theoretical > 0.0f && next_L < min_L_theoretical) {
                next_L = min_L_theoretical;
            }
            if (next_L < current_box) {
                float scale_factor = next_L / current_box;
                current_box = next_L;
                gpu_rescale_world(&soa, n_chains, n_polys, scale_factor);
                printf("[Epoch %d] Shrinking L -> %.4f (min %.4f)\n", e, current_box, min_L_theoretical);
            }
        }

        if (best_E <= -999999.0f) {
            printf("Perfect packing found at Epoch %d! Stopping.\n", e);
            break;
        }

        if (time(NULL) - last_ckpt > checkpoint_every) {
            gpu_sync_rng(&soa, h_rng, n_chains, false);
            save_checkpoint("run_checkpoint.bin", current_box, total_launches, &soa, h_t, h_accept, h_rng, n_chains, n_polys, STEPS_PER_LAUNCH);
            last_ckpt = time(NULL);
        }

        if (logf && time(NULL) - last_log > log_every) {
            float minE = 1e9f, maxE = -1e9f;
            for (int i = 0; i < n_chains; i++) {
                if (host_energies[i] < minE) minE = host_energies[i];
                if (host_energies[i] > maxE) maxE = host_energies[i];
            }
            float* tmp = (float*)malloc(n_chains * sizeof(float));
            memcpy(tmp, host_energies, n_chains * sizeof(float));
            for (int i = 0; i < n_chains; i++) {
                for (int j = i + 1; j < n_chains; j++) {
                    if (tmp[j] < tmp[i]) {
                        float t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t;
                    }
                }
            }
            float medianE = tmp[n_chains/2];
            float p10E = tmp[(int)(0.10f * (n_chains - 1))];
            float p90E = tmp[(int)(0.90f * (n_chains - 1))];
            free(tmp);

            float acceptHot = (float)h_accept[n_chains - 1] / (float)(total_launches + 1);
            float acceptCold = (float)h_accept[0] / (float)(total_launches + 1);

            fprintf(logf, "%ld,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.6f\n",
                    (long)time(NULL), total_launches, current_box,
                    minE, medianE, p10E, p90E,
                    acceptHot, acceptCold, 0, 0.0f);
            fflush(logf);
            last_log = time(NULL);
        }
    }

    gpu_free_soa(&soa);
    gpu_free_geometry();
    baked_geometry_free(&baked);
    free_convex_decomp(&D);
    free_triangulation(&T);
    free(tree.v);

    if (stats_csv) fclose(stats_csv);
    if (logf) fclose(logf);
    free(host_energies);
    free(hof_x);
    free(hof_y);
    free(hof_a);
    free(hof_energies);
    free(hof_fingerprints);
    free(snap_x);
    free(snap_y);
    free(snap_a);
    free(h_t);
    free(h_accept);
    free(h_rng);
    printf("[Host] Shutdown complete.\n");
    return 0;
}
