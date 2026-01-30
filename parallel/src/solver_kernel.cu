#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "gpu_data.h"
#include <stdio.h>

// --- 1. Constant Memory (Geometry) ---
__constant__ GpuBakedGeometry c_geo;

// Host wrapper to upload geometry
extern "C" void gpu_upload_geometry(GpuBakedGeometry* host_geo) {
    cudaMemcpyToSymbol(c_geo, host_geo, sizeof(GpuBakedGeometry));
}

// --- 2. Math Helpers (Device) ---
#define PI 3.1415926535f
#define SAT_EPSILON 1e-5f

__device__ inline float wrap_angle(float a) {
    while (a > PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

__device__ inline float reflect_val(float val, float limit) {
    if (val > limit) return limit - (val - limit);
    if (val < -limit) return -limit + (-limit - val);
    return val;
}

__device__ inline void apply_pose(float lx, float ly, float tx, float ty, float c, float s, float* wx, float* wy) {
    *wx = lx * c - ly * s + tx;
    *wy = lx * s + ly * c + ty;
}

// --- 3. Collision Logic (SAT + Broadphase) ---
__device__ float check_pair_overlap(
    float ax, float ay, float aa,
    float bx, float by, float ba
) {
    float ca = cosf(aa), sa = sinf(aa);
    float cb = cosf(ba), sb = sinf(ba);

    for(int i=0; i < c_geo.n_parts; i++) {
        for(int j=0; j < c_geo.n_parts; j++) {
            // --- Broadphase: radius check ---
            float dx = ax - bx;
            float dy = ay - by;
            float r = c_geo.part_radius[i] + c_geo.part_radius[j];
            if (dx*dx + dy*dy > r*r) continue;

            bool separated = false;

            int a_start = c_geo.part_start_axis[i];
            int a_num   = c_geo.part_num_axes[i];
            int v_start_a = c_geo.part_start_vert[i];
            int v_num_a   = c_geo.part_num_verts[i];

            int b_start = c_geo.part_start_axis[j];
            int b_num   = c_geo.part_num_axes[j];
            int v_start_b = c_geo.part_start_vert[j];
            int v_num_b   = c_geo.part_num_verts[j];

            // Check A's axes
            for(int k=0; k<a_num; k++) {
                float nx = c_geo.axes_x[a_start+k] * ca - c_geo.axes_y[a_start+k] * sa;
                float ny = c_geo.axes_x[a_start+k] * sa + c_geo.axes_y[a_start+k] * ca;

                float min_a = 1e9f, max_a = -1e9f;
                for(int v=0; v<v_num_a; v++) {
                    float vx, vy;
                    apply_pose(c_geo.verts_x[v_start_a+v], c_geo.verts_y[v_start_a+v], ax, ay, ca, sa, &vx, &vy);
                    float p = vx*nx + vy*ny;
                    if(p < min_a) min_a = p;
                    if(p > max_a) max_a = p;
                }

                float min_b = 1e9f, max_b = -1e9f;
                for(int v=0; v<v_num_b; v++) {
                    float vx, vy;
                    apply_pose(c_geo.verts_x[v_start_b+v], c_geo.verts_y[v_start_b+v], bx, by, cb, sb, &vx, &vy);
                    float p = vx*nx + vy*ny;
                    if(p < min_b) min_b = p;
                    if(p > max_b) max_b = p;
                }

                if (max_a < min_b - SAT_EPSILON || max_b < min_a - SAT_EPSILON) {
                    separated = true;
                    break;
                }
            }
            if(separated) continue;

            // Check B's axes
            for(int k=0; k<b_num; k++) {
                float nx = c_geo.axes_x[b_start+k] * cb - c_geo.axes_y[b_start+k] * sb;
                float ny = c_geo.axes_x[b_start+k] * sb + c_geo.axes_y[b_start+k] * cb;

                float min_a = 1e9f, max_a = -1e9f;
                for(int v=0; v<v_num_a; v++) {
                    float vx, vy;
                    apply_pose(c_geo.verts_x[v_start_a+v], c_geo.verts_y[v_start_a+v], ax, ay, ca, sa, &vx, &vy);
                    float p = vx*nx + vy*ny;
                    if(p < min_a) min_a = p;
                    if(p > max_a) max_a = p;
                }

                float min_b = 1e9f, max_b = -1e9f;
                for(int v=0; v<v_num_b; v++) {
                    float vx, vy;
                    apply_pose(c_geo.verts_x[v_start_b+v], c_geo.verts_y[v_start_b+v], bx, by, cb, sb, &vx, &vy);
                    float p = vx*nx + vy*ny;
                    if(p < min_b) min_b = p;
                    if(p > max_b) max_b = p;
                }

                if (max_a < min_b - SAT_EPSILON || max_b < min_a - SAT_EPSILON) {
                    separated = true;
                    break;
                }
            }
            if(separated) continue;

            return 1.0f;
        }
    }

    return 0.0f;
}

// --- 4. The Main Kernel ---
__global__ void k_anneal_n200(
    DeviceSoA soa,
    int n_chains,
    int n_polys,
    float box_size,
    int n_steps_per_launch
) {
    int chain_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (chain_id >= n_chains) return;

    curandState local_rng = soa.rng[chain_id];
    float T = soa.temperature[chain_id];
    float E = soa.energy[chain_id];
    int accepts = 0;
    float limit = box_size * 0.5f;

    for(int step=0; step < n_steps_per_launch; step++) {
        int p_idx = curand(&local_rng) % n_polys;

        int mem_idx = SOA_IDX(p_idx, chain_id, n_chains);
        float old_x = soa.pos_x[mem_idx];
        float old_y = soa.pos_y[mem_idx];
        float old_a = soa.angle[mem_idx];

        float dx = (curand_uniform(&local_rng) - 0.5f) * 1.0f;
        float dy = (curand_uniform(&local_rng) - 0.5f) * 1.0f;
        float da = (curand_uniform(&local_rng) - 0.5f) * 0.5f;

        float new_x = reflect_val(old_x + dx, limit);
        float new_y = reflect_val(old_y + dy, limit);
        float new_a = wrap_angle(old_a + da);

        float dE = 0.0f;

        for(int j=0; j<n_polys; j++) {
            if(p_idx == j) continue;

            int n_mem_idx = SOA_IDX(j, chain_id, n_chains);
            float n_x = soa.pos_x[n_mem_idx];
            float n_y = soa.pos_y[n_mem_idx];
            float n_a = soa.angle[n_mem_idx];

            dE -= check_pair_overlap(old_x, old_y, old_a, n_x, n_y, n_a);
            dE += check_pair_overlap(new_x, new_y, new_a, n_x, n_y, n_a);
        }

        bool accept = false;
        if (dE <= 0.0f) {
            accept = true;
        } else {
            float r = curand_uniform(&local_rng);
            if (r < expf(-dE / T)) accept = true;
        }

        if (accept) {
            soa.pos_x[mem_idx] = new_x;
            soa.pos_y[mem_idx] = new_y;
            soa.angle[mem_idx] = new_a;
            E += dE;
            accepts++;
        }
    }

    soa.rng[chain_id] = local_rng;
    soa.energy[chain_id] = E;
    soa.chain_energies[chain_id] = E;
    soa.accept_count[chain_id] += accepts;
}

// --- 5. Host Launcher ---
extern "C" void gpu_launch_anneal(
    DeviceSoA* soa,
    int n_chains,
    int n_polys,
    float box_size,
    int steps
) {
    int blockSize = 128;
    int numBlocks = (n_chains + blockSize - 1) / blockSize;

    k_anneal_n200<<<numBlocks, blockSize>>>(*soa, n_chains, n_polys, box_size, steps);
    cudaDeviceSynchronize();
}

__global__ void k_overwrite_chain(DeviceSoA data, int src_chain, int dst_chain, int n_polys, int n_chains) {
    int poly_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (poly_idx >= n_polys) return;

    int src_ptr = SOA_IDX(poly_idx, src_chain, n_chains);
    int dst_ptr = SOA_IDX(poly_idx, dst_chain, n_chains);

    data.pos_x[dst_ptr] = data.pos_x[src_ptr];
    data.pos_y[dst_ptr] = data.pos_y[src_ptr];
    data.angle[dst_ptr] = data.angle[src_ptr];
}

extern "C" void gpu_overwrite_chain(DeviceSoA* data, int src_chain, int dst_chain, int n_polys, int n_chains) {
    int threads = 128;
    int blocks = (n_polys + threads - 1) / threads;
    k_overwrite_chain<<<blocks, threads>>>(*data, src_chain, dst_chain, n_polys, n_chains);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Overwrite Kernel Failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("!!! CLONE KERNEL FAILED: %s !!!\n", cudaGetErrorString(err));
    }
}
