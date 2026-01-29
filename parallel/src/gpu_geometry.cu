#include "gpu_interface.h"
#include "geometry_bake.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

static float* d_verts_x = NULL;
static float* d_verts_y = NULL;
static float* d_axes_x = NULL;
static float* d_axes_y = NULL;
static int* d_part_start_vert = NULL;
static int* d_part_num_verts = NULL;
static int* d_part_start_axis = NULL;
static int* d_part_num_axes = NULL;
static float* d_part_radius = NULL;

extern "C" void gpu_upload_geometry_from_baked(const BakedGeometry* baked)
{
    if (!baked) return;

    int nParts = baked->nParts;
    int totalVerts = baked->totalVerts;

    float* h_vx = (float*)malloc(sizeof(float) * (size_t)totalVerts);
    float* h_vy = (float*)malloc(sizeof(float) * (size_t)totalVerts);
    float* h_ax = (float*)malloc(sizeof(float) * (size_t)totalVerts);
    float* h_ay = (float*)malloc(sizeof(float) * (size_t)totalVerts);
    float* h_pr = (float*)malloc(sizeof(float) * (size_t)nParts);

    int* h_psv = (int*)malloc(sizeof(int) * (size_t)nParts);
    int* h_pnv = (int*)malloc(sizeof(int) * (size_t)nParts);
    int* h_psa = (int*)malloc(sizeof(int) * (size_t)nParts);
    int* h_pna = (int*)malloc(sizeof(int) * (size_t)nParts);

    for (int i = 0; i < totalVerts; i++) {
        h_vx[i] = (float)baked->verts[i].x;
        h_vy[i] = (float)baked->verts[i].y;
        h_ax[i] = (float)baked->axes[i].x;
        h_ay[i] = (float)baked->axes[i].y;
    }

    for (int p = 0; p < nParts; p++) {
        h_psv[p] = baked->partStart[p];
        h_pnv[p] = baked->partCount[p];
        h_psa[p] = baked->partStart[p];
        h_pna[p] = baked->partCount[p];

        double r2 = 0.0;
        int start = baked->partStart[p];
        int count = baked->partCount[p];
        for (int v = 0; v < count; v++) {
            double x = baked->verts[start + v].x;
            double y = baked->verts[start + v].y;
            double d2 = x*x + y*y;
            if (d2 > r2) r2 = d2;
        }
        h_pr[p] = (float)sqrt(r2);
    }

    CUDA_CHECK(cudaMalloc((void**)&d_verts_x, totalVerts * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_verts_y, totalVerts * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_axes_x, totalVerts * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_axes_y, totalVerts * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_part_start_vert, nParts * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_part_num_verts, nParts * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_part_start_axis, nParts * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_part_num_axes, nParts * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_part_radius, nParts * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_verts_x, h_vx, totalVerts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_verts_y, h_vy, totalVerts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_axes_x, h_ax, totalVerts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_axes_y, h_ay, totalVerts * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part_start_vert, h_psv, nParts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part_num_verts, h_pnv, nParts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part_start_axis, h_psa, nParts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part_num_axes, h_pna, nParts * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part_radius, h_pr, nParts * sizeof(float), cudaMemcpyHostToDevice));

    GpuBakedGeometry g;
    g.n_parts = nParts;
    g.total_verts = totalVerts;
    g.verts_x = d_verts_x;
    g.verts_y = d_verts_y;
    g.axes_x = d_axes_x;
    g.axes_y = d_axes_y;
    g.part_start_vert = d_part_start_vert;
    g.part_num_verts = d_part_num_verts;
    g.part_start_axis = d_part_start_axis;
    g.part_num_axes = d_part_num_axes;
    g.part_radius = d_part_radius;

    gpu_upload_geometry(&g);

    free(h_vx); free(h_vy); free(h_ax); free(h_ay); free(h_pr);
    free(h_psv); free(h_pnv); free(h_psa); free(h_pna);
}

extern "C" void gpu_free_geometry(void)
{
    cudaFree(d_verts_x);
    cudaFree(d_verts_y);
    cudaFree(d_axes_x);
    cudaFree(d_axes_y);
    cudaFree(d_part_start_vert);
    cudaFree(d_part_num_verts);
    cudaFree(d_part_start_axis);
    cudaFree(d_part_num_axes);
    cudaFree(d_part_radius);

    d_verts_x = NULL;
    d_verts_y = NULL;
    d_axes_x = NULL;
    d_axes_y = NULL;
    d_part_start_vert = NULL;
    d_part_num_verts = NULL;
    d_part_start_axis = NULL;
    d_part_num_axes = NULL;
    d_part_radius = NULL;
}
