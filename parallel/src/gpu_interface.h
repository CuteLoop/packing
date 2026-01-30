#ifndef GPU_INTERFACE_H
#define GPU_INTERFACE_H

#include <stddef.h>

typedef struct BakedGeometry BakedGeometry;
typedef struct DeviceSoA DeviceSoA;

// GPU-ready baked geometry layout (device pointers).
typedef struct {
	int n_parts;
	int total_verts;

	// Flattened vertex/axis arrays
	const float* verts_x;
	const float* verts_y;
	const float* axes_x;
	const float* axes_y;

	// Per-part ranges into vertex/axis arrays
	const int* part_start_vert;
	const int* part_num_verts;
	const int* part_start_axis;
	const int* part_num_axes;

	// Broadphase helpers (per-part radius)
	const float* part_radius;
} GpuBakedGeometry;

// C linkage for host-side callers
#ifdef __cplusplus
extern "C" {
#endif

// Wrapper for the annealing kernel
void gpu_launch_anneal(
	DeviceSoA* soa,
	int n_chains,
	int n_polys,
	float box_size,
	int steps
);

void gpu_overwrite_chain(DeviceSoA* data, int src_chain, int dst_chain, int n_polys, int n_chains);
void gpu_rescale_world(DeviceSoA* data, int n_chains, int n_polys, float scale_factor);
void gpu_audit_memory(DeviceSoA* data, int n_chains, int n_polys);

// Wrapper for geometry upload
void gpu_upload_geometry(GpuBakedGeometry* host_geo);
void gpu_upload_geometry_from_baked(const BakedGeometry* baked);
void gpu_free_geometry(void);

void gpu_download_energies(DeviceSoA* data, float* host_dst, int n_chains);
void gpu_download_chain_geometry(DeviceSoA* data, int chain_idx, float* h_x, float* h_y, float* h_ang, int n_polys, int n_chains);
void gpu_upload_chain_geometry(DeviceSoA* data, int chain_idx, const float* h_x, const float* h_y, const float* h_ang, int n_polys, int n_chains);

void run_simulation(int n_chains, int n_polys, int n_epochs);
void run_sweep_step(int n_polys, int n_epochs, void* csv_file_ptr);

#ifdef __cplusplus
}
#endif

#endif // GPU_INTERFACE_H
