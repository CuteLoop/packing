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

// Wrapper for geometry upload
void gpu_upload_geometry(GpuBakedGeometry* host_geo);
void gpu_upload_geometry_from_baked(const BakedGeometry* baked);
void gpu_free_geometry(void);

#ifdef __cplusplus
}
#endif

#endif // GPU_INTERFACE_H
