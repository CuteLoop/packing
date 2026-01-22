#include "../include/spatial_hash.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

static inline int clampi(int v, int lo, int hi) {
	if (v < lo) return lo;
	if (v > hi) return hi;
	return v;
}

int grid_index(const Grid *g, int ix, int iy) {
	return iy * g->nx + ix;
}

void grid_cell_xy(const Grid *g, double x, double y, int *ix, int *iy) {
	double fx = (x + g->half) / g->cell;
	double fy = (y + g->half) / g->cell;
	int cx = (int)floor(fx);
	int cy = (int)floor(fy);
	cx = clampi(cx, 0, g->nx - 1);
	cy = clampi(cy, 0, g->ny - 1);
	*ix = cx; *iy = cy;
}

void grid_init(Grid *g, int N, double L, double cell) {
	g->N = N;
	g->L = L;
	g->half = 0.5 * L;
	g->cell = cell;

	g->nx = (int)ceil(L / cell);
	g->ny = (int)ceil(L / cell);
	if (g->nx < 1) g->nx = 1;
	if (g->ny < 1) g->ny = 1;

	int nc = g->nx * g->ny;

	g->head = (int*)malloc((size_t)nc * sizeof(int));
	g->next = (int*)malloc((size_t)g->N * sizeof(int));
	g->prev = (int*)malloc((size_t)g->N * sizeof(int));
	g->cell_id = (int*)malloc((size_t)g->N * sizeof(int));

	if (!g->head || !g->next || !g->prev || !g->cell_id) {
		fprintf(stderr, "grid alloc failed\n");
		exit(1);
	}

	for (int c = 0; c < nc; c++) g->head[c] = -1;
	for (int i = 0; i < g->N; i++) {
		g->next[i] = -1;
		g->prev[i] = -1;
		g->cell_id[i] = -1;
	}
}

void grid_free(Grid *g) {
	free(g->head); free(g->next); free(g->prev); free(g->cell_id);
	g->head = g->next = g->prev = g->cell_id = NULL;
	g->nx = g->ny = 0;
}

void grid_insert(Grid *g, int i, double x, double y) {
	int ix, iy;
	grid_cell_xy(g, x, y, &ix, &iy);
	int cid = grid_index(g, ix, iy);

	int h = g->head[cid];
	g->prev[i] = -1;
	g->next[i] = h;
	if (h != -1) g->prev[h] = i;
	g->head[cid] = i;
	g->cell_id[i] = cid;
}

void grid_remove(Grid *g, int i) {
	int cid = g->cell_id[i];
	if (cid < 0) return;

	int pi = g->prev[i];
	int ni = g->next[i];

	if (pi != -1) g->next[pi] = ni;
	else g->head[cid] = ni;

	if (ni != -1) g->prev[ni] = pi;

	g->prev[i] = g->next[i] = -1;
	g->cell_id[i] = -1;
}

void grid_update(Grid *g, int i, double x, double y) {
	int ix, iy;
	grid_cell_xy(g, x, y, &ix, &iy);
	int new_cid = grid_index(g, ix, iy);
	int old_cid = g->cell_id[i];

	if (old_cid == new_cid) return;
	if (old_cid != -1) grid_remove(g, i);
	grid_insert(g, i, x, y);
}

void grid_rebuild(Grid *g, int N, double L, double cell, const double *cx, const double *cy) {
	grid_free(g);
	grid_init(g, N, L, cell);
	for (int i = 0; i < N; i++) grid_insert(g, i, cx[i], cy[i]);
}

int grid_R_cells(const State *s) {
	int R = (int)ceil((2.0 * s->br) / s->grid.cell) + 1;
	if (R < 1) R = 1;
	return R;
}

void aabb_to_cell_range(const Grid *g, const AABB *b, int *ix0, int *iy0, int *ix1, int *iy1) {
	int x0, y0, x1, y1;
	grid_cell_xy(g, b->minx, b->miny, &x0, &y0);
	grid_cell_xy(g, b->maxx, b->maxy, &x1, &y1);
	if (ix0) *ix0 = x0; if (iy0) *iy0 = y0; if (ix1) *ix1 = x1; if (iy1) *iy1 = y1;
}

void grid_query_neighbors(const Grid *g, const State *s, int k, cell_visit_fn visit, void *ctx) {
	int cid = g->cell_id[k];
	if (cid < 0) {
		// no cell; brute force
		for (int j = 0; j < s->N; j++) {
			if (j == k) continue;
			if (!visit(j, ctx)) return;
		}
		return;
	}

	int kx = cid % g->nx;
	int ky = cid / g->nx;

	int R = (int)ceil((2.0 * s->br) / g->cell) + 1;
	if (R < 1) R = 1;

	for (int dy = -R; dy <= R; dy++) {
		int yy = ky + dy;
		if (yy < 0 || yy >= g->ny) continue;
		for (int dx = -R; dx <= R; dx++) {
			int xx = kx + dx;
			if (xx < 0 || xx >= g->nx) continue;
			int c = grid_index(g, xx, yy);
			for (int j = g->head[c]; j != -1; j = g->next[j]) {
				if (j == k) continue;
				if (!visit(j, ctx)) return;
			}
		}
	}
}
