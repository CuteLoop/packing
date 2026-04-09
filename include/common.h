#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>

// Constants for the base polygon
#define NV 15
#define NTRI 13

typedef struct { double x, y; } Vec2;

typedef struct { double minx, miny, maxx, maxy; } AABB;

typedef struct { int a, b, c; } Tri;

// Triangle indexing and base polygon vertices are defined in one C file
extern const Tri TRIS[NTRI];
extern const Vec2 BASE_V[NV];

// Grid structure used by spatial hash (also exposed via State.grid in some translations)
typedef struct {
	double L;
	double cell;
	int nx, ny;
	double half;

	int *head;
	int *next;
	int *prev;
	int *cell_id;
	int N;
} Grid;

// Totals returned by energy computations
typedef struct {
	double overlap_total;
	double out_total;
} Totals;

typedef struct {
	double alpha_L;
	double lambda_ov;
	double mu_out;
} Weights;

// Shared packing `State` and helpers (moved here so modules compile)
typedef struct {
	int N;
	double L;

	double *cx;
	double *cy;
	double *th;

	Vec2  *world;     // N*NV
	AABB  *aabb;      // N
	AABB  *tri_aabb;  // N*NTRI

	double br;

	// Grid (declared here so spatial module and monolith share the same type)
	Grid grid;
} State;

// Small helpers (inline for header visibility)
static inline Vec2* W(State *s, int i) { return &s->world[(size_t)i * (size_t)NV]; }
static inline const Vec2* Wc(const State *s, int i) { return &s->world[(size_t)i * (size_t)NV]; }

static inline AABB* TA(State *s, int i) { return &s->tri_aabb[(size_t)i * (size_t)NTRI]; }
static inline const AABB* TAc(const State *s, int i) { return &s->tri_aabb[(size_t)i * (size_t)NTRI]; }

static inline int aabb_overlap(const AABB *a, const AABB *b) {
    if (a->maxx < b->minx || b->maxx < a->minx) return 0;
    if (a->maxy < b->miny || b->maxy < a->miny) return 0;
    return 1;
}

static inline double outside_penalty_aabb(const AABB *b, double L) {
    double half = 0.5 * L;
    double pen = 0.0;
    if (b->minx < -half) { double d = (-half - b->minx); pen += d * d; }
    if (b->maxx >  half) { double d = (b->maxx - half);  pen += d * d; }
    if (b->miny < -half) { double d = (-half - b->miny); pen += d * d; }
    if (b->maxy >  half) { double d = (b->maxy - half);  pen += d * d; }
    return pen;
}

#endif // COMMON_H