#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "common.h"

// Geometry helpers (to be implemented during refactor)
void build_world_verts(Vec2 *world, double cx, double cy, double theta);
double base_polygon_area(void);
double base_bounding_radius(void);
void update_instance(State *s, int i);
void update_instance_rw(const double *cx, const double *cy, const double *th,
						Vec2 *world, AABB *aabb, AABB *tri_aabb, int i);

#endif // GEOMETRY_H
