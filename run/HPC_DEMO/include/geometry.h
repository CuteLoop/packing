#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "common.h"

// Geometry helpers (to be implemented during refactor)
void build_world_verts(Vec2 *world, double cx, double cy, double theta);
double base_polygon_area(void);
double base_bounding_radius(void);

#endif // GEOMETRY_H
