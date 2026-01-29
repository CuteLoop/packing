// aabb.h - axis-aligned bounding box helpers
#ifndef AABB_H
#define AABB_H

#include "geom_vec2.h"

typedef struct { Vec2 min, max; } AABB;

static inline AABB aabb_empty(void){
    AABB b; b.min=v2(1e300,1e300); b.max=v2(-1e300,-1e300); return b;
}
static inline void aabb_add_point(AABB *b, Vec2 p){
    if(p.x < b->min.x) b->min.x = p.x;
    if(p.y < b->min.y) b->min.y = p.y;
    if(p.x > b->max.x) b->max.x = p.x;
    if(p.y > b->max.y) b->max.y = p.y;
}
static inline int aabb_overlap(AABB a, AABB b){
    return !(a.max.x < b.min.x || b.max.x < a.min.x || a.max.y < b.min.y || b.max.y < a.min.y);
}
static inline Vec2 aabb_center(AABB b){
    return v2(0.5*(b.min.x+b.max.x), 0.5*(b.min.y+b.max.y));
}

#endif // AABB_H
