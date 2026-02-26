// geom_vec2.h - small 2D vector and pose helpers
#ifndef GEOM_VEC2_H
#define GEOM_VEC2_H

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { double x,y; } Vec2;

static inline double frand01(void){ return (double)rand() / (double)RAND_MAX; }

static inline Vec2 v2(double x,double y){ Vec2 v={x,y}; return v; }
static inline Vec2 add(Vec2 a, Vec2 b){ return v2(a.x+b.x,a.y+b.y); }
static inline Vec2 sub(Vec2 a, Vec2 b){ return v2(a.x-b.x,a.y-b.y); }
static inline Vec2 mul(Vec2 a, double s){ return v2(a.x*s,a.y*s); }

static inline double dot(Vec2 a, Vec2 b){ return a.x*b.x + a.y*b.y; }
static inline double cross(Vec2 a, Vec2 b){ return a.x*b.y - a.y*b.x; }

static inline double len2(Vec2 a){ return dot(a,a); }
static inline Vec2 perp(Vec2 a){ return v2(-a.y, a.x); }
static inline Vec2 norm_or_zero(Vec2 a){
    const double eps=1e-18;
    double l2 = len2(a);
    if(l2 < eps) return v2(0.0, 0.0);
    double inv = 1.0/sqrt(l2);
    return mul(a, inv);
}

typedef struct {
    Vec2 t;      // translation
    double ang;  // radians
} Pose;

static inline Vec2 rot(Vec2 p, double ang){
    double c = cos(ang), s=sin(ang);
    return v2(c*p.x - s*p.y, s*p.x + c*p.y);
}
static inline Vec2 apply_pose(Vec2 p, Pose pose){
    return add(rot(p, pose.ang), pose.t);
}

#endif // GEOM_VEC2_H
