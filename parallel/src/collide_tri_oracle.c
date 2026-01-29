// collide_tri_oracle.c - triangle-triangle intersection and driver
#include "collide_tri_oracle.h"
#include "geom_vec2.h"
#include <math.h>

static double orient(Vec2 a, Vec2 b, Vec2 c){ return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x); }

static int point_in_triangle(Vec2 p, Vec2 a, Vec2 b, Vec2 c){
    const double eps=1e-12;
    double o1 = orient(a,b,p);
    double o2 = orient(b,c,p);
    double o3 = orient(c,a,p);
    int has_neg = (o1 < -eps) || (o2 < -eps) || (o3 < -eps);
    int has_pos = (o1 > eps) || (o2 > eps) || (o3 > eps);
    return !(has_neg && has_pos);
}

static int seg_intersect(Vec2 a, Vec2 b, Vec2 c, Vec2 d){
    const double eps=1e-12;
    Vec2 r = sub(b,a), s = sub(d,c);
    double rxs = r.x*s.y - r.y*s.x;
    double qpxr = (c.x-a.x)*r.y - (c.y-a.y)*r.x;

    if(fabs(rxs) < eps && fabs(qpxr) < eps){
        double rr = dot(r,r) + eps;
        double t0 = dot(sub(c,a), r) / rr;
        double t1 = t0 + dot(s, r) / rr;
        if(t0 > t1){ double tmp=t0; t0=t1; t1=tmp; }
        return !(t1 < 0.0-eps || t0 > 1.0+eps);
    }
    if(fabs(rxs) < eps && fabs(qpxr) >= eps) return 0;
    double t = cross(sub(c,a), s) / (rxs);
    double u = cross(sub(c,a), r) / (rxs);
    return (t >= -eps && t <= 1.0+eps && u >= -eps && u <= 1.0+eps);
}

int tri_intersect(Vec2 a, Vec2 b, Vec2 c, Vec2 d, Vec2 e, Vec2 f){
    Vec2 t1[3]={a,b,c}, t2[3]={d,e,f};
    for(int i=0;i<3;i++){
        Vec2 p=t1[i], q=t1[(i+1)%3];
        for(int j=0;j<3;j++){
            Vec2 r=t2[j], s=t2[(j+1)%3];
            if(seg_intersect(p,q,r,s)) return 1;
        }
    }
    if(point_in_triangle(a,d,e,f)) return 1;
    if(point_in_triangle(d,a,b,c)) return 1;
    return 0;
}
