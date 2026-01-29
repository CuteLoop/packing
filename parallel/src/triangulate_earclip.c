// triangulate_earclip.c
#include "triangulate_earclip.h"
#include <stdlib.h>
#include <string.h>
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
    Vec2 r = {b.x-a.x, b.y-a.y};
    Vec2 s = {d.x-c.x, d.y-c.y};
    double rxs = r.x*s.y - r.y*s.x;
    double qpxr = (c.x-a.x)*r.y - (c.y-a.y)*r.x;

    if(fabs(rxs) < eps && fabs(qpxr) < eps){
        double rr = r.x*r.x + r.y*r.y + eps;
        double t0 = ((c.x-a.x)*r.x + (c.y-a.y)*r.y) / rr;
        double t1 = t0 + (s.x*r.x + s.y*r.y) / rr;
        if(t0 > t1){ double tmp=t0; t0=t1; t1=tmp; }
        return !(t1 < 0.0-eps || t0 > 1.0+eps);
    }
    if(fabs(rxs) < eps && fabs(qpxr) >= eps) return 0;
    double t = ((c.x-a.x)*s.y - (c.y-a.y)*s.x) / rxs;
    double u = ((c.x-a.x)*r.y - (c.y-a.y)*r.x) / rxs;
    return (t >= -eps && t <= 1.0+eps && u >= -eps && u <= 1.0+eps);
}

static int is_ear(const Poly *p, const int *idx, int m, int i){
    int i0 = idx[(i-1+m)%m];
    int i1 = idx[i];
    int i2 = idx[(i+1)%m];
    Vec2 a=p->v[i0], b=p->v[i1], c=p->v[i2];
    if(orient(a,b,c) <= 1e-12) return 0;
    for(int k=0;k<m;k++){
        int iv = idx[k];
        if(iv==i0||iv==i1||iv==i2) continue;
        if(point_in_triangle(p->v[iv], a,b,c)) return 0;
    }
    return 1;
}

Triangulation triangulate_earclip(const Poly *pin){
    Poly p = *pin;
    Vec2 *owned = NULL;

    if(!(poly_area(pin) > 0)){
        owned = (Vec2*)malloc(sizeof(Vec2)*(size_t)pin->n);
        for(int i=0;i<pin->n;i++) owned[i] = pin->v[pin->n-1-i];
        p.v = owned;
    }

    int n = p.n;
    int *idx = (int*)malloc(sizeof(int)*(size_t)n);
    for(int i=0;i<n;i++) idx[i]=i;

    Tri *tris = (Tri*)malloc(sizeof(Tri)*(size_t)(n-2));
    int tcount=0;

    int m=n;
    int guard=0;
    while(m > 2 && guard < 10000){
        int ear_found=0;
        for(int i=0;i<m;i++){
            if(is_ear(&p, idx, m, i)){
                int i0 = idx[(i-1+m)%m];
                int i1 = idx[i];
                int i2 = idx[(i+1)%m];
                tris[tcount++] = (Tri){i0,i1,i2};
                memmove(&idx[i], &idx[i+1], sizeof(int)*(size_t)(m-i-1));
                m--;
                ear_found=1;
                break;
            }
        }
        if(!ear_found) break;
        guard++;
    }

    free(idx);
    if(owned) free(owned);

    Triangulation T;
    T.nTris = tcount;
    T.tris = tris;
    return T;
}
