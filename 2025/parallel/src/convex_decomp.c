// convex_decomp.c - merge triangles into convex parts
#include "convex_decomp.h"
#include "triangulate_earclip.h"
#include <stdlib.h>
#include <math.h>

static double orient(Vec2 a, Vec2 b, Vec2 c){ return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x); }

static int is_polygon_convex(const Vec2 *pts, int n){
    const double eps=1e-12;
    int sign=0;
    for(int i=0;i<n;i++){
        Vec2 a=pts[i], b=pts[(i+1)%n], c=pts[(i+2)%n];
        double o = orient(a,b,c);
        if(fabs(o) <= eps) continue;
        int s = (o > 0) ? 1 : -1;
        if(sign==0) sign=s;
        else if(sign != s) return 0;
    }
    return 1;
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

static int merge_two_tris_to_quad(const Poly *p, Tri t1, Tri t2, Vec2 outQuad[4]){
    int a1[3]={t1.a,t1.b,t1.c};
    int a2[3]={t2.a,t2.b,t2.c};

    int shared[2], ns=0;
    int u1=-1, u2=-1;
    for(int i=0;i<3;i++){
        int v=a1[i];
        int found=0;
        for(int j=0;j<3;j++) if(a2[j]==v){ found=1; break; }
        if(found){ if(ns<2) shared[ns]=v; ns++; } else u1=v;
    }
    for(int j=0;j<3;j++){
        int v=a2[j];
        int found=0;
        for(int i=0;i<3;i++) if(a1[i]==v){ found=1; break; }
        if(!found) u2=v;
    }
    if(ns!=2 || u1<0 || u2<0) return 0;

    int cand[4] = {u1, shared[0], u2, shared[1]};
    int perms[6][4] = {{cand[0],cand[1],cand[2],cand[3]},{cand[0],cand[1],cand[3],cand[2]},{cand[0],cand[2],cand[1],cand[3]},{cand[0],cand[2],cand[3],cand[1]},{cand[0],cand[3],cand[1],cand[2]},{cand[0],cand[3],cand[2],cand[1]}};

    for(int k=0;k<6;k++){
        Vec2 q[4] = { p->v[perms[k][0]], p->v[perms[k][1]], p->v[perms[k][2]], p->v[perms[k][3]] };
        if(seg_intersect(q[0],q[1],q[2],q[3])) continue;
        if(seg_intersect(q[1],q[2],q[3],q[0])) continue;
        if(!is_polygon_convex(q,4)) continue;
        for(int i=0;i<4;i++) outQuad[i]=q[i];
        return 1;
    }
    return 0;
}

static void convex_part_build_axes(ConvexPart *p){
    const double eps=1e-12;
    p->axis = (Vec2*)malloc(sizeof(Vec2)*(size_t)p->n);
    for(int i=0;i<p->n;i++){
        Vec2 p0=p->v[i], p1=p->v[(i+1)%p->n];
        Vec2 e = {p1.x-p0.x, p1.y-p0.y};
        Vec2 a = {-e.y, e.x};
        double len = sqrt(a.x*a.x + a.y*a.y);
        if(len < eps) a = v2(1.0, 0.0);
        else a = v2(a.x/len, a.y/len);
        p->axis[i] = a;
    }
}

ConvexDecomp convex_decomp_merge_tris(const Poly *p, const Triangulation *T){
    int n = T->nTris;
    int *used = (int*)calloc((size_t)n, sizeof(int));
    ConvexPart *parts = (ConvexPart*)malloc(sizeof(ConvexPart)*(size_t)n);
    int pc=0;

    for(int i=0;i<n;i++){
        if(used[i]) continue;
        int merged=0;
        for(int j=i+1;j<n;j++){
            if(used[j]) continue;
            Vec2 quad[4];
            if(merge_two_tris_to_quad(p, T->tris[i], T->tris[j], quad)){
                used[i]=used[j]=1;
                parts[pc].n=4;
                parts[pc].v=(Vec2*)malloc(sizeof(Vec2)*4);
                for(int k=0;k<4;k++) parts[pc].v[k]=quad[k];
                parts[pc].axis=NULL;
                convex_part_build_axes(&parts[pc]);
                pc++; merged=1; break;
            }
        }
        if(!merged){
            used[i]=1;
            parts[pc].n=3;
            parts[pc].v=(Vec2*)malloc(sizeof(Vec2)*3);
            parts[pc].v[0]=p->v[T->tris[i].a];
            parts[pc].v[1]=p->v[T->tris[i].b];
            parts[pc].v[2]=p->v[T->tris[i].c];
            parts[pc].axis=NULL;
            convex_part_build_axes(&parts[pc]);
            pc++;
        }
    }

    free(used);
    ConvexDecomp D; D.nParts=pc; D.parts=parts; return D;
}
