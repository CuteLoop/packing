// shape_tree.c - example polygon factory
#include "shape_tree.h"
#include <stdlib.h>

Poly make_tree_poly_local(void){
    Poly P;
    P.n = 15;
    P.v = (Vec2*)malloc(sizeof(Vec2)*(size_t)P.n);
    int k=0;

    double trunk_w=0.15, trunk_h=0.2;
    double base_w=0.7, mid_w=0.4, top_w=0.25;
    double tip_y=0.8, tier1=0.5, tier2=0.25, base_y=0.0, trunk_bottom=-trunk_h;

    P.v[k++] = v2(0.0, tip_y);
    P.v[k++] = v2((top_w/2.0), tier1);
    P.v[k++] = v2((top_w/4.0), tier1);
    P.v[k++] = v2((mid_w/2.0), tier2);
    P.v[k++] = v2((mid_w/4.0), tier2);
    P.v[k++] = v2((base_w/2.0), base_y);
    P.v[k++] = v2((trunk_w/2.0), base_y);
    P.v[k++] = v2((trunk_w/2.0), trunk_bottom);
    P.v[k++] = v2(-(trunk_w/2.0), trunk_bottom);
    P.v[k++] = v2(-(trunk_w/2.0), base_y);
    P.v[k++] = v2(-(base_w/2.0), base_y);
    P.v[k++] = v2(-(mid_w/4.0), tier2);
    P.v[k++] = v2(-(mid_w/2.0), tier2);
    P.v[k++] = v2(-(top_w/4.0), tier1);
    P.v[k++] = v2(-(top_w/2.0), tier1);

    return P;
}

double poly_area(const Poly *p){
    double a=0;
    for(int i=0;i<p->n;i++){
        Vec2 A=p->v[i], B=p->v[(i+1)%p->n];
        a += (A.x*B.y - A.y*B.x);
    }
    return 0.5*a;
}
