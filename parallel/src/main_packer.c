#include "solver.h"
#include "convex_decomp.h"
#include "geom_vec2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    srand(time(NULL));

    // 1. Create a simple "L" shape (2 rectangles)
    ConvexPart *parts = malloc(sizeof(ConvexPart)*2);

    // Part 0: Vertical (0,0) to (1,4)
    parts[0].n = 4;
    parts[0].v = malloc(sizeof(Vec2)*4);
    parts[0].v[0]=v2(0,0); parts[0].v[1]=v2(1,0); parts[0].v[2]=v2(1,4); parts[0].v[3]=v2(0,4);
    parts[0].axis = malloc(sizeof(Vec2)*4);
    parts[0].axis[0]=v2(0,1); parts[0].axis[1]=v2(1,0); parts[0].axis[2]=v2(0,-1); parts[0].axis[3]=v2(-1,0);

    // Part 1: Horizontal (0,0) to (3,1)
    parts[1].n = 4;
    parts[1].v = malloc(sizeof(Vec2)*4);
    parts[1].v[0]=v2(0,0); parts[1].v[1]=v2(3,0); parts[1].v[2]=v2(3,1); parts[1].v[3]=v2(0,1);
    parts[1].axis = malloc(sizeof(Vec2)*4);
    parts[1].axis[0]=v2(0,1); parts[1].axis[1]=v2(1,0); parts[1].axis[2]=v2(0,-1); parts[1].axis[3]=v2(-1,0);

    ConvexDecomp D;
    D.nParts = 2;
    D.parts = parts;

    // 2. Setup Scene
    int N = 50;
    Pose *poses = malloc(sizeof(Pose)*N);
    Container box = { .width = 20.0, .height = 20.0 };

    // Random Init
    for(int i=0; i<N; i++) {
        poses[i].t.x = (rand()%20) - 10;
        poses[i].t.y = (rand()%20) - 10;
        poses[i].ang = ((rand()%100)/100.0) * 6.28;
    }

    // 3. Run Solver
    SolverParams p;
    p.max_iter = 100000;
    p.initial_beta = 0.5;
    p.final_beta = 10.0;
    p.sigma_trans = 0.2;
    p.sigma_rot = 0.1;
    p.squeeze_interval = 5000;
    p.squeeze_factor = 0.98;

    run_solver(&D, poses, N, &box, p);

    return 0;
}
