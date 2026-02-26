// collision_benchmark.c
// Build:  gcc -O3 -march=native -std=c11 collision_benchmark.c -lm -o bench
// Run:    ./bench 2000 2000   (instances, pair checks)
//
#define _POSIX_C_SOURCE 199309L
#define MAX_PART_VERTS 32

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "src/geom_vec2.h"
#include "src/aabb.h"
#include "src/shape_tree.h"
#include "src/triangulate_earclip.h"
#include "src/convex_decomp.h"
#include "src/collide_tri_oracle.h"
#include "src/collide_sat.h"

// Types `Poly`, `Tri`, `Triangulation`, `ConvexPart`, and `ConvexDecomp`
// are provided by the module headers included above.

/* ----- Benchmark statistics ----- */
typedef struct {
    long long pair_total;
    long long pair_aabb_pass;
    long long narrow_calls;
    long long collisions;
    double time_sec;
} PairStats;

static void print_pair_stats(const char *name, const PairStats *S){
    double mpairs = (S->time_sec > 0.0) ? ((double)S->pair_total / S->time_sec / 1e6) : 0.0;
    double pass_rate = (S->pair_total>0) ? ((double)S->pair_aabb_pass / (double)S->pair_total) : 0.0;
    printf("%s: pairs=%lld, broadpass=%lld, narrow_calls=%lld, collisions=%lld, time=%.6fs, Mpairs/s=%.3f, broadpass_rate=%.3f\n",
           name, S->pair_total, S->pair_aabb_pass, S->narrow_calls, S->collisions, S->time_sec, mpairs, pass_rate);
}

static void print_narrow_stats(const NarrowStats *N){
    double hint_reject_rate = (N->hint_axis_tests>0) ? ((double)N->hint_axis_rejects / (double)N->hint_axis_tests) : 0.0;
    double sat_rate = (N->sat_full_calls>0) ? ((double)N->sat_full_hits / (double)N->sat_full_calls) : 0.0;
    double avg_axes = (N->sat_full_calls>0) ? ((double)N->sat_axes_tested / (double)N->sat_full_calls) : 0.0;
    double part_pass_rate = (N->part_pair_total>0) ? ((double)N->part_aabb_pass / (double)N->part_pair_total) : 0.0;
    printf("Narrowphase breakdown:\n");
    printf("  part pairs=%lld, part aabb pass=%lld, part_pass_rate=%.3f\n", N->part_pair_total, N->part_aabb_pass, part_pass_rate);
    printf("  hint tests=%lld, hint rejects=%lld, hint_reject_rate=%.3f\n", N->hint_axis_tests, N->hint_axis_rejects, hint_reject_rate);
    printf("  sat full calls=%lld, sat hits=%lld, sat_rate=%.3f, sat_axes_tested=%lld, avg_axes_per_sat=%.3f\n",
           N->sat_full_calls, N->sat_full_hits, sat_rate, N->sat_axes_tested, avg_axes);
}

// `Pose`, `rot`, and `apply_pose` are provided by src/geom_vec2.h

// ---------------------- Geometry helpers ----------------------

// Polygon and triangle helper functions moved to modules in src/ (triangulation and triangle oracle).

// ---------------------- Ear clipping triangulation ----------------------

// is_ear moved to src/triangulate_earclip.c

// `triangulate_earclip` implemented in src/triangulate_earclip.c

// ---------------------- Convex merge of triangles ----------------------

// Convex decomposition logic moved to src/convex_decomp.c

// ---------------------- Shape definition (your tree) ----------------------

// `make_tree_poly_local` implemented in src/shape_tree.c

// ---------------------- AABB for transformed shapes (baseline) ----------------------

static AABB aabb_of_tris_pose(const Poly *base, const Triangulation *T, Pose pose){
    AABB b=aabb_empty();
    for(int t=0;t<T->nTris;t++){
        Vec2 a=apply_pose(base->v[T->tris[t].a], pose);
        Vec2 c=apply_pose(base->v[T->tris[t].b], pose);
        Vec2 d=apply_pose(base->v[T->tris[t].c], pose);
        aabb_add_point(&b,a); aabb_add_point(&b,c); aabb_add_point(&b,d);
    }
    return b;
}

static AABB aabb_of_convex_pose_baseline(const ConvexDecomp *D, Pose pose){
    AABB b=aabb_empty();
    for(int i=0;i<D->nParts;i++){
        const ConvexPart *cp=&D->parts[i];
        for(int j=0;j<cp->n;j++){
            Vec2 w=apply_pose(cp->v[j], pose);
            aabb_add_point(&b,w);
        }
    }
    return b;
}

// ---------------------- Collision between two placed instances ----------------------

static int collide_by_tris(const Poly *base, const Triangulation *T, Pose A, Pose B){
    for(int i=0;i<T->nTris;i++){
        Vec2 a1=apply_pose(base->v[T->tris[i].a], A);
        Vec2 b1=apply_pose(base->v[T->tris[i].b], A);
        Vec2 c1=apply_pose(base->v[T->tris[i].c], A);
        for(int j=0;j<T->nTris;j++){
            Vec2 a2=apply_pose(base->v[T->tris[j].a], B);
            Vec2 b2=apply_pose(base->v[T->tris[j].b], B);
            Vec2 c2=apply_pose(base->v[T->tris[j].c], B);
            if(tri_intersect(a1,b1,c1,a2,b2,c2)) return 1;
        }
    }
    return 0;
}

// Baseline convex SAT: transforms vertices each call and computes axes on the fly.
static int collide_by_convex_sat_baseline(const ConvexDecomp *D, Pose A, Pose B){
    Vec2 WA[MAX_PART_VERTS];
    Vec2 WB[MAX_PART_VERTS];

    for(int i=0;i<D->nParts;i++){
        const ConvexPart *pA=&D->parts[i];
        if(pA->n > MAX_PART_VERTS){ fprintf(stderr,"MAX_PART_VERTS too small\n"); exit(1); }
        for(int a=0;a<pA->n;a++) WA[a]=apply_pose(pA->v[a], A);

        for(int j=0;j<D->nParts;j++){
            const ConvexPart *pB=&D->parts[j];
            if(pB->n > MAX_PART_VERTS){ fprintf(stderr,"MAX_PART_VERTS too small\n"); exit(1); }
            for(int b=0;b<pB->n;b++) WB[b]=apply_pose(pB->v[b], B);

            if(convex_sat_intersect(WA,pA->n,WB,pB->n)) return 1;
        }
    }
    return 0;
}

// Cached instance geometry and hint/SAT helpers moved into src/instance_cache.c and src/collide_sat.c

// collide_cached_convex moved to src/collide_sat.c

// ---------------------- Timing ----------------------

static double now_sec(void){
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
#endif
}

// ---------------------- Main ----------------------

int main(int argc, char **argv){
    int nInstances = 2000;
    int nPairs = 2000;
    unsigned long seed = 1;
    const char *csv_out = NULL;
    int check_agreement = 0; /* number of pairs to check */
    enum { MODE_ALL=0, MODE_TRI=1, MODE_BASE=2, MODE_CACHE=3 } run_mode = MODE_ALL;

    /* Simple CLI parsing for long options */
    for(int i=1;i<argc;i++){
        if(strcmp(argv[i], "--instances")==0 && i+1<argc){ nInstances = atoi(argv[++i]); }
        else if(strcmp(argv[i], "--pairs")==0 && i+1<argc){ nPairs = atoi(argv[++i]); }
        else if(strcmp(argv[i], "--seed")==0 && i+1<argc){ seed = (unsigned long)atoi(argv[++i]); }
        else if(strcmp(argv[i], "--csv")==0 && i+1<argc){ csv_out = argv[++i]; }
        else if(strcmp(argv[i], "--check_agreement")==0 && i+1<argc){ check_agreement = atoi(argv[++i]); }
        else if(strcmp(argv[i], "--mode")==0 && i+1<argc){
            const char *m = argv[++i];
            if(strcmp(m,"all")==0) run_mode = MODE_ALL;
            else if(strcmp(m,"tri")==0) run_mode = MODE_TRI;
            else if(strcmp(m,"sat_base")==0) run_mode = MODE_BASE;
            else if(strcmp(m,"sat_cache")==0) run_mode = MODE_CACHE;
        }
    }

    srand((unsigned int)seed);

    Poly tree = make_tree_poly_local();
    printf("Tree polygon vertices: %d\n", tree.n);
    printf("Area: %.12f\n", poly_area(&tree));

    Triangulation T = triangulate_earclip(&tree);
    printf("Triangulation triangles: %d\n", T.nTris);

    ConvexDecomp D = convex_decomp_merge_tris(&tree, &T);
    printf("Convex parts (tri-merge): %d\n", D.nParts);

    // Random poses
    Pose *poses = (Pose*)malloc(sizeof(Pose)*(size_t)nInstances);
    AABB *aabbT = (AABB*)malloc(sizeof(AABB)*(size_t)nInstances);
    AABB *aabbC_baseline = (AABB*)malloc(sizeof(AABB)*(size_t)nInstances);

    for(int i=0;i<nInstances;i++){
        poses[i].t = v2((frand01()*10.0-5.0), (frand01()*10.0-5.0));
        poses[i].ang = (frand01()*2.0*M_PI);
        aabbT[i] = aabb_of_tris_pose(&tree, &T, poses[i]);
        aabbC_baseline[i] = aabb_of_convex_pose_baseline(&D, poses[i]);
    }

    // Build cached world geometry for convex parts (fast CPU path)
    double t_cache_build0 = now_sec();
    InstanceCache C = build_instance_cache(&D, poses, nInstances);
    double t_cache_build1 = now_sec();
    double cache_build_time = t_cache_build1 - t_cache_build0;

    // Validate instance cache offsets to catch corruption
    size_t totalVertsCheck = 0, totalAxisCheck = 0;
    for(int p=0;p<C.nParts;p++){ totalVertsCheck += (size_t)D.parts[p].n; totalAxisCheck += (size_t)D.parts[p].n; }
    totalVertsCheck *= (size_t)C.nInstances;
    totalAxisCheck *= (size_t)C.nInstances;
    for(int i=0;i<C.nInstances;i++){
        for(int p=0;p<C.nParts;p++){
            int vo = C.vertOffset[i*C.nParts + p];
            int ao = C.axisOffset[i*C.nParts + p];
            if(vo < 0 || ao < 0 || (size_t)vo >= totalVertsCheck || (size_t)ao >= totalAxisCheck){
                fprintf(stderr, "ERROR: cache offset OOB i=%d p=%d vertOffset=%d axisOffset=%d totalVerts=%zu totalAxis=%zu\n",
                        i, p, vo, ao, totalVertsCheck, totalAxisCheck);
                return 3;
            }
        }
    }

    // Random pair list
    int *pairA=(int*)malloc(sizeof(int)*(size_t)nPairs);
    int *pairB=(int*)malloc(sizeof(int)*(size_t)nPairs);
    for(int k=0;k<nPairs;k++){
        int ii = rand()%nInstances;
        int jj = rand()%nInstances;
        if(jj==ii) jj=(jj+1)%nInstances;
        pairA[k]=ii; pairB[k]=jj;
    }

    // Prepare stats
    PairStats S_tri = {0}, S_base = {0}, S_cache = {0};
    NarrowStats N_cache = {0};
    S_tri.pair_total = S_base.pair_total = S_cache.pair_total = (long long)nPairs;

    /* Helper timers */
    double tri_broad_time=0.0, tri_narrow_time=0.0;
    double base_broad_time=0.0, base_narrow_time=0.0;
    double cache_broad_time=0.0, cache_narrow_time=0.0;

    // Benchmark triangles
    for(int k=0;k<nPairs;k++){
        int ii=pairA[k], jj=pairB[k];
        double t0 = now_sec();
        int pass = aabb_overlap(aabbT[ii], aabbT[jj]);
        double t1 = now_sec(); tri_broad_time += t1 - t0;
        if(!pass) continue;
        S_tri.pair_aabb_pass++;
        S_tri.narrow_calls++;
        double t2 = now_sec();
        if(collide_by_tris(&tree, &T, poses[ii], poses[jj])) S_tri.collisions++;
        double t3 = now_sec(); tri_narrow_time += t3 - t2;
    }
    S_tri.time_sec = tri_broad_time + tri_narrow_time;

    // Benchmark convex SAT baseline (no caching, on-the-fly axes)
    for(int k=0;k<nPairs;k++){
        int ii=pairA[k], jj=pairB[k];
        double t0 = now_sec();
        int pass = aabb_overlap(aabbC_baseline[ii], aabbC_baseline[jj]);
        double t1 = now_sec(); base_broad_time += t1 - t0;
        if(!pass) continue;
        S_base.pair_aabb_pass++;
        S_base.narrow_calls++;
        double t2 = now_sec();
        if(collide_by_convex_sat_baseline(&D, poses[ii], poses[jj])) S_base.collisions++;
        double t3 = now_sec(); base_narrow_time += t3 - t2;
    }
    S_base.time_sec = base_broad_time + base_narrow_time;

    // Benchmark convex SAT cached (cached world vertices + cached world axes + cached AABB + per-part AABB + hint axes)
    for(int k=0;k<nPairs;k++){
        int ii=pairA[k], jj=pairB[k];
        double t0 = now_sec();
        int pass = aabb_overlap(C.aabb[ii], C.aabb[jj]);
        double t1 = now_sec(); cache_broad_time += t1 - t0;
        if(!pass) continue;
        S_cache.pair_aabb_pass++;
        S_cache.narrow_calls++;
        double t2 = now_sec();
        if(collide_cached_convex(&D, &C, ii, jj, &N_cache)) S_cache.collisions++;
        double t3 = now_sec(); cache_narrow_time += t3 - t2;
    }
    S_cache.time_sec = cache_broad_time + cache_narrow_time;

    printf("\nPairs total: %d\n", nPairs);
        print_pair_stats("Triangles", &S_tri);
        printf("  broad_time=%.6fs, narrow_time=%.6fs, build_time=%.6fs\n", tri_broad_time, tri_narrow_time, 0.0);
        print_pair_stats("ConvexSAT base", &S_base);
        printf("  broad_time=%.6fs, narrow_time=%.6fs, build_time=%.6fs\n", base_broad_time, base_narrow_time, 0.0);
        print_pair_stats("ConvexSAT cache", &S_cache);
        printf("  broad_time=%.6fs, narrow_time=%.6fs, build_time=%.6fs\n", cache_broad_time, cache_narrow_time, cache_build_time);

        // Cached narrowphase breakdown
        print_narrow_stats(&N_cache);

        // Speedups (time ratios)
        double tri_time = S_tri.time_sec > 0.0 ? S_tri.time_sec : 1e-30;
        double base_time = S_base.time_sec > 0.0 ? S_base.time_sec : 1e-30;
        double cache_time = S_cache.time_sec > 0.0 ? S_cache.time_sec : 1e-30;
        printf("Speedups (time ratios): tri/base=%.3f, base/cache=%.3f, tri/cache=%.3f\n",
            tri_time/base_time, base_time/cache_time, tri_time/cache_time);

    // Collision equality check
    if(!(S_tri.collisions == S_base.collisions && S_base.collisions == S_cache.collisions)){
        fprintf(stderr, "WARNING: collision counts differ: tri=%lld base=%lld cache=%lld\n",
                S_tri.collisions, S_base.collisions, S_cache.collisions);
    }

    // Optional CSV output
    if(csv_out){
        const char *csvpath = csv_out;
        FILE *fp = fopen(csvpath, "a+");
        if(fp){
            // check if empty -> write header
            fseek(fp, 0, SEEK_END);
            long pos = ftell(fp);
            if(pos <= 0){
                fprintf(fp, "timestamp,nInstances,nPairs,tri_time,base_time,cache_time,tri_aabb_pass,base_aabb_pass,cache_aabb_pass,tri_collisions,base_collisions,cache_collisions,cache_part_pair_total,cache_part_aabb_pass,cache_hint_tests,cache_hint_rejects,cache_sat_calls,cache_sat_hits,cache_sat_axes_tested\n");
            }
            long long ts = (long long)time(NULL);
            fprintf(fp, "%lld,%d,%d,%.6f,%.6f,%.6f,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
                    ts, nInstances, nPairs,
                    S_tri.time_sec, S_base.time_sec, S_cache.time_sec,
                    S_tri.pair_aabb_pass, S_base.pair_aabb_pass, S_cache.pair_aabb_pass,
                    S_tri.collisions, S_base.collisions, S_cache.collisions,
                    N_cache.part_pair_total, N_cache.part_aabb_pass,
                    N_cache.hint_axis_tests, N_cache.hint_axis_rejects,
                    N_cache.sat_full_calls, N_cache.sat_full_hits, N_cache.sat_axes_tested);
            fclose(fp);
        } else {
            fprintf(stderr, "ERROR: could not open CSV file %s for append\n", argv[3]);
        }
    }

    // Optional agreement checks between triangle oracle and cached SAT
    if(check_agreement > 0){
        fprintf(stderr, "[agree] starting agreement check K=%d\n", check_agreement);
        int K = check_agreement;
        int found = 0;
        for(int k=0;k<nPairs && found < K;k++){
            int ii = pairA[k], jj = pairB[k];
            if(!aabb_overlap(C.aabb[ii], C.aabb[jj])) continue;
            fprintf(stderr, "[agree] sample %d pairIdx=%d ii=%d jj=%d\n", found, k, ii, jj);
            int tri = collide_by_tris(&tree, &T, poses[ii], poses[jj]);
            int cache = collide_cached_convex(&D, &C, ii, jj, NULL);
            if(tri != cache){
#ifdef DEBUG_ORACLE
                fprintf(stderr, "DEBUG_ORACLE MISMATCH pair %d: ii=%d jj=%d tri=%d cache=%d\n", k, ii, jj, tri, cache);
                fprintf(stderr, "poses: A=(%.6f,%.6f,%.6f) B=(%.6f,%.6f,%.6f)\n",
                        poses[ii].t.x, poses[ii].t.y, poses[ii].ang,
                        poses[jj].t.x, poses[jj].t.y, poses[jj].ang);
                exit(2);
#else
                fprintf(stderr, "WARNING: mismatch pair %d: ii=%d jj=%d tri=%d cache=%d\n", k, ii, jj, tri, cache);
#endif
            }
            found++;
        }
        printf("Agreement check: sampled %d part-passing pairs (requested %d)\n", found, K);
    }

    // Cleanup
    free_instance_cache(&C);

    free(tree.v);
    free(T.tris);

    for(int i=0;i<D.nParts;i++){
        free(D.parts[i].v);
        free(D.parts[i].axis);
    }
    free(D.parts);

    free(poses);
    free(aabbT);
    free(aabbC_baseline);
    free(pairA);
    free(pairB);

    return 0;
}
