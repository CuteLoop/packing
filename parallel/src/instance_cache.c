#include <stdlib.h>
#include <math.h>
#include "instance_cache.h"

InstanceCache build_instance_cache(const ConvexDecomp *D, const Pose *poses, int nInstances){
    InstanceCache C;
    C.nInstances = nInstances;
    C.nParts = D->nParts;

    C.vertOffset = (int*)malloc(sizeof(int)*(size_t)nInstances*(size_t)C.nParts);
    C.axisOffset = (int*)malloc(sizeof(int)*(size_t)nInstances*(size_t)C.nParts);

    C.aabb       = (AABB*)malloc(sizeof(AABB)*(size_t)nInstances);

    C.partAabb   = (AABB*)malloc(sizeof(AABB)*(size_t)nInstances*(size_t)C.nParts);
    C.partCenter = (Vec2*)malloc(sizeof(Vec2)*(size_t)nInstances*(size_t)C.nParts);

    C.cang       = (double*)malloc(sizeof(double)*(size_t)nInstances);
    C.sang       = (double*)malloc(sizeof(double)*(size_t)nInstances);

    // total verts/axes across all (instance, part)
    size_t totalVerts = 0;
    size_t totalAxis  = 0;
    for(int i=0;i<nInstances;i++){
        for(int p=0;p<C.nParts;p++){
            totalVerts += (size_t)D->parts[p].n;
            totalAxis  += (size_t)D->parts[p].n;
        }
    }
    C.worldVerts = (Vec2*)malloc(sizeof(Vec2)*totalVerts);
    C.worldAxis  = (Vec2*)malloc(sizeof(Vec2)*totalAxis);

    size_t vcur = 0;
    size_t acur = 0;

    for(int i=0;i<nInstances;i++){
        double c = cos(poses[i].ang), s = sin(poses[i].ang);
        C.cang[i]=c; C.sang[i]=s;

        AABB instBox = aabb_empty();

        for(int p=0;p<C.nParts;p++){
            const ConvexPart *cp = &D->parts[p];

            C.vertOffset[i*C.nParts + p] = (int)vcur;
            C.axisOffset[i*C.nParts + p] = (int)acur;

            // world vertices (rotate + translate) + per-part AABB
            AABB partBox = aabb_empty();
            for(int k=0;k<cp->n;k++){
                Vec2 lv = cp->v[k];
                Vec2 w  = v2(c*lv.x - s*lv.y + poses[i].t.x,
                             s*lv.x + c*lv.y + poses[i].t.y);
                C.worldVerts[vcur++] = w;
                aabb_add_point(&partBox, w);
                aabb_add_point(&instBox, w);
            }
            C.partAabb[i*C.nParts + p] = partBox;
            C.partCenter[i*C.nParts + p] = aabb_center(partBox);

            // world axes (rotate only; translation irrelevant)
            for(int k=0;k<cp->n;k++){
                Vec2 la = cp->axis[k];
                Vec2 wa = v2(c*la.x - s*la.y,
                             s*la.x + c*la.y);
                C.worldAxis[acur++] = wa;
            }
        }

        C.aabb[i]=instBox;
    }
    return C;
}

void free_instance_cache(InstanceCache *C){
    if(!C) return;
    free(C->worldVerts);
    free(C->worldAxis);
    free(C->vertOffset);
    free(C->axisOffset);
    free(C->aabb);
    free(C->partAabb);
    free(C->partCenter);
    free(C->cang);
    free(C->sang);

    C->worldVerts=NULL;
    C->worldAxis=NULL;
    C->vertOffset=NULL;
    C->axisOffset=NULL;
    C->aabb=NULL;
    C->partAabb=NULL;
    C->partCenter=NULL;
    C->cang=NULL;
    C->sang=NULL;
}

// Optional: update single instance (not highly optimized)
void cache_update_one(const ConvexDecomp *D, InstanceCache *C, int i, Pose newPose){
    double c = cos(newPose.ang), s = sin(newPose.ang);
    C->cang[i]=c; C->sang[i]=s;

    AABB instBox = aabb_empty();
    for(int p=0;p<C->nParts;p++){
        const ConvexPart *cp = &D->parts[p];
        int vo = C->vertOffset[i*C->nParts + p];
        int ao = C->axisOffset[i*C->nParts + p];
        AABB partBox = aabb_empty();
        for(int k=0;k<cp->n;k++){
            Vec2 lv = cp->v[k];
            Vec2 w  = v2(c*lv.x - s*lv.y + newPose.t.x,
                         s*lv.x + c*lv.y + newPose.t.y);
            C->worldVerts[vo + k] = w;
            aabb_add_point(&partBox, w);
            aabb_add_point(&instBox, w);
        }
        C->partAabb[i*C->nParts + p] = partBox;
        C->partCenter[i*C->nParts + p] = aabb_center(partBox);
        for(int k=0;k<cp->n;k++){
            Vec2 la = cp->axis[k];
            Vec2 wa = v2(c*la.x - s*la.y,
                         s*la.x + c*la.y);
            C->worldAxis[ao + k] = wa;
        }
    }
    C->aabb[i] = instBox;
}
