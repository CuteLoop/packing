#ifndef CHIMERA_SYSTEM_H
#define CHIMERA_SYSTEM_H

#include <stddef.h>
#include "gpu_data.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ChimeraContext {
    DeviceSoA soa;
    int active_chains;
    int active_polys;
    size_t total_vram_bytes;
} ChimeraContext;

int chimera_init(ChimeraContext* ctx, int active_chains, int active_polys);
void chimera_free(ChimeraContext* ctx);

#ifdef __cplusplus
}
#endif

#endif
