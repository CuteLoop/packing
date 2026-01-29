#include <stdio.h>
#include <assert.h>

#define SOA_IDX(poly_idx, chain_idx, num_chains) ((poly_idx) * (num_chains) + (chain_idx))

int main() {
    int num_chains = 32;
    int num_polys = 4;

    printf("Testing Access Pattern for Coalescing...\n");

    for(int tid=0; tid<num_chains; tid++) {
        int addr = SOA_IDX(0, tid, num_chains);
        if (tid > 0) {
            int prev_addr = SOA_IDX(0, tid-1, num_chains);
            assert(addr == prev_addr + 1);
        }
    }

    printf("PASS: Addresses are sequential (Coalescing active).\n");
    return 0;
}
