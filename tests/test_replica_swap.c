#include "../include/replica.h"
#include "../include/utils.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    ReplicaState a = {0};
    ReplicaState b = {0};
    ReplicaState c = {0};

    a.cx[0] = 1.0;
    a.replica_id = 1;
    a.rng_s = 1;

    b.cx[0] = 2.0;
    b.replica_id = 2;
    b.rng_s = 3;

    replica_swap(&a, &b);

    assert(a.cx[0] == 2.0);
    assert(a.replica_id == 2);

    assert(b.cx[0] == 1.0);
    assert(b.replica_id == 1);

    replica_clone(&c, &a, 7);
    assert(c.replica_id == a.replica_id);
    assert(c.rng_s != a.rng_s);

    printf("test_replica_swap ok\n");
    return 0;
}
