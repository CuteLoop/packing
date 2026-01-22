#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../run/HPC_DEMO/include/utils.h"

static void fail(const char *msg) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); }

int main(void) {
    const char *d = "tests_tmp_dir";
    ensure_dir(d);
    if (!file_exists(d)) {
        // directory exists but file_exists checks fopen; create a file inside
    }

    const char *f = "tests_tmp_dir/tmpfile.txt";
    FILE *fp = fopen(f, "w");
    if (!fp) fail("could not create temp file");
    fprintf(fp, "hello\n"); fclose(fp);

    if (!file_exists(f)) fail("file_exists returned false for created file");

    remove(f);
    rmdir(d);

    printf("test_io: OK\n");
    return 0;
}
