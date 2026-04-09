#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>

static __thread int g_depth = 0;
static int g_enabled = -1;
static int g_max_depth = 0;
static int g_show_unknown = 0;

__attribute__((no_instrument_function))
static int trace_enabled(void) {
    if (g_enabled == -1) {
        const char *env = getenv("TRACE_CALLS");
        g_enabled = (env && env[0] != '\0') ? 1 : 0;
        const char *md = getenv("TRACE_MAX_DEPTH");
        g_max_depth = md ? atoi(md) : 0;
        const char *su = getenv("TRACE_SHOW_UNKNOWN");
        g_show_unknown = (su && su[0] != '\0') ? 1 : 0;
    }
    return g_enabled;
}

__attribute__((no_instrument_function))
static void trace_print(const char *event, void *func) {
    if (!trace_enabled()) return;
    if (g_max_depth > 0 && g_depth > g_max_depth) return;

    Dl_info info;
    const char *name = NULL;
    if (dladdr(func, &info) && info.dli_sname) name = info.dli_sname;
    if (!name && !g_show_unknown) return;

    for (int i = 0; i < g_depth; i++) fputc(' ', stderr);
    fprintf(stderr, "%s %s\n", event, name ? name : "(unknown)");
}

void __attribute__((no_instrument_function))
__cyg_profile_func_enter(void *func, void *caller) {
    (void)caller;
    trace_print(">>", func);
    g_depth++;
}

void __attribute__((no_instrument_function))
__cyg_profile_func_exit(void *func, void *caller) {
    (void)caller;
    if (g_depth > 0) g_depth--;
    trace_print("<<", func);
}
