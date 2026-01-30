# ==========================================
# Chimera CUDA Solver - Clean Build
# ==========================================

CC      = nvcc
CFLAGS  = -O3 -I./src
ARCH    = -arch=sm_60
LDFLAGS = -lm

# --- EXPLICIT SOURCE LIST ---
# Only compile what we use.

CUDA_SRCS = src/main.cu \
            src/solver_kernel.cu \
            src/gpu_memory.cu \
            src/gpu_geometry.cu \
            src/gpu_rng.cu

C_SRCS    = src/triangulate_earclip.c \
            src/convex_decomp.c \
            src/shape_tree.c \
            src/geometry_bake.c

OBJS = $(CUDA_SRCS:.cu=.o) $(C_SRCS:.c=.o)

TARGET = chimera_run

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo ">> Linking Final Executable..."
	$(CC) $(ARCH) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cu
	@echo ">> Compiling CUDA: $<"
	$(CC) $(CFLAGS) $(ARCH) -c $< -o $@

%.o: %.c
	@echo ">> Compiling Host C: $<"
	$(CC) $(CFLAGS) -Xcompiler -std=c99 $(ARCH) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET)
	@echo ">> Cleaned."
