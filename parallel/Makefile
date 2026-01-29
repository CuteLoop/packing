# ==========================================
# Chimera Solver Makefile
# ==========================================
# Prerequisite: Load your CUDA module before running make
# Example: module load cuda11/11.8
# ==========================================

# 1. Compiler Settings
# We use nvcc for linking and compiling both host and device code
CC      = nvcc
CFLAGS  = -O3 -I./src
LDFLAGS = -lm

# 2. GPU Architecture
# Adjust this for your specific card to get best performance:
# sm_60 = P100
# sm_70 = V100
# sm_75 = T4 (common on Colab/AWS G4)
# sm_80 = A100
ARCH    = -arch=sm_60

# 3. Source Discovery
# Find all .c and .cu files
ALL_C   = $(wildcard src/*.c)
ALL_CU  = $(wildcard src/*.cu)

# 4. Filter Sources
# Exclude extra mains and broken tests
EXCLUDE = src/accept.c \
		  src/anneal.c \
		  src/anneal_joint.c \
		  src/audit_pairs.c \
		  src/bounds.c \
		  src/collide_sat.c \
		  src/collide_tri_oracle.c \
		  src/energy.c \
		  src/export.c \
		  src/grid_hash.c \
		  src/instance_cache.c \
		  src/io.c \
		  src/main_packer.c \
		  src/main_unified.c \
		  src/metropolis.c \
		  src/propose.c \
		  src/rng.c \
		  src/solve_feasible.c \
		  src/solver.c \
		  src/solver_parallel.c \
		  src/solver_serial.c \
		  src/test_invariants.c \
		  src/test_soa_logic.c

SRCS_C  = $(filter-out $(EXCLUDE), $(ALL_C))
SRCS_CU = $(ALL_CU)
OBJS    = $(SRCS_C:.c=.o) $(SRCS_CU:.cu=.o)

# 4. Main Target
TARGET  = chimera_run

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@"
	$(CC) $(ARCH) $(OBJS) -o $@ $(LDFLAGS)

# 5. Compilation Rules
# Compile .c files (Host)
%.o: %.c
	@echo "Compiling Host Code: $<"
	$(CC) $(CFLAGS) -Xcompiler -std=c99 $(ARCH) -c $< -o $@

# Compile .cu files (Device)
%.o: %.cu
	@echo "Compiling Device Code: $<"
	$(CC) $(CFLAGS) $(ARCH) -c $< -o $@

# 6. Cleanup
clean:
	rm -f src/*.o $(TARGET)
