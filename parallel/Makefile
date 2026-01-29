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
# sm_70 = V100
# sm_75 = T4 (common on Colab/AWS G4)
# sm_80 = A100
ARCH    = -arch=sm_75

# 3. Source Discovery
# Automatically find all .c and .cu files in src/
SRCS_C  = $(wildcard src/*.c)
SRCS_CU = $(wildcard src/*.cu)
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
	$(CC) $(CFLAGS) $(ARCH) -c $< -o $@

# Compile .cu files (Device)
%.o: %.cu
	@echo "Compiling Device Code: $<"
	$(CC) $(CFLAGS) $(ARCH) -c $< -o $@

# 6. Cleanup
clean:
	rm -f src/*.o $(TARGET)
