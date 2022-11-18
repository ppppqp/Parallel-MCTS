# Compiler
NVCC = nvcc
CXX = g++
FLAGS = -O3
# PROG = multi_gpu
# SRC = multi_gpu.cu
CUDA_PROG = mcts_cuda
CUDA_SRC = main.cpp mcts.cu

PROG = mcts
SRC = main.cpp mcts.cpp
$(PROG):$(OBJS)
	$(NVCC) $(FLAGS) $(SRC) -pg -o $(PROG)

all:$(PROG)

cuda:
	$(NVCC) $(CUDA_SRC) -o $(CUDA_PROG)

clean:
	rm -rf $(PROG) $(CUDA_PROG) *.o slurm*.out
