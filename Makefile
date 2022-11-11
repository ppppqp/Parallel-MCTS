# Compiler
CXX = g++
FLAGS = -O3
# PROG = multi_gpu
# SRC = multi_gpu.cu

PROG = mcts
SRC = main.cpp mcts.cpp
$(PROG):$(OBJS)
	$(CXX) $(FLAGS) $(SRC) -o $(PROG)

all:$(PROG)

clean:
	rm -rf $(PROG) *.o slurm*.out
