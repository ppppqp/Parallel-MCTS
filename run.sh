#!/bin/bash
#SBATCH --job-name=example1
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:01:00
#SBATCH --account=eecs587f22_class
#SBATCH --partition=gpu




#make sure to load the cuda module before running
module load cuda
#make sure to compile your program using nvcc
#nvcc -o example1 example1.cu
# nvprof --print-gpu-trace -f -o results.nvprof ./mcts_cuda
# nsys profile ./mcts_cuda
./mcts_cuda
# ./mcts_cuda