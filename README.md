# Parallelized Monte Carol Tree Search
This is the repo for EECS 587 course project.

## Prerequisit
- gcc 10.3.0^
- CUDA 11.6^


## Versions
|Version|Branch|Compile Command|Run Command| Verify command
|---|---|---|---|---|
|Baseline|single-gpu | `make` | `./mcts` | `./verify.sh`|
|Parallel-Simulation| single-gpu | `make cuda` | `./mcts_cuda` | `./verify.sh -g`
|Completely Parallel| multi-thread|`make cuda`| `./mcts_cuda` | `./verify.sh -g`
|Heterogeous Parallel | hybrid | `make cuda`| `./mcts` |`./verify.sh -g`


