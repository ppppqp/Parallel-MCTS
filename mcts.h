#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


#ifndef MCTS_H
#define MCTS_H
#include <vector>
#include "tree.h"
#include "game.h"
#include "logger.h"
#include <time.h>
#include <curand_kernel.h>

const double GPU_PERCENT = 0.9;
const int MAX_SIM_STEP = 100;
const int SIM_TIMES = 20;
const int MAX_EXPAND_STEP = 70;
const int MILLION = 1000000;
const long long BILLION = 1000000000;
const int MAX_TIME = 1000; // each step takes 1 second

using namespace std;

__device__ void backprop_device(int *score, int *n, int level, int new_score, int new_n);
__device__ void board_initialize(uint16_t *path, int path_len, uint8_t *s_board, ROLE* current_role);
__device__ void expand_device(uint8_t *s_board, ROLE *role, uint16_t *children);
__device__ void use_this(curandState *state, uint16_t* act, int* count, uint8_t x, uint8_t y );
__device__ uint16_t get_random_action(uint8_t *s_board, ROLE *role);
__device__ void update_board(uint8_t *s_board, uint8_t act_x, uint8_t act_y, ROLE *role);
__device__ Result get_result(uint8_t *s_board);
__device__ void simulate_device(uint8_t *s_board, ROLE *current_role, uint16_t *children, int children_len, int *win, int *sim, int*result);
__global__ void simulate_kernel(uint16_t *path, int path_len, uint16_t *children, int children_len, int*result);

__global__ void traverse_kernel(uint16_t *path, int path_len, uint16_t *children, uint *children_len, int *score, int *n);

class MCTSProfiler{
public:
    int nodesTraversed;
    int nodesExpanded;
    int nodesSimulated;
    int totalSimulations;
};

class BackPropObj{
public:
    int wins;
    int sims;
    CUDA_HOSTDEV BackPropObj(Result r){
        wins = (r == Result::WIN) ? 1 : 0;
        sims = 1;
    }
    CUDA_HOSTDEV BackPropObj():wins(0), sims(0){};
    CUDA_HOSTDEV void add(Result r){
        wins += (r == Result::WIN) ? 1 : 0;
        sims += 1;
    }
};


class MCTS{
public:
    Node *root;
    vector<Action> init_path;
    Board init_board;
    bool abort;
    struct timespec start, end;
    MCTS(vector<Action>&path):init_path(path), abort(false){
        root = new Node(path);
    };
    MCTS(Board b): init_board(b), abort(false){

    };
    ~MCTS(){
        delete root;
    };
    Action run(Logger& logger); // get an optimal action.
    Result simulate(Node *root); // simulate a node
    Node* select(Node* root);
    void expand(Node* node);
    void backprop(Node* node, BackPropObj result);
    bool rollout(Board &b); // given a board and role, randomly simulate one step
    void traverse(Node *root, vector<Action> &path, Board& board); // traverse the tree to find a node to simulate
    bool checkAbort();
};

#endif