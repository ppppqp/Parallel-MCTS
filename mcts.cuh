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
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
const int MAX_SIM_STEP = 100;
const int SIM_TIMES = 20;
const int MAX_EXPAND_STEP = 70;
const int MILLION = 1000000;
const long long BILLION = 1000000000;
const int MAX_TIME = 1000; // each step takes 1 second
const int SPECULATE_NUM = 2;
using namespace std;
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
    mutex node_lock;
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
    void traverse(Node *root, vector<Action> &path, Board& board, int tid, cudaStream_t& stream); // traverse the tree to find a node to simulate
    bool checkAbort();
};

#endif