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
#include <time.h>
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
    Action run(); // get an optimal action.
    Result simulate(Node *root); // simulate a node
    Node* select(Node* root);
    void expand(Node* node);
    void backprop(Node* node, BackPropObj result);
    bool rollout(Board &b); // given a board and role, randomly simulate one step
    void traverse(Node *root, vector<Action> &path, Board& board); // traverse the tree to find a node to simulate
    bool checkAbort();
};

#endif