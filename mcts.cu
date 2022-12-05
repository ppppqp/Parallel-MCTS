#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
#include <curand_kernel.h>
using namespace std;

#define GPU_PERCENT 0.5

Action MCTS::run(Logger& logger){
    logger.setGPU();
    clock_gettime(CLOCK_REALTIME, &start);
    Board b;    // NOTE: duplicate of the board in main. Can we remove it?

    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }

    uint16_t * d_path;
    int path_len = init_path.size();
    uint16_t * d_children;
    uint * d_children_len;
    int * d_score;
    int * d_n;
    cudaMalloc(&d_path, path_len*sizeof(uint16_t));
    cudaMalloc(&d_children, BOARD_SIZE*BOARD_SIZE*sizeof(uint16_t));
    cudaMalloc(&d_children_len, sizeof(uint));
    cudaMalloc(&d_score, BOARD_SIZE*BOARD_SIZE*sizeof(int));
    cudaMalloc(&d_n, BOARD_SIZE*BOARD_SIZE*sizeof(int));

    uint16_t* path_buffer = new uint16_t[path_len];
    for(int i = 0; i < path_len; i++){
        Action a = init_path[i];
        path_buffer[i] = (a.x << 8) + a.y;
    }

    uint16_t * h_children = new uint16_t[BOARD_SIZE*BOARD_SIZE];
    uint h_children_len = 0;
    int * h_score = new int[BOARD_SIZE*BOARD_SIZE];
    int * h_n = new int[BOARD_SIZE*BOARD_SIZE];

    cudaMemcpy(d_path, path_buffer, path_len*sizeof(uint16_t), cudaMemcpyHostToDevice);

    int GPU_work = (int)((double)MAX_EXPAND_STEP * GPU_PERCENT);
    int CPU_work = MAX_EXPAND_STEP - GPU_work;

    dim3 DimGrid(1, 1, GPU_work);
    dim3 DimBlock(BOARD_SIZE, BOARD_SIZE, 1);
        
    traverse_kernel<<<DimGrid, DimBlock>>>(d_path, path_len, d_children, d_children_len, d_score, d_n);
    
    for (int i = 0; i < CPU_work; ++i) {
        traverse(root, init_path, b);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(&h_children_len, d_children_len, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_children, d_children, h_children_len*sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_score, d_score, h_children_len*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_n, d_n, h_children_len*sizeof(int), cudaMemcpyDeviceToHost);

    // cout << "h_children_len: " << h_children_len << endl;
    // for (int i = 0; i < h_children_len; ++i) {
    //     cout << (h_children[i] & 0xFF) << "," << ((h_children[i] >> 8) & 0xFF) << "  ";
    // }
    // cout << endl;

    Action bestMove(0,0);
    double maxv = 0;
    int best_i = 0;
    for(int i = 0; i < h_children_len; ++i){
        double v = (double)h_score[i] / ((double)h_n[i] + EPSILON);
        if(v >= maxv){
            maxv = v;
            best_i = i;
        }
    }
    bestMove.x = (h_children[best_i] >> 8) & 0xFFU;
    bestMove.y = h_children[best_i] & 0xFFU;
    // get the best move and return

    delete[] path_buffer;
    delete[] h_children;
    delete[] h_score;
    delete[] h_n;
    cudaFree(d_path);
    cudaFree(d_children);
    cudaFree(d_children_len);
    cudaFree(d_score);
    cudaFree(d_n);

    // get time
    uint64_t diff;
    clock_gettime(CLOCK_REALTIME, &end);
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    logger.log("time used:" + to_string(diff/MILLION));
    logger.record_time(diff/MILLION);

    return bestMove;
}


void MCTS::traverse(Node *root, vector<Action> &path, Board &b){
    stack<Node*> S;
    S.push(root);
    int iter_step = 0;
    while(!S.empty()){
        // cout << iter_step << endl;
        iter_step++;
        Node* node = S.top();
        
        S.pop();
        // Node *child = nullptr;
        if(!node->expandable){
            if(node->children.empty()){
                // this is an terminal state
                backprop(node, simulate(node));
            } else{
                S.push(select(node));
            }
        } else{
            node->expandable = false;
            expand(node);

            for(auto child : node->children){
                backprop(node, simulate(child));
            }
        }
        // if(checkAbort()) return;
    }
}

Node* MCTS::select(Node* node){
    // cout << "enter select" << endl;
    double maxn = -1;
    Node* child = nullptr;   
    for(auto c : node->children){
        // cout << c << endl;
        double UCB = c->UCB;
        if(UCB > maxn){
            child = c;
            maxn = UCB;
        }
    }
    // cout << child << endl;
    // cout << "exit select" << endl;
    return child;
}

void MCTS::expand(Node * node){
    Board b;
    b.batch_update(node->path);
    vector<Action> actions = b.get_actions();
    // cout << "action size" << actions.size() << endl;
    for(auto action : actions){
        // cout << act_y << act_x << endl;
        node->add_child(new Node(node->path, action));
    }
}


void MCTS::backprop(Node *node, BackPropObj result){
        // cout << "enter backprop" << endl;
    bool shouldUpdate = false;
    while(node->parent){
        node = node->parent;
        if(shouldUpdate) node->score += result.wins;
        node->n += result.sims;
        shouldUpdate = !shouldUpdate;
    }
}


Result MCTS::simulate(Node *root){
    Board b;
    // cout << "enter simulate" << endl;

    for(auto action:root->path){
        b.update(action);
    }
    int step = 0;
    while(step < MAX_SIM_STEP){
        step++;
        if(!rollout(b)){
            return b.get_result();
        }
    }
    return Result::DRAW;
}
bool MCTS::rollout(Board &b){
    vector<Action> actions = b.get_actions();
    if(actions.empty()) return false;
    shuffle(actions.begin(), actions.end(), std::default_random_engine(42));
    b.update(actions[0]);
    return true;
}