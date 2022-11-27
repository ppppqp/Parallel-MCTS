#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
#include <curand_kernel.h>
using namespace std;


Action MCTS::run(Logger& logger){
    logger.setGPU();
    clock_gettime(CLOCK_REALTIME, &start);
    Board b;    // NOTE: duplicate of the board in main. Can we remove it?

    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }
    int step = 0;
    while(step < MAX_EXPAND_STEP){
        // cout << "traverse step:" << cstep << endl;
        traverse(root, init_path, b);   // TODO: add another CUDA version
        step += 1;
    }
    Action bestMove(0,0);
    double maxv = 0;
    for(auto child : root->children){
        double v = child->score / (child->n + EPSILON);
        if(v >= maxv){
            maxv = v;
            bestMove = child->path.back();
        }
    }
    // get the best move and return


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
    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(BOARD_SIZE, BOARD_SIZE, 1);
    while(!S.empty()){
        // cout << iter_step << endl;
        iter_step++;
        Node* node = S.top();
        
        S.pop();
        // Node *child = nullptr;

        if(!node->expandable){
            if(node->children.empty()){
                // this is an terminal state
                backprop(node, BackPropObj(simulate(node)));
            } else{
                S.push(select(node));
            }
        } else{
            node->expandable = false;
            expand(node);

            uint16_t * d_path;
            int path_len = node->path.size();
            uint16_t * d_children;
            int children_len = node->children.size();
            int *d_result;
            cudaMalloc(&d_path, path_len*sizeof(uint16_t));
            cudaMalloc(&d_children, children_len*sizeof(uint16_t));
            cudaMalloc(&d_result, 2 * sizeof(int));
            
            uint16_t* children_buffer = new uint16_t[children_len];
            for(int i = 0; i < children_len; i++){
                Action a = node->children[i]->path.back();
                children_buffer[i] = (a.x << 8) + a.y;
            }


            uint16_t* path_buffer = new uint16_t[path_len];
            for(int i = 0; i < path_len; i++){
                Action a = node->path[i];
                path_buffer[i] = (a.x << 8) + a.y;
            }


            cudaMemcpy( d_children, children_buffer, children_len*sizeof(uint16_t), cudaMemcpyHostToDevice);
            cudaMemcpy( d_path, path_buffer, path_len*sizeof(uint16_t), cudaMemcpyHostToDevice);

            int* result_buffer = new int[2];
            simulate_kernel<<<DimGrid, DimBlock>>>(d_path, path_len, d_children, children_len, d_result);
            cudaMemcpy( result_buffer, d_result, 2*sizeof(int), cudaMemcpyDeviceToHost);

            BackPropObj obj;
            obj.wins = result_buffer[0];
            obj.sims = result_buffer[1];
            backprop(node, obj);
            
            cudaFree(d_path);
            cudaFree(d_children);
            cudaFree(d_result);
            delete[] children_buffer;
            delete[] path_buffer;
            delete[] result_buffer;
        }
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