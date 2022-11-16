#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
using namespace std;

#define MAX_SIM_STEP 100
#define MAX_EXPAND_STEP 100

#define BOARD_W 9
#define BLOCK_W 9

#define D_NONE 0
#define D_WHITE 1
#define D_BLACK 2

// act:
//  15:8 = x
//   7:0 = y
__device__ uint16_t git_random_action(){
    uint16_t actions[BOARD_W * BOARD_W];
    uint16_t act = 0;


    return act;
}



__device__ void update_board(uint8_t *s_board, uint8_t act_x, uint8_t act_y, ROLE role){
    uint8_t myStone = (role == ROLE::BLACK) ? D_BLACK : D_WHITE;
    uint8_t opponentStone = (role == ROLE::BLACK) ? D_WHITE : D_BLACK;

    uint8_t y = 0;
    uint8_t x = 0;

    // top
    y = act_y-1;
    x = act_x;
    while(y >= 0 && s_board[y * BOARD_W + x] == opponentStone){
        y--;
    }
    if(y >= 0 && s_board[y * BOARD_W + x] == myStone){
        for(int i = act_y-1; i > y; i--){
            s_board[i * BOARD_W + x] = myStone;
        }
    }

    // bottom
    y = act_y+1;
    x = act_x;
    while(y < BOARD_W && s_board[y * BOARD_W + x] == opponentStone){
        y++;
    }
    if(y < BOARD_W && s_board[y * BOARD_W + x] == myStone){
        for(int i = act_y+1; i < y; i++){
            s_board[i * BOARD_W + x] = myStone;
        }
    }

    // right
    y = act_y;
    x = act_x+1;
    while(x < BOARD_W && s_board[y * BOARD_W + x] == opponentStone){
        x++;
    }
    if(x < BOARD_W && s_board[y * BOARD_W + x] == myStone){
        for(int i = act_x+1; i < x; i++){
            s_board[y * BOARD_W + i] = myStone;
        }
    }

    // left
    y = act_y;
    x = act_x-1;
    while(x >= 0 && s_board[y * BOARD_W + x] == opponentStone){
        x--;
    }
    if(x >= 0 && s_board[y * BOARD_W + x] == myStone){
        for(int i = act_x-1; i > x; i--){
            s_board[y * BOARD_W + i] = myStone;
        }
    }

    // top-left
    y = act_y-1;
    x = act_x-1;
    int count = 0;
    while(x >= 0 && y >= 0 && s_board[y * BOARD_W + x] == opponentStone){
        x--;
        y--;
        count ++;
    }
    if(x >=0 && y >= 0 && s_board[y * BOARD_W + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y-1-i) * BOARD_W + act_x-1-i] = myStone;
        }
    }

    // top-right
    y = act_y-1;
    x = act_x+1;
    count = 0;
    while(x < BOARD_W && y >= 0 && s_board[y * BOARD_W + x] == opponentStone){
        x++;
        y--;
        count ++;
    }
    if(x < BOARD_W && y >= 0 && s_board[y * BOARD_W + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y-1-i) * BOARD_W + act_x+1+i] = myStone;
        }
    }

    // bottom-right
    y = act_y+1;
    x = act_x+1;
    count = 0;
    while(x < BOARD_W && y < BOARD_W && s_board[y * BOARD_W + x] == opponentStone){
        x++;
        y++;
        count ++;
    }
    if(x < BOARD_W && y < BOARD_W && s_board[y * BOARD_W + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y+1+i) * BOARD_W + act_x+1+i] = myStone;
        }
    }

    // bottom-left
    y = act_y+1;
    x = act_x-1;
    count = 0;
    while(x >=0 && y < BOARD_W && s_board[y * BOARD_W + x] == opponentStone){
        x--;
        y++;
        count ++;
    }
    if(x >= 0 && y < BOARD_W && s_board[y * BOARD_W + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y+1+i) * BOARD_W + act_x-1-i] = myStone;
        }
    }
}

// INPUTS:
//  path[i][15:8]: act_x
//  path[i][ 7:0]: act_y
//  children: the action added for each child, same decode as path
// OUTPUTS:
//  win: the number of wins (new results from the simulation) for every node on the path
__global__ simulate_kernel(uint16_t *path, int path_len, 
                           uint16_t *children, int children_len,
                           int *win){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory to update the total wins on the path
    __shared__ int s_win[BOARD_W * BOARD_W];
    
    for (int s = 0; tid + s < BOARD_W * BOARD_W; s += blockDim.x * blockDim.y) {
        s_win[tid + s] = 0;
    }
    __syncthreads();

    // every block shares an initial board
    __shared__ uint8_t s_board[BOARD_W * BOARD_W];

    for (int s_x = 0; threadIdx.x + s_x < BOARD_W; s_x += blockDim.x) {
        for (int s_y = 0; threadIdx.y + s_y < BOARD_W; s_y += blockDim.y) {
            int tsy = threadIdx.y + s_y;
            int tsx = threadIdx.x + s_x
            s_board[tsy * BOARD_W + tsx] = D_NONE;
            if ((threadIdx.y + s_y == 3 && threadIdx.x + s_x == 3) || 
                (threadIdx.y + s_y == 4 && threadIdx.x + s_x == 4)
                s_board[tsy * BOARD_W + tsx] = D_BLACK;
            if ((threadIdx.y + s_y == 3 && threadIdx.x + s_x == 4) || 
                (threadIdx.y + s_y == 4 && threadIdx.x + s_x == 3)
                s_board[tsy * BOARD_W + tsx] = D_WHITE;
        }
    }
    __syncthreads();

    ROLE current_role = ROLE::WHITE;

    // Let one thread do all the initialization of the board
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < path_len; ++i) {
            uint8_t act_x = (uint8_t)(path[i] >> 8) & 0xFFU;
            uint8_t act_y = (uint8_t)path[i] & 0xFFU;

            update_board(s_board, act_x, act_y, current_role);

            current_role = (current_role == ROLE::WHITE) ? ROLE::BLACK : ROLE::WHITE;   // flip the role
        }
    }

    
Board b;
    // cout << "enter simulate" << endl;

    for(auto action:root->path){
        b.update(action);
    }
    int step = 0;
    while(step < MAX_SIM_STEP){
        step++;
        vector<Action> actions = b.get_actions();
        if(actions.empty()) return false;
        shuffle(actions.begin(), actions.end(), std::default_random_engine(42));
        b.update(actions[0]);
        return true;
        if(!rollout(b)){
            return b.get_result();
        }
    }
    return Result::DRAW;
}


Action MCTS::run(){
    Board b;    // NOTE: duplicate of the board in main. Can we remove it?

    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }
    int step = 0;
    while(step < MAX_EXPAND_STEP){
        // cout << "traverse step:" << cstep << endl;
        traverse(root, init_path, b);
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
        Node *child = nullptr;
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

            for(auto child : node->children){   // NOTE: can be parallelized
                backprop(node, simulate(child));
            }
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

void MCTS::backprop(Node *node, Result result){
        // cout << "enter backprop" << endl;
    bool shouldUpdate = false;
    while(node->parent){
        node = node->parent;
        if(result == Result::WIN) node->score += 1;
        node->n += 1;
        shouldUpdate = !shouldUpdate;   // NOTE: not used?
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