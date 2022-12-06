#include "mcts.cuh"
#include "game.h"
#include "timer.h"
#include <stack>
#include <algorithm>    
#include <random>       
#include <thread>
#include <curand_kernel.h>
#include <mutex>
using namespace std;

#define D_NONE 0
#define D_WHITE 1
#define D_BLACK 2

const bool MULTITHREAD = false;
const int nStreams = 10;
// act:
//  15:8 = x
//   7:0 = y
// 0xFFFF: no act is available
__device__ void use_this(curandState *state, uint16_t* act, int* count, uint8_t x, uint8_t y ){
    *count += 1;
    if((int)(curand_uniform(state) * *count) % *count == 0){
        *act = (x << 8) + y;
    }
}


__device__ uint16_t get_random_action(uint8_t *s_board, ROLE *role){
    uint16_t act = 0xFFFFU;
    curandState state;
    ROLE current_role = *role;
    curand_init((unsigned int)clock64(), threadIdx.y * blockDim.x + threadIdx.x, 0, &state);
    int myStone = (current_role == ROLE::BLACK) ? D_BLACK : D_WHITE;
    int opponentStone = (current_role == ROLE::BLACK) ? D_WHITE : D_BLACK;
    int count = 0;
    for(int i = 0; i < BOARD_SIZE; i++){
        for(int j = 0; j < BOARD_SIZE; j++){
            if(s_board[i*BOARD_SIZE+j] == myStone){
                // top
                int y = i-1;
                int x = j;
                while(y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                    y--;
                }
                if(y >= 0 && y != i-1 &&  s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);

                // bottom
                y = i+1;
                x = j;
                while(y < BOARD_SIZE && s_board[y*BOARD_SIZE+x] == opponentStone){
                    y++;
                }
                if(y < BOARD_SIZE && y != i+1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);

                // right
                y = i;
                x = j+1;
                while(x < BOARD_SIZE && s_board[y*BOARD_SIZE+x] == opponentStone){
                    x++;
                }
                if(x < BOARD_SIZE && x != j+1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);

                // left
                y = i;
                x = j-1;
                while(x >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                    x--;
                }
                if(x >= 0 && x != j-1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);


                // top left
                y = i-1;
                x = j-1;
                while(x >= 0 && y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                    x--;
                    y--;
                }
                if(x >= 0 && y >= 0 && x != j-1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);

                // top right
                y = i-1;
                x = j+1;
                while(x < BOARD_SIZE && y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                    x++;
                    y--;
                }
                if(x < BOARD_SIZE && y >= 0 && x!= j+1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);

                // bottom left
                y = i+1;
                x = j-1;
                while(x >= 0 && y < BOARD_SIZE && s_board[y*BOARD_SIZE+x] == opponentStone){
                    x--;
                    y++;
                }
                if(x >= 0 && y < BOARD_SIZE && x != j-1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);


                // bottom right
                y = i+1;
                x = j+1;
                while(x >= 0 && y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                    x++;
                    y++;
                }
                if(x < BOARD_SIZE && y < BOARD_SIZE && x != j+1 && s_board[y*BOARD_SIZE+x] == D_NONE) use_this(&state, &act, &count, x, y);

            }
        }
    }
    return act;
}







__device__ void update_board(uint8_t *s_board, uint8_t act_x, uint8_t act_y, ROLE *role){
    uint8_t myStone = (*role == ROLE::BLACK) ? D_BLACK : D_WHITE;
    uint8_t opponentStone = (*role == ROLE::BLACK) ? D_WHITE : D_BLACK;

    int8_t y = 0;
    int8_t x = 0;

    // top
    y = act_y-1;
    x = act_x;
    while(y >= 0 && s_board[y * BOARD_SIZE + x] == opponentStone){
        y--;
    }
    if(y >= 0 && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = act_y-1; i > y; i--){
            s_board[i * BOARD_SIZE + x] = myStone;
        }
    }

    // bottom
    y = act_y+1;
    x = act_x;
    while(y < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == opponentStone){
        y++;
    }
    if(y < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = act_y+1; i < y; i++){
            s_board[i * BOARD_SIZE + x] = myStone;
        }
    }

    // right
    y = act_y;
    x = act_x+1;
    while(x < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == opponentStone){
        x++;
    }
    if(x < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = act_x+1; i < x; i++){
            s_board[y * BOARD_SIZE + i] = myStone;
        }
    }

    // left
    y = act_y;
    x = act_x-1;
    while(x >= 0 && s_board[y * BOARD_SIZE + x] == opponentStone){
        x--;
    }
    if(x >= 0 && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = act_x-1; i > x; i--){
            s_board[y * BOARD_SIZE + i] = myStone;
        }
    }

    // top-left
    y = act_y-1;
    x = act_x-1;
    int count = 0;
    while(x >= 0 && y >= 0 && s_board[y * BOARD_SIZE + x] == opponentStone){
        x--;
        y--;
        count ++;
    }
    if(x >=0 && y >= 0 && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y-1-i) * BOARD_SIZE + act_x-1-i] = myStone;
        }
    }

    // top-right
    y = act_y-1;
    x = act_x+1;
    count = 0;
    while(x < BOARD_SIZE && y >= 0 && s_board[y * BOARD_SIZE + x] == opponentStone){
        x++;
        y--;
        count ++;
    }
    if(x < BOARD_SIZE && y >= 0 && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y-1-i) * BOARD_SIZE + act_x+1+i] = myStone;
        }
    }

    // bottom-right
    y = act_y+1;
    x = act_x+1;
    count = 0;
    while(x < BOARD_SIZE && y < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == opponentStone){
        x++;
        y++;
        count ++;
    }
    if(x < BOARD_SIZE && y < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y+1+i) * BOARD_SIZE + act_x+1+i] = myStone;
        }
    }

    // bottom-left
    y = act_y+1;
    x = act_x-1;
    count = 0;
    while(x >=0 && y < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == opponentStone){
        x--;
        y++;
        count ++;
    }
    if(x >= 0 && y < BOARD_SIZE && s_board[y * BOARD_SIZE + x] == myStone){
        for(int i = 0; i < count; i++){
            s_board[(act_y+1+i) * BOARD_SIZE + act_x-1-i] = myStone;
        }
    }

    // flip the role
    *role = (*role == ROLE::WHITE) ? ROLE::BLACK : ROLE::WHITE;
}



__device__ Result get_result(uint8_t *s_board){
    int count = 0;
    for(int i = 0; i < BOARD_SIZE; i++){
        for(int j = 0; j < BOARD_SIZE; j++){
            if(s_board[i*BOARD_SIZE+j] == D_BLACK) count ++;
            if(s_board[i*BOARD_SIZE+j] == D_WHITE) count --;
        }
    }
    if(count > 0) return Result::WIN;
    if(count == 0) return Result::DRAW;
    if(count < 0) return Result::LOSE;
    return Result::DRAW;
}
// Every thread calculates one child
// INPUTS:
//  path[i][15:8]: act_x
//  path[i][ 7:0]: act_y
//  children: the action added for each child, same decode as path
// OUTPUTS:
//  win: the number of wins (new results from the simulation) for every node on the path
__global__ void simulate_kernel(uint16_t *path, int path_len, 
                           uint16_t *children, int children_len, int*result){

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory to update the total wins on the path
    __shared__ int s_win[BOARD_SIZE * BOARD_SIZE];
    __shared__ int s_sim[BOARD_SIZE * BOARD_SIZE];
    // for (int s = 0; tid + s < BOARD_SIZE * BOARD_SIZE; s += blockDim.x * blockDim.y) {
    //     s_win[tid + s] = 0;
    // }
    // __syncthreads();

    // every block shares an initial board
    __shared__ uint8_t s_board[BOARD_SIZE * BOARD_SIZE];

    for (int s_y = 0; threadIdx.y + s_y < BOARD_SIZE; s_y += blockDim.y) {
        for (int s_x = 0; threadIdx.x + s_x < BOARD_SIZE; s_x += blockDim.x) {
            int tsy = threadIdx.y + s_y;
            int tsx = threadIdx.x + s_x;
            s_board[tsy * BOARD_SIZE + tsx] = D_NONE;
            if ((threadIdx.y + s_y == BOARD_SIZE/2-1 && threadIdx.x + s_x == BOARD_SIZE/2-1) || 
                (threadIdx.y + s_y == BOARD_SIZE/2 && threadIdx.x + s_x == BOARD_SIZE/2))
                s_board[tsy * BOARD_SIZE + tsx] = D_BLACK;
            if ((threadIdx.y + s_y == BOARD_SIZE/2-1 && threadIdx.x + s_x == BOARD_SIZE/2) || 
                (threadIdx.y + s_y == BOARD_SIZE/2 && threadIdx.x + s_x == BOARD_SIZE/2-1))
                s_board[tsy * BOARD_SIZE + tsx] = D_WHITE;
        }
    }


    __shared__ ROLE current_role;
    if(threadIdx.x == 0 && threadIdx.y == 0){
        current_role = ROLE::WHITE;
    }

    __syncthreads();
    // Let one thread do all the initialization of the board
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < path_len; ++i) {
            uint8_t act_x = (uint8_t)(path[i] >> 8) & 0xFFU;
            uint8_t act_y = (uint8_t)path[i] & 0xFFU;

            update_board(s_board, act_x, act_y, &current_role);
        }
    }
    __syncthreads();

    // every thread gets a private copy of the board
    uint8_t p_board[BOARD_SIZE * BOARD_SIZE];
    for (int y = 0; y < BOARD_SIZE; ++y) {
        for (int x = 0; x < BOARD_SIZE; ++x) {
            // TODO: remove bank conflicts
            p_board[y * BOARD_SIZE + x] = s_board[y * BOARD_SIZE + x];
        }
    }

    // update the private board based on the child
    // every thread also gets a private copy of the children
    if (tid < children_len) {
        uint8_t child_x = (uint8_t)(children[tid] >> 8) & 0xFFU;
        uint8_t child_y = (uint8_t)(children[tid]) & 0xFFU;
        update_board(p_board, child_x, child_y, &current_role);
        // every thread gets a new private ROLE variable
        ROLE p_role = current_role;
        int step = 0;
        while(step < MAX_SIM_STEP){
            step++;
            uint16_t rand_act = get_random_action(p_board, &p_role);
            if (rand_act != 0xFFFFU) {
                uint8_t rand_x = (uint8_t)(rand_act >> 8) & 0xFFU;
                uint8_t rand_y = (uint8_t)(rand_act) & 0xFFU;
                update_board(p_board, rand_x, rand_y, &p_role);
            } else {    // game finishes
                // TODO: get result
                Result r = get_result(s_board);
                if(r == Result::WIN) s_win[child_y * BOARD_SIZE + child_x] ++;
                s_sim[child_y * BOARD_SIZE + child_x] ++;
            }
        }
    }
    __syncthreads();
    // reduce
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        result[0] = 0;
        result[1] = 0;
        for(int i = 0; i < BOARD_SIZE; i++){
            for(int j = 0; j < BOARD_SIZE; j++){
                result[0] += s_win[i*BOARD_SIZE + j];
                result[1] += s_sim[i*BOARD_SIZE + j];
            }
        }
    }

}


Action MCTS::run(Logger& logger){
    logger.setGPU();
    Board b;    // NOTE: duplicate of the board in main. Can we remove it?

    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }
    int step = 0;
    vector<thread> vt;

    Timer timer;
    timer.start();
    // create cuda streams
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i ++){
        cudaStreamCreate(&streams[i]);
    }

    while(step < MAX_EXPAND_STEP){
        if(MULTITHREAD){
            thread t(&MCTS::traverse, this, std::ref(root), std::ref(init_path), std::ref(b), step, std::ref(streams[step%nStreams]));
            vt.push_back(move(t));
        } else{
            traverse(root, init_path, b, step, streams[0]);
        }
        step += 1;
    }
    if(MULTITHREAD){
        for(thread& t : vt){
            if(t.joinable()) t.join();
        }
    }

    //delete cuda stream
    for (int i = 0; i < nStreams; i ++){
        cudaStreamDestroy(streams[i]);
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
    timer.stop();
    logger.log(to_string(timer.time()));
    logger.record_time(timer.time());

    return bestMove;
}



void MCTS::traverse(Node *root, vector<Action> &path, Board &b, int tid, cudaStream_t& stream){
    stack<Node*> S;
    S.push(root);
    Timer timer;
    timer.start();
    int iter_step = 0;
    dim3 DimGrid(BOARD_SIZE, BOARD_SIZE, 1);
    dim3 DimBlock(1, 1, 1);
    while(!S.empty()){
        // cout << iter_step << endl;
        iter_step++;
        Node* node = S.top();
        
        S.pop();
        // Node *child = nullptr;

        if(!node->expandable){
            if(node->children.empty()){
                // this is an terminal state
                node_lock.lock();
                backprop(node, BackPropObj(simulate(node)));
                node_lock.unlock();
            } else{
                S.push(select(node)[0]);
            }
        } else{
            node_lock.lock();
            node->expandable = false;
            expand(node);
            node_lock.unlock();

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


            cudaMemcpyAsync( d_children, children_buffer, children_len*sizeof(uint16_t), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync( d_path, path_buffer, path_len*sizeof(uint16_t), cudaMemcpyHostToDevice, stream);

            int* result_buffer = new int[2];
            simulate_kernel<<<DimGrid, DimBlock, 0, stream>>>(d_path, path_len, d_children, children_len, d_result);
            cudaMemcpyAsync( result_buffer, d_result, 2*sizeof(int), cudaMemcpyDeviceToHost, stream);

            BackPropObj obj;
            obj.wins = result_buffer[0];
            obj.sims = result_buffer[1];
            
            node_lock.lock();
            backprop(node, obj);
            node_lock.unlock();


            cudaFree(d_path);
            cudaFree(d_children);
            cudaFree(d_result);
            delete[] children_buffer;
            delete[] path_buffer;
            delete[] result_buffer;
        }
    }
    timer.stop();
}

deque<Node*> MCTS::select(Node* node){
    double maxn = -1;
    Node* child = nullptr;
    deque<Node*> v(SPECULATE_NUM, nullptr);
    int vsize = 0;   
    for(auto c : node->children){
        double UCB = c->UCB;
        if(UCB > maxn){
            child = c;
            maxn = UCB;
            if(vsize < SPECULATE_NUM){
                v[vsize] = c;
                vsize ++;
            }else{
                v.pop_back();
                v.push_front(c);
            }
        }
    }

    return v;
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