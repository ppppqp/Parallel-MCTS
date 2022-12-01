#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
#include <curand_kernel.h>
using namespace std;

#define D_NONE 0
#define D_WHITE 1
#define D_BLACK 2

__device__ void board_initialize(uint16_t *path, int path_len, uint8_t *s_board, ROLE* current_role){
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

    __syncthreads();
    // Let one thread do all the initialization of the board
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < path_len; ++i) {
            uint8_t act_x = (uint8_t)(path[i] >> 8) & 0xFFU;
            uint8_t act_y = (uint8_t)path[i] & 0xFFU;

            update_board(s_board, act_x, act_y, current_role);
        }
    }
    __syncthreads();
}

__device__ void expand_device(uint8_t *s_board, ROLE *role, uint16_t *children, int *children_len){
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j == 0 && i == 0) {
        *children_len = 0;
    }
    __syncthreads();
    int index = 0;
    ROLE current_role = *role;
    int myStone = (current_role == ROLE::BLACK) ? D_BLACK : D_WHITE;
    int opponentStone = (current_role == ROLE::BLACK) ? D_WHITE : D_BLACK;
    if (i < BOARD_SIZE && j < BOARD_SIZE){
        if(s_board[i*BOARD_SIZE+j] == myStone){
            // top
            int y = i-1;
            int x = j;
            while(y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                y--;
            }
            if(y >= 0 && y != i-1 &&  s_board[y*BOARD_SIZE+x] == D_NONE) {
                children[index++] = (x << 8) + y;
            }

            // bottom
            y = i+1;
            x = j;
            while(y < BOARD_SIZE && s_board[y*BOARD_SIZE+x] == opponentStone){
                y++;
            }
            if(y < BOARD_SIZE && y != i+1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;

            // right
            y = i;
            x = j+1;
            while(x < BOARD_SIZE && s_board[y*BOARD_SIZE+x] == opponentStone){
                x++;
            }
            if(x < BOARD_SIZE && x != j+1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;

            // left
            y = i;
            x = j-1;
            while(x >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                x--;
            }
            if(x >= 0 && x != j-1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;


            // top left
            y = i-1;
            x = j-1;
            while(x >= 0 && y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                x--;
                y--;
            }
            if(x >= 0 && y >= 0 && x != j-1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;

            // top right
            y = i-1;
            x = j+1;
            while(x < BOARD_SIZE && y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                x++;
                y--;
            }
            if(x < BOARD_SIZE && y >= 0 && x!= j+1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;

            // bottom left
            y = i+1;
            x = j-1;
            while(x >= 0 && y < BOARD_SIZE && s_board[y*BOARD_SIZE+x] == opponentStone){
                x--;
                y++;
            }
            if(x >= 0 && y < BOARD_SIZE && x != j-1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;


            // bottom right
            y = i+1;
            x = j+1;
            while(x >= 0 && y >= 0 && s_board[y*BOARD_SIZE+x] == opponentStone){
                x++;
                y++;
            }
            if(x < BOARD_SIZE && y < BOARD_SIZE && x != j+1 && s_board[y*BOARD_SIZE+x] == D_NONE) children[index++] = (x << 8) + y;
        }
    }
    *children_len = index;
}

__device__ void backprop_device(int *score, int *n, int level, int new_score, int new_n){
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < level) {
        bool shouldUpdate = (level - tid) % 2 == 0 ? true : false;
        if (shouldUpdate) {
            score[tid] += new_score;
        }
        n[tid] += new_n;
    }
}

// path
__global__ void traverse_kernel(uint16_t *path, int path_len, uint16_t *children, uint *children_len, int *score, int *n){
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ int s_cur_score[BOARD_SIZE * BOARD_SIZE];
    __shared__ int s_cur_n[BOARD_SIZE * BOARD_SIZE];

    __shared__ int s_score[BOARD_SIZE * BOARD_SIZE];
    __shared__ int s_n[BOARD_SIZE * BOARD_SIZE];

    if (tid < BOARD_SIZE * BOARD_SIZE) {
        s_score[tid] = 0;
        s_n[tid] = 0;
    }
    __syncthreads();

    __shared__ uint8_t s_board[BOARD_SIZE * BOARD_SIZE];
    __shared__ ROLE s_current_role;

    if(threadIdx.x == 0 && threadIdx.y == 0){
        s_current_role = ROLE::WHITE;
    }
    __syncthreads();
    
    board_initialize(path, path_len, s_board, &s_current_role);

    __shared__ uint16_t s_children[BOARD_SIZE * BOARD_SIZE];
    __shared__ int s_children_len;

    if (tid == 0)
        expand_device(s_board, &s_current_role, s_children, &s_children_len); // write to children
    __syncthreads();

    int level = 0;

    __shared__ int s_result[2];

    // while (level < 10 && s_children_len > 0) {
    // while (s_children_len != 0) {
        // simulate_device(s_board, &s_current_role, s_children, s_children_len, s_cur_score, s_cur_n, s_result);
        __syncthreads();
        if (tid == 0) {
            s_score[level] += s_result[0];
            s_n[level] += s_result[1];
        }

        if (level == 0) {
            if (tid < s_children_len){
                children[tid] = s_children[tid];
                score[tid] += s_cur_score[tid];
                n[tid] += s_cur_n[tid];
            }
            if (tid == 0) *children_len = s_children_len;
        }
/*        __syncthreads();

        // select
        if (tid == 0) {
            uint16_t child = s_children[0];
            double maxn = 0.0;
            int parent_score = score[level];
            for(int i = 0; i < s_children_len; ++i){
                double UCB = s_cur_score[i]/(s_cur_n[i] + EPSILON) + C * sqrt(log(parent_score + EPSILON)/(s_cur_n[i] + EPSILON));
                if(UCB > maxn){
                    child = s_children[i];
                    maxn = UCB;
                }
            }
            uint8_t act_x = (child >> 8) & 0xFFU;
            uint8_t act_y = child & 0xFFU;
            // update_board(s_board, act_x, act_y, &s_current_role);
            // expand_device(s_board, &s_current_role, s_children, &s_children_len);
        }
        __syncthreads();
        // backprop_device(s_score, s_n, level, s_result[0], s_result[1]);
        __syncthreads();
        level++;
    // }
*/
}


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


__device__ void simulate_device(uint8_t *s_board, ROLE *current_role, uint16_t *children, int children_len, int *win, int *sim, int*result) {

    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // shared memory to update the total wins on the path
    __shared__ int s_win[BOARD_SIZE * BOARD_SIZE];
    __shared__ int s_sim[BOARD_SIZE * BOARD_SIZE];
    for (int s = 0; tid + s < BOARD_SIZE * BOARD_SIZE; s += blockDim.x * blockDim.y) {
        s_win[tid + s] = 0;
        s_sim[tid + s] = 0;
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
        update_board(p_board, child_x, child_y, current_role);
        // every thread gets a new private ROLE variable
        ROLE p_role = *current_role;
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
    __syncthreads();
    if (tid < children_len) {
        uint8_t tmp_x = (children[tid] >> 8) & 0xFFU;
        uint8_t tmp_y = children[tid] & 0xFFU;
        win[tid] = s_win[tmp_y * BOARD_SIZE + tmp_x];
        sim[tid] = s_sim[tmp_y * BOARD_SIZE + tmp_x];
    }
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

    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    // every block shares an initial board
    __shared__ uint8_t s_board[BOARD_SIZE * BOARD_SIZE];
    __shared__ ROLE current_role;

    if(tid == 0){
        current_role = ROLE::WHITE;
    }
    __syncthreads();
    
    board_initialize(path, path_len, s_board, &current_role);

    // s_win and s_sim are not used by this kernel
    __shared__ int s_win[BOARD_SIZE * BOARD_SIZE];
    __shared__ int s_sim[BOARD_SIZE * BOARD_SIZE];

    simulate_device(s_board, &current_role, children, children_len, s_win, s_sim, result);

}