#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
using namespace std;
const int MAX_SIM_STEP = 100;
const int MAX_EXPAND_STEP = 100;


// bool MCTS::shouldAbort(){

// }
Action MCTS::run(){
    Board b;
    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }
    int step = 0;
    while(step < MAX_EXPAND_STEP){
        traverse(root, init_path, b);
        step += 1;
    }
    Action bestMove(0,0,ROLE::WHITE);
    // get the best move and return
    return bestMove;
}



void MCTS::traverse(Node *root, vector<Action> &path, Board &b){

    stack<Node*> S;
    S.push(root);
    while(!S.empty()){
        Node* node = S.top();
        Node *child = nullptr;
        
        if(!node->expandable()){
            // selection
            double maxn = 0;   
            for(auto c : node->children){
                double UCB = c->UCB;
                if(UCB > maxn){
                    child = c;
                    maxn = UCB;
                }
            }
            S.push(child);
        } else{
            expand(node);
            backprop(node, simulate(node));
        }
    }



    // double maxn = 0;
    // for(auto c: root->children){
    //     // SELECT PHASE
    //     double UCB = c->UCB;
    //     if(UCB > maxn){
    //         child = c;
    //         maxn = UCB;
    //     }
    // }
    // path.push_back(child->action);
    // b.update(child->action);
    // if(b.get_actions(ROLE::WHITE).empty()){
    //     // this is a terminal state
    //     return b.get_result();
    // }

    // if(!child->children.empty()){
    //     // this child is not a leaf
    //     // traverse this child
    //     // BACK_PROP the score back to the root
    //     Result r = MCTS::traverse(child, path, b);
    //     if(r == Result::WIN) child->score += 1;
    //     child->n += 1;
    //     child->update_UCB();
    // }
    // // this child is not a leaf
    // Result r;
    // if(child->n == 0){
    //     // this child has not been investigated before
    //     // SIMULATE PHASE
    //     r = MCTS::simulate(child, path);
    //     if(r == Result::WIN) child->score += 1;
    //     child->n += 1;
    //     return r;

    // } else{
    //     // this child has been investigated
    //     for(auto action : b.get_actions(ROLE::BLACK)){
    //         Node *new_node = new Node(action);
    //         child->children.push_back(new_node);
    //         r = MCTS::simulate(new_node, path);
    //         if(r == Result::WIN) child->score += 1;
    //         child->n += 1;
    //         return r;
    //     }
    // }
    // return r;
}

void MCTS::expand(Node * node){
    Board b;
    b.batch_update(node->path);
    vector<Action> actions = b.get_actions(ROLE::BLACK);
    for(auto action : actions){
        node->add_child(new Node(node->path, action));
    }
}

void MCTS::backprop(Node *node, Result result){
    bool shouldUpdate = false;
    while(node->parent){
        node = node->parent;
        if(result == Result::WIN) node->score += 1;
        node->n += 1;
        shouldUpdate = !shouldUpdate;
    }
}


Result MCTS::simulate(Node *root){
    Board b;
    ROLE role = ROLE::BLACK;

    for(auto action:root->path){
        b.update(action);
    }
    int step = 0;
    while(step < MAX_SIM_STEP){
        step++;
        if(!rollout(b, role)){
            // reach terminal state
            return b.get_result();
        }
        role = (role == ROLE::BLACK) ? ROLE::WHITE : ROLE::BLACK;
    }
    return Result::DRAW;
}
bool MCTS::rollout(Board &b, ROLE role){
    vector<Action> actions = b.get_actions(role);
    if(actions.empty()) return false;
    shuffle(actions.begin(), actions.end(), std::default_random_engine(42));
    b.update(actions[0]);
    return true;
}