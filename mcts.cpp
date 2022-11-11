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
    while(!S.empty()){
        Node* node = S.top();
        Node *child = nullptr;
        
        if(!node->expandable){
            // selection

            S.push(select(root));
        } else{
            node->expandable = false;
            expand(node);
            for(auto child : node->children){
                backprop(node, simulate(child));
            }
        }
    }
}

Node* MCTS::select(Node* node){
    double maxn = 0;
    Node* child = nullptr;   
    for(auto c : node->children){
        double UCB = c->UCB;
        if(UCB > maxn){
            child = c;
            maxn = UCB;
        }
    }
    return child;
}

void MCTS::expand(Node * node){
    Board b;
    b.batch_update(node->path);
    vector<Action> actions = b.get_actions();
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

    for(auto action:root->path){
        b.update(action);
    }
    int step = 0;
    while(step < MAX_SIM_STEP){
        step++;
        if(!rollout(b)){
            // reach terminal state
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