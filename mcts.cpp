#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
using namespace std;


bool MCTS::checkAbort(){
    // if(!abort){
    //     uint64_t diff;
	// 	clock_gettime(CLOCK_REALTIME, &end);
	// 	diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
	// 	abort = diff / MILLION > MAX_TIME;
    //     // cout << diff / MILLION << endl;
    // }
    // return abort;
    return false;
}


Action MCTS::run(Logger& logger){
    logger.setCPU();
    Board b;
    clock_gettime(CLOCK_REALTIME, &start);
    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }
    int step = 0;
    while(step < MAX_EXPAND_STEP){
        // cout << "traverse step:" << cstep << endl;
        traverse(root, init_path, b);
        step += 1;
        if(checkAbort()) break;
    }
    Action bestMove(0,0);
    double maxv = 0;
    for(auto child : root->children){
        double v = child->score / (child->n + EPSILON);
        // cout << "node:" << child << " move:" << child->path.back().y <<  child->path.back().x  << " times:" << child->n << " value:" << v << endl;
        if(v >= maxv){
            maxv = v;
            bestMove = child->path.back();
        }
    }
    // get time:
    uint64_t diff;
    clock_gettime(CLOCK_REALTIME, &end);
    diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
    // cout << "time used:" << diff / MILLION << endl;
    logger.log("time used:" + to_string(diff/MILLION));
    logger.record_time(diff/MILLION);
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
                // backprop(node, BackPropObj(simulate(node)));
            } else{
                Node* next = select(node);
                S.push(next);
            }
        } else{
            node->expandable = false;
            expand(node);
            BackPropObj bp;
            for(auto child : node->children){   // NOTE: can be parallelized
                for(int i = 0; i < SIM_TIMES; i++){
                    Result r = simulate(child);
                    bp.add(r);
                }
                backprop(child, bp);
            }
            // cout << "role: " << "updated node:" << node << " with wins:" << bp.wins << "and sims:" << bp.sims << endl;
            
        }
        if(checkAbort()) return;
    }
}

Node* MCTS::select(Node* node){
    double maxn = -1;
    Node* child = nullptr;   
    for(auto c : node->children){
        double UCB = c->score/(c->n + EPSILON) + 2 * sqrt(log(node->n+EPSILON)/(c->n + EPSILON));
        // cout << "child: " << c << " UCB: " << UCB << " first part:" << c->score/(c->n + EPSILON) << " second part:" << 2 * sqrt(log(node->n+EPSILON)/(c->n + EPSILON)) <<  endl;
        if(UCB > maxn){
            child = c;
            maxn = UCB;
        }
    }
    // cout << "select " << child << " of " << node << " with UCB " << maxn << endl;
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

void MCTS::backprop(Node *node, BackPropObj result){
        // cout << "enter backprop" << endl;
    bool shouldUpdate = true;
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
    ROLE role = b.current_role;
    int step = 0;
    while(step < MAX_SIM_STEP){
        step++;
        if(!rollout(b)){
            return b.get_result(role);
        }
        if(checkAbort()) return Result::DRAW;
    }
    return Result::DRAW;
}
bool MCTS::rollout(Board &b){
    vector<Action> actions = b.get_actions();
    if(actions.empty()) return false;
    b.update(actions[rand()%actions.size()]);
    return true;
}