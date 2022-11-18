#include "mcts.h"
#include "game.h"
#include <stack>
#include <algorithm>    
#include <random>       
#include<thread>
using namespace std;
const int MAX_SIM_STEP = 100;
const int MAX_EXPAND_STEP = 100;
const int MILLION = 1000000;
const long long BILLION = 1000000000;
const int MAX_TIME = 1000; // each step takes 1 second
bool MCTS::checkAbort(){
    if(!abort){
        uint64_t diff;
		clock_gettime(CLOCK_REALTIME, &end);
		diff = BILLION * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
		abort = diff / MILLION > MAX_TIME;
        // cout << diff / MILLION << endl;
    }
    return abort;
}


Action MCTS::run(){
    Board b;
    clock_gettime(CLOCK_REALTIME, &start);
    for(auto action:init_path){
        // initialize the board with history actions
        b.update(action);
    }
    // int step = 0;
    while(true){
        // cout << "traverse step:" << cstep << endl;
        traverse(root, init_path, b);
        // step += 1;
        if(checkAbort()) break;
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
                backprop(node, BackPropObj(simulate(node, nullptr)));
            } else{
                S.push(select(node));
            }
        } else{
            node->expandable = false;
            expand(node);
            vector<thread> threads;
            vector<Result> results(node->children.size(), Result::DRAW);
            vector <Result>::iterator it = results.begin();
            for(auto child : node->children){
                // map to different threads
                thread t(MCTS::simulate, child, it);
                threads.push_back(move(t));
                it ++;
            }
            for(std::thread & th : threads){
                th.join();
            }
            // reduece and backprop
            BackPropObj bp;
            for(auto r: results){
                bp.add(r);
            }
            backprop(node, bp);
        }
        if(checkAbort()) return;
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
        // cout << action.y << action.x << endl;
        node->add_child(new Node(node->path, action));
    }
}

void MCTS::backprop(Node *node, BackPropObj result){
        // cout << "enter backprop" << endl;
    bool shouldUpdate = false;
    while(node->parent){
        node = node->parent;
        node->score += result.wins;
        node->n += result.sims;
        shouldUpdate = !shouldUpdate;
    }
}

Result MCTS::simulate(Node *root, Result* result=nullptr){
    Board b;
    Result r = Result::DRAW;
    // cout << "enter simulate" << endl;
    for(auto action:root->path){
        b.update(action);
    }
    // int step = 0;
    while(true){
        // step++;
        if(!rollout(b)){
            r = b.get_result();
            if(result) *result = r;
            break;
        }
    }
    return r;
}
bool MCTS::rollout(Board &b){
    vector<Action> actions = b.get_actions();
    if(actions.empty()) return false;
    shuffle(actions.begin(), actions.end(), std::default_random_engine(42));
    b.update(actions[0]);
    return true;
}