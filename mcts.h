#ifndef MCTS_H
#define MCTS_H
#include <vector>
#include "tree.h"
#include "game.h"
using namespace std;
class MCTS{
public:
    Node *root;
    vector<Action> init_path;
    Board init_board;
    MCTS(vector<Action>&path):init_path(path){
        root = new Node(path);
    };
    MCTS(Board b): init_board(b){

    };
    ~MCTS(){
        delete root;
    };
    Action run(); // get an optimal action.
    Result simulate(Node *root); // simulate a node
    Node* select(Node* root);
    void expand(Node* node);
    void backprop(Node* node, Result result);
    bool rollout(Board &b); // given a board and role, randomly simulate one step
    void traverse(Node *root, vector<Action> &path, Board& board); // traverse the tree to find a node to simulate
};

#endif