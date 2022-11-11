#ifndef TREE_H
#define TREE_H
#include <vector>
#include "game.h"
#include <cmath>
using namespace std;
/* 
each tree node represent a state of the board
*/
const double EPSILON = 1e-6;
const double C = 2;// exploration factor




class Node{
public:
    double score; // number of wins
    double UCB;    // UCB value of this node
    double n; // number of simualtions
    bool expandable;
    vector<Node *> children;
    Node* parent;
    vector<Action> path;
    void update_UCB(){
        if(parent == nullptr) return;
        UCB = score/(n + EPSILON) + C * sqrt(log(parent->score + EPSILON)/(n + EPSILON));
    };
    void add_child(Node* child){
        children.push_back(child);
        child->parent = this;
    }
    // constructor
    Node(vector<Action> _path, Action action):score(0), UCB(0), n(0), parent(nullptr), path(_path), expandable(true){
        path.push_back(action);
    }
    Node(vector<Action> _path):score(0), UCB(0), n(0), parent(nullptr), path(_path), expandable(true){
    }
    ~Node(){
        for(auto child:children){
            delete child; 
        }
    }


};

#endif