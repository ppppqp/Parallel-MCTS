
#ifndef GAME_H
#define GAME_H
#include <vector>
#include <iostream>
using namespace std;
enum class ROLE{
    BLACK,
    WHITE
};


class Action{
public:
    int x;
    int y;
    ROLE role;
    Action(int _y, int _x, ROLE _role):x(_x), y(_y), role(_role){};
};



enum class Result{
    WIN,
    LOSE,
    DRAW
};
enum class State{
    WHITE,
    BLACK,
    NONE
};




class Board{
public:
    State s[9][9];
    ROLE current_role;
    int remain;
    Board():current_role(ROLE::BLACK){
        for(auto i = 0; i < 9; i++){
            for(auto j = 0; j < 9; j++){
                s[i][j] = State::NONE;
            }
        }
        remain = 81;
    };
    void update(Action action){
        State state = (action.role == ROLE::BLACK) ? State::BLACK : State::WHITE;
        s[action.y][action.x] = state;
        remain --;
    }; // TODO
    void batch_update(vector<Action>& path){
        for(auto action : path ) update(action);
    }
    vector<Action> get_actions(ROLE role){
        return vector<Action>();
    }; // TODO
    Result get_result(){
        return Result::WIN;
    }  // TODO
    bool check_end(ROLE role){
        if(get_actions(role).empty()){
            Result r = get_result();
            if(r == Result::WIN) cout << "WIN!" << endl;
            if(r == Result::LOSE) cout << "LOSE!" << endl;
            if(r == Result::DRAW) cout << "DRAW!" << endl;
            return true;
        }
        return false;
    }
};

#endif