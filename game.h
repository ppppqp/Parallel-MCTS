
#ifndef GAME_H
#define GAME_H
#include <vector>
#include <iostream>
#include <string>
#include <set>
using namespace std;
enum class ROLE{
    BLACK,
    WHITE
};


class Action{
public:
    int x;
    int y;
    Action(int _y, int _x):x(_x), y(_y){};
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
    Board():current_role(ROLE::WHITE){
        for(auto i = 0; i < 8; i++){
            for(auto j = 0; j < 8; j++){
                s[i][j] = State::NONE;
            }
        }
        s[3][3] = State::BLACK;
        s[3][4] = State::WHITE;
        s[4][3] = State::WHITE;
        s[4][4] = State::BLACK;
    };
    bool update(Action action){
        if(action.y >= 8 || action.x >= 8 || action.y < 0 || action.x < 0) throw "Out of board!";
        if(s[action.y][action.x] != State::NONE) throw "This position is already filled!";
        bool legal = false;
        vector<Action> v = get_actions();
        for(auto a : v){
            if(a.x == action.x && a.y == action.y) legal = true;
        }
        if(!legal) throw "Illegal move!";
        s[action.y][action.x] = (current_role == ROLE::BLACK) ? State::BLACK : State::WHITE;
        // flip

        State myStone = (current_role == ROLE::BLACK) ? State::BLACK : State::WHITE;
        State opponentStone = (current_role == ROLE::BLACK) ? State::WHITE : State::BLACK;
        // top
        int y = action.y-1;
        int x = action.x;
        while(y >= 0 && s[y][x] == opponentStone){
            y--;
        }
        if(y >= 0 && s[y][x] == myStone){
            for(int i = action.y-1; i > y; i--){
                s[i][x] = myStone;
            }
        }

        // bottom
        y = action.y+1;
        x = action.x;
        while(y < 8 && s[y][x] == opponentStone){
            y++;
        }
        if(y < 8 && s[y][x] == myStone){
            for(int i = action.y+1; i < y; i++){
                s[i][x] = myStone;
            }
        }

        // right
        y = action.y;
        x = action.x+1;
        while(x < 8 && s[y][x] == opponentStone){
            x++;
        }
        if(x < 8 && s[y][x] == myStone){
            for(int i = action.x+1; i < x; i++){
                s[y][i] = myStone;
            }
        }

        // left
        y = action.y;
        x = action.x-1;
        while(x >= 0 && s[y][x] == opponentStone){
            x--;
        }
        if(x >= 0 && s[y][x] == myStone){
            for(int i = action.x-1; i > x; i--){
                s[y][i] = myStone;
            }
        }

        // top-left
        y = action.y-1;
        x = action.x-1;
        int count = 0;
        while(x >= 0 && y >= 0 && s[y][x] == opponentStone){
            x--;
            y--;
            count ++;
        }
        if(x >=0 && y >= 0 && s[y][x] == myStone){
            for(int i = 0; i < count; i++){
                s[action.y-1-i][action.x-1-i] = myStone;
            }
        }

        // top-right
        y = action.y-1;
        x = action.x+1;
        count = 0;
        while(x < 8 && y >= 0 && s[y][x] == opponentStone){
            x++;
            y--;
            count ++;
        }
        if(x < 8 && y >= 0 && s[y][x] == myStone){
            for(int i = 0; i < count; i++){
                s[action.y-1-i][action.x+1+i] = myStone;
            }
        }

        // bottom-right
        y = action.y+1;
        x = action.x+1;
        count = 0;
        while(x < 8 && y < 8 && s[y][x] == opponentStone){
            x++;
            y++;
            count ++;
        }
        if(x < 8 && y < 8 && s[y][x] == myStone){
            for(int i = 0; i < count; i++){
                s[action.y+1+i][action.x+1+i] = myStone;
            }
        }


        // bottom-left
        y = action.y+1;
        x = action.x-1;
        count = 0;
        while(x >=0 && y < 8 && s[y][x] == opponentStone){
            x--;
            y++;
            count ++;
        }
        if(x >= 0 && y < 8 && s[y][x] == myStone){
            for(int i = 0; i < count; i++){
                s[action.y+1+i][action.x-1-i] = myStone;
            }
        }

    

        current_role = (current_role == ROLE::BLACK) ? ROLE::WHITE : ROLE::BLACK;
        return true;
    }; // TODO
    void batch_update(vector<Action>& path){
        try{
            for(auto action : path ) update(action);
        }
        catch(string s){
            cout << s  << endl;
        }
    }
    vector<Action> get_actions(){
        auto cmp = [](const Action& a, const Action& b) { if(a.x == b.x) return a.y < b.y; return a.x < b.x; };
        set<Action, decltype(cmp)> sa(cmp);
        
        State myStone = (current_role == ROLE::BLACK) ? State::BLACK : State::WHITE;
        State opponentStone = (current_role == ROLE::BLACK) ? State::WHITE : State::BLACK;
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                if(s[i][j] == myStone){
                    // top
                    int y = i-1;
                    int x = j;
                    while(y >= 0 && s[y][x] == opponentStone){
                        y--;
                    }
                    if(y >= 0 && y != i-1 &&  s[y][x] == State::NONE) sa.emplace(y,x);

                    // bottom
                    y = i+1;
                    x = j;
                    while(y < 8 && s[y][x] == opponentStone){
                        y++;
                    }
                    if(y < 8 && y != i+1 && s[y][x] == State::NONE) sa.emplace(y,x);

                    // right
                    y = i;
                    x = j+1;
                    while(x < 8 && s[y][x] == opponentStone){
                        x++;
                    }
                    if(x < 8 && x != j+1 && s[y][x] == State::NONE) sa.emplace(y,x);

                    // left
                    y = i;
                    x = j-1;
                    while(x >= 0 && s[y][x] == opponentStone){
                        x--;
                    }
                    if(x >= 0 && x != j-1 && s[y][x] == State::NONE) sa.emplace(y,x);


                    // top left
                    y = i-1;
                    x = j-1;
                    while(x >= 0 && y >= 0 && s[y][x] == opponentStone){
                        x--;
                        y--;
                    }
                    if(x >= 0 && y >= 0 && x != j-1 && s[y][x] == State::NONE) sa.emplace(y,x);

                    // top right
                    y = i-1;
                    x = j+1;
                    while(x < 8 && y >= 0 && s[y][x] == opponentStone){
                        x++;
                        y--;
                    }
                    if(x < 8 && y >= 0 && x!= j+1 && s[y][x] == State::NONE) sa.emplace(y,x);

                    // bottom left
                    y = i+1;
                    x = j-1;
                    while(x >= 0 && y < 8 && s[y][x] == opponentStone){
                        x--;
                        y++;
                    }
                    if(x >= 0 && y < 8 && x != j-1 && s[y][x] == State::NONE) sa.emplace(y,x);


                    // bottom right
                    y = i+1;
                    x = j+1;
                    while(x >= 0 && y >= 0 && s[y][x] == opponentStone){
                        x++;
                        y++;
                    }
                    if(x < 8 && y < 8 && x != j+1 && s[y][x] == State::NONE) sa.emplace(y,x);
                }
            }
        }

        return vector<Action>(sa.begin(), sa.end());
    }; // TODO
    Result get_result(){
        int count = 0;
        for(int i = 0; i < 8; i++){
            for(int j = 0; j < 8; j++){
                if(s[i][j] == State::BLACK) count ++;
                if(s[i][j] == State::WHITE) count --;
            }
        }
        if(count > 0) return Result::WIN;
        if(count == 0) return Result::DRAW;
        if(count < 0) return Result::LOSE;
        return Result::DRAW;
    }  // TODO
    bool check_end(){
        if(get_actions().empty()){
            Result r = get_result();
            if(r == Result::WIN) cout << "WIN!" << endl;
            if(r == Result::LOSE) cout << "LOSE!" << endl;
            if(r == Result::DRAW) cout << "DRAW!" << endl;
            return true;
        }
        return false;
    }
    void print(){
        cout << "  0 1 2 3 4 5 6 7" << endl;
        for(int i = 0; i < 8; i++){
            cout << i << ' ';
            for(int j = 0; j < 8; j++){

                if(s[i][j] == State::BLACK) cout << 'x';
                if(s[i][j] == State::WHITE) cout << 'o';
                if(s[i][j] == State::NONE) cout << '-';
                cout << ' ';
            }
            cout << endl;
        }

    }
};

#endif