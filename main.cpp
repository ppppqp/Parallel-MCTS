#include "mcts.h"
#include "game.h"
#include <iostream>
int main(){
    Board b;
    vector<Action> path;
    while(true){
        // b.print();
        int x;
        int y;
        cin >> y >> x;
        Action input_action(y, x, ROLE::WHITE);
        b.update(input_action);
        path.push_back(input_action);
        if(b.check_end(ROLE::BLACK)){
            return 0;
        }
        MCTS mcts(path);
        Action action = mcts.run();
        b.update(action);
        if(b.check_end(ROLE::WHITE)){
            return 0;
        }
        path.push_back(action);
    }
    return 0;
}