#include "mcts.h"
#include "game.h"
#include <iostream>
int main(){
    Board b;
    vector<Action> path;
    while(true){
        b.print();
        int x;
        int y;

        bool success = false;
        while(!success){
            try{
                cout << "WHITE MOVE:";
                cin >> y >> x;
                Action input_action(y,x);
                success = b.update(input_action);
                path.push_back(input_action);
            }catch(const char* s){
                cout << s << endl;
            }
        }
        b.print();
        if(b.check_end()){
            return 0;
        }
        MCTS mcts(path);
        Action action = mcts.run();
        b.update(action);
        cout << "BLACK MOVE:" <<  action.y << ' ' <<  action.x << endl;
        if(b.check_end()){
            return 0;
        }
        path.push_back(action);
    }
    return 0;
}