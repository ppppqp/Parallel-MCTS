#include "mcts.h"
#include "game.h"
#include <iostream>
#include <algorithm>    
#include <random>    
#include <chrono> 
const bool AUTO = true;
int main(){
    Board b;
    vector<Action> path;
    Logger logger("log");
    try{
        while(true){
                b.print();
                int x;
                int y;

                bool success = false;
                if(AUTO){
                    // get a random action
                    vector<Action> actions = b.get_actions();
                    shuffle(actions.begin(), actions.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
                    b.update(actions[0]);
                    path.push_back(actions[0]);
                }else{
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
                }
                b.print();
                if(b.check_end()){
                    logger.flush();
                    return 0;
                }
                MCTS mcts(path);
                Action action = mcts.run(logger);   // NOTE: RUN
                b.update(action);
                cout << "BLACK MOVE:" <<  action.y << ' ' <<  action.x << endl;
                if(b.check_end()){
                    b.print();
                    logger.flush();
                    return 0;
                }
                path.push_back(action);
            }
    }catch(char * c){
        cout << c << endl;
    }
    logger.flush();
    return 0;
}