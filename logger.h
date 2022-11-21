#include <iostream>
#include <fstream>
#include <string>
using namespace std;
class Logger{
public:
    string name;
    string content;
    vector<int> times;
    int total_time;
    Logger(string _name): name(_name), content(""), total_time(0){};
    void log(string s){
        content += s + "\n";
    }
    void record_time(int time){
        times.push_back(time);
        total_time += time;
    }
    void setGPU(){
        name = "gpu.log";
    }
    void setCPU(){
        name = "cpu.log";
    }
    void flush(){
        ofstream file;
        file.open(name, ios::out);
        file << content;
        if(times.size() > 0) file << "average time:" << total_time/times.size() << endl;
        file.close();
    }
};