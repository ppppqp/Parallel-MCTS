#include <iostream>
#include <fstream>
#include <string>
using namespace std;
class Logger{
public:
    int counter = 0;
    int total_time_no_first = 0;
    double average_20 = 0.0;
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
        counter++;
        if (counter != 1) total_time_no_first += time;
        if (counter == 21) {
            average_20 = (double)total_time_no_first / 20.0;
        }
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
        if(times.size() > 0) {
            file << "Average time (first 20): "<< average_20 << endl;
            file << "average time:" << total_time/times.size() << endl;
        }
        file.close();
    }
};