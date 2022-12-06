#ifndef TIMEER_H
#define TIMER_H
#include <time.h>
const int MILLION = 1000000;
const long long BILLION = 1000000000;
class Timer{
    struct timespec start_time;
    struct timespec end_time;
public:
    void start(){
        clock_gettime(CLOCK_REALTIME, &start_time);
    }
    void stop(){
        clock_gettime(CLOCK_REALTIME, &end_time);
    }
    uint64_t time(){
        uint64_t diff = BILLION * (end_time.tv_sec - start_time.tv_sec) + end_time.tv_nsec - start_time.tv_nsec;
        return diff/(MILLION);
    }
    uint64_t get_start(){
        return start_time.tv_sec * BILLION + start_time.tv_nsec;
    }
    uint64_t get_end(){
        return (end_time.tv_sec * BILLION + end_time.tv_nsec)/(MILLION);
    }
};
#endif