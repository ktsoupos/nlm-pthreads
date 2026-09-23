#include "timer.h"

void timer_start(stopwatch_t *t){
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

double timer_elapsed_sec(const stopwatch_t *t){
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double sec  = (double)(now.tv_sec  - t->start.tv_sec);
    double nsec = (double)(now.tv_nsec - t->start.tv_nsec);
    return sec + nsec * 1e-9;
}
