#ifndef TIMER_H
#define TIMER_H

#include <time.h>

/* Not named timer_t -- that is a POSIX type. */
typedef struct {
    struct timespec start;
} stopwatch_t;

void   timer_start(stopwatch_t *t);
double timer_elapsed_sec(const stopwatch_t *t);

#endif
