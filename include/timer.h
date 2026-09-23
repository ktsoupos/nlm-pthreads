#ifndef TIMER_H
#define TIMER_H

#include <time.h>

/** @brief Monotonic stopwatch. Not named timer_t -- that's a POSIX type. */
typedef struct {
    struct timespec start;
} stopwatch_t;

/** @brief Start (or restart) the stopwatch. */
void timer_start(stopwatch_t *t);

/** @brief Seconds elapsed since timer_start(), as a double. */
double timer_elapsed_sec(const stopwatch_t *t);

#endif
