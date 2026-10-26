#include "denoise.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
double monotonic_seconds(void) {
    LARGE_INTEGER now, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)frequency.QuadPart;
}
#else
#include <time.h>
double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif
