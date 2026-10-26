#include "denoise.h"
#include <math.h>
#include <stdint.h>

static uint64_t rng_next(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}
static double uniform_open(uint64_t *state) {
    return ((rng_next(state) >> 11) + 1.0) / 9007199254740993.0;
}
int image_add_gaussian_noise(const Image *clean, Image *noisy, float sigma, uint64_t seed) {
    size_t i;
    uint64_t state;
    int status;
    if (!clean || !clean->pixels || !noisy || sigma < 0.0f || !isfinite(sigma))
        return DENOISE_INVALID;
    status = image_alloc(noisy, clean->width, clean->height);
    if (status != DENOISE_OK)
        return status;
    state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
    for (i = 0; i < clean->count; i += 2) {
        double radius = sqrt(-2.0 * log(uniform_open(&state)));
        double angle = 6.2831853071795864769 * uniform_open(&state);
        double z[2] = {radius * cos(angle), radius * sin(angle)};
        size_t j;
        for (j = 0; j < 2 && i + j < clean->count; ++j) {
            float value = clean->pixels[i + j] + sigma * (float)z[j];
            noisy->pixels[i + j] = value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value);
        }
    }
    return DENOISE_OK;
}
