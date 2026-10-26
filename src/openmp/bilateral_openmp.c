#include "denoise.h"
#include <math.h>
#include <stdlib.h>
#include <omp.h>

int filter_openmp(const Image *in, Image *out, const BilateralConfig *c, int threads,
                  TimingInfo *timing) {
    int r, dy, dx;
    long long y;
    float *spatial;
    double start;
    if (!in || !in->pixels || !out || !bilateral_config_valid(c) || threads < 1)
        return DENOISE_INVALID;
    if (image_alloc(out, in->width, in->height) != DENOISE_OK)
        return DENOISE_ALLOC;
    r = c->window_size / 2;
    spatial = (float *)malloc((size_t)c->window_size * c->window_size * sizeof(float));
    if (!spatial) {
        image_free(out);
        return DENOISE_ALLOC;
    }
    for (dy = -r; dy <= r; ++dy)
        for (dx = -r; dx <= r; ++dx)
            spatial[(dy + r) * c->window_size + dx + r] =
                expf(-(float)(dx * dx + dy * dy) / (2.0f * c->sigma_spatial * c->sigma_spatial));
    start = monotonic_seconds();
#pragma omp parallel for schedule(static) num_threads(threads)
    for (y = 0; y < (long long)in->height; ++y) {
        size_t x;
        int local_dy, local_dx;
        for (x = 0; x < in->width; ++x) {
            float center = in->pixels[(size_t)y * in->width + x], weighted = 0.0f, weights = 0.0f;
            for (local_dy = -r; local_dy <= r; ++local_dy)
                for (local_dx = -r; local_dx <= r; ++local_dx) {
                    long long sy = y + local_dy, sx = (long long)x + local_dx;
                    float value, diff, weight;
                    if (sy < 0)
                        sy = 0;
                    if (sy >= (long long)in->height)
                        sy = (long long)in->height - 1;
                    if (sx < 0)
                        sx = 0;
                    if (sx >= (long long)in->width)
                        sx = (long long)in->width - 1;
                    value = in->pixels[(size_t)sy * in->width + (size_t)sx];
                    diff = center - value;
                    weight = spatial[(local_dy + r) * c->window_size + local_dx + r] *
                             expf(-(diff * diff) / (2.0f * c->sigma_range * c->sigma_range));
                    weighted += weight * value;
                    weights += weight;
                }
            out->pixels[(size_t)y * in->width + x] =
                (weights > 0.0f && isfinite(weights)) ? weighted / weights : center;
        }
    }
    if (timing) {
        timing->kernel_ms = (monotonic_seconds() - start) * 1000.0;
        timing->wall_ms = timing->kernel_ms;
    }
    free(spatial);
    return DENOISE_OK;
}
