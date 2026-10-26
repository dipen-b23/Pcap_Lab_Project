#include "denoise.h"
#include <math.h>
#include <string.h>

uint8_t pixel_to_u8(float value) {
    if (!isfinite(value) || value <= 0.0f) return 0;
    if (value >= 255.0f) return 255;
    return (uint8_t)floorf(value + 0.5f);
}

void validate_images(const Image *a, const Image *b, double tolerance, ValidationResult *r) {
    size_t i, within = 0, exact = 0;
    double sum = 0.0, squared = 0.0, maximum = 0.0;
    memset(r, 0, sizeof(*r));
    if (!a || !b || !a->pixels || !b->pixels || a->count != b->count || a->count == 0) return;
    for (i = 0; i < a->count; ++i) {
        double error = fabs((double)a->pixels[i] - (double)b->pixels[i]);
        if (error > maximum) maximum = error;
        sum += error; squared += error * error;
        if (error <= tolerance) ++within;
        if (pixel_to_u8(a->pixels[i]) == pixel_to_u8(b->pixels[i])) ++exact;
    }
    r->max_abs_error = maximum;
    r->mean_abs_error = sum / (double)a->count;
    r->rmse = sqrt(squared / (double)a->count);
    r->within_tolerance_percent = 100.0 * (double)within / (double)a->count;
    r->exact_u8_percent = 100.0 * (double)exact / (double)a->count;
    r->passed = maximum <= tolerance;
}

void quality_metrics(const Image *a, const Image *b, QualityMetrics *r) {
    size_t i;
    double sum = 0.0;
    r->mse = r->psnr = NAN;
    if (!a || !b || !a->pixels || !b->pixels || a->count != b->count || !a->count) return;
    for (i = 0; i < a->count; ++i) {
        double d = (double)a->pixels[i] - b->pixels[i];
        sum += d * d;
    }
    r->mse = sum / (double)a->count;
    r->psnr = r->mse == 0.0 ? INFINITY : 10.0 * log10(65025.0 / r->mse);
}
