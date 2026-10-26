#ifndef DENOISE_H
#define DENOISE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum DenoiseStatus {
    DENOISE_OK = 0,
    DENOISE_INVALID = 2,
    DENOISE_ALLOC = 3,
    DENOISE_IO = 4,
    DENOISE_UNAVAILABLE = 5,
    DENOISE_RUNTIME = 6
};

typedef struct {
    size_t width, height, count;
    float *pixels;
} Image;
typedef struct {
    int window_size;
    float sigma_spatial, sigma_range;
} BilateralConfig;
typedef struct {
    double wall_ms, kernel_ms, h2d_ms, d2h_ms, communication_ms;
    double distribution_ms, halo_ms, gather_ms;
} TimingInfo;
typedef struct {
    double max_abs_error, mean_abs_error, rmse;
    double within_tolerance_percent, exact_u8_percent;
    int passed;
} ValidationResult;
typedef struct {
    double mse, psnr;
} QualityMetrics;

int image_alloc(Image *image, size_t width, size_t height);
void image_free(Image *image);
int image_clone(const Image *source, Image *destination);
int image_load_pnm(const char *path, Image *image);
int image_save_pgm(const char *path, const Image *image);
int image_generate(Image *image, size_t width, size_t height);
int image_add_gaussian_noise(const Image *clean, Image *noisy, float sigma, uint64_t seed);
int bilateral_config_valid(const BilateralConfig *config);
double monotonic_seconds(void);

/* Filter functions allocate output->pixels; the caller owns it and calls image_free. */
int filter_serial(const Image *input, Image *output, const BilateralConfig *config,
                  TimingInfo *timing);
int filter_openmp(const Image *input, Image *output, const BilateralConfig *config,
                  int thread_count, TimingInfo *timing);
int filter_cuda(const Image *input, Image *output, const BilateralConfig *config,
                TimingInfo *timing);
int filter_hybrid(const Image *input, Image *output, const BilateralConfig *config,
                  int thread_count, int cpu_percent, TimingInfo *timing);
int cuda_runtime_available(void);

void validate_images(const Image *reference, const Image *candidate, double tolerance,
                     ValidationResult *result);
void quality_metrics(const Image *reference, const Image *candidate, QualityMetrics *result);
uint8_t pixel_to_u8(float value);

#ifdef __cplusplus
}
#endif
#endif
