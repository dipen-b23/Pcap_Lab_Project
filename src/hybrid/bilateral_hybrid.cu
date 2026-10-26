#include "denoise.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

__global__ static void hybrid_gpu_kernel(const float *in, float *out, int w, int h, int row0,
                                         int rows, int r, float ss, float sr) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x),
        ly = (int)(blockIdx.y * blockDim.y + threadIdx.y), y = row0 + ly;
    if (x >= w || ly >= rows)
        return;
    float center = in[y * w + x], sum = 0, ws = 0;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            int yy = y + dy, xx = x + dx;
            if (yy < 0)
                yy = 0;
            if (yy >= h)
                yy = h - 1;
            if (xx < 0)
                xx = 0;
            if (xx >= w)
                xx = w - 1;
            float v = in[yy * w + xx], d = center - v,
                  q = expf(-(float)(dx * dx + dy * dy) / (2 * ss * ss)) *
                      expf(-(d * d) / (2 * sr * sr));
            sum += q * v;
            ws += q;
        }
    out[ly * w + x] = ws > 0 ? sum / ws : center;
}
static void cpu_region(const Image *in, Image *out, const BilateralConfig *c, int rows,
                       int threads) {
    int r = c->window_size / 2;
    long long y;
#pragma omp parallel for schedule(static) num_threads(threads)
    for (y = 0; y < rows; ++y)
        for (size_t x = 0; x < in->width; ++x) {
            float center = in->pixels[(size_t)y * in->width + x], sum = 0, ws = 0;
            for (int dy = -r; dy <= r; ++dy)
                for (int dx = -r; dx <= r; ++dx) {
                    long long yy = y + dy, xx = (long long)x + dx;
                    if (yy < 0)
                        yy = 0;
                    if (yy >= (long long)in->height)
                        yy = (long long)in->height - 1;
                    if (xx < 0)
                        xx = 0;
                    if (xx >= (long long)in->width)
                        xx = (long long)in->width - 1;
                    float v = in->pixels[(size_t)yy * in->width + (size_t)xx], d = center - v,
                          q = expf(-(float)(dx * dx + dy * dy) /
                                   (2 * c->sigma_spatial * c->sigma_spatial)) *
                              expf(-(d * d) / (2 * c->sigma_range * c->sigma_range));
                    sum += q * v;
                    ws += q;
                }
            out->pixels[(size_t)y * in->width + x] = ws > 0 ? sum / ws : center;
        }
}
extern "C" int filter_hybrid(const Image *in, Image *out, const BilateralConfig *c, int threads,
                             int cpu_percent, TimingInfo *t) {
    int devices = 0, cpu_rows, gpu_rows, status = DENOISE_OK;
    float *di = NULL, *dout = NULL;
    double start;
    if (!in || !in->pixels || !out || !bilateral_config_valid(c) || threads < 1 ||
        cpu_percent < 0 || cpu_percent > 100)
        return DENOISE_INVALID;
    if (cpu_percent == 100)
        return filter_openmp(in, out, c, threads, t);
    if (cpu_percent == 0)
        return filter_cuda(in, out, c, t);
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1)
        return DENOISE_UNAVAILABLE;
    if (image_alloc(out, in->width, in->height) != DENOISE_OK)
        return DENOISE_ALLOC;
    cpu_rows = (int)(in->height * (size_t)cpu_percent / 100);
    gpu_rows = (int)in->height - cpu_rows;
    start = monotonic_seconds();
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            cpu_region(in, out, c, cpu_rows, threads > 1 ? threads - 1 : 1);
        }
#pragma omp section
        {
            size_t all = in->count * sizeof(float),
                   part = (size_t)gpu_rows * in->width * sizeof(float);
            if (cudaMalloc((void **)&di, all) != cudaSuccess ||
                cudaMalloc((void **)&dout, part) != cudaSuccess)
                status = DENOISE_RUNTIME;
            else if (cudaMemcpy(di, in->pixels, all, cudaMemcpyHostToDevice) != cudaSuccess)
                status = DENOISE_RUNTIME;
            else {
                dim3 b(16, 16),
                    g((unsigned)((in->width + 15) / 16), (unsigned)((gpu_rows + 15) / 16));
                hybrid_gpu_kernel<<<g, b>>>(di, dout, (int)in->width, (int)in->height, cpu_rows,
                                            gpu_rows, c->window_size / 2, c->sigma_spatial,
                                            c->sigma_range);
                if (cudaDeviceSynchronize() != cudaSuccess ||
                    cudaMemcpy(out->pixels + (size_t)cpu_rows * in->width, dout, part,
                               cudaMemcpyDeviceToHost) != cudaSuccess)
                    status = DENOISE_RUNTIME;
            }
            if (di)
                cudaFree(di);
            if (dout)
                cudaFree(dout);
        }
    }
    if (status) {
        image_free(out);
        return status;
    }
    if (t) {
        t->wall_ms = (monotonic_seconds() - start) * 1000.0;
        t->kernel_ms = t->wall_ms;
    }
    return DENOISE_OK;
}
