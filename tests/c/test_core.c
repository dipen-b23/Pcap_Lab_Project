#include "denoise.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

static void test_constant(void) {
    Image in = {0}, out = {0};
    BilateralConfig c = {5, 2, 20};
    TimingInfo t = {0};
    image_alloc(&in, 7, 5);
    for (size_t i = 0; i < in.count; ++i)
        in.pixels[i] = 73.0f;
    CHECK(filter_serial(&in, &out, &c, &t) == 0);
    for (size_t i = 0; i < out.count; ++i)
        CHECK(fabsf(out.pixels[i] - 73.0f) < 1e-5f);
    CHECK(t.kernel_ms >= 0);
    image_free(&in);
    image_free(&out);
}
static void test_tiny_and_boundaries(void) {
    Image in = {0}, out = {0};
    BilateralConfig c = {9, 3, 25};
    image_alloc(&in, 1, 1);
    in.pixels[0] = 201;
    CHECK(filter_serial(&in, &out, &c, NULL) == 0);
    CHECK(fabsf(out.pixels[0] - 201) < 1e-4f);
    image_free(&in);
    image_free(&out);
}
static void test_impulse_edge(void) {
    Image in = {0}, out = {0};
    BilateralConfig c = {3, 1, 100};
    image_alloc(&in, 5, 5);
    in.pixels[12] = 255;
    CHECK(filter_serial(&in, &out, &c, NULL) == 0);
    CHECK(out.pixels[12] < 255 && out.pixels[12] > 0);
    CHECK(out.pixels[0] >= 0);
    image_free(&in);
    image_free(&out);
}
static void test_invalid(void) {
    Image in = {0}, out = {0};
    BilateralConfig even = {4, 2, 20}, zero = {3, 0, 20};
    image_alloc(&in, 2, 2);
    CHECK(filter_serial(&in, &out, &even, NULL) == DENOISE_INVALID);
    CHECK(filter_serial(&in, &out, &zero, NULL) == DENOISE_INVALID);
    CHECK(image_alloc(&out, 0, 2) == DENOISE_INVALID);
    image_free(&in);
}
static void test_noise(void) {
    Image in = {0}, a = {0}, b = {0};
    image_alloc(&in, 8, 8);
    for (size_t i = 0; i < in.count; ++i)
        in.pixels[i] = 128;
    CHECK(image_add_gaussian_noise(&in, &a, 25, 42) == 0);
    CHECK(image_add_gaussian_noise(&in, &b, 25, 42) == 0);
    CHECK(memcmp(a.pixels, b.pixels, a.count * sizeof(float)) == 0);
    image_free(&in);
    image_free(&a);
    image_free(&b);
}
static void test_metrics(void) {
    Image a = {0}, b = {0};
    QualityMetrics q;
    ValidationResult v;
    image_alloc(&a, 2, 2);
    image_alloc(&b, 2, 2);
    quality_metrics(&a, &b, &q);
    CHECK(q.mse == 0 && isinf(q.psnr));
    b.pixels[0] = 1;
    validate_images(&a, &b, 0.001, &v);
    CHECK(!v.passed && fabs(v.max_abs_error - 1) < 1e-9 && v.exact_u8_percent == 75);
    image_free(&a);
    image_free(&b);
}
static void test_openmp(void) {
    Image in = {0}, a = {0}, b = {0};
    BilateralConfig c = {5, 2, 25};
    TimingInfo t;
    ValidationResult v;
    image_generate(&in, 31, 23);
    CHECK(filter_serial(&in, &a, &c, &t) == 0);
    int s = filter_openmp(&in, &b, &c, 4, &t);
    if (s == 0) {
        validate_images(&a, &b, 1e-3, &v);
        CHECK(v.passed);
    } else
        CHECK(s == DENOISE_UNAVAILABLE);
    image_free(&in);
    image_free(&a);
    image_free(&b);
}
int main(void) {
    test_constant();
    test_tiny_and_boundaries();
    test_impulse_edge();
    test_invalid();
    test_noise();
    test_metrics();
    test_openmp();
    if (failures) {
        fprintf(stderr, "%d tests failed\n", failures);
        return 1;
    }
    puts("all C tests passed");
    return 0;
}
