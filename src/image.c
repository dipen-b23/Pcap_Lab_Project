#include "denoise.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

int image_alloc(Image *image, size_t width, size_t height) {
    if (!image || width == 0 || height == 0 || width > SIZE_MAX / height ||
        width * height > SIZE_MAX / sizeof(float))
        return DENOISE_INVALID;
    image->width = width;
    image->height = height;
    image->count = width * height;
    image->pixels = (float *)calloc(image->count, sizeof(float));
    if (!image->pixels) {
        memset(image, 0, sizeof(*image));
        return DENOISE_ALLOC;
    }
    return DENOISE_OK;
}

void image_free(Image *image) {
    if (image) {
        free(image->pixels);
        memset(image, 0, sizeof(*image));
    }
}

int image_clone(const Image *source, Image *destination) {
    int status;
    if (!source || !source->pixels || !destination)
        return DENOISE_INVALID;
    status = image_alloc(destination, source->width, source->height);
    if (status == DENOISE_OK)
        memcpy(destination->pixels, source->pixels, source->count * sizeof(float));
    return status;
}

static int next_token(FILE *file, char *token, size_t capacity) {
    int c;
    size_t n = 0;
    do {
        c = fgetc(file);
        if (c == '#')
            while (c != '\n' && c != EOF)
                c = fgetc(file);
    } while (isspace(c));
    if (c == EOF)
        return 0;
    while (c != EOF && !isspace(c)) {
        if (n + 1 >= capacity)
            return 0;
        token[n++] = (char)c;
        c = fgetc(file);
    }
    token[n] = '\0';
    return n > 0;
}

int image_load_pnm(const char *path, Image *image) {
    FILE *file;
    char token[64];
    int ascii, channels, max_value;
    size_t width, height, i;
    if (!path || !image)
        return DENOISE_INVALID;
    file = fopen(path, "rb");
    if (!file)
        return DENOISE_IO;
    if (!next_token(file, token, sizeof(token))) {
        fclose(file);
        return DENOISE_IO;
    }
    ascii = strcmp(token, "P2") == 0;
    channels = strcmp(token, "P6") == 0 ? 3 : 1;
    if (!ascii && strcmp(token, "P5") != 0 && channels != 3) {
        fclose(file);
        return DENOISE_IO;
    }
    if (!next_token(file, token, sizeof(token))) {
        fclose(file);
        return DENOISE_IO;
    }
    width = (size_t)strtoull(token, NULL, 10);
    if (!next_token(file, token, sizeof(token))) {
        fclose(file);
        return DENOISE_IO;
    }
    height = (size_t)strtoull(token, NULL, 10);
    if (!next_token(file, token, sizeof(token))) {
        fclose(file);
        return DENOISE_IO;
    }
    max_value = atoi(token);
    if (max_value < 1 || max_value > 255 || image_alloc(image, width, height) != DENOISE_OK) {
        fclose(file);
        return DENOISE_IO;
    }
    for (i = 0; i < image->count; ++i) {
        int r, g, b;
        if (ascii) {
            if (!next_token(file, token, sizeof(token)))
                goto fail;
            r = atoi(token);
            g = b = r;
        } else {
            r = fgetc(file);
            g = channels == 3 ? fgetc(file) : r;
            b = channels == 3 ? fgetc(file) : r;
            if (r == EOF || g == EOF || b == EOF)
                goto fail;
        }
        image->pixels[i] = channels == 3
                               ? (float)(0.299 * r + 0.587 * g + 0.114 * b) * 255.0f / max_value
                               : (float)r * 255.0f / max_value;
    }
    fclose(file);
    return DENOISE_OK;
fail:
    fclose(file);
    image_free(image);
    return DENOISE_IO;
}

int image_save_pgm(const char *path, const Image *image) {
    FILE *file;
    size_t i;
    if (!path || !image || !image->pixels)
        return DENOISE_INVALID;
    file = fopen(path, "wb");
    if (!file)
        return DENOISE_IO;
    if (fprintf(file, "P5\n%zu %zu\n255\n", image->width, image->height) < 0) {
        fclose(file);
        return DENOISE_IO;
    }
    for (i = 0; i < image->count; ++i)
        if (fputc(pixel_to_u8(image->pixels[i]), file) == EOF) {
            fclose(file);
            return DENOISE_IO;
        }
    return fclose(file) == 0 ? DENOISE_OK : DENOISE_IO;
}

int image_generate(Image *image, size_t width, size_t height) {
    size_t x, y;
    int status = image_alloc(image, width, height);
    if (status != DENOISE_OK)
        return status;
    for (y = 0; y < height; ++y)
        for (x = 0; x < width; ++x) {
            float gradient = 180.0f * (float)x / (float)(width > 1 ? width - 1 : 1);
            float checker = (((x / 32) + (y / 32)) & 1) ? 45.0f : 0.0f;
            long long cx = (long long)x - (long long)width / 2,
                      cy = (long long)y - (long long)height / 2;
            long long radius = (long long)width / 5;
            float circle = (cx * cx + cy * cy < radius * radius) ? 30.0f : 0.0f;
            float value = gradient + checker + circle;
            image->pixels[y * width + x] = value > 255.0f ? 255.0f : value;
        }
    return DENOISE_OK;
}
