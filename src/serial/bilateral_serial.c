#include "denoise.h"
#include <math.h>
#include <stdlib.h>

int bilateral_config_valid(const BilateralConfig *c) {
    return c && c->window_size >= 3 && (c->window_size & 1) &&
           c->sigma_spatial > 0.0f && isfinite(c->sigma_spatial) &&
           c->sigma_range > 0.0f && isfinite(c->sigma_range);
}
int filter_serial(const Image *in, Image *out, const BilateralConfig *c, TimingInfo *timing) {
    int r, dy, dx; size_t y, x; float *spatial; double start;
    if (!in || !in->pixels || !out || !bilateral_config_valid(c)) return DENOISE_INVALID;
    if (image_alloc(out, in->width, in->height) != DENOISE_OK) return DENOISE_ALLOC;
    r = c->window_size / 2;
    spatial = (float *)malloc((size_t)c->window_size * c->window_size * sizeof(float));
    if (!spatial) { image_free(out); return DENOISE_ALLOC; }
    for (dy=-r;dy<=r;++dy) for(dx=-r;dx<=r;++dx)
        spatial[(dy+r)*c->window_size+dx+r]=expf(-(float)(dx*dx+dy*dy)/(2.0f*c->sigma_spatial*c->sigma_spatial));
    start = monotonic_seconds();
    for (y=0;y<in->height;++y) for(x=0;x<in->width;++x) {
        float center=in->pixels[y*in->width+x], weighted=0.0f, weights=0.0f;
        for(dy=-r;dy<=r;++dy) for(dx=-r;dx<=r;++dx) {
            size_t yy = dy < 0 && (size_t)(-dy)>y ? 0 : (y+(size_t)(dy>0?dy:0)-(size_t)(dy<0?-dy:0));
            size_t xx = dx < 0 && (size_t)(-dx)>x ? 0 : (x+(size_t)(dx>0?dx:0)-(size_t)(dx<0?-dx:0));
            float value, diff, weight;
            if(yy>=in->height) yy=in->height-1; if(xx>=in->width) xx=in->width-1;
            value=in->pixels[yy*in->width+xx]; diff=center-value;
            weight=spatial[(dy+r)*c->window_size+dx+r]*expf(-(diff*diff)/(2.0f*c->sigma_range*c->sigma_range));
            weighted += weight*value; weights += weight;
        }
        out->pixels[y*in->width+x]=(weights>0.0f&&isfinite(weights))?weighted/weights:center;
    }
    if(timing) { timing->kernel_ms=(monotonic_seconds()-start)*1000.0; timing->wall_ms=timing->kernel_ms; }
    free(spatial); return DENOISE_OK;
}
