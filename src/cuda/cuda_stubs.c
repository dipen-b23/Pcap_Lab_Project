#include "denoise.h"
int filter_cuda(const Image*a,Image*b,const BilateralConfig*c,TimingInfo*d){(void)a;(void)b;(void)c;(void)d;return DENOISE_UNAVAILABLE;}
int filter_hybrid(const Image*a,Image*b,const BilateralConfig*c,int d,int e,TimingInfo*f){(void)a;(void)b;(void)c;(void)d;(void)e;(void)f;return DENOISE_UNAVAILABLE;}
int cuda_runtime_available(void){return 0;}
