#include "denoise.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ static void bilateral_kernel(const float *in,float *out,int width,int height,int row0,int rows,int radius,float ss,float sr){
    int x=(int)(blockIdx.x*blockDim.x+threadIdx.x),local_y=(int)(blockIdx.y*blockDim.y+threadIdx.y),y=row0+local_y;
    if(x>=width||local_y>=rows)return;
    float center=in[y*width+x],sum=0.0f,weights=0.0f;
    for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){
        int yy=y+dy,xx=x+dx;if(yy<0)yy=0;if(yy>=height)yy=height-1;if(xx<0)xx=0;if(xx>=width)xx=width-1;
        float value=in[yy*width+xx],diff=center-value;
        float w=expf(-(float)(dx*dx+dy*dy)/(2.0f*ss*ss))*expf(-(diff*diff)/(2.0f*sr*sr));sum+=w*value;weights+=w;
    }out[local_y*width+x]=(weights>0.0f&&isfinite(weights))?sum/weights:center;
}
extern "C" int cuda_runtime_available(void){int count=0;return cudaGetDeviceCount(&count)==cudaSuccess&&count>0;}

extern "C" int filter_cuda(const Image *in,Image *out,const BilateralConfig *c,TimingInfo *timing){
    float *di=NULL,*dout=NULL;cudaEvent_t a,b;size_t bytes;double start;float elapsed;int devices=0;cudaError_t e;
    dim3 block(16,16);
    dim3 grid((unsigned)((in->width+15)/16),
          (unsigned)((in->height+15)/16));
    if(!in||!in->pixels||!out||!bilateral_config_valid(c))return DENOISE_INVALID;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices<1)return DENOISE_UNAVAILABLE;
    if(image_alloc(out,in->width,in->height)!=DENOISE_OK)return DENOISE_ALLOC;bytes=in->count*sizeof(float);start=monotonic_seconds();
    if(cudaEventCreate(&a)!=cudaSuccess||cudaEventCreate(&b)!=cudaSuccess)goto fail;
    if(cudaMalloc((void**)&di,bytes)!=cudaSuccess||cudaMalloc((void**)&dout,bytes)!=cudaSuccess)goto fail;
    cudaEventRecord(a);e=cudaMemcpy(di,in->pixels,bytes,cudaMemcpyHostToDevice);cudaEventRecord(b);cudaEventSynchronize(b);if(e!=cudaSuccess)goto fail;
    cudaEventElapsedTime(&elapsed,a,b);if(timing)timing->h2d_ms=elapsed;
    cudaEventRecord(a);
    bilateral_kernel<<<grid,block>>>(di,dout,(int)in->width,(int)in->height,0,(int)in->height,c->window_size/2,c->sigma_spatial,c->sigma_range);
    if(cudaGetLastError()!=cudaSuccess)goto fail;cudaEventRecord(b);cudaEventSynchronize(b);cudaEventElapsedTime(&elapsed,a,b);if(timing)timing->kernel_ms=elapsed;
    cudaEventRecord(a);e=cudaMemcpy(out->pixels,dout,bytes,cudaMemcpyDeviceToHost);cudaEventRecord(b);cudaEventSynchronize(b);if(e!=cudaSuccess)goto fail;
    cudaEventElapsedTime(&elapsed,a,b);if(timing){timing->d2h_ms=elapsed;timing->wall_ms=(monotonic_seconds()-start)*1000.0;}
    cudaFree(di);cudaFree(dout);cudaEventDestroy(a);cudaEventDestroy(b);return DENOISE_OK;
fail:
    if(di)cudaFree(di);if(dout)cudaFree(dout);cudaEventDestroy(a);cudaEventDestroy(b);image_free(out);return DENOISE_RUNTIME;
}
