#include "denoise.h"
#include "build_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef struct {
    const char *mode, *input, *output, *clean_output, *noisy_output, *csv;
    size_t width, height; int threads, cpu_percent, repetitions;
    float noise_sigma; uint64_t seed; BilateralConfig filter;
} Options;

static void defaults(Options *o) {
    memset(o,0,sizeof(*o)); o->mode="serial"; o->width=o->height=256; o->threads=4;
    o->cpu_percent=25;o->repetitions=5;o->noise_sigma=25.0f;o->seed=42;
    o->filter.window_size=9;o->filter.sigma_spatial=3.0f;o->filter.sigma_range=25.0f;
}
static const char *value(int argc,char **argv,int *i){return *i+1<argc?argv[++*i]:NULL;}
static int parse(int argc,char **argv,Options *o){int i;const char*v;for(i=2;i<argc;++i){
    if(!strcmp(argv[i],"--mode")){if(!(v=value(argc,argv,&i)))return 0;o->mode=v;}
    else if(!strcmp(argv[i],"--input")){if(!(o->input=value(argc,argv,&i)))return 0;}
    else if(!strcmp(argv[i],"--output")){if(!(o->output=value(argc,argv,&i)))return 0;}
    else if(!strcmp(argv[i],"--clean-output")){if(!(o->clean_output=value(argc,argv,&i)))return 0;}
    else if(!strcmp(argv[i],"--noisy-output")){if(!(o->noisy_output=value(argc,argv,&i)))return 0;}
    else if(!strcmp(argv[i],"--csv")){if(!(o->csv=value(argc,argv,&i)))return 0;}
    else if(!strcmp(argv[i],"--width")){if(!(v=value(argc,argv,&i)))return 0;o->width=(size_t)strtoull(v,NULL,10);}
    else if(!strcmp(argv[i],"--height")){if(!(v=value(argc,argv,&i)))return 0;o->height=(size_t)strtoull(v,NULL,10);}
    else if(!strcmp(argv[i],"--threads")){if(!(v=value(argc,argv,&i)))return 0;o->threads=atoi(v);}
    else if(!strcmp(argv[i],"--cpu-percent")){if(!(v=value(argc,argv,&i)))return 0;o->cpu_percent=atoi(v);}
    else if(!strcmp(argv[i],"--repetitions")){if(!(v=value(argc,argv,&i)))return 0;o->repetitions=atoi(v);}
    else if(!strcmp(argv[i],"--noise-sigma")){if(!(v=value(argc,argv,&i)))return 0;o->noise_sigma=(float)atof(v);}
    else if(!strcmp(argv[i],"--seed")){if(!(v=value(argc,argv,&i)))return 0;o->seed=(uint64_t)strtoull(v,NULL,10);}
    else if(!strcmp(argv[i],"--window")){if(!(v=value(argc,argv,&i)))return 0;o->filter.window_size=atoi(v);}
    else if(!strcmp(argv[i],"--sigma-spatial")){if(!(v=value(argc,argv,&i)))return 0;o->filter.sigma_spatial=(float)atof(v);}
    else if(!strcmp(argv[i],"--sigma-range")){if(!(v=value(argc,argv,&i)))return 0;o->filter.sigma_range=(float)atof(v);}
    else return 0;
}return o->width>0&&o->height>0&&o->threads>0&&o->repetitions>0&&o->noise_sigma>=0&&isfinite(o->noise_sigma)&&bilateral_config_valid(&o->filter);}

static int prepare(const Options *o,Image *clean,Image *noisy){int s=o->input?image_load_pnm(o->input,clean):image_generate(clean,o->width,o->height);if(s)return s;
    s=image_add_gaussian_noise(clean,noisy,o->noise_sigma,o->seed);if(s){image_free(clean);return s;}
    if(o->clean_output&&(s=image_save_pgm(o->clean_output,clean)))return s;
    if(o->noisy_output&&(s=image_save_pgm(o->noisy_output,noisy)))return s;return DENOISE_OK;}
static int run_mode(const Options*o,const Image*in,Image*out,TimingInfo*t){memset(t,0,sizeof(*t));
    if(!strcmp(o->mode,"serial"))return filter_serial(in,out,&o->filter,t);
    if(!strcmp(o->mode,"openmp"))return filter_openmp(in,out,&o->filter,o->threads,t);
    if(!strcmp(o->mode,"cuda"))return filter_cuda(in,out,&o->filter,t);
    if(!strcmp(o->mode,"hybrid"))return filter_hybrid(in,out,&o->filter,o->threads,o->cpu_percent,t);
    return DENOISE_UNAVAILABLE;}
static void print_usage(void){fprintf(stderr,"Usage: denoise info | denoise <denoise|validate|benchmark> [--mode serial|openmp|cuda|hybrid] [options]\n");}
static int compare_double(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static void print_json_string(FILE *stream,const char *value){const unsigned char *p=(const unsigned char*)value;fputc('"',stream);for(;*p;++p){switch(*p){case '"':fputs("\\\"",stream);break;case '\\':fputs("\\\\",stream);break;case '\b':fputs("\\b",stream);break;case '\f':fputs("\\f",stream);break;case '\n':fputs("\\n",stream);break;case '\r':fputs("\\r",stream);break;case '\t':fputs("\\t",stream);break;default:if(*p<0x20)fprintf(stream,"\\u%04x",(unsigned)*p);else fputc(*p,stream);}}fputc('"',stream);}

static int command_denoise(const Options*o){Image clean={0},noisy={0},out={0};TimingInfo t;QualityMetrics q;int s;
    if(!o->output){fprintf(stderr,"--output is required\n");return DENOISE_INVALID;}s=prepare(o,&clean,&noisy);if(s)goto done;
    s=run_mode(o,&noisy,&out,&t);if(s)goto done;s=image_save_pgm(o->output,&out);if(s)goto done;quality_metrics(&clean,&out,&q);
    printf("{\"status\":\"ok\",\"mode\":\"%s\",\"width\":%zu,\"height\":%zu,\"wall_ms\":%.6f,\"kernel_ms\":%.6f,\"h2d_ms\":%.6f,\"d2h_ms\":%.6f,\"mse\":%.9g,\"psnr\":%.9g,\"output\":",o->mode,clean.width,clean.height,t.wall_ms,t.kernel_ms,t.h2d_ms,t.d2h_ms,q.mse,q.psnr);print_json_string(stdout,o->output);puts("}");
done:if(s)fprintf(stderr,"mode %s failed with status %d\n",o->mode,s);image_free(&clean);image_free(&noisy);image_free(&out);return s;}

static int command_validate(const Options*o){Image clean={0},noisy={0},reference={0},candidate={0};TimingInfo a,b;ValidationResult v;int s=prepare(o,&clean,&noisy);if(s)goto done;
    s=filter_serial(&noisy,&reference,&o->filter,&a);if(s)goto done;s=run_mode(o,&noisy,&candidate,&b);if(s)goto done;
    validate_images(&reference,&candidate,1e-3,&v);printf("{\"status\":\"%s\",\"mode\":\"%s\",\"tolerance\":0.001,\"max_abs_error\":%.9g,\"mean_abs_error\":%.9g,\"rmse\":%.9g,\"within_tolerance_percent\":%.9g,\"exact_u8_percent\":%.9g}\n",v.passed?"pass":"fail",o->mode,v.max_abs_error,v.mean_abs_error,v.rmse,v.within_tolerance_percent,v.exact_u8_percent);if(!v.passed)s=DENOISE_RUNTIME;
done:if(s&&s!=DENOISE_RUNTIME)fprintf(stderr,"validation failed with status %d\n",s);image_free(&clean);image_free(&noisy);image_free(&reference);image_free(&candidate);return s;}

static int command_benchmark(const Options*o){Image clean={0},noisy={0},out={0};TimingInfo t;double *times,sum=0,mean,variance=0,min,max,median,throughput;FILE*f=NULL;int s,i;
    s=prepare(o,&clean,&noisy);if(s)goto done;times=(double*)malloc((size_t)o->repetitions*sizeof(double));if(!times){s=DENOISE_ALLOC;goto done;}
    s=run_mode(o,&noisy,&out,&t);image_free(&out);if(s){free(times);goto done;}
    for(i=0;i<o->repetitions;++i){s=run_mode(o,&noisy,&out,&t);if(s){free(times);goto done;}times[i]=t.wall_ms;sum+=times[i];image_free(&out);}
    min=max=times[0];for(i=0;i<o->repetitions;++i){if(times[i]<min)min=times[i];if(times[i]>max)max=times[i];}mean=sum/o->repetitions;
    for(i=0;i<o->repetitions;++i){double d=times[i]-mean;variance+=d*d;}variance=sqrt(variance/o->repetitions);qsort(times,(size_t)o->repetitions,sizeof(double),compare_double);median=times[o->repetitions/2];if(!(o->repetitions&1))median=(times[o->repetitions/2-1]+median)/2.0;
    throughput=(double)(clean.width*clean.height)/(median*1000.0);
    if(o->csv){f=fopen(o->csv,"wb");if(!f){free(times);s=DENOISE_IO;goto done;}fprintf(f,"mode,width,height,window,threads,repetitions,median_ms,min_ms,max_ms,stddev_ms,throughput_mpix_s\n%s,%zu,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",o->mode,clean.width,clean.height,o->filter.window_size,o->threads,o->repetitions,median,min,max,variance,throughput);fclose(f);}
    printf("{\"status\":\"ok\",\"mode\":\"%s\",\"width\":%zu,\"height\":%zu,\"window\":%d,\"threads\":%d,\"repetitions\":%d,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"stddev_ms\":%.6f,\"throughput_mpix_s\":%.6f}\n",o->mode,clean.width,clean.height,o->filter.window_size,o->threads,o->repetitions,median,min,max,variance,throughput);free(times);
done:if(s)fprintf(stderr,"benchmark failed with status %d\n",s);image_free(&clean);image_free(&noisy);image_free(&out);return s;}

int main(int argc,char**argv){Options o;defaults(&o);if(argc<2){print_usage();return DENOISE_INVALID;}
    if(!strcmp(argv[1],"info")){int gpu=cuda_runtime_available();printf("{\"status\":\"ok\",\"compiler\":\"%s\",\"serial\":{\"compiled\":true,\"runtime\":true},\"openmp\":{\"compiled\":%s,\"runtime\":%s},\"mpi\":{\"compiled\":%s,\"runtime\":false},\"cuda\":{\"compiled\":%s,\"runtime\":%s},\"hybrid\":{\"compiled\":%s,\"runtime\":%s}}\n",BUILD_C_COMPILER,BUILD_HAVE_OPENMP?"true":"false",BUILD_HAVE_OPENMP?"true":"false",BUILD_HAVE_MPI?"true":"false",
#ifdef HAVE_CUDA
    "true",gpu?"true":"false","true",(gpu&&BUILD_HAVE_OPENMP)?"true":"false"
#else
    "false","false","false","false"
#endif
    );return 0;}
    if(!parse(argc,argv,&o)){print_usage();return DENOISE_INVALID;}
    if(!strcmp(argv[1],"denoise"))return command_denoise(&o);if(!strcmp(argv[1],"validate"))return command_validate(&o);if(!strcmp(argv[1],"benchmark"))return command_benchmark(&o);print_usage();return DENOISE_INVALID;}
