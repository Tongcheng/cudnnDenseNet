#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

const NUM_THREADS = 512;
inline int NUM_BLOCKS(int N){
    return (N + NUM_THREADS - 1)/NUM_THREADS;
}

// From Caffe Implementation:
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

void printGPUArray(float* gpuArray,int len){
    float* cpuArray = (float*) malloc(len * sizeof(float));
    cudaMemcpy(cpuArray,gpuArray,len * sizeof(float),cudaMemcpyDeviceToHost);
    for (int i=0;i<len;++i){
        printf("%f\n",cpuArray[i]);
    }
}

void printCPUArray(float* cpuArray,int len){
    for (int i=0;i<len;++i) printf("%f\n",cpuArray[i]);
}

/*helper function for computing per-channel of 4D tensor of N*C*H*W*/
//compute mean of the channel given by channelIdx
//longConstVec_gpu 's element has value 1.0/spatialDimension
//dataRegion pointer: start Ptr of 4d tensor in N*C*H*W with C gap
//channelStartIdx is inclusive, channelEndIdx is not inclusive.
void tensorMean(int channelStartIdx,int channelEndIdx,float* dataRegion_gpu,int N,int C,int H,int W,float* spatialOneVec_gpu,float* output_gpuVec){
    int numChannels = channelEndIdx - channelStartIdx;
    int spatialDim = H * W;
    float* localAlpha = (float*) malloc(sizeof(float)); localAlpha[0] = 1.0;
    float* localBeta = (float*) malloc(sizeof(float)); localBeta[0] = 0.0;
    float* localBeta2 = (float*) malloc(sizeof(float)); localBeta2[0] = 1.0;
    //outer loop: sum over images in miniBatch
    for (int n=0;n<N;++n){
        float* localgpu_sum_n;
        cudaMalloc(&localgpu_sum_n,(channelEndIdx - channelStartIdx) * sizeof(float));
	cudaMemset(localgpu_sum_n,0,(channelEndIdx - channelStartIdx) * sizeof(float));
	float* localA = dataRegion_gpu + (n*C*H*W) + (channelStartIdx * H * W);	
        //sgemv: sum over all spatial dim
        //multiply M: all covered channels per image and V: spatialOneVec_gpu 
        cublasHandle_t handle; 
	cublasCreate(&handle);
	cublasSgemv(handle,CUBLAS_OP_T,
	    spatialDim,numChannels,
	    localAlpha,localA,spatialDim,
	    spatialOneVec_gpu,1,
	    localBeta,localgpu_sum_n,1
	);
        //printf("Iteration for image %d\n",n);
	//printGPUArray(localgpu_sum_n,numChannels);
	//vec-add
        cublasSaxpy(handle,channelEndIdx - channelStartIdx,localAlpha,localgpu_sum_n,1,output_gpuVec,1);
	//printGPUArray(output_gpuVec,numChannels);
	cublasDestroy(handle);
	cudaFree(localgpu_sum_n);
    }        
}

/*Remove mean using per-channel vector from a 4D tensor of N*C*H*W in-place*/
//It not want in-place, just copy first before removeMean
void removeMean(int N,int C,int H,int W,int channelStartIdx,int channelEndIdx,float* dataRegion_gpu,float* channelMean_gpu,float* spatialOneVec_gpu){
    //loop over each continuous region (one image), do rank-1 update
    for (int n=0;n<N;++n){
        float* localStartPtr = dataRegion_gpu + (n*C*H*W) + (channelStartIdx*H*W);
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSger(handle,channelEndIdx - channelStartIdx,H*W,-1.0,channelMean_gpu,1.0,spatialOneVec_gpu,1.0,dataRegion_gpu,channelEndIdx - channelStartIdx);	
    }
}

//in-place square kernel 
__global__ void square_Kernel(int N,float* A){
    CUDA_KERNEL_LOOP(index, n) {
	A[index] = A[index] * A[index];
    } 
}

//in-place square operation
void tensorSquare(int N,float* dataRegion_gpu){
    square_Kernel<<<NUM_BLOCKS(N),NUM_THREADS>>>(N,dataRegion_gpu);
}

/*
void testTensorMean(int N,int C,int H,int W,int channelStart,int channelEnd,float* dataRegion_cpu,float* output_cpuVec){
    int spatialDim = H * W;
    float* longConstVec_cpu = (float*)malloc(spatialDim * sizeof(float));
    for (int i=0;i<H*W;++i) longConstVec_cpu[i] = 1.0 / (spatialDim * N);
    float* longConstVec_gpu;
    cudaMalloc(&longConstVec_gpu,H*W * sizeof(float));
    cudaMemcpy(longConstVec_gpu,longConstVec_cpu,H*W * sizeof(float),cudaMemcpyHostToDevice); 
    float* dataRegion_gpu;
    cudaMalloc(&dataRegion_gpu,N*C*H*W*sizeof(float));
    cudaMemcpy(dataRegion_gpu,dataRegion_cpu,N * C * H * W * sizeof(float),cudaMemcpyHostToDevice);
    float* output_gpuVec;
    cudaMalloc(&output_gpuVec,(channelEnd - channelStart) * sizeof(float));
    cudaMemset(output_gpuVec,0,(channelEnd - channelStart) * sizeof(float));
    tensorMean(channelStart,channelEnd,dataRegion_gpu,N,C,H,W,longConstVec_gpu,output_gpuVec);
    cudaMemcpy(output_cpuVec,output_gpuVec,(channelEnd - channelStart) * sizeof(float),cudaMemcpyDeviceToHost);
    printGPUArray(output_gpuVec,(channelEnd- channelStart));
    printCPUArray(output_cpuVec,(channelEnd- channelStart));
}
*/

//both floatParams and intParams is array on CPU, stands for configuration of this BatchNormalization Op
//intParams[0]: global stats usage, 1 iff use.
//
//floatParams[0]: moving average fraction
//floatParams[1]: epsilon
void BatchNormalizeOp_Forward(float* dataRegion_gpu,int channel_gap,int N,int C,int H,int W,int* intParams,float* floatParams){
    //get Op params
    float moving_average_fraction = floatParams[0];float epsilon = floatParams[1];
    int use_gloabl_stats = intParams[0];
    //compute mean
    if (use_global_stats){}
    else {}
    //subtract by mean
   

    if (!use_global_stats){
        //compute variance
        
    }    
}

void BatchNormalizeOp_Backward(float* dataRegion,int channel_gap,int N,int C,int H,int W,int* intParams,float* floatParams){

}
