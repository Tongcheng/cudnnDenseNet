#include "cudnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <cuda_runtime.h>


//////////////////////////////////////////////////////////////////////////////
// Error handling
// Adapted from the CUDNN classification code 
// sample: https://developer.nvidia.com/cuDNN

#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

//
void scatterCudaMemcpy(float* srcPtr,float* desPtr,int n,int singleSize,int gapSrc,int gapDes,cudaMemcpyKind kind){
    for (int i=0;i<n;++i){
	cudaMemcpy(desPtr + i*gapDes, srcPtr + i*gapSrc,singleSize * sizeof(float),kind);
    }
}

//a wrapper for cudnnConvolutionForward
//alpha,beta in host memory
//srcData and destData allow nStride, 
//filterData is continuous
void layerConvolutionForward(cudnnHandle_t* globalHandle,
    float* alpha,float* beta,
    cudnnTensorDescriptor_t* srcDesc,float* srcData,
    cudnnFilterDescriptor_t* filterDesc, float* filterData,
    cudnnTensorDescriptor_t* destDesc, float* destData, 
    cudnnConvolutionDescriptor_t* convDesc, cudnnConvolutionFwdAlgo_t convAlgo,
    void* workspace, size_t workspaceSize){
      checkCUDNN(  cudnnConvolutionForward(*globalHandle,alpha,*srcDesc,srcData,*filterDesc,filterData,*convDesc,convAlgo,workspace,workspaceSize,beta,*destDesc,destData) ); 

}

void printGPUArray(float* A,int len1,int len2,int len3,int len4){
    int len = len1 * len2 * len3 * len4;
    float* hostA = new float[len];
    cudaMemcpy(hostA,A,len * sizeof(float),cudaMemcpyDeviceToHost);
    for (int i1=0;i1<len1;++i1){
      for (int i2=0;i2<len2;++i2){
	for (int i3=0;i3<len3;++i3){
	  for (int i4=0;i4<len4;++i4){
            printf("%f,",hostA[i4+i3*len4+i2*len3*len4+i1*len2*len3*len4]);
	  }
	  printf("\n");
	}
	printf("\n");
      }
      printf("\n");
    }
    printf("\n");
};

//first test a convolution with stride, make sure everything works
//both return and inputs are in host memory
//continuous host pointer as input
//by Reading cudnn.h, xStride is just diff (subtraction)
float* testCUDNN(int n_miniBatch,int c_channels,int h_img,int w_img,float* input_host,int h_filter,int w_filter,float* filter_host,int pad_h,int pad_w,int conv_horizentalStride, int conv_verticalStride){
    //test especially, the cudnnConvolutionForward
    cudnnHandle_t* handlePtr = new cudnnHandle_t;
    cudnnCreate(handlePtr); 
    float* alphaPtr = new float[1]; alphaPtr[0] = 1.0;
    float* betaPtr = new float[1]; betaPtr[0] = 0.0;
    
    //prepare data for two tensor
    float* allTensorData;
    int singleImgGap_host = c_channels * h_img * w_img;
    int singleImgGap_gpu = 2 * c_channels * h_img * w_img;
    cudaMalloc(&allTensorData,n_miniBatch * singleImgGap_gpu * sizeof(float)); 
    cudaMemset(allTensorData,0,n_miniBatch * singleImgGap_gpu*sizeof(float));
    //for each i in n_miniBatch, do a cudaMemcpy
    scatterCudaMemcpy(input_host,allTensorData,n_miniBatch,singleImgGap_host,singleImgGap_host,singleImgGap_gpu,cudaMemcpyHostToDevice);
    float* inputDataPtr = allTensorData;
    float* outputDataPtr = allTensorData + c_channels * h_img * w_img;
    //prepare two descriptors for Tensor
    cudnnTensorDescriptor_t* inputDescriptor = new cudnnTensorDescriptor_t;
    cudnnTensorDescriptor_t* outputDescriptor = new cudnnTensorDescriptor_t;
    cudnnCreateTensorDescriptor(inputDescriptor);
    cudnnCreateTensorDescriptor(outputDescriptor);
    cudnnSetTensor4dDescriptorEx(*inputDescriptor,CUDNN_DATA_FLOAT,n_miniBatch,c_channels,h_img,w_img,2*c_channels*h_img*w_img,h_img*w_img,w_img,1);
    cudnnSetTensor4dDescriptorEx(*outputDescriptor,CUDNN_DATA_FLOAT,n_miniBatch,c_channels,h_img,w_img,2*c_channels*h_img*w_img,h_img*w_img,w_img,1);
    
    //prepare data for filter
    float* allFilterData;
    cudaMalloc(&allFilterData,c_channels*c_channels*h_filter*w_filter*sizeof(float));
    cudaMemcpy(allFilterData,filter_host,c_channels*c_channels*h_filter*w_filter*sizeof(float),cudaMemcpyHostToDevice);

    //prepare descriptor for filter 
    cudnnFilterDescriptor_t* filterDescriptor = new cudnnFilterDescriptor_t;
    cudnnCreateFilterDescriptor(filterDescriptor);
    cudnnSetFilter4dDescriptor(*filterDescriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,c_channels,c_channels,h_filter,w_filter);
    
    //prepare convolution descriptor
    cudnnConvolutionDescriptor_t* convolutionDescriptor = new cudnnConvolutionDescriptor_t;
    cudnnCreateConvolutionDescriptor(convolutionDescriptor);
    cudnnSetConvolution2dDescriptor(*convolutionDescriptor,pad_h,pad_w,conv_verticalStride,conv_horizentalStride,1,1,CUDNN_CONVOLUTION); 
    //400 MB of workspace
    float* workspace;
    int workspaceSize = 100000000*sizeof(float);
    cudaMalloc(&workspace, workspaceSize);
   
    printGPUArray(allTensorData,n_miniBatch,2*c_channels,h_img,w_img); 
    //call the function   
    layerConvolutionForward(handlePtr,alphaPtr,betaPtr,
        inputDescriptor,inputDataPtr,filterDescriptor,allFilterData,
        outputDescriptor,outputDataPtr,
        convolutionDescriptor,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        workspace, workspaceSize 
    );

    printf("after convolution\n");
    printGPUArray(allTensorData,n_miniBatch,2*c_channels,h_img,w_img); 
    
    //data copy back to host 
    float* output_host = new float[n_miniBatch * c_channels * h_img * w_img];
    scatterCudaMemcpy(allTensorData + c_channels*h_img*w_img,output_host,n_miniBatch,singleImgGap_host,singleImgGap_gpu,singleImgGap_host,cudaMemcpyDeviceToHost); 
    //for test purposes, we want to see what is inside allTensorData
    float* allTensor_host = new float[n_miniBatch*singleImgGap_gpu*sizeof(float)];
    cudaMemcpy(allTensor_host,allTensorData,n_miniBatch*singleImgGap_gpu*sizeof(float),cudaMemcpyDeviceToHost);

    return output_host;
    //return allTensor_host;//just for test  
}
