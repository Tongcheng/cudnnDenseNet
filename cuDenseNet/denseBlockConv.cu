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

void print1D_Array(float* A,int len){
    float* hostA = new float[len];
    cudaMemcpy(hostA,A,len * sizeof(float),cudaMemcpyDeviceToHost);
    for (int i=0;i<len;++i) {
        printf("%f,",hostA[i]);
    }
    printf("\n");
}

void print4D_Array(float* A,int len1,int len2,int len3,int len4){
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

//TODO: currently do nothing: identity transform
void BatchNormalize(int curLevel,float* dataRegion_preBN,float* dataRegion_postBN,int numLayer,int n,int c,int h,int w){
    int commonGap = numLayer * c * h * w;
    float* srcStartPtr = dataRegion_preBN + curLevel*c*h*w; 
    float* desStartPtr = dataRegion_postBN + curLevel*c*h*w;
    int chunkSize = c*h*w;
    scatterCudaMemcpy(srcStartPtr,desStartPtr,n,chunkSize,commonGap,commonGap,cudaMemcpyDeviceToDevice);
}

float* BlockConvolutionForward(int numLayer,int n_miniBatch,int c_channels,int h_img,int w_img,float* input_host,int h_filter,int w_filter,float* filter_host,int pad_h,int pad_w,int conv_horizentalStride, int conv_verticalStride){
    //test especially, the cudnnConvolutionForward
    cudnnHandle_t* handlePtr = new cudnnHandle_t;
    cudnnCreate(handlePtr);
    float* alphaPtr = new float[1]; alphaPtr[0] = 1.0;
    float* betaPtr = new float[1]; betaPtr[0] = 0.0;
    //deploy data to GPU:
    //deploy input
    float* allTensorData_preBN;
    float* allTensorData;
    int singleImgGap_host = c_channels * h_img * w_img;
    int singleImgGap_gpu = numLayer * c_channels * h_img * w_img;
    //preBN data region
    cudaMalloc(&allTensorData_preBN,n_miniBatch * singleImgGap_gpu * sizeof(float));
    cudaMemset(allTensorData_preBN, 0 ,n_miniBatch * singleImgGap_gpu * sizeof(float));
    scatterCudaMemcpy(input_host,allTensorData_preBN,n_miniBatch,singleImgGap_host,singleImgGap_host,singleImgGap_gpu,cudaMemcpyHostToDevice);  
    //postBN data region
    cudaMalloc(&allTensorData,n_miniBatch * singleImgGap_gpu * sizeof(float));
    cudaMemset(allTensorData, 0 ,n_miniBatch * singleImgGap_gpu * sizeof(float)); 
    //deploy filter param
    float* allFilterData;
    cudaMalloc(&allFilterData,(numLayer - 1) * c_channels * c_channels * h_filter * w_filter * sizeof(float));
    cudaMemcpy(allFilterData,filter_host,(numLayer - 1)*c_channels*c_channels*h_filter*w_filter * sizeof(float),cudaMemcpyHostToDevice);
    //prepare workspace:about 400MB     
    float* workspace;
    int workspaceSize = 100000000*sizeof(float);
    cudaMalloc(&workspace, workspaceSize);
    //do the transformation where # of transform = # of layer - 1
    for (int transformIdx=0;transformIdx < numLayer - 1;++transformIdx){
        //phase 1: BN
        BatchNormalize(transformIdx,allTensorData_preBN,allTensorData,numLayer,n_miniBatch,c_channels,h_img,w_img);
        //printf("After BN in transform %d\n",transformIdx);
        //print1D_Array(allTensorData,n_miniBatch * singleImgGap_gpu);
        //printf("\n\n");
        //phase 2: ReLU: in place
        float* ReLU_DataPtr = allTensorData + transformIdx*c_channels*h_img*w_img;
	cudnnTensorDescriptor_t* ReLU_Descriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(ReLU_Descriptor);
        cudnnSetTensor4dDescriptorEx(*ReLU_Descriptor,CUDNN_DATA_FLOAT,n_miniBatch,c_channels,h_img,w_img,numLayer*c_channels*h_img*w_img,h_img*w_img,w_img,1);
        cudnnActivationDescriptor_t* activationDescPtr = new cudnnActivationDescriptor_t;
        cudnnCreateActivationDescriptor(activationDescPtr);
        cudnnSetActivationDescriptor(*activationDescPtr,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0.0);
        cudnnActivationForward(*handlePtr,*activationDescPtr,alphaPtr,
            *ReLU_Descriptor,ReLU_DataPtr,betaPtr,*ReLU_Descriptor,ReLU_DataPtr	
	);       
 
        //phase 3: Convolution
        //prepare dataPtrs:
        float* inputDataPtr = allTensorData + transformIdx*c_channels*h_img*w_img;
        float* outputDataPtr = allTensorData_preBN + (transformIdx+1)*c_channels*h_img*w_img;
        float* filterDataPtr = allFilterData + transformIdx*c_channels*c_channels*h_filter*w_filter;
        //print1D_Array(filterDataPtr,c_channels * c_channels * h_filter * w_filter);
        //prepare descriptors
        //in/out descriptor
        cudnnTensorDescriptor_t* inputDescriptor = new cudnnTensorDescriptor_t;
        cudnnTensorDescriptor_t* outputDescriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(inputDescriptor);
        cudnnCreateTensorDescriptor(outputDescriptor); 
        cudnnSetTensor4dDescriptorEx(*inputDescriptor,CUDNN_DATA_FLOAT,n_miniBatch,c_channels,h_img,w_img,numLayer*c_channels*h_img*w_img,h_img*w_img,w_img,1);
        cudnnSetTensor4dDescriptorEx(*outputDescriptor,CUDNN_DATA_FLOAT,n_miniBatch,c_channels,h_img,w_img,numLayer*c_channels*h_img*w_img,h_img*w_img,w_img,1);
        //filter descriptor
        cudnnFilterDescriptor_t* filterDescriptor = new cudnnFilterDescriptor_t;
        cudnnCreateFilterDescriptor(filterDescriptor);
        cudnnSetFilter4dDescriptor(*filterDescriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,c_channels,c_channels,h_filter,w_filter);
        //convolution descriptor
        cudnnConvolutionDescriptor_t* convolutionDescriptor = new cudnnConvolutionDescriptor_t;
        cudnnCreateConvolutionDescriptor(convolutionDescriptor);
        cudnnSetConvolution2dDescriptor(*convolutionDescriptor,pad_h,pad_w,conv_verticalStride,conv_horizentalStride,1,1,CUDNN_CONVOLUTION);
        layerConvolutionForward(handlePtr,alphaPtr,betaPtr,
	    inputDescriptor,inputDataPtr,
            filterDescriptor,filterDataPtr,
	    outputDescriptor,outputDataPtr,
	    convolutionDescriptor,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
	    workspace, workspaceSize
        );
        //print1D_Array(allTensorData_preBN, n_miniBatch * singleImgGap_gpu); 
    }
    //convolution output data after the DenseBlock
    float* output_host = new float[n_miniBatch * c_channels * h_img * w_img];
    scatterCudaMemcpy(allTensorData_preBN+(numLayer-1)*c_channels*h_img*w_img,output_host,n_miniBatch,c_channels*h_img*w_img,singleImgGap_gpu,singleImgGap_host,cudaMemcpyDeviceToHost); 
    return output_host;
}
