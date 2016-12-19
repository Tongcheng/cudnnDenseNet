#include "cudnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

float** GPU_filterDeploy(float** filter_host,int numTransform,int initChannel,int growthRate,int N,int filter_H,int filter_W){
    float** output_ptrs = new float*[numTransform];
    for (int transformIdx=0;transformIdx < numTransform;++transformIdx){
        float* output_gpu_local;
	int outputNumChannels = growthRate;
	int inputNumChannels = transformIdx * growthRate + initChannel;
	int localSize_bytes = outputNumChannels * inputNumChannels * H * W * sizeof(float);
        cudaMalloc(&output_gpu_local,localSize_bytes);
	cudaMemcpy(output_gpu_local,filter_host[transformIdx],localSize_bytes,cudaMemcpyHostToDevice);
        output_ptrs[transformIdx] = output_gpu_local; 
    }
    return output_ptrs;
}

float** GPU_miscDeploy(float* BNScaler_host,float* BNBias_host,int numTransform,int initChannel,int growthRate,int N,int H,int W,int workspaceSize){
    float** output_ptrs = new float*[10];
    int totalNumChannel = initChannel+numTransform*growthRate;  
    //index 0 is BN_Scaler_Vec
    float* output_BN_Scaler;
    cudaMalloc(&output_BN_Scaler,totalNumChannel*sizeof(float));
    cudaMemcpy(output_BN_Scaler,BNScaler_host,totalNumChannel*sizeof(float),cudaMemcpyHostToDevice);
    output_ptrs[0] = output_BN_Scaler;
    //index 1 is BN_Bias_Vec
    float* output_BN_Bias;
    cudaMalloc(&output_BN_Bias,totalNumChannel*sizeof(float));
    cudaMemcpy(output_BN_Bias,);
    output_ptrs[1] = output_BN_Bias;
    //index 2 is ResultRunningMean
    float* output_ResultRunningMean;
    cudaMalloc(&output_ResultRunningMean,totalNumChannel*sizeof(float));
    cudaMemset(output_ResultRunningMean,0,totalNumChannel*sizeof(float));
    output_ptrs[2] = output_ResultRunningMean;    
    //index 3 is ResultRunningVariance
    float* output_ResultRunningVariance;
    cudaMalloc(&output_ResultRunningVariance,totalNumChannel*sizeof(float));
    cudaMemset(output_ResultRunningVariance,0,totalNumChannel*sizeof(float)); 
    output_ptrs[3] = output_ResultRunningVariance; 
    //index 4 is ResultSaveMean
    float* output_ResultSaveMean;
    cudaMalloc(&output_ResultSaveMean,totalNumChannel*sizeof(float));
    cudaMemset(output_ResultSaveMean,0,totalNumChannel*sizeof(float));
    output_ptrs[4] = output_ResultSaveMean;
    //index 5 is ResultSaveInvVariance
    float* output_ResultSaveInvVariance;
    cudaMalloc(&output_ResultSaveInvVariance,totalNumChannel*sizeof(float));
    cudaMemset(output_ResultSaveInvVariance,0,totalNumChannel*sizeof(float));
    output_ptrs[5] = output_ResultSaveInvVariance;
    //index 6 is postConv_dataRegion
    int postSize = N*(initChannel+growthRate*numTransform)*H*W*sizeof(float); 
    float* postConv_dataPtr;
    cudaMalloc(&postConv_dataPtr,postSize); 
    cudaMemset(postConv_dataPtr,0,postSize);
    output_ptrs[6] = postConv_dataPtr; 
    //index 7 is postBN_dataRegion
    float* postBN_dataPtr;
    cudaMalloc(&postBN_dataPtr,postSize);
    cudaMemset(postBN_dataPtr,0,postSize);
    output_ptrs[7] = postBN_dataPtr;
    //index 8 is postReLU_dataRegion
    float* postReLU_dataPtr;
    cudaMalloc(&postReLU_dataPtr,postSize);
    cudaMemset(postReLU_dataPtr,0,postSize);
    output_ptrs[8] = postReLU_dataPtr; 
    //index 9 is workspace
    float* workspacePtr;
    cudaMalloc(&workspacePtr,workspaceSize);
    cudaMemset(workspacePtr,0,workspaceSize);
    output_ptrs[9] = workspacePtr; 
    //done and return
    return output_ptrs;  
}

/*DenseLayer: For each small transition within DenseLayer, do BN->ReLU->Convolution*/
//Input: # of channel = k0 + k(Order - 1)
//Output: # of channel = k
//testMode: 1 if test, 0 if train
//trainCycleIdx: the idx for current training cycle, related to EMA of BN, doesn't matter if in test
//BNScalerVec, BNBiasVec: Scaler and Bias per channel.
//resultRunningMean, resultRunningVariance: per channel.
//resultSaveMean, resultSaveInvVariance: null in testing phase.
//numTransition: number of BN->ReLU->Convolutions
//filter_transform: filter_transform on cpu,filter_transform[i] is on gpu 
void DenseBlockForward(int initChannel,int growthRate,int numTransition,
  int N,int H,int W,int pad_h,int pad_w,int conv_verticalStride,int conv_horizentalStride,
  int testMode, int trainCycleIdx,
  float* BNScalerVec, float* BNBiasVec,float* resultRunningMean, float* resultRunningVariance, float* resultSaveMean, float* resultSaveInvVariance, 
  float* postConv_dataRegion, float* postBN_dataRegion, float* postReLU_dataRegion,
  float** filter_transform,int filter_H,int filter_W,
  float* workspace_gpu,int workspaceSize
    ){
    cudnnHandle_t* handlePtr = new cudnnHandle_t;
    cudnnCreate(handlePtr);
    float* oneScalerPtr = new float[1]; oneScalerPtr[0] = 1.0;
    float* zeroScalerPtr = new float[1]; zeroScalerPtr[0] = 0.0; 
    for (int transitionIdx=0;transitionIdx < numTransition;++transitionIdx){
	//BN transform
	cudnnTensorDescriptor_t* BN_x_Descriptor = new cudnnTensorDescriptor_t;
   	cudnnTensorDescriptor_t* BN_y_Descriptor = new cudnnTensorDescriptor_t;
    	cudnnTensorDescriptor_t* BN_param_Descriptor = new cudnnTensorDescriptor_t;
        cudnnTensorDescriptor_t* ReLU_y_Descriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(BN_x_Descriptor);
    	cudnnCreateTensorDescriptor(BN_y_Descriptor);
        cudnnCreateTensorDescriptor(BN_param_Descriptor);
        cudnnCreateTensorDescriptor(ReLU_y_Descriptor);
	//same channel size pre and post Mapping
        int numChannelTransform = growthRate;
        if (transitionIdx==0){numChannelTransform = initChannel;}
	
        cudnnSetTensor4dDescriptorEx(*BN_x_Descriptor,CUDNN_DATA_FLOAT,N,numChannelTransform,H,W,
          (numTransition*growthRate+initChannel)*H*W,H*W,W,1
        );
        cudnnSetTensor4dDescriptorEx(*BN_y_Descriptor,CUDNN_DATA_FLOAT,N,numChannelTransform,H,W,
	  (numTransition*growthRate+initChannel)*H*W,H*W,W,1
	);
        cudnnSetTensor4dDescriptorEx(*ReLU_y_Descriptor,CUDNN_DATA_FLOAT,N,numChannelTransform,H,W,
	  (numTransition*growthRate+initChannel)*H*W,H*W,W,1
	);      
 
	if (transitionIdx==0){
            cudnnSetTensor4dDescriptor(*BN_param_Descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,initChannel,1,1);
        } else {
	    cudnnSetTensor4dDescriptor(*BN_param_Descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,growthRate,1,1);
        }
        int channelsBefore_noself = (transformIdx==0?0:initChannel+(transformIdx-1) * growth);
	int channelsBefore_self = initChannel + transitionIdx * growth;
        float* BN_x_ptr = postConv_dataRegion+channelsBefore_noself*H*W;
	float* BN_y_ptr = postBN_dataRegion+channelsBefore_noself*H*W;
        float* BN_scaler_local = BNScalerVec + channelsBefore_noself;
	float* BN_bias_local = BNBiasVec + channelsBefore_noself; 
        float* BN_mean_local = resultRunningMean + channelsBefore_noself; 
	float* BN_var_local = resultRunningVariance + channelsBefore_noself;
	if (testMode){
	    cudnnBatchNormalizationForwardInference(*handlePtr,CUDNN_BATCHNORM_SPATIAL,oneScalerPtr,zeroScalerPtr,*BN_x_Descriptor,BN_x_ptr,*BN_y_Descriptor,BN_y_ptr,*BN_param_Descriptor,BN_scaler_local,BN_bias_local,BN_mean_local,BN_var_local,CUDNN_BN_MIN_EPSILON);
        }
	else {
	    float* resultSaveMean_local = resultSaveMean + channelsBefore_noself;
	    float* resultSaveInvVariance_local = resultSaveInvVariance + channelsBefore_noself;
            float exponentialMovingAverageFactor = 1.0/(1+trainCycleIdx);
	    cudnnBatchNormalizationForwardTraining(*handlePtr,CUDNN_BATCHNORM_SPATIAL,oneScalerPtr,zeroScalerPtr,*BN_x_Descriptor,BN_x_ptr,*BN_y_Descriptor,BN_y_ptr,*BN_param_Descriptor,BN_scaler_local,BN_bias_local,BN_mean_local,BN_var_local,CUDNN_BN_MIN_EPSILON,resultSaveMean_local,resultSaveInvVariance_local);
        }
	//ReLU transform
        float* ReLU_y_ptr = postBN_dataRegion+channelsBefore_noself*H*W; 
	cudnnActivationDescriptor_t* activationDescPtr = new cudnnActivationDescriptor_t;
	cudnnCreateActivationDescriptor(activationDescPtr);
	cudnnSetActivationDescriptor(*activationDescPtr,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0.0);
        cudnnActivationForward(*handlePtr,*activationDescPtr,oneScalerPtr,*BN_y_Descriptor,BN_y_ptr,zeroScalerPtr,ReLU_y_Descriptor,ReLU_y_ptr);
        //Convolution
	//Convolution::tensor Descriptor
        cudnnTensorDescriptor_t* Conv_x_Descriptor = new cudnnTensorDescriptor_t;
        cudnnTensorDescriptor_t* Conv_y_Descriptor = new cudnnTensorDescriptor_t; 
        cudnnCreateTensorDescriptor(Conv_x_Descriptor);
        cudnnCreateTensorDescriptor(Conv_y_Descriptor);
	cudnnSetTensor4dDescriptorEx(*Conv_x_Descriptor,CUDNN_DATA_FLOAT,N,channelsBefore_self,H,W,(numTransition*growthRate+initChannel)*H*W,H*W,W,1);
        cudnnSetTensor4dDescriptorEx(*Conv_y_Descriptor,CUDNN_DATA_FLOAT,N,growthRate,H,W,(numTransition*growthRate+initChannel)*H*W,H*W,W,1);
	//Convolution::tensor Ptr
        int delayChannels = initChannel+growthRate*transitionIdx;
	float* conv_x_local = postConv_dataRegion; 
	float* conv_y_local = postConv_dataRegion + delayChannels*H*W;
        //Convolution::filter Descriptor
	cudnnFilterDescriptor_t* filterDescriptor = new cudnnFilterDescriptor_t;
	cudnnCreateFilterDescriptor(filterDescriptor);
	cudnnSetFilter4dDescriptor(*filterDescriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,growthRate,channelsBefore_self,filter_H,filter_W);   	
	//Convolution::convolution Descriptor	
        cudnnConvolutionDescriptor_t* convolutionDescriptor = new cudnnConvolutionDescriptor_t;
        cudnnCreateConvolutionDescriptor(convolutionDescriptor);
        cudnnSetConvolution2dDescriptor(*convolutionDescriptor,pad_h,pad_w,conv_verticalStride,conv_horizentalStride,1,1,CUDNN_CONVOLUTION);  
       
        cudnnConvolutionForward(*handlePtr,oneScalerPtr,Conv_x_Descriptor,conv_x_local,filterDescriptor,filter_transform[transformIdx],convolutionDescriptor,CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,workspace_gpu,workspace,zeroScalerPtr,Conv_y_Descriptor,conv_y_local); 
    }
 
}

void DenseBlockBackward(float* postConv_data,float* postBN_data,float* postReLU_data,
  float* postConv_grad,float* postBN_grad,float* postReLU_grad,
  float* BNscaler_grad, float* BNbias_grad, float** filter_grad,
  float* BNscaler_data, float* BNbias_data, float** filter_data,
  float* resultSaveMean, float* resultSaveInvVariance,
  int numTransition,int N,int H,int W,int initChannel,int growthRate,
  int pad_h,int pad_w,int conv_verticalStride,int conv_horizentalStride,
  int filter_H,int filter_W,
  float* workspace_gpu, int workspaceSize 
){
    cudnnHandle_t* handlePtr = new cudnnHandle_t;
    cudnnCreate(handlePtr);
    float* oneScalePtr = new float[1]; oneScalePtr[0] = 1.0;
    float* zeroScalePtr = new float[1]; zeroScalePtr[0] = 0.0;
    
    for (int transitionIdx = numTransition-1;transitionIdx>=0;--transitionIdx){
        int channelsBefore_self = initChannels + transitionIdx * growthRate; 
        int channelsBefore_noself = (transitionIdx>0?initChannel:0)+(transitionIdx-1)*growthRate;
	//Conv backward::Preparation
        cudnnFilterDescriptor_t* filterDesc = new cudnnFilterDescriptor_t;
        cudnnCreateFilterDescriptor(filterDesc);
	cudnnSetFilter4dDescriptor(*filterDesc,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,growthRate,channelsBefore_self,filter_H,filter_W);
        cudnnConvolutionDescriptor_t* convolutionDescriptor = new cudnnConvolutionDescriptor_t;
        cudnnCreateConvolutionDescriptor(convolutionDescriptor);
        cudnnSetConvolution2dDescriptor(*convolutionDescriptor,pad_h,pad_w,conv_verticalStride,conv_horizentalStride,1,1,CUDNN_CONVOLUTION);  
        cudnnTensorDescriptor_t* Conv_x_Descriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(Conv_x_Descriptor);
        cudnnSetTensor4dDescriptorEx(*Conv_x_Descriptor,CUDNN_DATA_FLOAT,N,channelsBefore_self,H,W,(numTransition*growthRate+initChannel)*H*W,H*W,W,1);
        cudnnTensorDescriptor_t* Conv_y_Descriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(Conv_y_Descriptor);
        cudnnSetTensor4dDescriptorEx(*Conv_y_Descriptor,CUDNN_DATA_FLOAT,N,growthRate,H,W,(numTransition*growthRate+initChannel)*H*W,H*W,W,1);
        //Conv backward::filter grad
        float* filterGrad_local = filter_grad[transitionIdx];
        float* conv_x_ptr = postReLU_data;
        float* conv_dy_ptr = postConv_grad + channelsBefore_self*H*W;
        cudnnConvolutionBackwardFilter(*handlePtr,oneScalePtr,*Conv_x_Descriptor,conv_x_ptr,
          *Conv_y_Descriptor,conv_dy_ptr,*convolutionDescriptor,CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,workspace_gpu,workspaceSize,
	  zeroScalePtr,*filterDesc,filterGrad_local
        );       
	//Conv backward::data grad
        float* filterData_local = filter_data[transitionIdx]; 
	cudnnConvolutionBackwardData(*handlePtr,oneScalePtr,*filterDesc,filterData_local,
            *Conv_y_Descriptor,conv_dy_ptr,*convolutionDescriptor,CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,workspace_gpu,workspaceSize,oneScalePtr,*Conv_x_Descriptor,postReLU_grad
        ); 	
	//ReLU backward
        int numChannelTransform = (transformIdx==0?initChannel:growthRate);
        cudnnActivationDescriptor_t* activationDescPtr = new cudnnActivationDescriptor_t;
	cudnnCreateActivationDescriptor(activationDescPtr);
	cudnnSetActivationDescriptor(*activationDescPtr,CUDNN_ACTIVATION_RELU,CUDNN_NOT_PROPAGATE_NAN,0.0);
        cudnnTensorDescriptor_t* Bijective_Tensor_Descriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(Bijective_Tensor_Descriptor);
    	cudnnSetTensor4dDescriptorEx(*Bijective_Tensor_Descriptor,CUDNN_DATA_FLOAT,N,numChannelTransform,H,W,
	  (numTransition*growthRate+initChannel)*H*W,H*W,W,1
	);

	float* ReLU_y_ptr = postReLU_data + channelsBefore_noself*H*W;
	float* ReLU_x_ptr = postBN_data + channelsBefore_noself*H*W; 
	float* ReLU_dy_ptr = postReLU_grad + channelsBefore_noself*H*W;
	float* ReLU_dx_ptr = postBn_grad + channelsBefore_noself*H*W; 
	cudnnActivationBackward(*handlePtr,*activationDescPtr,oneScalePtr,*Bijective_Tensor_Descriptor,ReLU_y_ptr,*Bijective_Tensor_Descriptor,ReLU_dy_ptr,*Bijective_Tensor_Descriptor,ReLU_x_ptr,zeroScalePtr,*Bijective_Tensor_Descriptor,ReLU_dx_ptr);	
	//BN backward
	float* BN_x_ptr = postConv_data + channelsBefore_noself*H*W;
	float* BN_dx_ptr = postConv_grad + channelsBefore_noself*H*W;
        cudnnTensorDescriptor_t* BN_param_Descriptor = new cudnnTensorDescriptor_t;
        cudnnCreateTensorDescriptor(BN_param_Descriptor);
    	cudnnSetTensor4dDescriptor(*BN_param_Descriptor,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,numChannelTransform,1,1);
        float* BNscaler_data_localPtr = BNscaler_data + channelsBefore_noself;
	float* BNscaler_grad_localPtr = BNscaler_grad + channelsBefore_noself;
	float* Bnbias_grad_localPtr = BNbias_grad + channelsBefore_noself;      
        float* saveMean_local = resultSaveMean + channelsBefore_noself;
	float* saveInvVar_local = resultSaveInvVariance + channelsBefore_noself; 
        cudnnBatchNormalizationBackward(*handlePtr,CUDNN_BATCHNORM_SPATIAL,oneScalePtr,zeroScalePtr,oneScalePtr,zeroScalePtr,*BijectiveTensorDescriptor,BN_x_ptr,*BijectiveTensorDescriptor,ReLU_dx_ptr,*BijectiveTensorDescriptor,BN_dx_ptr,*BN_param_Descriptor,BNscaler_data_localPtr,BNscaler_grad_localPtr,BNbias_grad_localPtr,CUDNN_BN_MIN_EPSILON,saveMean_local,saveInvVar_local); 
    }
}




