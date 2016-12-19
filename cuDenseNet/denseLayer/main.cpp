#include <string> 
#include <iostream>

float** GPU_filterDeploy(float** filter_host,int numTransform,int initChannel,int growthRate,int N,int filter_H,int filter_W); 
float** GPU_miscDeploy(float* BNScaler_host,float* BNBiasVec_host,int numTransform,int initChannel,int growthRate,int N,int H,int W,int workspaceSize_gpu);

struct DenseBlock{
    int initChannel,growthRate,numTransition;
    int N,H,W;
    int pad_h,pad_w,conv_verticalStride,conv_horizentalStride;
    int filter_H,filter_W;
    int testMode,trainCycleIdx;
    float* BN_Scaler_Vec_gpu; //inited in GPU, 
    float* BN_Bias_Vec_gpu; //inited in GPU
    float* ResultRunningMean_gpu; //inited in GPU
    float* ResultRunningVariance_gpu; //inited in GPU
    float* ResultSaveMean_gpu; //inited in GPU
    float* ResultSaveInvVariance_gpu; //inited in GPU
    float* postConv_dataRegion_gpu;//inited in GPU, need CPU input for test
    float* postBN_dataRegion_gpu;//inited in GPU
    float* postReLU_dataRegion_gpu;//inited in GPU
    float** filter_gpu; //inited in GPU, need CPU filter input for test
    float* workspace_gpu; //inited in GPU 
    int workspace_size_bytes;   

    DenseBlock(int initChannel_in,int growthRate_in,int numTransition_in,
        int N_in,int H_in,int W_in,int testMode_in,
        float* BNScalerVec_in,float* BNBiasVec_in,float** filter_host_in,
        int workspaceSize
        ):
        initChannel(initChannel_in),growthRate(growthRate_in),
    	numTransition(numTransition_in),N(N_in),H(H_in),W(W_in),
        pad_h(1),pad_w(1),conv_verticalStride(1),conv_horizentalStride(1),
	filter_H(3),filter_W(3),
        testMode(testMode_in),trainCycleIdx(0),
    { 
	GPU_Init(BNScalerVec_in,BNBiasVec_in,filter_host_in,numTransform,initChannel,growthRate,N,H,W,filter_H,filter_W,workspaceSize);
    } 
    
    void GPU_Init(float* BNScalerVec_host,float* BNBiasVec_host,float** filter_host_in,int numTransform,int initChannel,int growthRate,int N,int H,int W,int filter_H,int filter_W,int workspaceSize){
        this->filter_gpu = GPU_filterDeploy(filter_host_in,numTransform,initChannel,growthRate,N,filter_H,filter_W);
        float** misc_MetaPtr = GPU_miscDeploy(BNScalerVec_host,BNBiasVec_host,numTransform,initChannel,growthRate,N,H,W,workspaceSize);
        this->BN_Scaler_Vec_gpu = misc_MetaPtr[0];
	this->BN_Bias_Vec_gpu = misc_MetaPtr[1];
        this->ResultRunningMean_gpu = misc_MetaPtr[2];
        this->ResultRunningVariance_gpu = misc_MetaPtr[3];
        this->ResultSaveMean_gpu = misc_MetaPtr[4];
	this->ResultSaveInvVariance_gpu = misc_MetaPtr[5];
	this->postConv_dataRegion_gpu = misc_MetaPtr[6];
        this->postBN_dataRegion_gpu = misc_MetaPtr[7];
	this->postReLU_dataRegion_gpu = misc_MetaPtr[8];
	this->workspace_gpu = misc_MetaPtr[9]; 
    }
   
    void GPU_inputDeploy(float* PreConv_Img){

    }
};

int main(){
    DenseBlock* db = new DenseBlock(3,2,2,2,5,5,1,1); 
}

