#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

float*** GPU_filterDeploy(float** filter_host,int numTransform,int initChannel,int growthRate,int N,int filter_H,int filter_W);

float** GPU_miscDeploy(float* BNScaler_host,float* BNBiasVec_host,int numTransform,int initChannel,int growthRate,int N,int H,int W,int workspaceSize_gpu);

void GPU_inputDeploy(float* inputData_host,float* inputData_device,int N,int numTransform,int initChannel,int growthRate,int H,int W);

void GPU_topGradDeploy(float* topGrad_host,float* postConv_grad_gpu,int N,int numTransition,int initChannel,int growthRate,int H,int W);

void GPU_deployInferenceMeanVar(int numTransform,int initChannel,int growthRate,float* infMean_gpu,float* infVar_gpu,float* infMean_host,float* infVar_host);

void DenseBlockForward(int initChannel,int growthRate,int numTransition,
		  int N,int H,int W,int pad_h,int pad_w,int conv_verticalStride,int conv_horizentalStride,
		  int testMode, int trainCycleIdx,
		  float* BNScalerVec, float* BNBiasVec,float* resultRunningMean, float* resultRunningVariance, float* resultSaveMean, float* resultSaveInvVariance,
		  float* postConv_dataRegion, float* postBN_dataRegion, float* postReLU_dataRegion,
		  float** filter_transform,int filter_H,int filter_W,float* workspace_gpu,int workspaceSize);

void DenseBlockBackward(float* postConv_data,float* postBN_data,float* postReLU_data,
  float* postConv_grad,float* postBN_grad,float* postReLU_grad,
  float* BNscaler_grad, float* BNbias_grad, float** filter_grad,
  float* BNscaler_data, float* BNbias_data, float** filter_data,
  float* resultSaveMean, float* resultSaveInvVariance,
  int numTransition,int N,int H,int W,int initChannel,int growthRate,
  int pad_h,int pad_w,int conv_verticalStride,int conv_horizentalStride,
  int filter_H,int filter_W,
  float* workspace_gpu, int workspaceSize 
);

float* GPU_transferPtr(int bufferSize,float* gpuPtr);

float** GPU_getBufferState(int bufferSize,float* postConv_gpuPtr,float* postBN_gpuPtr,float* postReLU_gpuPtr);

void printTensor(float* tensor,int tensorLen){
    for (int i=0;i<tensorLen;++i) cout<< tensor[i]<<",";
    cout<<endl;
}

void writeTensor(float* tensor,int tensorLen,string fileName){
    std::ofstream outWriter(fileName);
    for (int i=0;i<tensorLen;++i) outWriter<<tensor[i]<<",";
    outWriter<<endl;
}

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
    //Bwd Parts
    float* postConv_grad_gpu;
    float* postBN_grad_gpu;
    float* postReLU_grad_gpu;
    float* BNscaler_grad_gpu;
    float* BNbias_grad_gpu;
    float** filter_grad_gpu;

    DenseBlock(int initChannel_in,int growthRate_in,int numTransition_in,
        int N_in,int H_in,int W_in,int testMode_in,
        float* BNScalerVec_in,float* BNBiasVec_in,float** filter_host_in,
        int workspaceSize,int trainCycleIdx_in = 0
        ):
        initChannel(initChannel_in),growthRate(growthRate_in),
    	numTransition(numTransition_in),N(N_in),H(H_in),W(W_in),
        pad_h(1),pad_w(1),conv_verticalStride(1),conv_horizentalStride(1),
	filter_H(3),filter_W(3),
        testMode(testMode_in),trainCycleIdx(trainCycleIdx_in)
    {
	this->GPU_Init(BNScalerVec_in,BNBiasVec_in,filter_host_in,numTransition_in,initChannel_in,growthRate_in,N_in,H_in,W_in,this->filter_H,this->filter_W,workspaceSize);
    }

    void GPU_Init(float* BNScalerVec_host,float* BNBiasVec_host,float** filter_host_in,int numTransform,int initChannel,int growthRate,int N,int H,int W,int filter_H,int filter_W,int workspaceSize){
        //this->filter_gpu = GPU_filterDeploy(filter_host_in,numTransform,initChannel,growthRate,N,filter_H,filter_W);
        float*** filter_totalPtr = GPU_filterDeploy(filter_host_in,numTransform,initChannel,growthRate,N,filter_H,filter_W);
	this->filter_gpu = filter_totalPtr[0];
	this->filter_grad_gpu = filter_totalPtr[1];
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
	this->postConv_grad_gpu = misc_MetaPtr[10];
	this->postBN_grad_gpu = misc_MetaPtr[11];
	this->postReLU_grad_gpu = misc_MetaPtr[12];
	this->BNscaler_grad_gpu = misc_MetaPtr[13];
	this->BNbias_grad_gpu = misc_MetaPtr[14];	
    }

    void inferenceMeanVarDeploy(float* infMean_host,float* infVar_host){
        GPU_deployInferenceMeanVar(this->numTransition,this->initChannel,this->growthRate,this->ResultRunningMean_gpu,this->ResultRunningVariance_gpu,infMean_host,infVar_host);
    }

    void denseBlockInputDeploy(float* initData_host){
        GPU_inputDeploy(initData_host,this->postConv_dataRegion_gpu,this->N,this->numTransition,this->initChannel,this->growthRate,this->H,this->W);        	
    }

    void denseBlockGradientDeploy(float* TopGrad_host){
        GPU_topGradDeploy(TopGrad_host,this->postConv_grad_gpu,this->N,this->numTransition,this->initChannel,this->growthRate,this->H,this->W);
    }

    void cu_denseBlockForward(){
        DenseBlockForward(this->initChannel,this->growthRate,this->numTransition,
	    this->N,this->H,this->W,this->pad_h,this->pad_w,this->conv_verticalStride,this->conv_horizentalStride,
	    this->testMode,this->trainCycleIdx,
	    this->BN_Scaler_Vec_gpu,this->BN_Bias_Vec_gpu,this->ResultRunningMean_gpu,this->ResultRunningVariance_gpu,
	    this->ResultSaveMean_gpu,this->ResultSaveInvVariance_gpu,
	    this->postConv_dataRegion_gpu,this->postBN_dataRegion_gpu,this->postReLU_dataRegion_gpu,
	    this->filter_gpu,this->filter_H,this->filter_W,
	    this->workspace_gpu,this->workspace_size_bytes
	);   
	if (this->testMode==0) this->trainCycleIdx++;
    }

    void cu_denseBlockBackward(){
        DenseBlockBackward(this->postConv_dataRegion_gpu,this->postBN_dataRegion_gpu,this->postReLU_dataRegion_gpu,this->postConv_grad_gpu,this->postBN_grad_gpu,this->postReLU_grad_gpu,this->BNscaler_grad_gpu,this->BNbias_grad_gpu,this->filter_grad_gpu,this->BN_Scaler_Vec_gpu,this->BN_Bias_Vec_gpu,this->filter_gpu,this->ResultSaveMean_gpu,this->ResultSaveInvVariance_gpu,this->numTransition,this->N,this->H,this->W,this->initChannel,this->growthRate,this->pad_h,this->pad_w,this->conv_verticalStride,this->conv_horizentalStride,this->filter_H,this->filter_W,this->workspace_gpu,this->workspace_size_bytes);
    }

    void logInternalState(string rootDir){
        int bufferSize = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W; //the number of values within buffer
	float** bufferStates_host = GPU_getBufferState(bufferSize,this->postConv_dataRegion_gpu,this->postBN_dataRegion_gpu,this->postReLU_dataRegion_gpu);
	float* bufferState_postConv_host = bufferStates_host[0];
	float* bufferState_postBN_host = bufferStates_host[1];
	float* bufferState_postReLU_host = bufferStates_host[2];
        writeTensor(bufferState_postConv_host,bufferSize,rootDir+"/postConv_cpp");		
	writeTensor(bufferState_postBN_host,bufferSize,rootDir+"/postBN_cpp");
	writeTensor(bufferState_postReLU_host,bufferSize,rootDir+"/postReLU_cpp");
    }

    void logGradients(string rootDir){
        int bufferSize = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W; //the number of values within buffer
        float* postConvGrad_host = GPU_transferPtr(bufferSize,this->postConv_grad_gpu);
	float* postBNGrad_host = GPU_transferPtr(bufferSize,this->postBN_grad_gpu);
	float* postReLUGrad_host = GPU_transferPtr(bufferSize,this->postReLU_grad_gpu);
	writeTensor(postConvGrad_host,bufferSize,rootDir+"/ConvGrad_cpp");
	writeTensor(postBNGrad_host,bufferSize,rootDir+"/BNGrad_cpp");
	writeTensor(postReLUGrad_host,bufferSize,rootDir+"/ReLUGrad_cpp");

	int numChannelTotal = this->initChannel+this->growthRate*this->numTransition;
	float* BNscalerGrad_host = GPU_transferPtr(numChannelTotal,this->BNscaler_grad_gpu);
	float* BNbiasGrad_host = GPU_transferPtr(numChannelTotal,this->BNbias_grad_gpu);
	writeTensor(BNscalerGrad_host,numChannelTotal,rootDir+"/BNscalerGrad_cpp");
	writeTensor(BNbiasGrad_host,numChannelTotal,rootDir+"/BNbiasGrad_cpp");

        //log filter grad
        for (int localTransitionIdx=0;localTransitionIdx < this->numTransition;++localTransitionIdx){
	    string filterName = "filterGrad"+to_string(localTransitionIdx)+"_cpp";
	    int localFilterNumValues = this->growthRate*(this->initChannel+localTransitionIdx*this->growthRate)*this->filter_H*this->filter_W;
	    float* filter_localTransition_host = GPU_transferPtr(localFilterNumValues,this->filter_grad_gpu[localTransitionIdx]);
	    writeTensor(filter_localTransition_host,localFilterNumValues,rootDir+"/"+filterName);
	}
    }

    void logResultMeanVar(string rootDir){
        int bufferSize = this->initChannel + this->growthRate * this->numTransition;
	float* resultMean_cpu = GPU_transferPtr(bufferSize,this->ResultRunningMean_gpu);
	float* resultVar_cpu = GPU_transferPtr(bufferSize,this->ResultRunningVariance_gpu);
	writeTensor(resultMean_cpu,bufferSize,rootDir+"/Mean_cpp");
	writeTensor(resultVar_cpu,bufferSize,rootDir+"/Var_cpp");
    }
};

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
	std::vector<std::string>   result;
	std::string                line;
	std::getline(str,line);

	std::stringstream          lineStream(line);
	std::string                cell;

	while(std::getline(lineStream,cell, ','))
	{
		if (cell.size()>0) result.push_back(cell);
	}
	return result;
}

float* stringVec2floatPtr(vector<string> vecString){
    float* output = new float[vecString.size()];
    for (int i=0;i<vecString.size();++i) output[i] = stof(vecString[i]);
    return output;
}

float* floatVec2floatPtr(vector<float> vecFloat){
    float* output = new float[vecFloat.size()];
    for (int i=0;i<vecFloat.size();++i) output[i] = vecFloat[i];
    return output;
}

float** generate_filter(vector<string> vecNames,int numTransform){
    float** output = new float*[numTransform];
    for (int transformIdx=0;transformIdx<numTransform;++transformIdx){
        string localFileName = vecNames[transformIdx];
	ifstream localReader(localFileName);
	vector<string> localContent = getNextLineAndSplitIntoTokens(localReader);
	float* localOutput = stringVec2floatPtr(localContent);
	output[transformIdx] = localOutput;
    }
    return output;
}

float* generate_data(string initDataFileName){
    ifstream inputFileReader(initDataFileName);
    vector<string> vecStr = getNextLineAndSplitIntoTokens(inputFileReader);
    float* output = stringVec2floatPtr(vecStr);  
    return output;
}

//case_1 : forward inference test
void testCase1(){
    int workspaceSize = 10000000;
    string rootDir = "test_case_1";
    vector<float> scalerVec = {1,2,3,4,5,6,7};
    vector<float> biasVec = {3,2,1,0,-1,-2,-3};
    vector<float> popMeanVec = {0,1,-1,0,0,0,0};
    vector<float> popVarVec = {1,2,3,4,5,6,7};
    float* scalerPtr_host = floatVec2floatPtr(scalerVec);
    float* biasPtr_host = floatVec2floatPtr(biasVec);
    float* popMeanPtr_host = floatVec2floatPtr(popMeanVec);
    float* popVarPtr_host = floatVec2floatPtr(popVarVec);
    vector<string> filterNames = {rootDir+"/Filter1_py.txt",rootDir+"/Filter2_py.txt"};
    float** filter_cpu = generate_filter(filterNames,2);
    float* initData_cpu = generate_data(rootDir+"/InitTensor_py.txt");
    DenseBlock* db = new DenseBlock(3,2,2,2,5,5,1,scalerPtr_host,biasPtr_host,filter_cpu,workspaceSize);
    db->inferenceMeanVarDeploy(popMeanPtr_host,popVarPtr_host);
    db->denseBlockInputDeploy(initData_cpu);
    db->cu_denseBlockForward();
    db->logInternalState(rootDir);
}

//case_2: forward inference training : 
//trainCycle_0
void testCase2(){
    int workspaceSize = 10000000;
    string rootDir = "test_case_2";
    vector<float> scalerVec = {1,2,3,4,5,6,7};
    vector<float> biasVec = {3,2,1,0,-1,-2,-3};
    vector<float> popMeanVec = {0,1,-1,0,0,0,0};
    vector<float> popVarVec = {1,2,3,4,5,6,7};
    float* scalerPtr_host = floatVec2floatPtr(scalerVec);
    float* biasPtr_host = floatVec2floatPtr(biasVec);
    float* popMeanPtr_host = floatVec2floatPtr(popMeanVec);
    float* popVarPtr_host = floatVec2floatPtr(popVarVec);
    vector<string> filterNames = {rootDir+"/Filter1_py.txt",rootDir+"/Filter2_py.txt"};
    float** filter_cpu = generate_filter(filterNames,2);
    float* initData_cpu = generate_data(rootDir+"/InitTensor_py.txt");
    DenseBlock* db = new DenseBlock(3,2,2,2,5,5,0,scalerPtr_host,biasPtr_host,filter_cpu,workspaceSize);
    db->inferenceMeanVarDeploy(popMeanPtr_host,popVarPtr_host);
    db->denseBlockInputDeploy(initData_cpu);
    db->cu_denseBlockForward();
    db->logInternalState(rootDir);
    db->logResultMeanVar(rootDir);
}

//case_3: forward inferencetraining : 
//trainCycle_3
void testCase3(){
    int workspaceSize = 10000000;
    string rootDir = "test_case_3";
    vector<float> scalerVec = {1,2,3,4,5,6,7};
    vector<float> biasVec = {3,2,1,0,-1,-2,-3};
    vector<float> popMeanVec = {0,1,-1,0,0,0,0};
    vector<float> popVarVec = {1,2,3,4,5,6,7};
    float* scalerPtr_host = floatVec2floatPtr(scalerVec);
    float* biasPtr_host = floatVec2floatPtr(biasVec);
    float* popMeanPtr_host = floatVec2floatPtr(popMeanVec);
    float* popVarPtr_host = floatVec2floatPtr(popVarVec);
    vector<string> filterNames = {rootDir+"/Filter1_py.txt",rootDir+"/Filter2_py.txt"};
    float** filter_cpu = generate_filter(filterNames,2);
    float* initData_cpu = generate_data(rootDir+"/InitTensor_py.txt");
    DenseBlock* db = new DenseBlock(3,2,2,2,5,5,0,scalerPtr_host,biasPtr_host,filter_cpu,workspaceSize,10000);
    db->inferenceMeanVarDeploy(popMeanPtr_host,popVarPtr_host);
    db->denseBlockInputDeploy(initData_cpu);
    db->cu_denseBlockForward();
    db->logInternalState(rootDir);
    db->logResultMeanVar(rootDir);
}

//case_4: Backward Training:
void testCase4(){
    int workspaceSize = 10000000;
    string rootDir = "test_case_4";
    vector<float> scalerVec = {1,2,3,4,5,6,7};
    vector<float> biasVec = {3,2,1,0,-1,-2,-3};
    vector<float> popMeanVec = {0,1,-1,0,0,0,0};
    vector<float> popVarVec = {1,2,3,4,5,6,7};
    float* scalerPtr_host = floatVec2floatPtr(scalerVec);
    float* biasPtr_host = floatVec2floatPtr(biasVec);
    float* popMeanPtr_host = floatVec2floatPtr(popMeanVec);
    float* popVarPtr_host = floatVec2floatPtr(popVarVec);
    vector<string> filterNames = {rootDir+"/Filter1_py.txt",rootDir+"/Filter2_py.txt"};
    float** filter_cpu = generate_filter(filterNames,2);
    float* initData_cpu = generate_data(rootDir+"/InitTensor_py.txt");
    float* topGrad_cpu = generate_data(rootDir+"/TopGrad_py.txt");
    DenseBlock* db = new DenseBlock(3,2,2,2,5,5,0,scalerPtr_host,biasPtr_host,filter_cpu,workspaceSize);
    db->inferenceMeanVarDeploy(popMeanPtr_host,popVarPtr_host);
    db->denseBlockInputDeploy(initData_cpu);
    db->cu_denseBlockForward();
    db->logInternalState(rootDir);
    db->denseBlockGradientDeploy(topGrad_cpu);
    db->cu_denseBlockBackward();
    db->logGradients(rootDir);
}

int main(){
    testCase4(); 
}
