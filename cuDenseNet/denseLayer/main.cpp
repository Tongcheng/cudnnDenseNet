#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

float** GPU_filterDeploy(float** filter_host,int numTransform,int initChannel,int growthRate,int N,int filter_H,int filter_W);

float** GPU_miscDeploy(float* BNScaler_host,float* BNBiasVec_host,int numTransform,int initChannel,int growthRate,int N,int H,int W,int workspaceSize_gpu);

void GPU_inputDeploy(float* inputData_host,float* inputData_device,int N,int numTransform,int initChannel,int growthRate,int H,int W);

void GPU_deployInferenceMeanVar(int numTransform,int initChannel,int growthRate,float* infMean_gpu,float* infVar_gpu,float* infMean_host,float* infVar_host);

void DenseBlockForward(int initChannel,int growthRate,int numTransition,
		  int N,int H,int W,int pad_h,int pad_w,int conv_verticalStride,int conv_horizentalStride,
		  int testMode, int trainCycleIdx,
		  float* BNScalerVec, float* BNBiasVec,float* resultRunningMean, float* resultRunningVariance, float* resultSaveMean, float* resultSaveInvVariance,
		  float* postConv_dataRegion, float* postBN_dataRegion, float* postReLU_dataRegion,
		  float** filter_transform,int filter_H,int filter_W,float* workspace_gpu,int workspaceSize)

float** GPU_getBufferState(int bufferSize,float* postConv_gpuPtr,float* postBN_gpuPtr,float* postReLU_gpuPtr);

void printTensor(float* tensor,int tensorLen){
    for (int i=0;i<tensorLen;++i) cout<< tensor[i]<<",";
    cout<<endl;
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

    void inferenceMeanVarDeploy(float* infMean_host,float* infVar_host){
        GPU_deployInferenceMeanVar(this->numTransition,this->initChannel,this->growthRate,this->ResultRunningMean_gpu,this->ResultRunningVariance_gpu,infMean_host,infVar_host);
    }

    void denseBlockInputDeploy(float* initData_host){
        GPU_inputDeploy(initData_host,this->postConv_dataRegion_gpu,this->N,this->numTransform,this->initChannel,this->growthRate,this->H,this->W);        	
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

    
    void logInternalState(){
        int bufferSize = this->N * (this->initChannel + this->growthRate * this->numTransition) * this->H * this->W; //the number of values within buffer
	float** bufferStates_host = GPU_getBufferState(bufferSize,this->postConv_dataRegion_gpu,this->postBN_dataRegion_gpu,this->postReLU_dataRegion_gpu);
	float* bufferState_postConv_host = bufferStates_host[0];
	float* bufferState_postBN_host = bufferStates_host[1];
	float* bufferState_postReLU_host = bufferStates_host[2];
        out<< "postConv"<<endl;
        printTensor(bufferState_postConv_host,bufferSize);		
	cout<< "postBN"<<endl;
	printTensor(bufferState_postBN_host,bufferSize);
	cout<< "postReLU"<<endl;
	printTensor(bufferState_postReLU_host,bufferSize);
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
	float* localOutput = string2float(localContent);
	output[transformIdx] = localOutput;
    }
    return output;
}

float* generate_data(string initDataFileName){
    vector<string> vecStr = getNextLineAndSplitIntoTokens(initDataFileName);
    float* output = stringVec2floatPtr(vecStr);  
    return output;
}

int main(){
    int workspaceSize = 10000000;
    vector<float> scalerVec = {1,2,3,4,5,6,7};
    vector<float> biasVec = {3,2,1,0,-1,-2,-3};
    vector<float> popMeanVec = {0,1,-1,0,0,0,0};
    vector<float> popVarVec = {1,2,3,4,5,6,7};
    float* scalerPtr_host = floatVec2floatPtr(scalerVec);
    float* biasPtr_host = floatVec2floatPtr(biasVec);
    float* popMeanPtr_host = floatVec2floatPtr(popMeanVec);
    float* popVarPtr_host = floatVec2floatPtr(popVarVec);
    vector<string> filterNames = {"Filter1_py.txt","Filter2_py.txt"};
    float** filter_cpu = generate_filter(filterNames,2);
    float* initData_cpu = generate_data("InitTensor_py.txt");
    DenseBlock* db = new DenseBlock(3,2,2,2,5,5,1,scalerPtr_host,biasPtr_host,filter_cpu,workspaceSize);
    db->inferenceMeanVarDeploy(popMeanPtr_host,popVarPtr_host);
    db->denseBlockInputDeploy(initData_cpu);
    db->cu_denseBlockForward();
    db->logInternalState();
}
