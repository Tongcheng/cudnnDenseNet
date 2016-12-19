#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

using namespace std;

//read from istream
std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream& str)
{
    std::vector<std::string>   result;
    std::string                line;
    std::getline(str,line);

    std::stringstream          lineStream(line);
    std::string                cell;

    while(std::getline(lineStream,cell, ','))
    {
        result.push_back(cell);
    }
    return result;
}

void writeArray(string fileName,float* A,int len){
    std::ofstream outWriter(fileName);
    for (int i=0;i<len;++i) outWriter<<A[i]<<",";
    outWriter.flush();
}

float* BlockConvolutionForward(int numLayer,int n_miniBatch,int c_channels,int h_img,int w_img,float* input_host,int h_filter,int w_filter,float* filter_host,int pad_h,int pad_w,int conv_horizentalStride,int conv_verticalStride);

struct DenseBlock{
    int numLayer; //numTransform = numLayer - 1
    int n_miniBatch, c_channels, w_img, h_img;
    int h_filter, w_filter;
    int pad_h, pad_w;
    int conv_horizentalStride, conv_verticalStride;
    float* inputTensorPtr;
    float* filterTensorPtr;
    DenseBlock(int in_numLayer,int in_n_miniBatch,int in_c_channels, int in_w_img,int in_h_img, int in_h_filter, int in_w_filter, int in_pad_h,int in_pad_w,int in_conv_horizentalStride, int in_conv_verticalStride) : numLayer(in_numLayer),n_miniBatch(in_n_miniBatch),c_channels(in_c_channels),w_img(in_w_img),h_img(in_h_img),h_filter(in_h_filter),w_filter(in_w_filter),pad_h(in_pad_h),pad_w(in_pad_w),conv_horizentalStride(in_conv_horizentalStride),conv_verticalStride(in_conv_verticalStride) {
        int n_inputSize = n_miniBatch * c_channels * h_img * w_img;
        int n_filterSize = (numLayer - 1) * c_channels * c_channels * h_filter * w_filter;
        inputTensorPtr = new float[n_inputSize];
        filterTensorPtr = new float[n_filterSize];
    } 
    float* denseBlockConvolutionForward(){
        return BlockConvolutionForward(numLayer,n_miniBatch,c_channels,h_img,w_img,inputTensorPtr,h_filter,w_filter,filterTensorPtr,pad_h,pad_w,conv_horizentalStride,conv_verticalStride);
    }   
};

int main(){
    DenseBlock* db1 = new DenseBlock(2,3,2,5,5,3,3,1,1,1,1);     
 
    std::ifstream inputReader("inputTensor.txt");
    std::ifstream filterReader("filterTensor.txt");
    vector<string> inputVec = getNextLineAndSplitIntoTokens(inputReader);
    vector<string> filterVec = getNextLineAndSplitIntoTokens(filterReader);
    for (int i=0;i<inputVec.size();++i){db1->inputTensorPtr[i] = stof(inputVec[i]);}
    for (int i=0;i<filterVec.size();++i){db1->filterTensorPtr[i] = stof(filterVec[i]);}     
    float* output = db1->denseBlockConvolutionForward(); 
    writeArray("outCUDA.txt",output,3*2*5*5); 
    
    return 0;
}
