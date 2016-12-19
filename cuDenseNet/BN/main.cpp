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

void testTensorMean(int N,int C,int H,int W,int channelStart,int channelEnd,float* dataRegion_cpu,float* output_cpuVec);

int main(){
    int N=2,C=3,H=5,W=7;
    int channelStart=1, channelEnd=3;
    std::ifstream inputReader("testInput_mean.txt");
    vector<string> inputVec = getNextLineAndSplitIntoTokens(inputReader);
    float* dataRegion_cpu = new float[N*C*H*W];
    float* output_cpuVec = new float[channelEnd - channelStart];
    for (int i=0;i< N*C*H*W;++i) dataRegion_cpu[i] = stoi(inputVec[i]);
    testTensorMean(N,C,H,W,channelStart,channelEnd,dataRegion_cpu,output_cpuVec);
    for (int cIdx = channelStart;cIdx < channelEnd;++cIdx){
	int localCIdx = cIdx - channelStart;
	cout<< output_cpuVec[localCIdx] <<endl;
    } 
    //writeArray("outCUDA.txt",output,3*2*5*5); 
    
    return 0;
}
