import numpy as np
from scipy.ndimage.filters import convolve

#output a 2D array for single image
def pyConvolution_single_Fwd(c_output,c_input,h_img,w_img,h_filter,w_filter,inputData,filterData):
    localChannels = []
    for outChannelIdx in range(c_output):
        outChannel = np.zeros((h_img,w_img))
        for inChannelIdx in range(c_input):
            localFilter = filterData[outChannelIdx][inChannelIdx]
            localData = inputData[inChannelIdx]
            localOutput = convolve(localData,localFilter,mode='constant')
            outChannel += localOutput
        localChannels.append(outChannel)
    return localChannels

def pyConvolution_batch_Fwd(n,c_output,c_input,h_img,w_img,h_filter,w_filter,inputBatchData,filterData):
    output = []
    for i in range(n):
        localChannels = pyConvolution_single_Fwd(c_output,c_input,h_img,w_img,h_filter,w_filter,inputBatchData[i],filterData)
    	output.append(localChannels)
    return output

def pyReLU_batch_Fwd(inputData,n,c,h_img,w_img):
    outputData = np.zeros((n,c,h_img,w_img))
    for n_i in range(n):
      for i in range(c):
        for j in range(h_img):
	  for k in range(w_img):
	    localElement = inputData[n_i][i][j][k]
	    outLocal = localElement if localElement > 0 else 0
	    outputData[n_i][i][j][k] = outLocal
    return outputData
    
def pyBN_inf_Fwd(inputData,n,c,h_img,w_img,popMeanVec,popVarVec,scalerVec,biasVec):
    epsilon = 1e-5
        
    output = np.zeros((n,c,h_img,w_img))
    for imgIdx in range(n):
        for channelIdx in range(c):
            inputLocalFeatureMap = inputData[imgIdx][channelIdx]
	    tmp = (inputLocalFeatureMap - popMeanVec[channelIdx]) / np.sqrt(popVarVec[channelIdx] + epsilon)
	     
            outputLocalFeatureMap = scalerVec[channelIdx]*tmp + biasVec[channelIdx]
            output[imgIdx][channelIdx] = outputLocalFeatureMap

    return output

def writeTensor1DToFile(tensor,length,fileName):
    for i in range(length):
        with open(fileName,'a') as wFile:
            wFile.write(`tensor[i]`+",")

def writeTensor2DToFile(tensor,len1,len2,fileName):
    for i in range(len1):
        writeTensor1DToFile(tensor[i],len2,fileName)

def writeTensor3DToFile(tensor,len1,len2,len3,fileName):
    for i in range(len1):
        writeTensor2DToFile(tensor[i],len2,len3,fileName)

def writeTensor4DToFile(tensor,len1,len2,len3,len4,fileName):
    for i in range(len1):
        writeTensor3DToFile(tensor[i],len2,len3,len4,fileName)
            
if __name__ == '__main__':
    #N=2,C=3->2->2,H=W=5
    N,H,W=2,5,5
    InitC,growthRate=3,2
    popMeanVec=[0,1,-1,0,0,0,0]
    popVarVec =[1,2,3,4,5,6,7]
    scalerVec =[1,2,3,4,5,6,7]
    biasVec = [3,2,1,0,-1,-2,-3]
    
    InitMat = np.random.normal(0,2,(N,InitC,H,W)) #This is the matrix as input for the Convolution
    
    HConv,WConv = 3,3
    #Filter Transition one is 2*3*HConv*WConv
    Filter1 = np.random.normal(0,1,(2,3,HConv,WConv))
    Filter2 = np.random.normal(0,1,(2,5,HConv,WConv))
    AllFilters = [Filter1,Filter2]
    
    #BatchNorm then ReLU
    BN1_output = pyBN_inf_Fwd(InitMat,N,InitC,H,W,popMeanVec[:InitC],popVarVec[:InitC],scalerVec[:InitC],biasVec[:InitC])
    ReLU1_output = pyReLU_batch_Fwd(BN1_output,N,InitC,H,W)
    Conv1_output = pyConvolution_batch_Fwd(2,2,3,H,W,HConv,WConv,ReLU1_output,Filter1)
    
    BN2_output = pyBN_inf_Fwd(Conv1_output,N,growthRate,H,W,popMeanVec[InitC:InitC+growthRate],popVarVec[InitC:InitC+growthRate],scalerVec[InitC:InitC+growthRate],biasVec[InitC:InitC+growthRate])
    ReLU2_output = pyReLU_batch_Fwd(BN2_output,N,growthRate,H,W)
    #Merge ReLU1_output and ReLU2_output as input for second convolution
    Conv2_input = []
    for n in range(N):
        localN = []
	#print ReLU1_output.shape
        for c in range(InitC):
	    localN.append(ReLU1_output[n][c])
        for c in range(growthRate):
            localN.append(ReLU2_output[n][c])
        Conv2_input.append(localN)
    
    Conv2_output = pyConvolution_batch_Fwd(2,2,5,H,W,HConv,WConv,Conv2_input,Filter2)
    
    writeTensor4DToFile(Filter1,2,3,HConv,WConv,"Filter1_py.txt")
    writeTensor4DToFile(Filter2,2,5,HConv,WConv,"Filter2_py.txt")
    writeTensor4DToFile(InitMat,N,InitC,H,W,'InitTensor_py.txt')
    writeTensor4DToFile(BN1_output,N,InitC,H,W,'BN1_py.txt')
    writeTensor4DToFile(ReLU1_output,N,InitC,H,W,'ReLU1_py.txt')
    writeTensor4DToFile(Conv1_output,N,growthRate,H,W,'Conv1_py.txt')
    writeTensor4DToFile(BN2_output,N,growthRate,H,W,'BN2_py.txt')
    writeTensor4DToFile(ReLU2_output,N,growthRate,H,W,'ReLU2_py.txt')
    writeTensor4DToFile(Conv2_input,N,InitC+growthRate,H,W,'Conv2pre_py.txt')
    writeTensor4DToFile(Conv2_output,N,growthRate,H,W,'Conv2_py.txt')
    
    
