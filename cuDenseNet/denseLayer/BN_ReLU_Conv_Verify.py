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


