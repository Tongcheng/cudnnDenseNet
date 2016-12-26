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

def inExRange(x,exUpperRange):
    return (x>=0) and (x<exUpperRange)

#bottomData, topGrad: n*c*h_img*w_img
#filterData: c_output*c_input*h_filter*w_filter
def pyConv_batch_Bwd(n,c_output,c_input,h_img,w_img,h_filter,w_filter,bottomData,topGrad,filterData):
    filterGrad = np.zeros((c_output,c_input,h_filter,w_filter))
    bottomGrad = np.zeros((n,c_input,h_img,w_img))
    #compute filter grad
    for coutIdx in range(c_output):
        for cinIdx in range(c_input):
            for x in range(h_filter):
                for y in range(w_filter):
                    localGradSum = 0
                    for nIdx in range(n):
                        for i in range(h_img):
                            for j in range(w_img):
                                if inExRange(i+1-x,h_img) and inExRange(j+1-y,w_img):
                                    localGradSum += topGrad[nIdx][coutIdx][i][j] * bottomData[nIdx][cinIdx][i+1-x][j+1-y]
                    filterGrad[coutIdx][cinIdx][x][y] += localGradSum
                    
    #compute bottom grad
    for nIdx in range(n):
        for cinIdx in range(c_input):
            for i in range(h_img):
                for j in range(w_img):
                    localGradSum = 0
                    for coutIdx in range(c_output):
                        for x in range(h_img):
                            for y in range(w_img):
                                if inExRange(x-i+1,h_filter) and inExRange(y-j+1,w_filter):
                                    localGradSum += topGrad[nIdx][coutIdx][x][y] * filterData[coutIdx][cinIdx][x-i+1][y-j+1]
                    bottomGrad[nIdx][cinIdx][i][j] += localGradSum

    return bottomGrad,filterGrad


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

#bottomData: n*c*h_img*w_img
#topGrad: n*c*h_img*w_img
def pyReLU_batch_Bwd(bottomData,topGrad,n,c,h_img,w_img):
    outputGrad = np.zeros((n,c,h_img,w_img))
    for nIdx in range(n):
        for cIdx in range(c):
            for hIdx in range(h_img):
                for wIdx in range(w_img):
                    bottomLocal = bottomData[nIdx][cIdx][hIdx][wIdx]
                    if bottomLocal > 0:
                        outputGrad[nIdx][cIdx][hIdx][wIdx] = topGrad[nIdx][cIdx][hIdx][wIdx]
    return outptuGrad


def pyBN_train_Fwd(inputData,n,c,h_img,w_img,inMeanVec,inVarVec,scalerVec,biasVec,trainCycleIdx):
    epsilon = 1e-5
        
    output = np.zeros((n,c,h_img,w_img))
    #update output Mean and Var
    output_Mean = np.zeros(len(inMeanVec))
    output_Var = np.zeros(len(inVarVec))
    exponentialAverageFactor = 1.0/(1+trainCycleIdx)
    local_MeanList = []
    local_VarList = []
    for channelIdx in range(c):
        localChannelAll = []
        for imgIdx in range(n):
            localChannelAll.append(inputData[imgIdx][channelIdx])
        
        variance_adjust_m = n*h_img*w_img
        Mean_miniBatch = np.mean(localChannelAll)
        Var_miniBatch =  (variance_adjust_m / (variance_adjust_m - 1.0)) * np.var(localChannelAll)
        local_MeanList.append(Mean_miniBatch)
        local_VarList.append(Var_miniBatch)
        output_Mean[channelIdx] = (1-exponentialAverageFactor)*inMeanVec[channelIdx] + exponentialAverageFactor*Mean_miniBatch
        output_Var[channelIdx] = (1-exponentialAverageFactor)*inVarVec[channelIdx] + exponentialAverageFactor*Var_miniBatch
    
    for imgIdx in range(n):
        for channelIdx in range(c):
            inputLocalFeatureMap = inputData[imgIdx][channelIdx]
	    tmp = (inputLocalFeatureMap - local_MeanList[channelIdx]) / np.sqrt(local_VarList[channelIdx] + epsilon)

	    output_xhat = tmp
            outputLocalFeatureMap = scalerVec[channelIdx]*tmp + biasVec[channelIdx]
            output[imgIdx][channelIdx] = outputLocalFeatureMap

    batch_Mean, batch_Var = local_MeanList, local_VarList
    return output, output_xhat, output_Mean, output_Var, batch_Mean, batch_Var

def pyBN_train_Bwd(bottomData,bottomXHatData,topGrad,n,c,h_img,w_img,batchMean,batchVar,scalerVec,biasVec):
    epsilon = 1e-5

    #compute biasGrad and scalerGrad
    biasGrad = np.zeros(c)
    scalerGrad = np.zeros(c)
    for channelIdx in range(c):
        for nIdx in range(n):
            for hIdx in range(h_img):
                for wIdx in range(w_img):
                    biasGrad[channelIdx] += topGrad[nIdx][channelIdx][hIdx][wIdx]
                    scalerGrad[channelIdx] += topGrad[nIdx][channelIdx][hIdx][wIdx] * bottomXHatData[nIdx][channelIdx][hIdx][wIdx]

    #compute bottomDataGrad
    bottomDataGrad = np.zeros((n,c,h_img,w_img))
    #Helper 1: XHat gradient
    XHatGrad = np.zeros((n,c,h_img,w_img))
    for nIdx in range(n):
        for cIdx in range(c):
            for hIdx in range(h):
                for wIdx in range(w):
                    XHatGrad[nIdx][cIdx][hIdx][wIdx] = topGrad[nIdx][cIdx][hIdx][wIdx] * scalerVec[cIdx]

    #Helper 2: Var Gradient
    varGrad = np.zeros(c)
    for channelIdx in range(c):
        for nIdx in range(n):
            for hIdx in range(h):
                for wIdx in range(w):
                    varGrad[channelIdx] += XHatGrad[nIdx][channelIdx][hIdx][wIdx]*(bottomData[nIdx][channelIdx][hIdx][wIdx]-batchMean[channelIdx])*(-0.5)*np.power(batchVar[channelIdx]+epsilon,-1.5)
    
    #Helper 3: Mean Gradient
    meanGrad = np.zeros(c)
    for channelIdx in range(c):
        for nIdx in range(n):
            for hIdx in range(h):
                for wIdx in range(w):
                    meanGrad[channelIdx] += bottomDataGrad[nIdx][channelIdx][hIdx][wIdx] * (-1.0 / np.sqrt(batchVar[channelIdx] + epsilon))

    m = float(n * h_img * w_img)
    #Now main for calculate bottomDataGrad
    for nIdx in range(n):
        for cIdx in range(c):
            for hIdx in range(h):
                for wIdx in range(w):
                    term1 = XHatGrad[nIdx][cIdx][hId][wIdx]*np.power(batchVar[cIdx]+epsilon,-0.5)
                    term2 = varGrad[cIdx] * 2 * (bottomData[nIdx][cIdx][hIdx][wIdx] - batchMean[cIdx]) / m
                    term3 = meanGrad[cIdx] / m
                    bottomDataGrad[nIdx][cIdx][hIdx][wIdx] += term1 + term2 + term3
    
    return bottomDataGrad,scalerGrad,biasGrad 

    
    
