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

    
if __name__ == '__main__':
    #N=2,C=3->2->2,H=W=5
    N,H,W=2,5,5
    InitC,growthRate=3,2
    popMeanVec=[0,1,-1,0,0,0,0]
    popVarVec =[1,2,3,4,5,6,7]
    scalerVec =[1,2,3,4,5,6,7]
    biasVec = [3,2,1,0,-1,-2,-3]
    HConv,WConv = 3,3
    
    InitMat = np.random.normal(0,2,(N,InitC,H,W)) #This is the matrix as input for the Convolution
    """Fwd Phase"""
    #Filter Transition one is 2*3*HConv*WConv
    Filter1 = np.random.normal(0,1,(2,3,HConv,WConv))
    Filter2 = np.random.normal(0,1,(2,5,HConv,WConv))
    AllFilters = [Filter1,Filter2]
    
    #BatchNorm then ReLU
    BN1_output,BN1_Xhat,BN1_outputMean,BN1_outputVar,BN1_batchMean,BN1_batchVar = pyBN_train_Fwd(InitMat,N,InitC,H,W,popMeanVec[:InitC],popVarVec[:InitC],scalerVec[:InitC],biasVec[:InitC],10000)
    ReLU1_output = pyReLU_batch_Fwd(BN1_output,N,InitC,H,W)
    Conv1_output = pyConvolution_batch_Fwd(2,2,3,H,W,HConv,WConv,ReLU1_output,Filter1)
    
    BN2_output,BN2_Xhat,BN2_outputMean,BN2_outputVar,BN2_batchMean,BN2_batchVar  = pyBN_train_Fwd(Conv1_output,N,growthRate,H,W,popMeanVec[InitC:InitC+growthRate],popVarVec[InitC:InitC+growthRate],scalerVec[InitC:InitC+growthRate],biasVec[InitC:InitC+growthRate],10000)
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
    
    """Bwd phase"""
    TopGrad = np.random.normal(0,1,(N,growthRate,H,W))
    #Conv2 Bwd: Region2 -> Region1, Region0
    Conv2_bottomGrad,Conv2_filterGrad = pyConv_batch_Bwd(N,growthRate,InitC+growthRate,H,W,HConv,WConv,Conv2_input,TopGrad,Filter2)
    
    #ReLU2 Bwd: Region1
    Conv2_bottomGrad_region0 = Conv2_bottomGrad[n,:InitC,H,W]
    Conv2_bottomGrad_region1 = Conv2_bottomGrad[n,InitC:InitC+growthRate,H,W]
    ReLU2_bottomGrad = pyReLU_train_Bwd(BN2_output,Conv2_bottomGrad_region1,N,growthRate,H,W)

    #BN2 Bwd: Region1
    BN2_bottomGrad,BN2_scalerGrad,BN2_biasGrad = pyBN_train_Bwd(Conv1_output,BN2_Xhat,ReLU2_bottomGrad,N,growthRate,H,W,BN2_batchMean,BN2_batchVar,scalerVec[InitC:InitC+growthRate],biasVec[InitC:InitC+growthRate])

    #Conv1 Bwd: Region1 -> Region0
    Conv1_bottomGrad,Conv1_filterGrad = pyConv_batch_Bwd(N,growthRate,InitC,H,W,HConv,WConv,ReLU1_output,BN2_bottomGrad,Filter1)
    
    #ReLU1 Bwd: Region0
    Region0_MergedGrad = Conv2_botomGrad_region0 + Conv1_bottomGrad
    ReLU1_bottomGrad = pyReLU_train_Bwd(BN1_output,Region0_MergedGrad,N,InitC,H,W)

    #BN1 Bwd: Region0
    BN1_bottomGrad,BN1_scalerGrad,BN1_biasGrad = pyBN_train_Bwd(InitMat,BN1_Xhat,ReLU1_bottomGrad,N,InitC,H,W,BN1_batchMean,BN1_batchVar,scalerVec[:InitC],biasVec[:InitC])

    """Log Variables in Fwd Phase"""
    writeTensor4DToFile(Filter1,2,3,HConv,WConv,"Filter1_py.txt")
    writeTensor4DToFile(Filter2,2,5,HConv,WConv,"Filter2_py.txt")
    writeTensor4DToFile(InitMat,N,InitC,H,W,'InitTensor_py.txt')
    writeTensor4DToFile(BN1_output,N,InitC,H,W,'BN1_py.txt')
    writeTensor4DToFile(BN1_Xhat,N,InitC,H,W,'BN1_Xhat_py.txt')
    writeTensor1DToFile(BN1_outputMean,InitC,'BN1_outputMean_py.txt')
    writeTensor1DToFile(BN1_outputVar,InitC,'BN1_outputVar_py.txt')
    writeTensor1DToFile(BN1_batchMean,InitC,'BN1_batchMean_py.txt')
    writeTensor1DToFile(BN1_batchVar,InitC,'BN1_batchVar_py.txt')
    writeTensor4DToFile(ReLU1_output,N,InitC,H,W,'ReLU1_py.txt')
    writeTensor4DToFile(Conv1_output,N,growthRate,H,W,'Conv1_py.txt')
    writeTensor4DToFile(BN2_output,N,growthRate,H,W,'BN2_py.txt')
    writeTensor4DToFile(BN2_Xhat,N,growthRate,H,W,'BN2_Xhat_py.txt')
    writeTensor1DToFile(BN2_outputMean,growthRate,'BN2_outputMean_py.txt')
    writeTensor1DToFile(BN2_outputVar,growthRate,'BN2_outputVar_py.txt')
    writeTensor1DToFile(BN2_batchMean,growthRate,'BN2_batchMean_py.txt')
    writeTensor1DToFile(BN2_batchVar,growthRate,'BN2_batchVar_py.txt')
    writeTensor4DToFile(ReLU2_output,N,growthRate,H,W,'ReLU2_py.txt')
    writeTensor4DToFile(Conv2_input,N,InitC+growthRate,H,W,'Conv2pre_py.txt')
    writeTensor4DToFile(Conv2_output,N,growthRate,H,W,'Conv2_py.txt')

    """Log Variables in Bwd Phase"""
    writeTensor4DToFile(TopGrad,N,growthRate,H,W,'TopGrad_py.txt')
    writeTensor4DToFile(Conv2_bottomGrad_region1,N,growthRate,H,W,'Conv2_bottomGrad_region1_py.txt')
    writeTensor4DToFile(Conv2_filterGrad,growthRate,InitC+growthRate,HConv,WConv,'Filter2Grad_py.txt')
    writeTensor4DToFile(ReLU2_bottomGrad,N,growthRate,H,W,'ReLU2_bottomGrad_py.txt')
    writeTensor4DToFile(BN2_bottomGrad,N,growthRate,H,W,'BN2_bottomGrad_py.txt')
    writeTensor1DToFile(BN2_scalerGrad,growthRate,'BN2_scalergrad_py.txt')
    writeTensor1DToFile(BN2_biasgrad,growthRate,'BN2_biasGrad_py.txt')
    writeTensor4DToFile(Conv1_bottomGrad_region0,N,InitC,H,W,'Conv1_bottomGrad_region0_py.txt')
    writeTensor4DToFile(Region0_MergedGrad,N,InitC,H,W,'Region0_MergedGrad_py.txt')
    writeTensor4DToFile(Conv1_filterGrad,growthRate,InitC,HConv,WConv,'Filter1Grad_py.txt')
    writeTensor4DToFile(ReLU1_bottomGrad,N,InitC,H,W,'ReLU1_bottomGrad_py.txt')
    writeTensor4DToFile(BN1_bottomGrad,N,InitC,H,W,'BN1_bottomGrad_py.txt')
    writeTensor1DToFile(BN1_scalerGrad,InitC,'BN1_scalerGrad_py.txt')
    writeTensor1DToFile(BN1_biasGrad,InitC,'BN1_biasGrad_py.txt')

