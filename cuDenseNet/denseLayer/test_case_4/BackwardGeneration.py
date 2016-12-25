import numpy as np

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
    
    
