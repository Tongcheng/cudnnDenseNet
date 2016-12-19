import numpy as np
from scipy.ndimage.filters import convolve

def pyConvolution(n_img,c,h_img,w_img,h_filter,w_filter,inputData,filterData):
    output = []
    for i in range(n_img):
        localChannels = []
        for outChannelIdx in range(c):
            outChannel = np.zeros((h_img,w_img))
            for inChannelIdx in range(c):
	        #print filterData
		localFilter = filterData[outChannelIdx][inChannelIdx]
		localData = inputData[i][inChannelIdx]
		localOutput = convolve(localData,localFilter,mode='constant') 
	        outChannel += localOutput
	    localChannels.append(outChannel)
        output.append(localChannels)
    return output

def linearizeTensor(tensor4d,n,c,h,w):
  output = []
  for i1 in range(n):
    for i2 in range(c):
      for i3 in range(h):
	for i4 in range(w):
	  output.append(tensor4d[i1][i2][i3][i4])
  return output

def tensorizeLinear(linearArray,n,c,h,w):
  output = []
  for i1 in range(n):
    outputL1 = []
    for i2 in range(c):
      outputL2 = []
      for i3 in range(h):
        outputL3 = []
	for i4 in range(w):
	  localIdx = i1*(c*h*w) + i2*(h*w) + i3*w + i4
	  outputL3.append(linearArray[localIdx])
	outputL2.append(outputL3)
      outputL1.append(outputL2)
    output.append(outputL1)
  return output

def stringList(L):
  return ",".join(str(x) for x in L)

if __name__ == "__main__":

  n,c,h_img,w_img = 3,2,5,5 
  h_filter,w_filter = 3,3
  #4d input n*c*h*w
  inputArray = np.random.normal(size = (n*c*h_img*w_img))
  print "before RELU"
  print inputArray
  #RELU effect
  inputArray = map(lambda x:max(0,x),inputArray)  
  print "after RELU"
  print inputArray

  #4d filter c*c*h*w 
  filterArray = np.random.normal(size = (c*c*h_filter*w_filter))
    
  inputTensor = tensorizeLinear(inputArray,n,c,h_img,w_img)
  filterTensor = tensorizeLinear(filterArray,c,c,h_filter,w_filter)
  
  outputTensor = pyConvolution(n,c,h_img,w_img,h_filter,w_filter,inputTensor,filterTensor)

  outputArray = linearizeTensor(outputTensor,n,c,h_img,w_img)  
    
  with open("inputTensor.txt",'w') as fIn:
    inputStr=stringList(inputArray)
    fIn.write(inputStr+"\n")

  with open("filterTensor.txt",'w') as fFilter:
    filterStr=stringList(filterArray) 
    fFilter.write(filterStr+"\n")

  with open("outTensor.txt",'w') as fOut:
    outStr=stringList(outputArray)
    fOut.write(outStr+"\n")
