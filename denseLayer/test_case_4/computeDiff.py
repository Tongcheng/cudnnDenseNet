
fileNameA = "Filter1Grad_py.txt"
fileNameB = "filterGrad0_cpp"
A_offset = 0
B_offset = 0
rangeLen = 54

listA =  open(fileNameA,'r').readlines()[0].split(",")[:-1]
listB =  open(fileNameB,'r').readlines()[0].split(',')[:-1]
floatAL = map(lambda x:float(x),listA)
floatBL = map(lambda x:float(x),listB)

print floatAL[A_offset:A_offset+rangeLen]
print floatBL[B_offset:B_offset+rangeLen]
for i in range(rangeLen):
    numA,numB = floatAL[A_offset+i],floatBL[B_offset+i]
    print numA-numB
    if abs(numA-numB)>0.2:
        print i
    

globalMaxDiff = 0 
for i in range(rangeLen):
    aIdx,bIdx = A_offset+i,B_offset+i 
    globalMaxDiff = max(globalMaxDiff,abs(floatAL[aIdx]-floatBL[bIdx]))

print "global Max Diff is:"+`globalMaxDiff`
