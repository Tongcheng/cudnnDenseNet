
fileNameA = "BN2_py.txt"
fileNameB = "postBN_cpp"
A_offset = 0
B_offset = 75
rangeLen = 50

listA =  open(fileNameA,'r').readlines()[0].split(",")[:-1]
listB =  open(fileNameB,'r').readlines()[0].split(',')[:-1]
floatAL = map(lambda x:float(x),listA)
floatBL = map(lambda x:float(x),listB)

print floatAL[A_offset:A_offset+rangeLen]
print floatBL[B_offset:B_offset+rangeLen]

globalMaxDiff = 0 
for i in range(rangeLen):
    aIdx,bIdx = A_offset+i,B_offset+i
    globalMaxDiff = max(globalMaxDiff,abs(floatAL[aIdx]-floatBL[bIdx]))
    #test
    aVal,bVal = floatAL[aIdx],floatBL[bIdx]
    print abs(aVal - bVal) 

print "global Max Diff is:"+`globalMaxDiff`
