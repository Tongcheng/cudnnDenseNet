
fileNameA = "Conv1_py.txt"
fileNameB = "postConv_cpp"
A_offset = 0
B_offset = 75
rangeLen = 50

listA =  open(fileNameA,'r').readlines()[0].split(",")[:-1]
listB =  open(fileNameB,'r').readlines()[0].split(',')[:-1]
floatAL = map(lambda x:float(x),listA)
floatBL = map(lambda x:float(x),listB)

print floatAL[50:]
print floatBL[250:300]
globalMaxDiff = 0 
for i in range(rangeLen):
    aIdx,bIdx = A_offset+i,B_offset+i 
    globalMaxDiff = max(globalMaxDiff,abs(floatAL[aIdx]-floatBL[bIdx]))

print "global Max Diff is:"+`globalMaxDiff`
