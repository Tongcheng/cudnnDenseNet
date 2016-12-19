import random

N,C,H,W = 2,3,5,7
cStart,cEnd = 0,3
cMean = [0] * 3

sumNums = 0
Alist = []
for n in range(N):
  Blist = []
  for c in range(0,C):
    Clist = []
    for h in range(H):
      Dlist = []
      for w in range(W):
	localNum = 0
        if c in range(cStart,cEnd):
          localNum = random.randint(0,100) 
          cMean[c] += float(localNum)/(N*H*W)
	with open("testInput_mean.txt",'a') as wFile:
	  wFile.write(`localNum`+",")
	Dlist.append(localNum)
      #with open("testInput_mean.txt",'a') as wFile:
	#wFile.write("\n")
      Clist.append(Dlist)
    Blist.append(Clist)
  Alist.append(Blist)


print cMean
