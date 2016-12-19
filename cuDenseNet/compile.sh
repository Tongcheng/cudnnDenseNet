rm main
nvcc -std=c++11 -L/home/tl486/lib/cudnn/lib64 -lcudnn denseBlockConv.cu main.cpp -o main
