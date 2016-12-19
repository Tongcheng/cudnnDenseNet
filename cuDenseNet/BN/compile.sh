rm main
nvcc -std=c++11 -L/home/tl486/lib/cudnn/lib64 -L/usr/local/cuda/lib64 -lcudnn -lcublas BN_prototype.cu main.cpp -o main
