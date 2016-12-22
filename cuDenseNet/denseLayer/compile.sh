rm main
rm postConv_cpp
rm postBN_cpp
rm postReLU_cpp
nvcc -std=c++11 -L/home/tl486/lib/cudnn/lib64 -lcudnn denseLayer.cu main.cpp -o main
~
~

