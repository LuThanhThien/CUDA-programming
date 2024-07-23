#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define UTILS_IMPLEMENTATION
#include "../utils.h"


// Matmul with same size
__global__ void naiveMatMulKernel(float *input1, float *input2, float *output, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < width) {
        float outputValue = 0;
        for (int i = 0; i < width; ++i){
            outputValue += input1[row * width + i] * input2[i * width + col];
        }
        output[row * width + col] = outputValue;
    }
}


int main() {
    std::cout << "It is running!" << std::endl;
    
    return 0;
}
