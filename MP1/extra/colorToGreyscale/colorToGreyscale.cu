#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define UTILS_IMPLEMENTATION
#include "../utils.h"


__global__ void colorToGreyscaleKernel(unsigned char *input, unsigned char *output, int width, int height) {
    const float alpha = 0.21f;
    const float beta = 0.71f;
    const float gamma = 0.07f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
      int greyOffset = row * width + col;
      int rgbOffset = greyOffset * 3; // 3 channels (R, G, B)

      unsigned char r = input[rgbOffset];
      unsigned char g = input[rgbOffset + 1];
      unsigned char b = input[rgbOffset + 2];

      output[greyOffset] = alpha * r + beta * g + gamma * b;
    }
}

int main() {
    std::cout << "It is running!" << std::endl;

    // Load image
    int width, height, channels;
    unsigned char* input = loadImage("../sample.jpg", width, height, channels);

    // Allocate output for grayscale image
    int size = width * height * channels * sizeof(unsigned char);
    int graySize = width * height * sizeof(unsigned char);
    unsigned char* output = (unsigned char*)malloc(graySize);
    unsigned char *input_d, *output_d;

    // Allocate on device    
    cudaMalloc((void **)&input_d, size);
    cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&output_d, graySize);

    // launch kernel
    dim3 dimGrid(ceil(width / 32.0), ceil(height / 32.0), 1);
    dim3 dimBlock(32, 32, 1);
    printf("Dim grid: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Dim block: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    colorToGreyscaleKernel<<<dimGrid, dimBlock>>>(input_d, output_d, width, height);
    checkCudaError("kernel execution");

    cudaDeviceSynchronize();

    // copy back to host and free device
    cudaMemcpy(output, output_d, graySize, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

    printf("Conversion to grayscale done!\n");
    int nonzeroCount = 0;
    for (int i = 0; i < 50; i++) {
        std::cout << (int)output[i] << " " << (int)output[i+1000] << " ";
    }
    for (int i = 0; i < width * height; i++) {
        if ((int)output[i] != 0) {
            nonzeroCount ++;
        }
    }
    printf("\nNon zero counts for output: %d\n", nonzeroCount);

    // Save grayscale image
    int outChannel = 1;
    saveImage("../sample_greyscale.jpg", output, width, height, outChannel);
    printf("Saved grayscale image as sample_greyscale.jpg\n" );

    // Free memory
    imageFree(input);
    imageFree(output);

    return 0;
}
