#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define UTILS_IMPLEMENTATION
#include "../utils.h"


__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height) {
    const int BLUR_SIZE = 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        RGBPixel pixVal = createRGBPixel(0, 0, 0);
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    int curOffset = (curRow * width + curCol) * 3;
                    pixVal.r += input[curOffset];                    
                    pixVal.g += input[curOffset + 1];
                    pixVal.b += input[curOffset + 2];
                    pixels ++;
                }
            }
        }
        int offset = (row * width + col) * 3;
        output[offset] = (unsigned char) (pixVal.r / pixels);
        output[offset + 1] = (unsigned char) (pixVal.g / pixels);
        output[offset + 2] = (unsigned char) (pixVal.b / pixels);
    }
}

int main() {
    std::cout << "It is running!" << std::endl;

    // Load image
    int width, height, channels;
    unsigned char* input = loadImage("../sample.jpg", width, height, channels);

    // Allocate output for grayscale image
    int size = width * height * channels * sizeof(unsigned char);
    unsigned char* output = (unsigned char*)malloc(size);
    unsigned char *input_d, *output_d;

    // Allocate on device    
    cudaMalloc((void **)&input_d, size);
    cudaMemcpy(input_d, input, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&output_d, size);

    // launch kernel
    dim3 dimGrid(ceil(width / 256.0), ceil(height / 256.0), 1);
    dim3 dimBlock(256, 256, 1);
    printf("Dim grid: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("Dim block: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
    blurKernel<<<dimGrid, dimBlock>>>(input_d, output_d, width, height);
    checkCudaError("kernel execution");
    cudaDeviceSynchronize();

    // copy back to host and free device
    cudaMemcpy(output, output_d, size, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);

    // Save grayscale image
    saveImage("../sample_blur.jpg", output, width, height, channels);
    printf("Saved grayscale image as ../sample_blur.jpg\n" );

    // Free memory
    imageFree(input);
    imageFree(output);

    return 0;
}
