#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>
#include <corecrt_math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

unsigned char* loadImage(const char* imageSrc, int& width, int& height, int& channels) {
    unsigned char* input = stbi_load(imageSrc, &width, &height, &channels, STBI_rgb);
    if (input == NULL || channels != 3) {
        std::cerr << "Failed to load image: " << imageSrc << std::endl;
        exit(EXIT_FAILURE);
    }
    printf("Image dimensions: %d x %d x %d\n", width, height, channels);
    return input;
}

void saveImage(const char* saveSrc, unsigned char* image, int& width, int& height, int& channels) {
    stbi_write_jpg(saveSrc, width, height, channels, image, 100);
}

void imageFree(unsigned char* image) {
    stbi_image_free(image);
}


void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

typedef struct {
    int r;
    int g;
    int b;
} RGBPixel;

__host__ __device__ RGBPixel createRGBPixel(int r, int g, int b) {
    RGBPixel rgbPixel;
    rgbPixel.r = r;
    rgbPixel.g = g;
    rgbPixel.b = b;
    return rgbPixel;
}

#endif // UTILS_H
