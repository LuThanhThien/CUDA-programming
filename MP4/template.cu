#include "../wb.h"
#include "solution.h"
#include <cassert>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define KERNEL_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output,
                       const int z_size, const int y_size, const int x_size) {
    __shared__ float tile[TILE_WIDTH + KERNEL_WIDTH - 1]
                         [TILE_WIDTH + KERNEL_WIDTH - 1]
                         [TILE_WIDTH + KERNEL_WIDTH - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;
    int cha_o = blockIdx.z * TILE_WIDTH + tz;

    int row_i = row_o - KERNEL_WIDTH / 2;
    int col_i = col_o - KERNEL_WIDTH / 2;
    int cha_i = cha_o - KERNEL_WIDTH / 2;

    if (row_i >= 0 && row_i < y_size &&
        col_i >= 0 && col_i < x_size &&
        cha_i >= 0 && cha_i < z_size) {
        tile[tz][ty][tx] = input[(cha_i * y_size + row_i) * x_size + col_i];
    } else {
        tile[tz][ty][tx] = 0.0f;
    }

    __syncthreads();

    if (ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH &&
        cha_o < z_size && row_o < y_size && col_o < x_size) {
        float Pvalue = 0.0f;
        for (int i = 0; i < KERNEL_WIDTH; i++) {
            for (int j = 0; j < KERNEL_WIDTH; j++) {
                for (int k = 0; k < KERNEL_WIDTH; k++) {
                    Pvalue += tile[tz + k][ty + i][tx + j] *
                              deviceKernel[k * KERNEL_WIDTH * KERNEL_WIDTH + i * KERNEL_WIDTH + j];
                }
            }
        }
        output[(cha_o * y_size + row_o) * x_size + col_o] = Pvalue;
    }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  // float result = 0;
  // for (int i = 0; i < inputLength; i *= 2) {
  //   result += float(hostInput[i]) * float(hostInput[i+1]);
  // }
  // printf("%f", result);

  // return 0;

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is %d x %d x %d", z_size, y_size, x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 grid(ceil(x_size / (1.0 * TILE_WIDTH)), ceil(y_size / (1.0 * TILE_WIDTH)), ceil(z_size / (1.0 * TILE_WIDTH)));
  dim3 block(TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1);

  //@@ Launch the GPU kernel here
  conv3d<<<grid, block>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);


  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  for (int i = 3; i < 8; i++) {
    printf("%d %f \n", i, float(hostOutput[i]));
  }

  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceKernel);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
