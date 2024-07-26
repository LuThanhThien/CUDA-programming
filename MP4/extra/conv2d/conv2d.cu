#include "../../../wb.h"
#include "../../solution.h"
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
#define TILE_WIDTH 8
#define KERNEL_WIDTH 3

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH];


__global__ void conv2d(float *input, float *output,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float tile[TILE_WIDTH + KERNEL_WIDTH - 1][TILE_WIDTH + KERNEL_WIDTH - 1];
  // identify the index and load data
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // output indices
  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int col_o = blockIdx.x * TILE_WIDTH + tx;

  // input indices
  int row_i = row_o - KERNEL_WIDTH / 2;
  int col_i = col_o - KERNEL_WIDTH / 2;

  // check boundaries
  if (row_i >= 0 && row_i < y_size && col_i >= 0 && col_i < x_size) 
    tile[ty][tx] = input[row_i * x_size + col_i];
  else 
    tile[ty][tx] = 0.0f;

  __syncthreads();

  // perform computations
  if (ty < TILE_WIDTH && tx < TILE_WIDTH) {
    float Pvalue = 0; 
    for (int i = 0; i < KERNEL_WIDTH; i++) {
      for (int j = 0; j < KERNEL_WIDTH; j++) {
        Pvalue += tile[i + ty][j + tx] * deviceKernel[i * KERNEL_WIDTH + j];
      }
    }
    if (tx < x_size && ty < y_size) output[row_o * x_size + col_o] = Pvalue;  
  }
}


int main(int argc, char *argv[]) {
  wbArg_t args;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  y_size = hostInput[0];
  x_size = hostInput[1];
  wbLog(TRACE, "The input size is %d x %d", y_size, x_size);
  assert(y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here

  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
