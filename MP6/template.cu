// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include "../wb.h"
#include "solution.h"

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE  1024

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)



__global__ void KoggeStoneScan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[SECTION_SIZE];

  // Load the data
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) XY[threadIdx.x] = input[i];

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x - stride];
  }
  

  if (i < len) output[i] = XY[threadIdx.x];
}


__global__ void scanAdd(float *output, float *scanArray, int len) {
  // Element sum wise
  __shared__ float previous_sum;

  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x == 0) previous_sum = scanArray[blockIdx.x];

  if (i < len && blockIdx.x > 0) {
    output[i] += previous_sum;
    if (i + blockDim.x < len) {
      output[i + blockDim.x] += previous_sum;
    }
  }
}


__global__ void BrentKungScan(float *input, float *output, int len, float *scanArray, bool storeScan) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  // each thread load 2 values
  __shared__ float XY[SECTION_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) XY[threadIdx.x] = input[i]; 
  else XY[threadIdx.x] = 0.0f;
  if (i + blockDim.x < len) XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
  else XY[threadIdx.x + blockDim.x] = 0.0f;

  // Reduction
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) XY[index] += XY[index - stride];
  }

  // Reverse tree
  for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 2) * 2 * stride - 1;
    if (index + stride < SECTION_SIZE) XY[index + stride] += XY[index];
  }

  __syncthreads();
  if (i < len) output[i] = XY[threadIdx.x];
  if (i + blockDim.x < len) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
  if (storeScan == true && threadIdx.x == 0) scanArray[blockIdx.x] = XY[SECTION_SIZE - 1];
}

__global__ void scanStreaming(float *input, float *output, int len) {
  
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *scanArray;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is %d",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceInput, numElements * sizeof(float));
  checkCudaError("allocate GPU");
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  int numBlocks = ceil(numElements / (BLOCK_SIZE * 2.0));
  cudaMalloc((void **)&scanArray, numBlocks * sizeof(float));

  wbLog(CPU, "BLOCK_SIZE = %d; numBlocks = %d", BLOCK_SIZE, numBlocks);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  // // 1. Launch first kernel to do scan for each block
  KoggeStoneScan<<<ceil(numElements / (SECTION_SIZE * 1.0)), SECTION_SIZE>>>(deviceInput, deviceOutput, numElements); 
  // BrentKungScan<<<numBlocks, BLOCK_SIZE>>>(deviceInput, deviceOutput, numElements, scanArray, true);
  // checkCudaError("sectional brent-kung scan");
  
  // // // 2. Launch second kernel to do scan for the acummulated sum - global scan
  // BrentKungScan<<<1, ceil(numBlocks / 2.0)>>>(scanArray, scanArray, numBlocks, scanArray, false);
  // checkCudaError("global brent-kung scan");
  
  // // // 3. Launch third kernel to do scan for final result
  // scanAdd<<<numBlocks, BLOCK_SIZE>>>(deviceOutput, scanArray, numElements);
  // checkCudaError("element wise adding");

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  for (int i = 0; i < numBlocks; i++) {
    printf("%f ", float(hostOutput[i * SECTION_SIZE]));
  }
  printf("\n");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(scanArray);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
