// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include "../wb.h"
#include "solution.h"

#define BLOCK_SIZE 512 //@@ You can change this
#define SECTION_SIZE  BLOCK_SIZE*2

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
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) XY[threadIdx.x] = input[i];
  if (i + blockDim.x < len) XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];

  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    int index = threadIdx.x + stride;
    if (index < SECTION_SIZE) XY[index] += XY[index - stride];
  }

  output[i] = XY[threadIdx.x];
}


__global__ void scanAdd(float *output, float *scanArray, int len) {
  // Element sum wise
  __shared__ float XY[SECTION_SIZE];
  float previous_sum = scanArray[blockIdx.x];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) XY[threadIdx.x] = output[i];
  __syncthreads();

  if (threadIdx.x < blockDim.x - 1) XY[threadIdx.x] += previous_sum;  
  __syncthreads();

  if (i < len) output[i] = XY[threadIdx.x];
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
  if (i + blockDim.x) XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];

  // Reduction
  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE)
      XY[index] += XY[index - stride];
  }

  // Reverse tree
  for (int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 2) * 2 * stride - 1;
    if (index + stride < SECTION_SIZE) 
      XY[index + stride] += XY[index];
  }

  __syncthreads();
  if (i < len) output[i] = XY[threadIdx.x];
  if (i + blockDim.x < len) output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
  if (storeScan && threadIdx.x == 0) {
    scanArray[blockIdx.x] = XY[SECTION_SIZE - 1];
  }
  
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
  int numBlocks = numElements / (BLOCK_SIZE << 1);
  dim3 grid(numBlocks);
  dim3 block(BLOCK_SIZE);
  wbCheck(cudaMalloc((void **)&scanArray, numBlocks * sizeof(float)));

  wbLog(CPU, "BLOCK_SIZE = %d; numBlocks = %d", BLOCK_SIZE, numBlocks);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // 1. Launch first kernel to do scan for each block
  BrentKungScan<<<grid, block>>>(deviceInput, deviceOutput, numElements, scanArray, true);

  // 2. Launch second kernel to do scan for the acummulated sum - global scan
  BrentKungScan<<<1, grid>>>(scanArray, scanArray, numBlocks, scanArray, false);

  // 3. Launch third kernel to do scan for final result
  scanAdd<<<grid, block>>>(deviceOutput, scanArray, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
