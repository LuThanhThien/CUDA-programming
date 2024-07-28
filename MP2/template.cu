
#include "../wb.h"
#include "solution.h"


#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt %s", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  %s", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numCRows && col < numCColumns) {
    // Init the accumulated value (in registers)
    float Pvalue = 0;

    // for loop to calculate the accumulated value
    for (int k = 0; k < numCColumns; ++k) {
      Pvalue += (float) A[row * numCColumns + k] * B[k * numCColumns + col];
    }
    
    // Assign to the target output
    C[row * numCColumns + col] = Pvalue;
  }
}

// 
// mp2 -e ./data/0/output.raw -i ./data/0/input0.raw,./data/0/input1.raw -t vector
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  // numARows = numBCols; numACols = numBRows ==> numCRows = numARows; numCCols = numACols
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numAColumns;
  printf("numCRows x numCColumns = %d x %d\n", numCRows, numCColumns);
  //@@ Allocate the hostC matrix
  int sizeA = numARows * numAColumns * sizeof(float);
  int sizeB = numBRows * numBColumns * sizeof(float);
  int sizeC = numCRows * numCColumns * sizeof(float);
  hostC = (float *)malloc(sizeC);
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // for (int i = 0; i < numCColumns; i++) {
  //   printf("%f ", (float)hostA[i + numCColumns]);
  // }

  wbLog(TRACE, "The dimensions of A are %d x %d", numARows, numAColumns);
  wbLog(TRACE, "The dimensions of B are %d x %d", numBRows, numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  
  cudaMalloc((void **)&deviceA, sizeA);
  cudaMalloc((void **)&deviceB, sizeB);
  cudaMalloc((void **)&deviceC, sizeC);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numCColumns/32.0), ceil(numCRows/32.0), 1);
  dim3 dimBlock(32, 32, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<dimGrid, dimBlock>>>(
    deviceA, deviceB, deviceC,
    numARows, numAColumns, 
    numBRows, numBColumns, 
    numCRows, numCColumns
  );
  checkCudaError("Run matrix multiply");
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA); cudaFree(deviceB); cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  // for (int i = 0; i < numCColumns; i++) {
  //   printf("%f ", (float)hostC[i + numCColumns*2]);
  // }
  // printf("\n");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
