

#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}



void checkCudaError(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int main(void)
{
  int N = 1<<20;
  std::cout << N << std::endl;
  float *x = new float[N]; 
  float *y = new float[N];
  float *d_x, *d_y;

  // Allocate Unified Memory – accessible from CPU or GPU
  //cudaMallocManaged(&x, N*sizeof(float));
  //cudaMallocManaged(&y, N*sizeof(float));
  checkCudaError("Define float vars");

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }


  //cudaMalloc((void **)&d_x, N*sizeof(float));
  //cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  //cudaMalloc((void **)&d_y, N*sizeof(float));
  //cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // Run kernel on 1M elements on the GPU
  //add<<<1, 256>>>(N, d_x, d_y);

  // Wait for GPU to finish before accessing on host
  //cudaDeviceSynchronize();

  //cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);


  // Check for errors (all values should be 3.0f)
  //float maxError = 0.0f;
  //for (int i = 0; i < N; i++)
  //  maxError = fmax(maxError, fabs(y[i]-3.0f));
  //std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  //cudaFree(d_x);
  //cudaFree(d_y);
  
  return 0;
}