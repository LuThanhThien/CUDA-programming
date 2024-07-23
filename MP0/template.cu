#include "wb.h"

//@@ The purpose of this code is to become familiar with the submission
//@@ process. Do not worry if you do not understand all the details of
//@@ the code.

int main(int argc, char **argv) {
  int deviceCount;

  wbArg_read(argc, argv);

  cudaGetDeviceCount(&deviceCount);

  wbTime_start(GPU, "Getting GPU Data."); //@@ start a timer

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, dev);

    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
        wbLog(TRACE, "No CUDA GPU has been detected");
        return -1;
      } else if (deviceCount == 1) {
        //@@ WbLog is a provided logging API (similar to Log4J).
        //@@ The logging function wbLog takes a level which is either
        //@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
        //@@ message to be printed.
        wbLog(TRACE, "There is 1 device supporting CUDA");
      } else {
        wbLog(TRACE, "There are %d devices supporting CUDA", deviceCount);
      }
    }

    wbLog(TRACE, "Device %d name: %s", dev, deviceProp.name);
    wbLog(TRACE, " Computational Capabilities: %d.%d", deviceProp.major, deviceProp.minor);
    wbLog(TRACE, " Maximum global memory size: %d", deviceProp.totalGlobalMem);
    wbLog(TRACE, " Maximum constant memory size: %d", deviceProp.totalConstMem);
    wbLog(TRACE, " Maximum shared memory size per block: %d", deviceProp.sharedMemPerBlock);
    wbLog(TRACE, " Maximum threads per block: %d", deviceProp.maxThreadsPerBlock);
    wbLog(TRACE, " Maximum block dimensions: %d x %d x %d", 
      deviceProp.maxThreadsDim[0],  deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    wbLog(TRACE, " Maximum grid dimensions: %d x %d x %d", 
      deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    wbLog(TRACE, " Warp size: %d", deviceProp.warpSize);
  }

  wbTime_stop(GPU, "Getting GPU Data."); //@@ stop the timer

  return 0;
}
