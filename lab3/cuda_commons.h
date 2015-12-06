#ifndef _CUDA_COMMONS_H_
#define _CUDA_COMMONS_H_

#include <helper_cuda.h>
#include <helper_functions.h>   

#define __synchAndCheckErrors() {\
    cudaDeviceSynchronize(); \
    checkCudaErrors(cudaGetLastError()); \
}

#endif