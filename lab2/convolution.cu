#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "convultion.hpp"

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

// Constant memory for convolution kernel
__constant__ DataType dc_Kern[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

__host__ void copyKernelToConstantMemory(void *h_Kern, size_t kernDataSize) {
    cudaMemcpyToSymbol(dc_Kern, h_Kern, kernDataSize);
}

__global__ void convultionGlobalMemory(
    DataType *newGrid,
    DataType *grid,
    size_t rows,
    size_t cols,
    size_t kernSize)
{
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t idy = blockDim.y * blockIdx.y + threadIdx.y;
    size_t id = idy * blockDim.x * gridDim.x + idx;

    if (id >= rows * cols)
        return;

    const size_t kernRadius = kernSize / 2;
    const size_t realCols = cols + 2 * kernRadius;
    // recalculate id in real grid (with halo rows/cols)
    id = id + realCols * kernRadius + 2 * kernRadius * (id / cols) + kernRadius;

    DataType convSum = 0;
    for (int kRow = 0; kRow < kernSize; kRow++) {
        int shift = id + (kRow - kernRadius) * realCols;
        int kernelShift = kRow * kernSize;
        for (int kCol = 0; kCol < kernSize; kCol++) {
            convSum += grid[shift + (kCol - kernRadius)] * dc_Kern[kernelShift + kCol];
        }
    }

    newGrid[id] = convSum;

}

__global__ void convultionSharedMemory(
    DataType *newGrid,
    DataType *grid,
    size_t rows,
    size_t cols,
    size_t kernSize)
{
    // !!!!!! assert blockDim.x == blockDim.y !!!!!!!
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= cols || idy >= rows)
        return;

    const int kernRadius = kernSize / 2;

    const int realCols = cols + 2 * kernRadius;
    idx += kernRadius;
    idy += kernRadius;


    // 1) Copy data to shared area
    // From host code: 
    // area size: sharedSide * sharedSide 
    // size_t sharedSide = BLOCK_SQUARE_SIDE + 2 * kernRadius;
    extern __shared__ DataType area[];
    int areaSide = blockDim.x + 2 * kernRadius;


    // 1) fill shared area with global memory data
    // each thread fills 9 grid cells (up-left, up, up-right etc...)
    int areaId, areaIdx, areaIdy;
    int blockSize = blockDim.x;
#pragma unroll
    for (int areaPartY = -1; areaPartY <= 1; areaPartY++) {
#pragma unroll
        for (int areaPartX = -1; areaPartX <= 1; areaPartX++) {
            areaIdx = threadIdx.x + areaPartX * blockSize;
            areaIdy = threadIdx.y + areaPartY * blockSize;
            if (areaIdx < -kernRadius || areaIdx >(kernRadius + blockSize - 2) ||
                areaIdy < -kernRadius || areaIdy >(kernRadius + blockSize - 2))
                continue;

            areaId = (areaIdy + kernRadius) * areaSide + kernRadius + areaIdx;
            area[areaId] = grid[idx + areaPartX * blockSize + (idy + areaPartY * blockSize) * realCols];;
        }
    }


    // 2) Calc convultion
    int outId = idy * realCols + idx;
    idx = threadIdx.x;
    idy = threadIdx.y;

    DataType convSum = 0;
    for (int kRow = 0; kRow < kernSize; kRow++) {
        for (int kCol = 0; kCol < kernSize; kCol++) {
            convSum +=
                dc_Kern[kRow * kernSize + kCol]
                * area[(kRow + idy) * areaSide + kCol + idx];
        }
    }

    newGrid[outId] = convSum;
}
