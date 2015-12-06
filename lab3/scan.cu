#include "scan.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#ifndef __CUDACC__      
#define __CUDACC__
#endif
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

/* Scan algorithm for large input (numElems > BLOCK_SIZE nad blocks amount is more than 1) */
__host__ size_t exclusiveScanLarge(unsigned int* d_scanOut,
                                   const unsigned int* d_dataIn,
                                   size_t numElems,
                                   size_t blockSize,
                                   unsigned int* d_blockSums)
{
    const size_t TYPE_SIZE = sizeof(unsigned int);
    size_t dataSize = TYPE_SIZE * numElems;
    size_t gridSize = (size_t) ceil(float(numElems) / float(blockSize));

    cudaMemcpy(d_scanOut, d_dataIn, dataSize, cudaMemcpyDeviceToDevice);
    cudaMemset(d_blockSums, 0, gridSize * TYPE_SIZE);
    __synchAndCheckErrors();

    // 1) scan input data
    partialExclusiveBlellochScan << <gridSize, blockSize, TYPE_SIZE * blockSize >> >(d_scanOut, d_blockSums, numElems);
    __synchAndCheckErrors();

    size_t *d_totalSumOut; // to return value from kernel we must use variable in cuda memory
    cudaMalloc(&d_totalSumOut, sizeof(unsigned int));

    // 2) scan blocks sums
    partialExclusiveBlellochScan << <1, blockSize, TYPE_SIZE * blockSize >> >(d_blockSums, d_totalSumOut, gridSize);

    size_t h_totalSumOut = 0;
    cudaMemcpy(&h_totalSumOut, d_totalSumOut, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_totalSumOut);
    __synchAndCheckErrors();

    // 3) incriment partitial scans with block sums
    incrementBlellochScanWithBlockSums << <gridSize, blockSize >> >(d_scanOut, d_blockSums, numElems);
    __synchAndCheckErrors();

    return h_totalSumOut;
}

/* Blelloch scan algorithm for small arrays (numElems <= BLOCK_SIZE) */
__global__ void partialExclusiveBlellochScan(unsigned int* const d_data,
                                             unsigned int* const d_blockSums,
                                             const size_t numElems)
{
    extern __shared__ unsigned int s_blockScan[];

    const unsigned int tid = threadIdx.x;
    const unsigned int id = blockDim.x * blockIdx.x + tid;

    // copy to shared memory, pad tail block
    s_blockScan[tid] = id < numElems ? d_data[id] : 0;
    __syncthreads();

    // reduce
    unsigned int i;
    for (i = 2; i <= blockDim.x; i <<= 1) {
        if ((tid + 1) % i == 0) {
            unsigned int neighborOffset = i >> 1;
            s_blockScan[tid] += s_blockScan[tid - neighborOffset];
        }
        __syncthreads();
    }

    i >>= 1; // return i to last value before for loop exited

    // clear last element (sum of whole block) and save it to blokSums
    if (tid == (blockDim.x - 1)) {
        d_blockSums[blockIdx.x] = s_blockScan[tid];
        s_blockScan[tid] = 0;
    }
    __syncthreads();

    // traverse down tree & build scan 
    for (i = i; i >= 2; i >>= 1) {
        if ((tid + 1) % i == 0) {
            unsigned int neighborOffset = i >> 1;
            unsigned int oldNeighbor = s_blockScan[tid - neighborOffset];
            s_blockScan[tid - neighborOffset] = s_blockScan[tid]; // copy
            s_blockScan[tid] += oldNeighbor;
        }
        __syncthreads();
    }

    // copy result to global memory
    if (id < numElems) {
        d_data[id] = s_blockScan[tid];
    }
}


__global__ void incrementBlellochScanWithBlockSums(unsigned int* const d_predicateScan,
                                                   unsigned int* const d_blockSumScan,
                                                   const size_t numElems)
{
    const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems)
        return;

    d_predicateScan[id] += d_blockSumScan[blockIdx.x];
}