#ifndef _SCAN_H_
#define _SCAN_H_

#include "cuda_commons.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/* Scan algorithm for large input (numElems > BLOCK_SIZE nad blocks amount is more than 1) */
__host__ size_t exclusiveScanLarge(unsigned int* d_scanOut,
                                   const unsigned int* d_dataIn,
                                   size_t numElems,
                                   size_t blockSize,
                                   unsigned int* d_blockSums);


/* Blelloch scan algorithm for small arrays (numElems <= BLOCK_SIZE) */
__global__ void partialExclusiveBlellochScan(unsigned int* const d_data,
                                             unsigned int* const d_blockSums,
                                             const size_t numElems);

__global__ void incrementBlellochScanWithBlockSums(unsigned int* const d_predicateScan,
                                                   unsigned int* const d_blockSumScan,
                                                   const size_t numElems);

#endif