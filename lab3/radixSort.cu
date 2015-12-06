#include "radixSort.h"
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

const size_t BLOCK_SIZE = 256;
const size_t UNSIGNED_SIZE = sizeof(unsigned int);

__host__ void radixSort(unsigned int* h_data,
                        size_t numElems)
{
    unsigned int *d_data, *d_dataSorted;
    size_t dataSize = UNSIGNED_SIZE * numElems;

    cudaMalloc(&d_data, dataSize);
    cudaMalloc(&d_dataSorted, dataSize);

    cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice);

    radixSortCore(d_data, d_dataSorted, numElems); // <=========== CORE

    cudaMemcpy(h_data, d_dataSorted, dataSize, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_dataSorted);
}

/*
* ATTENTION: the method only works correctly when numElems <= 2^16
* params:
*   d_scanOut   - resulting scan of d_dataIn
*   d_dataIn    - input array (MUTABLE inside function, used as swap buffer)
*   numElems    - input array size
*   d_blockSums - user preallocated intermediate memory needed to hold block sums 
*/
/* NOTE: actually overflow can occur during the scan algorithm,
* so if we want to work 'radixSort' algorithm correctly (using scan algorithm)
* we should constrain input array numbers with 2^32/MAX_INPUT_ARRAY_SIZE = 2^16
* But here we do not check this... 
*/
__host__ void radixSortCore(unsigned int* const d_inputVals,
                            unsigned int* const d_outputVals,
                            const size_t numElems)
{
    size_t dataSize = UNSIGNED_SIZE * numElems;
    size_t gridSize = (size_t) ceil(numElems / float(BLOCK_SIZE));

    unsigned int *predicate;
    unsigned int *predicateTrueScan;
    unsigned int *predicateFalseScan;
    unsigned int *blockSums; // (intermediate, for large array (>blockSize) scan

    cudaMalloc(&predicate, dataSize);
    cudaMalloc(&predicateTrueScan, dataSize);
    cudaMalloc(&predicateFalseScan, dataSize);
    cudaMalloc(&blockSums, gridSize * UNSIGNED_SIZE);

    bool stepIsOdd = true;
    unsigned int *currentInput;
    unsigned int *currentOutput;

    const size_t MAX_BITS = 31;
    for (int currentBit = 0;
         currentBit < MAX_BITS;
         currentBit++, stepIsOdd = !stepIsOdd)
    {
        currentInput = stepIsOdd ? d_inputVals : d_outputVals;
        currentOutput = stepIsOdd ? d_outputVals : d_inputVals;

        unsigned int shiftedBit = 1 << currentBit;
        createInvertedPredicateTrue << <gridSize, BLOCK_SIZE >> >(currentInput,
                                                                  predicate,
                                                                  shiftedBit,
                                                                  numElems);
        __synchAndCheckErrors();


        size_t numberOfTrueElements
            = exclusiveScanLarge(predicateTrueScan, predicate, numElems, BLOCK_SIZE, blockSums);

        // transform predicateTrue -> predicateFalse
        flipBits << <gridSize, BLOCK_SIZE >> >(predicate, numElems);
        __synchAndCheckErrors();

        exclusiveScanLarge(predicateFalseScan, predicate, numElems, BLOCK_SIZE, blockSums);

        scatter << <gridSize, BLOCK_SIZE >> >(currentInput,
                                              currentOutput,
                                              predicateTrueScan,
                                              predicateFalseScan,
                                              predicate,
                                              numberOfTrueElements,
                                              numElems);
        __synchAndCheckErrors();
    }

    if (stepIsOdd) {
        cudaMemcpy(d_outputVals, currentOutput, dataSize, cudaMemcpyDeviceToDevice);
    }

    checkCudaErrors(cudaFree(predicate));
    checkCudaErrors(cudaFree(predicateTrueScan));
    checkCudaErrors(cudaFree(predicateFalseScan));
    checkCudaErrors(cudaFree(blockSums));
}


/* d_outputPredicate[id] will contain TRUE when
 * the bit of thread appropriate value is '0' */
__global__ void createInvertedPredicateTrue(unsigned int* const d_inputVals,
                                            unsigned int* const d_outputPredicate,
                                            const unsigned int bit,
                                            const size_t numElems)
{
    const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems)
        return;

    int predicate = (d_inputVals[id] & bit) == 0;
    d_outputPredicate[id] = predicate;
}

__global__ void flipBits(unsigned int* const d_data,
                         const size_t numElems)
{
    const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems)
        return;

    //d_data[id] = 1 - d_data[id]; // TODO: maybe so?
    d_data[id] = ((d_data[id] + 1) % 2);
}

__global__ void scatter(unsigned int* const d_input,
                        unsigned int* const d_output,
                        unsigned int* const d_predicateTrueScan,
                        unsigned int* const d_predicateFalseScan,
                        unsigned int* const d_predicateFalse,
                        unsigned int trueBitsNumber,
                        const size_t numElems)
{
    const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= numElems)
        return;

    unsigned int newId = d_predicateFalse[id] == 1
        ? d_predicateFalseScan[id] + trueBitsNumber
        : d_predicateTrueScan[id];

    d_output[newId] = d_input[id];
}

