#ifndef _RADIX_SORT_
#define _RADIX_SORT_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const size_t MAX_INPUT_ARRAY_SIZE = 65536;

__host__ void radixSort(unsigned int* h_data,
                        size_t numElems);

__host__  void radixSortCore(unsigned int* const d_inputVals,
                             unsigned int* const d_outputVals,
                             const size_t numElems);


__global__ void createInvertedPredicateTrue(unsigned int* const d_inputVals,
                                            unsigned int* const d_outputPredicate,
                                            const unsigned int bit,
                                            const size_t numElems);

__global__ void flipBits(unsigned int* const d_list,
                         const size_t numElems);

__global__ void scatter(unsigned int* const d_input,
                        unsigned int* const d_output,
                        unsigned int* const d_predicateTrueScan,
                        unsigned int* const d_predicateFalseScan,
                        unsigned int* const d_predicateFalse,
                        unsigned int trueBitsNumber,
                        const size_t numElems);

#endif