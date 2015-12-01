#include "commons.hpp"

const size_t MAX_KERNEL_SIZE = 9;

__host__ void copyKernelToConstantMemory(void *h_Kern, size_t kernDataSize);

__global__ void convultionGlobalMemory(
    DataType *newGrid,
    DataType *grid,
    size_t rows,
    size_t cols,
    size_t kernSize);

__global__ void convultionSharedMemory(
    DataType *newGrid,
    DataType *grid,
    size_t rows,
    size_t cols,
    size_t kernSize);