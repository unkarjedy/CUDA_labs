#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "convultion.hpp"
#include "cuda_timer.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <sstream>
#include <cassert>

using namespace std;


int main(int argc, char *argv[])
{
    string filename = argc > 1 
        ? argv[1] 
        : "input.txt";

    fstream input(filename);
    fstream output("output.txt", fstream::out);

    size_t gridSizeFromInput;       // matrix square side size
    size_t kernelSize;              // convolution kernel matrix square side size
    input >> gridSizeFromInput >> kernelSize;
    assert(kernelSize % 2 == 1);

    size_t kernelRadius;
    kernelRadius = kernelSize / 2;

    DataType *h_Grid, *h_Kern;
    DataType *d_Grid; // d_Kern is in constant memory
    DataType *d_GridNew;

    // for checking that 2 different CUDA kernels work equally
    DataType *h_GridCheck;


    const size_t CUDA_BLOCK_SIZE = 16;

    // gridSize ceil-rounded to k * CUDA_BLOCK_SIZE  to elluminate extra "if"s inside kernel
    size_t gridSize = ceilRound(gridSizeFromInput, CUDA_BLOCK_SIZE);

    // Allocate host & device memory
    size_t haloGridSize = (gridSize + 2 * kernelRadius);
    size_t gridDataSize = sizeof(DataType)* haloGridSize * haloGridSize; // with halo rows and columns
    size_t kernDataSize = sizeof(DataType)* kernelSize * kernelSize;

    h_Grid      = (DataType*)malloc(gridDataSize);
    h_GridCheck = (DataType*)malloc(gridDataSize);
    h_Kern      = (DataType*)malloc(kernDataSize);
    cudaMalloc(&d_Grid, gridDataSize);
    cudaMalloc(&d_GridNew, gridDataSize);


    // Read matrix and kernel from input
    readGridCustomBorders(
        h_Grid, input,
        haloGridSize, haloGridSize,
        kernelRadius, kernelRadius + gridSizeFromInput,
        kernelRadius, kernelRadius + gridSizeFromInput);
    readGrid(h_Kern, input, kernelSize, kernelSize);

    cudaMemcpy(d_Grid, h_Grid, gridDataSize, cudaMemcpyHostToDevice);
    copyKernelToConstantMemory(h_Kern, kernDataSize);

    // Calculate block/grid dimensions
    dim3 cudaBlockDim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE, 1);
    // actually gridSize is already multiple of CUDA_BLOCK_SIZE, but...
    size_t cudaGridSize = (size_t)ceil(gridSize / (float)CUDA_BLOCK_SIZE); 
    dim3 cudaGridDim(cudaGridSize, cudaGridSize, 1);


    /******************************************
    * Time measuring for 2 CUDA kernels
    ******************************************/
    CudaTimer timer;
    int atemps = 10;

    timer.start();
    for (int i = 0; i < atemps; i++) {
        convultionGlobalMemory << <cudaGridDim, cudaBlockDim >> >(
            d_GridNew, d_Grid,
            gridSize, gridSize,
            kernelSize);
    }
    timer.stop();
    cout << timer.getTime() / atemps << endl;
    cudaMemcpy(h_Grid, d_GridNew, gridDataSize, cudaMemcpyDeviceToHost);

    size_t sharedSide = CUDA_BLOCK_SIZE + 2 * kernelRadius;
    size_t sharedSize = sizeof(DataType)* sharedSide * sharedSide;
    timer.start();
    for (int i = 0; i < atemps; i++) {
        convultionSharedMemory << <cudaGridDim, cudaBlockDim, sharedSize >> >(
            d_GridNew, d_Grid,
            gridSize, gridSize,
            kernelSize);
    }
    timer.stop();
    cout << timer.getTime() / atemps << endl;
    cudaMemcpy(h_GridCheck, d_GridNew, gridDataSize, cudaMemcpyDeviceToHost);


    // Check if both methods generate equal results
    areGridsEqual(h_Grid, h_GridCheck, haloGridSize)
        ? cout << "Both methods generate equal results." << endl
        : cout << "Error: methods generate DIFFERENT results." << endl;


    // Print result
    printGridCustomBorders(
        h_Grid, output,
        haloGridSize,
        kernelRadius, kernelRadius + gridSizeFromInput,
        kernelRadius, kernelRadius + gridSizeFromInput);

    // Release memory
    free(h_GridCheck);
    cudaFree(d_GridNew);
    cudaFree(d_Grid);
    free(h_Kern);
    free(h_Grid);
    
    cin.get();
    return 0;
}
