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

STiming MesureMethodsTimings(istream &input, bool saveOutput = false);
int RunSimpleProgram(int argc, char ** argv);
int RunMesurings();


int main(int argc, char *argv[])
{
    int rc = 0;
    if (argc > 1 && string(argv[1]) == "-t") {
        rc = RunMesurings();
    }
    else {
        rc = RunSimpleProgram(argc, argv);
    }

    //cin.get();
    return rc;
}


/*
* 1. Reads input from input.txt or specified file as first argument
* 2. Runs stwo methods for convultion prints result matrxi to outputGlob.txt, outputShared.txt
* 3. Prints timings for both methods
*/
int RunSimpleProgram(int argc, char ** argv)
{
    string filename = argc > 1
        ? argv[1] : "input.txt";

    fstream input(filename);
    if (!input.is_open()){
        cerr << "File not found: " << filename << endl;
        return -1;
    }

    STiming timing = MesureMethodsTimings(input, true);
    cout << "Global: " << timing.method1 << endl;
    cout << "Shared: " << timing.method2;
    return 0;
}


/*
* Generate input data for different grid and kernel sizes, 
* mesures times for both methods and prints result
*/
int RunMesurings()
{
    for (size_t kernelSize = 3; kernelSize <= 9; kernelSize += 2){
        for (size_t gridSize = 10; gridSize <= 1000; gridSize *= 10) {
            stringstream input;
            generateInput(input, gridSize, kernelSize);
            STiming timing = MesureMethodsTimings(input);

            cout << "N: " << gridSize << endl;
            cout << "M: " << kernelSize << endl;
            cout << "Global: " << timing.method1 << endl;
            cout << "Shared: " << timing.method2 << endl;
        }
    }
    return 0;
}



STiming MesureMethodsTimings(istream &input, bool saveOutput /*= false*/)
{
    STiming timing;

    fstream outputShared("outputGlobal.txt", fstream::out);
    fstream outputGlobal("outputShared.txt", fstream::out);

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

    h_Grid = (DataType*)malloc(gridDataSize);
    h_GridCheck = (DataType*)malloc(gridDataSize);
    h_Kern = (DataType*)malloc(kernDataSize);
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
    timing.method1 = timer.getTime() / atemps;
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
    timing.method2 = timer.getTime() / atemps;
    cudaMemcpy(h_GridCheck, d_GridNew, gridDataSize, cudaMemcpyDeviceToHost);


    // Check if both methods generate equal results
    if (areGridsEqual(h_Grid, h_GridCheck, haloGridSize)){
        cout << "Both methods generate equal results." << endl;
    }
    else {
        cout << "Error: methods generate DIFFERENT results." << endl;
    }


    // Print result
    if (saveOutput){
        printGridCustomBorders(
            h_Grid, outputShared,
            haloGridSize,
            kernelRadius, kernelRadius + gridSizeFromInput,
            kernelRadius, kernelRadius + gridSizeFromInput);
        printGridCustomBorders(
            h_GridCheck, outputGlobal,
            haloGridSize,
            kernelRadius, kernelRadius + gridSizeFromInput,
            kernelRadius, kernelRadius + gridSizeFromInput);
    }
    

    // Release memory
    free(h_GridCheck);
    cudaFree(d_GridNew);
    cudaFree(d_Grid);
    free(h_Kern);
    free(h_Grid);

    return timing;
}