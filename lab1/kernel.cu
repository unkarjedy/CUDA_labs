#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;


// Forward declarations
void readGrid(int* hostGrid, fstream &input, int M, int N);
void printGrid(int* hostGrid, ostream &output, int M, int N);

__global__ void gameOfLifeStepKernel(int M, int N, int *grid, int *newGrid);


int main()
{
    fstream input("input.txt");
    fstream output("output.txt", fstream::out);

    int M, N;
    input >> M >> N;

    int* hostGrid;
    int* deviceGrid;
    int* deviceGridNew;

    // Allocate host & device memory
    size_t dataSize = sizeof(int) * (M + 2) * (N + 2); // extra 2 for ghost rows and columns


    hostGrid = (int*)malloc(dataSize);
    cudaMalloc(&deviceGrid, dataSize);
    cudaMalloc(&deviceGridNew, dataSize);

    readGrid(hostGrid, input, M, N);

    cudaMemcpy(deviceGrid, hostGrid, dataSize, cudaMemcpyHostToDevice);


    const int BLOCK_SIZE_LIN = 256;

    dim3 blockSize(BLOCK_SIZE_LIN, 1, 1);
    int linGrid = (int)ceil(M * N / (float)BLOCK_SIZE_LIN);
    dim3 gridSize(linGrid, 1, 1);

    gameOfLifeStepKernel << <gridSize, blockSize >> >(M, N, deviceGrid, deviceGridNew);

    // Print result
    cudaMemcpy(hostGrid, deviceGridNew, dataSize, cudaMemcpyDeviceToHost);
    printGrid(hostGrid, output, M, N);

    // Release memory
    cudaFree(deviceGridNew);
    cudaFree(deviceGrid);
    free(hostGrid);

    //cin.get();
    return 0;
}


void readGrid(int* hostGrid, fstream &input, int M, int N)
{
    memset(hostGrid, 0, sizeof(int)* (M + 2) * (N + 2));
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            input >> hostGrid[i *(N + 2) + j];
        }
    }

}

void printGrid(int* hostGrid, ostream &output, int M, int N)
{
    for (int i = 1; i <= M; i++) {
        for (int j = 1; j <= N; j++) {
            output << hostGrid[i * (N + 2) + j] << " ";
        }
        output << endl;
    }
}

__global__ void gameOfLifeStepKernel(int M, int N, int *grid, int *newGrid)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int id = idy * blockDim.x * gridDim.x + idx;

    if (id < M * N) {
        // recalculate id in real matrix (with ghost rows/cols)
        id = id + (N + 2) + 2 * (id / N) + 1;

        int neighbours = 0
            + grid[id - (N + 2)] + grid[id + (N + 2)]   // up & down
            + grid[id - 1] + grid[id + 1]               // left & right
            + grid[id - (N + 3)] + grid[id + (N + 3)]   // diagonals
            + grid[id - (N + 1)] + grid[id + (N + 1)];


        int cell = grid[id];

        // Game of life rules
        if (cell == 1 && (neighbours < 2 || neighbours > 3)){
            cell = 0;
        }
        else if (cell == 0 && neighbours == 3){
            cell = 1;
        }

        newGrid[id] = cell;
    }

}