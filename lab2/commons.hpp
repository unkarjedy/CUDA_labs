// Data type used in grid
typedef int DataType;

#include <iostream>

// Forward declarations
void readGrid(
    DataType* hostGrid,
    std::istream &input,
    size_t rows, size_t cols,
    size_t haloRows = 0,
    size_t haloCols = 0);

void readGridCustomBorders(
    DataType* hostGrid,
    std::istream &input,
    size_t rows, size_t cols,
    size_t startX, size_t endX,
    size_t startY, size_t endY);

void printGrid(
    DataType* hostGrid,
    std::ostream &output,
    size_t rows, size_t cols,
    size_t haloRows = 0,
    size_t haloCols = 0);

void printGridCustomBorders(
    DataType* hostGrid,
    std::ostream &output,
    size_t cols,
    size_t startX, size_t endX,
    size_t startY, size_t endY);


size_t ceilRound(size_t size, size_t blockSize);

bool areGridsEqual(DataType *grid1, DataType * grid2, size_t rows, size_t cols = 0);