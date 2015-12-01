#include "commons.hpp"
using namespace std;

void readGrid(
    DataType* hostGrid,
    std::istream &input,
    size_t rows, size_t cols,
    size_t haloRows /*= 0*/,
    size_t haloCols /*= 0*/)
{
    size_t realRows = rows + 2 * haloRows;
    size_t realCols = cols + 2 * haloCols;
    memset(hostGrid, 0, sizeof(DataType)* realRows * realCols);
    for (size_t row = haloRows; row < realRows - haloRows; row++) {
        for (size_t col = haloCols; col < realCols - haloCols; col++) {
            input >> hostGrid[row * realCols + col];
        }
    }
}

void readGridCustomBorders(
    DataType* hostGrid,
    std::istream &input,
    size_t rows, size_t cols,
    size_t startX, size_t endX,
    size_t startY, size_t endY)
{
    memset(hostGrid, 0, sizeof(DataType)* rows * cols);
    for (size_t row = startY; row < endY; row++) {
        for (size_t col = startX; col < endX; col++) {
            input >> hostGrid[row * cols + col];
        }
    }
}

void printGrid(
    DataType* hostGrid,
    std::ostream &output,
    size_t rows, size_t cols,
    size_t haloRows /*= 0*/,
    size_t haloCols /*= 0*/)
{
    size_t realRows = rows + 2 * haloRows;
    size_t realCols = cols + 2 * haloCols;
    for (size_t row = haloRows; row < realRows - haloRows; row++) {
        for (size_t col = haloCols; col < realCols - haloCols; col++) {
            output << hostGrid[row * realCols + col] << " ";
        }
        output << endl;
    }
}

void printGridCustomBorders(
    DataType* hostGrid,
    std::ostream &output,
    size_t cols,
    size_t startX, size_t endX,
    size_t startY, size_t endY)
{
    for (size_t row = startY; row < endY; row++) {
        for (size_t col = startX; col < endX; col++) {
            output << hostGrid[row * cols + col] << " ";
        }
        output << endl;
    }
}

size_t ceilRound(size_t size, size_t blockSize)
{
    int tail = size % blockSize;
    if (tail) {
        size += blockSize - tail;
    }
    return size;
}

bool areGridsEqual(DataType *grid1, DataType * grid2, size_t rows, size_t cols /* = 0 */)
{
    cols = (cols == 0) ? rows : cols;
    for (size_t row = 0; row < rows; row++){
        for (size_t col = 0; col < cols; col++){
            if (grid1[row * cols + col] !=
                grid2[row * cols + col])
                return false;
        }
    }
    return true;
}
