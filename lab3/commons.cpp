#include "commons.h"

#include <ctime>
using namespace std;

void readArray(
    DataType *data,
    istream &in,
    size_t size)
{
    for (size_t i = 0; i < size; i++){
        in >> data[i];
    }
}

void printArray(
    DataType *data,
    ostream &out,
    size_t size,
    char sep /* = '\n' */)
{
    for (size_t i = 0; i < size; i++){
        out << data[i] << sep;
    }
    cout << endl;
}

/* Checks if the array is monotonically non-decreasing */
bool isIncrSortedArray(DataType *h_data, size_t arraySize)
{
    for (unsigned i = 1; i < arraySize; i++){
        if (h_data[i] < h_data[i - 1])
            return false;
    }
    return true;
}
