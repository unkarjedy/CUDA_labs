#ifndef _COMMONS_H_
#define _COMMONS_H_

// Data type used in grid
typedef unsigned int DataType;

#include <iostream>
#include <ostream>

//void generateInput(std::ostream &out, size_t gridSize, size_t kernelSize);

void readArray(
    DataType *data,
    std::istream &in,
    size_t size);

void printArray(
    DataType *data,
    std::ostream &out,
    size_t size,
    char sep = ' ');

bool isIncrSortedArray(DataType *h_data, size_t arraySize);

#endif