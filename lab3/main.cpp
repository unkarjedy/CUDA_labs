#include "commons.h"
#include "radixSort.h"
#include "cuda_timer.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    string filename = argc > 1 ? argv[1] : "input.txt";

    ifstream input(filename);
    ofstream output("output.txt", fstream::out);
    if (!input.is_open()){
        cerr << "File not found: " << filename << endl;
        return -1;
    }

    size_t arraySize;
    input >> arraySize;
    if (arraySize > MAX_INPUT_ARRAY_SIZE) {
        cout << "Input array max size is: " << MAX_INPUT_ARRAY_SIZE;
        return -2;
    }

    // TODO: check for max arraySize

    auto h_data = shared_ptr<unsigned>(new unsigned[arraySize]);
    
    readArray(h_data.get(), input, arraySize);

    radixSort(h_data.get(), arraySize);

    //printArray(h_data.get(), cout, arraySize);
    printArray(h_data.get(), output, arraySize);

    assert(isIncrSortedArray(h_data.get(), arraySize));

    //cin.get();
    return 0;
}
