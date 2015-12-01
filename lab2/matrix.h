
struct Matrix {

    int* const data;
    const size_t rows, cols;

    Matrix(size_t rows, size_t cols) :
        data(new int[rows * cols]),
        rows(rows), cols(cols)
    {}

    ~Matrix(){
        delete[] data;
    }

};