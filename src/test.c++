#include <Eigen/Core>
#include <iostream>

template <typename MatrixType>
MatrixType rollRight(const MatrixType& matrix, int shift) {
    int numCols = matrix.cols();
    MatrixType result(matrix.rows(), numCols);
    shift = shift % numCols;  // Handle shift larger than number of columns

    if (shift == 0) {
        return matrix;
    }

    // Perform the shift
    result.leftCols(shift) = matrix.rightCols(shift);
    result.rightCols(numCols - shift) = matrix.leftCols(numCols - shift);

    return result;
}

int main() {
    Eigen::MatrixXi mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    std::cout << "Original matrix:\n" << mat << "\n";

    Eigen::MatrixXi shifted = rollRight(mat, 1);
    std::cout << "Matrix after rolling right by 1 column:\n" << shifted << "\n";

    return 0;
}
