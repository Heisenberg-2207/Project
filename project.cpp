#include <iostream>
#include <cmath>
#include <complex>

const double PI = 3.14159265358979323846;
#define N 4

// Function to calculate the 1D DFT
void dft(double x[], std::complex<double> X[]) {
    for (int k = 0; k < N; ++k) {
        X[k] = 0;
        for (int n = 0; n < N; ++n) {
            X[k] += x[n] * std::polar(1.0, -2 * PI * k * n / N);
        }
    }
}

// Function to calculate the 2D DFT
void dft2D(double x[][N], std::complex<double> X[][N]) {
    // Perform DFT on each row
    for (int i = 0; i < N; ++i) {
        dft(x[i], X[i]);
    }

    // Perform DFT on each column
    for (int j = 0; j < N; ++j) {
        double column[N];
        std::complex<double> columnDFT[N];

        // Extract the column values
        for (int i = 0; i < N; ++i) {
            column[i] = x[i][j];
        }

        // Perform DFT on the column
        dft(column, columnDFT);

        // Store the column DFT values in the output matrix
        for (int i = 0; i < N; ++i) {
            X[i][j] = columnDFT[i];
        }
    }
}

int main() {
    double x[N][N] = {{1, 2, 3, 4},
                      {5, 6, 7, 8},
                      {9, 10, 11, 12},
                      {13, 14, 15, 16}}; // Sample input signal
    std::complex<double> X[N][N]; // Output signal

    // Calculate 2D DFT
    dft2D(x, X);

    // Print the result
    std::cout << "2D DFT Result:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << "X[" << i << "][" << j << "] = " << X[i][j] << std::endl;
        }
    }

    return 0;
}
