#include <iostream>
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>
#include <cmath>
#include <time.h>

#define M_PI 3.14159265358979323846

// Define complex number type
typedef std::complex<double> Complex;

// Function to load signal from a CSV file
std::vector<std::vector<Complex>> loadSignalFromCSV(const std::string& filename) {
    std::vector<std::vector<Complex>> signal;
    std::ifstream inputFile(filename);
    if (inputFile.is_open()) {
        std::string line;
        while (getline(inputFile, line)) {
            std::vector<Complex> row;
            std::stringstream ss(line);
            std::string value;
            while (getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            signal.push_back(row);
        }
        inputFile.close();
    } else {
        std::cout << "Failed to open the input file" << std::endl;
    }
    return signal;
}

// Function to perform 1D DFT
std::vector<Complex> dft(const std::vector<Complex>& x) {
    int N = x.size();
    std::vector<Complex> X(N, 0);

    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            X[k] += x[n] * std::exp(Complex(0, -2 * M_PI * k * n / N));
        }
    }

    return X;
}

// Function to perform 2D DFT
std::vector<std::vector<Complex>> dft2D(const std::vector<std::vector<Complex>>& x) {
    int M = x.size();
    int N = x[0].size();
    // Perform row-wise DFT
    std::vector<std::vector<Complex>> X(M, std::vector<Complex>(N));
    for (int i = 0; i < M; ++i) {
        X[i] = dft(x[i]);
    }

    // Perform column-wise DFT
    std::vector<std::vector<Complex>> result(M, std::vector<Complex>(N));
    for (int j = 0; j < N; ++j) {
        std::vector<Complex> column(M);
        for (int i = 0; i < M; ++i) {
            column[i] = X[i][j];
        }
        std::vector<Complex> temp = dft(column);
        for (int i = 0; i < M; ++i) {
            result[i][j] = temp[i];
        }
    }

    return result;
}

// Function to perform inverse 1D DFT
std::vector<Complex> idft(const std::vector<Complex>& X) {
    int N = X.size();
    std::vector<Complex> x(N, 0);

    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < N; ++k) {
            x[n] += X[k] * std::exp(Complex(0, 2 * M_PI * k * n / N));
        }
        x[n] /= N;
    }

    return x;
}

// Function to perform inverse 2D DFT
std::vector<std::vector<Complex>> idft2D(const std::vector<std::vector<Complex>>& X) {
    int M = X.size();
    int N = X[0].size();
    // Perform row-wise IDFT
    std::vector<std::vector<Complex>> x(M, std::vector<Complex>(N));
    for (int i = 0; i < M; ++i) {
        x[i] = idft(X[i]);
    }

    // Perform column-wise IDFT
    std::vector<std::vector<Complex>> result(M, std::vector<Complex>(N));
    for (int j = 0; j < N; ++j) {
        std::vector<Complex> column(M);
        for (int i = 0; i < M; ++i) {
            column[i] = x[i][j];
        }
        std::vector<Complex> temp = idft(column);
        for (int i = 0; i < M; ++i) {
            result[i][j] = temp[i];
        }
    }

    return result;
}


int main() {
    
    /////////////////////////////////////////////////////////////////////
    std::vector<std::vector<Complex>> signal;
    double start, end;

    // Load signal from CSV file
    signal = loadSignalFromCSV("signal.csv");
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    start = clock();

    // Perform 2D DFT
    std::vector<std::vector<Complex>> dftSignal = dft2D(signal);

    end = clock();

    // Print the time taken for 2d dft
    std::cout << "Time taken for serial 2D DFT: " << (end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    // Perform inverse 2D DFT
    start = clock();

    std::vector<std::vector<Complex>> idftSignal = idft2D(dftSignal);

    end = clock();

    // Print the time taken for 2d dft
    std::cout << "Time taken for serial 2D IDFT: " << (end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    /////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////
    // Save the IDFT signal to a CSV file
    // std::ofstream outputFile("idft_signal.csv");
    // if (outputFile.is_open()) {
    //     for (const auto& row : idftSignal) {
    //         for (const auto& value : row) {
    //             outputFile << value.real() << ",";
    //         }
    //         outputFile << std::endl;
    //     }
    //     outputFile.close();
    //     std::cout << "IDFT signal saved to idft_signal.csv" << std::endl;
    // } else {
    //     std::cout << "Failed to open the output file" << std::endl;
    // }
    /////////////////////////////////////////////////////////////////////

    return 0;
}
