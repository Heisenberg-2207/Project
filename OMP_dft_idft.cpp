// Function to load signal from a CSV file
#include <iostream> // Add missing include directive
#include <fstream>
#include <sstream>
#include <complex>
#include <vector>
#include <cmath>
#include <omp.h>
#include <time.h>

#define thread_count 16

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

    #pragma omp parallel for num_threads(thread_count)
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
    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < M; ++i) {
        X[i] = dft(x[i]);
    }

    // Perform column-wise DFT
    std::vector<std::vector<Complex>> result(M, std::vector<Complex>(N));
    #pragma omp parallel for num_threads(thread_count)
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

    #pragma omp parallel for num_threads(thread_count)
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
    #pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < M; ++i) {
        x[i] = idft(X[i]);
    }

    // Perform column-wise IDFT
    std::vector<std::vector<Complex>> result(M, std::vector<Complex>(N));
    #pragma omp parallel for num_threads(thread_count)
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
    // Example 2D signal

    std::vector<std::vector<Complex>> signal, dftSignal, idftSignal;

    // Load signal from CSV file
    signal = loadSignalFromCSV("signal.csv");

    clock_t start, end;
    start = clock();
    // Perform 2D DFT
    dftSignal = dft2D(signal);

    // Perform inverse 2D DFT
    idftSignal = idft2D(dftSignal);

    end = clock();

    //Timing
    double duration = double((end - start) / CLOCKS_PER_SEC);
    std::cout << "Time taken: " << duration << " seconds" << std::endl;

    // Save the IDFT signal to a CSV file
    std::ofstream outputFile("idft_signal.csv");
    if (outputFile.is_open()) {
        for (const auto& row : idftSignal) {
            for (const auto& value : row) {
                outputFile << value.real() << ",";
            }
            outputFile << std::endl;
        }
        outputFile.close();
        std::cout << "IDFT signal saved to idft_signal.csv" << std::endl;
    } else {
        std::cout << "Failed to open the output file" << std::endl;
    }

    return 0;
}
