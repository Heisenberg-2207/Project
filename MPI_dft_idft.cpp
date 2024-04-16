// Function to load signal from a CSV file
#include <iostream> 
#include <mpi.h> 
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


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Example 2D signal
    std::vector<std::vector<Complex>> signal;

    // Load signal from CSV file
    if(rank == 0){
        signal = loadSignalFromCSV("signal.csv");
    }
    double start, end;
    start = clock();
    
    // Broadcast the signal to all other processes
    MPI_Bcast(signal.data(), signal.size() * signal[0].size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    // Calculate the number of rows to be processed by each process
    int rowsPerProcess = signal.size() / size;
    int remainder = signal.size() % size;

    // Calculate the starting and ending row indices for the current process
    int startRow = rank * rowsPerProcess;
    int endRow = startRow + rowsPerProcess;
    if (rank == size - 1) {
        endRow += remainder;
    }

    // Create a sub-signal for the current process
    std::vector<std::vector<Complex>> subSignal(signal.begin() + startRow, signal.begin() + endRow);

    // Perform 2D DFT on the sub-signal
    std::vector<std::vector<Complex>> dftSignal = dft2D(subSignal);

    // Gather the results from all processes
    std::vector<std::vector<Complex>> gatheredDftSignal;
    if (rank == 0) {
        gatheredDftSignal.resize(signal.size(), std::vector<Complex>(signal[0].size()));
    }
    MPI_Gather(dftSignal.data(), dftSignal.size() * dftSignal[0].size(), MPI_DOUBLE_COMPLEX,
               gatheredDftSignal.data(), dftSignal.size() * dftSignal[0].size(), MPI_DOUBLE_COMPLEX,
               0, MPI_COMM_WORLD);

    // Perform inverse 2D DFT on the gathered results
    // std::vector<std::vector<Complex>> idftSignal;
    // if (rank == 0) {
    //     idftSignal = idft2D(gatheredDftSignal);
    // }

    // // Broadcast the reconstructed signal to all processes
    // MPI_Bcast(idftSignal.data(), idftSignal.size() * idftSignal[0].size(), MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    end = clock();
    double duration = (end - start) / CLOCKS_PER_SEC;   
    // Save the IDFT signal to a CSV file
    if(rank == 0){
        std::cout << "Time taken: " << duration << " seconds" << std::endl;
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
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
