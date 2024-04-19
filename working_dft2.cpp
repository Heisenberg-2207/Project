#include <iostream>
#include <cmath>
#include <mpi.h>
#include <complex>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rows = 4;
    const int cols = 4;

    // Create a 2D array of complex numbers
    std::complex<double> array[rows][cols];
    if(rank == 0){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                array[i][j] = std::complex<double>(i * cols + j, 0.0);
            }
        }
    }

    // Perform row-wise decomposition
    int rows_per_process = rows / size;
    std::complex<double> local_rows[rows_per_process][cols];

    MPI_Scatter(&array[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &local_rows[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    std::complex<double> dft[rows][cols], local_dft[rows_per_process][cols];
    for (int i = 0; i < rows_per_process; i++) {
        for (int k = 0; k < cols; k++) {
            local_dft[i][k] = 0.0;
            for (int n = 0; n < cols; n++) {
                double angle = 2 * M_PI * k * n / cols;
                std::complex<double> complex_angle(cos(angle), -sin(angle));
                local_dft[i][k] += local_rows[i][n] * complex_angle;
            }
        }
    }

    MPI_Gather(&local_dft[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &dft[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);


    std::complex<double> transposed_dft[cols][rows];
    if(rank == 0){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << dft[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // Transpose the DFT matrix
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed_dft[j][i] = dft[i][j];
            }
        }
    }

    /////////////////////////////////

    std::complex<double> local_rows2[rows_per_process][cols];

    MPI_Scatter(&transposed_dft[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &local_rows2[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    std::complex<double> transposed_dft2[rows][cols], local_dft2[rows_per_process][cols];
    for (int i = 0; i < rows_per_process; i++) {
        for (int k = 0; k < cols; k++) {
            local_dft2[i][k] = 0.0;
            for (int n = 0; n < cols; n++) {
                double angle = 2 * M_PI * k * n / cols;
                std::complex<double> complex_angle(cos(angle), -sin(angle));
                local_dft2[i][k] += local_rows2[i][n] * complex_angle;
            }
        }
    }

    MPI_Gather(&local_dft2[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, &transposed_dft2[0][0], rows_per_process * cols, MPI_CXX_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

    std::complex<double> dft2[cols][rows];
    if(rank == 0){
        // Transpose the DFT matrix
        std::complex<double> transposed_dft[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dft2[j][i] = transposed_dft2[i][j];
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << dft2[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    
    MPI_Finalize();

    return 0;
}
