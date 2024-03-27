#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

#define M_PI 3.14159265358979323846

using namespace std;

// Define complex number type
typedef complex<double> Complex;

// Function to perform 1D DFT
vector<Complex> dft(const vector<Complex>& x) {
    int N = x.size();
    vector<Complex> X(N, 0);

    for (int k = 0; k < N; ++k) {
        for (int n = 0; n < N; ++n) {
            X[k] += x[n] * exp(Complex(0, -2 * M_PI * k * n / N));
        }
    }

    return X;
}

// Function to perform 2D DFT
vector<vector<Complex>> dft2D(const vector<vector<Complex>>& x) {
    int M = x.size();
    int N = x[0].size();
    // Perform row-wise DFT
    vector<vector<Complex>> X(M, vector<Complex>(N));
    for (int i = 0; i < M; ++i) {
        X[i] = dft(x[i]);
    }

    // Perform column-wise DFT
    vector<vector<Complex>> result(M, vector<Complex>(N));
    for (int j = 0; j < N; ++j) {
        vector<Complex> column(M);
        for (int i = 0; i < M; ++i) {
            column[i] = X[i][j];
        }
        vector<Complex> temp = dft(column);
        for (int i = 0; i < M; ++i) {
            result[i][j] = temp[i];
        }
    }

    return result;
}

int main() {
    // Example 2D signal
    vector<vector<Complex>> signal = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};

    // Print the signal vector
    // cout << "Signal vector:" << endl;
    // for (const auto& row : signal) {
    //     for (const auto& value : row) {
    //         cout << value << " ";
    //     }
    //     cout << endl;
    // }
    
    // Perform 2D DFT
    vector<vector<Complex>> dftSignal = dft2D(signal);

    // Output the result
    cout << "2D DFT result:" << endl;
    for (const auto& row : dftSignal) {
        for (const auto& value : row) {
            cout << value << " ";
        }
        cout << endl;
    }

    return 0;
}
