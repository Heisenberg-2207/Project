#include <iostream>
#include <complex>
#include <valarray>

#define M_PI 3.14159265358979323846

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

// Cooley-Tukey FFT (in-place, 2D)
// Higher memory consumption, but faster computation
void fft2(CArray& x)
{
    const size_t N = x.size();
    if (N <= 1) return;

    // Divide
    CArray even = x[std::slice(0, N/2, 2)];
    CArray odd = x[std::slice(1, N/2, 2)];

    // Conquer
    fft2(even);
    fft2(odd);

    // Combine
    for (size_t k = 0; k < N/2; ++k)
    {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k+N/2] = even[k] - t;
    }
}

int main()
{
    const int N = 4;
    CArray data(N*N);
    
    // Populate data with example 2D signal
    for(int i = 0; i < N*N; ++i) {
        data[i] = i + 1;
    }

    // 2D FFT
    fft2(data);

    // Output results
    std::cout << "FFT:" << std::endl;
    for(int i = 0; i < N*N; ++i) {
        std::cout << "FFT[" << i << "] = " << data[i] << std::endl;
    }

    return 0;
}
