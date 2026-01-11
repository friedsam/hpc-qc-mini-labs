// ===============================
// File: labs/openmp/cpp/omp_01_amdahl.cpp
// ===============================
// Purpose: demonstrate *algorithmic* limit (Amdahl's Law).
// A strictly serial section caps speedup regardless of threads.
//
// Build:
//   g++ -O3 -std=c++17 -fopenmp omp_01_amdahl.cpp -o omp_01_amdahl
// Run (try several):
//   OMP_NUM_THREADS=1 ./omp_01_amdahl
//   OMP_NUM_THREADS=2 ./omp_01_amdahl
//   OMP_NUM_THREADS=4 ./omp_01_amdahl
//   OMP_NUM_THREADS=8 ./omp_01_amdahl

#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>

static double serial_work(int n) {
    volatile double s = 0.0; // discourage over-optimization
    for (int i = 0; i < n; i++) {
        s += std::sin(i * 1e-6);
    }
    return (double)s;
}

static double parallel_work(int n) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; i++) {
        s += std::cos(i * 1e-6);
    }
    return s;
}

int main() {
    // Tune if runtime is too long/short on your machine
    const int Nserial = 50'000'000;   // strictly serial
    const int Npar    = 250'000'000;  // parallelizable

    int nt = 0;
    #pragma omp parallel
    #pragma omp single
    nt = omp_get_num_threads();

    auto t0 = std::chrono::high_resolution_clock::now();

    const double a = serial_work(Nserial);
    const double b = parallel_work(Npar);

    auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "threads=" << nt
              << "  a=" << a
              << "  b=" << b
              << "  wall=" << sec << "s\n";
    return 0;
}
