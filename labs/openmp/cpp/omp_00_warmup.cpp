// ===============================
// File: labs/openmp/cpp/omp_00_warmup.cpp
// ===============================
// Purpose: sanity check that OpenMP is enabled and running.
//
// Build (GCC):
//   g++ -O3 -std=c++17 -fopenmp omp_00_warmup.cpp -o omp_00_warmup
// Run:
//   OMP_NUM_THREADS=4 ./omp_00_warmup

#include <omp.h>
#include <cstdio>

int main() {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();

        #pragma omp critical
        std::printf("Hello from thread %d of %d\n", tid, nt);
    }
    return 0;
}
