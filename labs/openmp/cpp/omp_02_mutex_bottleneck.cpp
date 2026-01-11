// ==========================
// File: labs/openmp/cpp/omp_02_mutex_bottleneck.cpp
// ===============================
// Purpose: demonstrate a *shared-resource* bottleneck using a mutex
// (critical section). Models contention / queueing for a single slot.
//
// Build:
//   g++ -O3 -std=c++17 -fopenmp omp_02_mutex_bottleneck.cpp -o omp_02_mutex_bottleneck
// Run (try several):
//   OMP_NUM_THREADS=1 ./omp_02_mutex_bottleneck
//   OMP_NUM_THREADS=2 ./omp_02_mutex_bottleneck
//   OMP_NUM_THREADS=4 ./omp_02_mutex_bottleneck
//   OMP_NUM_THREADS=8 ./omp_02_mutex_bottleneck

#include <omp.h>
#include <chrono>
#include <iostream>
#include <thread>

int main() {
    const int iters = 200;            // total iterations
    const double parallel_ms = 2.0;   // parallel phase duration
    const double serial_ms   = 5.0;   // serialized (locked) phase

    int nt = 0;
    auto t0 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        #pragma omp single
        nt = omp_get_num_threads();

        #pragma omp for schedule(static)
        for (int i = 0; i < iters; i++) {
            // Parallel phase
            std::this_thread::sleep_for(
                std::chrono::duration<double, std::milli>(parallel_ms));

            // Serialized "one-slot" resource
            #pragma omp critical
            {
                std::this_thread::sleep_for(
                    std::chrono::duration<double, std::milli>(serial_ms));
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "threads=" << nt
              << "  iters=" << iters
              << "  parallel_ms=" << parallel_ms
              << "  serial_ms=" << serial_ms
              << "  wall=" << sec << "s\n";
    return 0;
}
