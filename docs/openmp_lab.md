# OpenMP Mini-Labs: Algorithmic Limits and In-Process Bottlenecks

These mini-labs are short, runnable **OpenMP (C++)** programs designed to build intuition about the limits of shared-memory parallelism.

They focus on questions that are especially important for learners coming from **quantum computing**:
- **Parallelism does not guarantee speedup.**
- **Some algorithms are fundamentally serial.**
- **Shared resources create hard stalls, not just overhead.**

All labs are runnable on a laptop and are intentionally simple.
The goal is **conceptual clarity**, not OpenMP tuning expertise.




## What these OpenMP labs are meant to teach (and what they are not)

The OpenMP examples are intentionally minimal. Their purpose is **not** to teach OpenMP tuning or performance engineering, but to expose two fundamental limits of classical parallelism that matter for hybrid HPC–quantum workflows.

First, the *serial fraction* example demonstrates an **algorithmic limit**: if part of an algorithm is inherently serial, adding more threads does not help. This is Amdahl’s Law in its purest form. No amount of classical hardware can accelerate work that cannot be parallelized in the first place. For quantum-centric learners, this is a crucial reality check: classical preprocessing or postprocessing must itself be well-parallelized to benefit from HPC at all.

Second, the *mutex / critical-section* example demonstrates a **shared-resource bottleneck**. Even if most work parallelizes, contention for a single resource forces threads to wait in line, creating queueing and idle time. This models situations such as limited licenses, serialized I/O, or a single accelerator slot. Unlike the serial fraction, this bottleneck is not inevitable—but it must be designed around.

Together, these examples highlight that classical parallelism can fail for **two very different reasons**: unavoidable algorithmic structure and avoidable resource contention.

The MPI labs that follow extend these same ideas to distributed systems. There, overhead arises from coordination and communication, and shared bottlenecks appear as external resources (e.g., accelerators) rather than mutexes. The underlying lessons, however, are the same: parallelism has costs, and good architectures start by understanding where those costs come from.




## Context: Why OpenMP comes before MPI (for QC-oriented learners)

For learners coming from QC, it is tempting to assume that classical HPC will “take care of” all classical preprocessing and postprocessing.

Before worrying about:
- MPI communication
- distributed coordination
- accelerators and queues

the **first question** should be:

*Does the classical algorithm parallelize well enough to justify HPC at all?*

OpenMP is the cleanest way to answer this, because:
- there is no network,
- no data distribution,
- no infrastructure complexity.

If an algorithm does not scale **within a single process**, it will not magically scale when distributed.

---




## Quick start


### Requirements
- C++ compiler with OpenMP support
- Linux: gcc (usually works out of the box)
- macOS: Homebrew gcc or clang + libomp
- Windows: WSL strongly recommended


### General advice
- Use small thread counts: `OMP_NUM_THREADS=1,2,4,8`
- Focus on qualitative behavior, not exact timings
- These are intuition labs, not benchmarks

---




## Lab overview (0,1,2)

| Lab | Script | What it teaches | Core idea |
|----:|--------|----------------|-----------|
| 0 | `omp_00_warmup.cpp` | OpenMP is working | Toolchain sanity check |
| 1 | `omp_01_amdahl.cpp` | Algorithmic limit | Serial fraction caps speedup |
| 2 | `omp_02_mutex_bottleneck.cpp` | Shared-resource bottleneck | Hard serialization via contention |




# Lab 0 — OpenMP warm-up (sanity check)


## Purpose

Verify that:
- OpenMP is enabled
- multiple threads are actually running

This lab exists only to confirm the toolchain works, especially given C++ / OpenMP platform differences.


## What the script does
- Spawns an OpenMP parallel region
- Each thread prints its thread ID
- Output order is intentionally nondeterministic


## Run
```bash
g++ -O3 -std=c++17 -fopenmp omp_00_warmup.cpp -o omp_00_warmup
OMP_NUM_THREADS=4 ./omp_00_warmup
```

## Results placeholder

[Paste your console output here]


## Takeaway

If this does not run, the problem is **setup**, not OpenMP concepts.
Do not proceed until this works.




# Lab 1 — Algorithmic limit (Amdahl’s Law)


## Purpose

Demonstrate that some algorithms **cannot scale**, regardless of how many threads are available.

This is an **algorithmic limit**, not a resource or overhead issue.


## What the script does (conceptual)
- Executes a **strictly serial** computation
- Executes a **parallelizable** computation
- Measures total wall time

Only part of the program can use multiple threads.


## Run
```bash
g++ -O3 -std=c++17 -fopenmp omp_01_amdahl.cpp -o omp_01_amdahl
OMP_NUM_THREADS=1 ./omp_01_amdahl
OMP_NUM_THREADS=2 ./omp_01_amdahl
OMP_NUM_THREADS=4 ./omp_01_amdahl
OMP_NUM_THREADS=8 ./omp_01_amdahl
```

## Results placeholder
[Paste your console output here]


## Expected results / takeaway
- Speedup quickly saturates
- Increasing threads beyond a point has little effect
- This limit is **mathematical**, not fixable by better hardware


### Key lesson:

If an algorithm has a large serial fraction, classical HPC cannot help — even before MPI or accelerators enter the picture.




# Lab 2 — Shared-resource bottleneck (mutex / critical section)


## Purpose

Show how **contention for a shared resource** creates **hard stalls** inside a parallel program.

This models situations such as:
- a single accelerator slot
- serialized I/O
- license servers
- protected global state

Unlike Lab 1, this bottleneck is **not inevitable**, but it must be designed around.


## What the script does (conceptual)

Each iteration has two phases:
1.	A parallel phase (threads work independently)
2.	A serialized phase protected by a mutex (critical)

Only one thread may enter the serialized region at a time.


## Run
```bash
g++ -O3 -std=c++17 -fopenmp omp_02_mutex_bottleneck.cpp -o omp_02_mutex_bottleneck
OMP_NUM_THREADS=1 ./omp_02_mutex_bottleneck
OMP_NUM_THREADS=2 ./omp_02_mutex_bottleneck
OMP_NUM_THREADS=4 ./omp_02_mutex_bottleneck
OMP_NUM_THREADS=8 ./omp_02_mutex_bottleneck
```


## Results placeholder
[Paste your console output here]


## Expected results / takeaway

- Adding threads helps only until contention dominates
- Threads spend increasing time waiting
- Parallelism collapses due to queueing


### Key lesson:

Shared resources turn parallel work into serialized queues unless explicitly managed.




## How these OpenMP lessons connect to MPI (and QCSC)

The OpenMP labs expose two fundamental failure modes of parallelism:
1. **Algorithmic limits**  
   Some work cannot be parallelized at all (Amdahl’s Law).

2. **Resource limits**  
   Some work is serialized because many workers want the same resource.

The MPI labs extend these same ideas to **distributed systems**:
- algorithmic limits appear as global synchronization
- resource limits appear as external accelerators (QPU, GPU, licenses)
- overhead appears as communication and coordination cost

The underlying lesson is the same:

**Parallelism has structure, costs, and limits — understanding them comes before scaling out.**




## Notes for contributors
- These labs are intentionally minimal
- Do not turn them into OpenMP tuning exercises
- The goal is architectural intuition, not API coverage
- Overlap with MPI lessons is intentional but not redundant

