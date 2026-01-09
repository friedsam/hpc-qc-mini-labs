# Introduction to High Performance Computing (HPC)

**Robert Loredo, Jose Hernandez, Claudia Friedsam, Jonah Sachs, Emre Camkerten, Kasirga Hamidi**

---

## 1. What is High Performance Computing?

High Performance Computing (HPC) refers to the use of many computational resources—typically CPUs, memory, and high-speed interconnects—working together to solve problems that are too large, too slow, or too memory-intensive for a single computer.

HPC is commonly used when:

- The problem size exceeds the memory of a single machine  
- The computation would take too long on a single CPU  
- Many similar tasks must be evaluated efficiently (ensembles, parameter sweeps)

Importantly, **HPC is not just a faster computer**.  
It is a different execution model that requires explicit consideration of how work and data are distributed.

---

## 2. Two Fundamental Parallel Programming Models

Most HPC applications are written using one (or both) of the following models.

---

### 2.1 MPI — Distributed Memory Parallelism

MPI (Message Passing Interface) is a standard for writing programs that run as multiple independent processes, called *ranks*. Programs scale by explicitly passing data between ranks using messages.

**Key characteristics:**

- Each rank has its own private memory  
- Data is exchanged explicitly via messages  
- Programs scale across multiple machines (nodes)

**Mental model:**  
Many independent workers that communicate by sending messages.

MPI is well-suited for:

- Large simulations  
- Parameter sweeps  
- Problems with natural data partitioning  

---

### 2.2 OpenMP — Shared Memory Parallelism

OpenMP (Open Multi-Processing) is a threading model for parallelism within a single process. It allows developers to parallelize loops and code regions across multiple CPU cores.

**Key characteristics:**

- Multiple threads share the same memory  
- Parallel regions are created explicitly  
- Synchronization (locks, barriers) is often required  

**Mental model:**  
Many workers sharing a single workspace.

OpenMP is well-suited for:

- Fine-grained parallelism  
- Loop-based numerical workloads  
- Single-node, multi-core systems  

---

## 3. Why More Parallelism Is Not Always Better

A common misconception is that adding more workers (cores, threads, or ranks) will always make a program faster. In practice, every parallel program has an **optimal degree of parallelism**.

Two fundamental limits apply.

---

### 3.1 Overhead

Parallel execution introduces costs:

- Communication between ranks (MPI)  
- Synchronization between threads (OpenMP)  
- Task scheduling and coordination  

When tasks are too small, overhead can dominate total runtime.

---

### 3.2 Serial Bottlenecks

Some parts of a program cannot be parallelized:

- Input/output  
- Shared resource access  
- Algorithmic dependencies  

These serial sections limit the maximum achievable speedup.

Even a small serial fraction can dominate total runtime.

---

## 4. Mini-Lab: MPI Scaling on a Laptop

To illustrate MPI behavior, we can run a simple MPI program locally and vary the number of ranks.

**What we observe:**

- Runtime improves initially as ranks increase  
- Speedup eventually plateaus or degrades  
- Communication overhead becomes dominant  

This demonstrates that:

- MPI scaling is not linear  
- Optimal rank count depends on task size and communication cost  

*(Results on a laptop differ from an HPC cluster, but the scaling behavior is the same.)*

---

## 5. Mini-Lab: OpenMP and Serial Bottlenecks

Using OpenMP, we can parallelize part of a computation while keeping another part strictly serial.

**What we observe:**

- Increasing thread count improves performance at first  
- Speedup saturates quickly  
- Serial sections dominate total runtime  

This illustrates **Amdahl’s Law** in practice:

> The maximum speedup is limited by the fraction of code that must run serially.

---

## 6. Shared Resource Bottlenecks (Conceptual Bridge)

Many real HPC workflows include shared resources:

- File systems  
- GPUs  
- Specialized accelerators  

If many workers must wait for a single shared resource:

- Overall runtime is dominated by that resource  
- Adding more workers provides little benefit  

This pattern is especially important for hybrid workflows that combine HPC with scarce accelerators.

---

## 7. Where Slurm Fits

Slurm is a workload manager used on many HPC systems.

Slurm:

- Allocates compute resources  
- Launches jobs  
- Enforces usage limits and scheduling policies  

Slurm does **not** change how MPI or OpenMP work.  
The same program can be run:

- Manually on a laptop  
- Via Slurm on an HPC cluster  

**Example (conceptual):**

```bash
#SBATCH -n 8
#SBATCH -c 4
srun ./my_program

```

Slurm is required only when running on a real cluster.

---

## 8. Why This Matters for Hybrid HPC–Quantum Workflows

Hybrid quantum–classical workflows often behave like MPI programs with a shared, serialized accelerator:

- Classical computation is parallel and scalable  
- Quantum hardware is scarce and externally managed  
- Jobs may queue or experience latency  

Understanding MPI, OpenMP, and bottlenecks is essential to:

- Predict performance  
- Avoid unrealistic expectations  
- Design efficient hybrid workflows  

---

## 9. Key Takeaways

- HPC requires explicit parallel design  
- More workers do not guarantee better performance  
- MPI and OpenMP address different parallelism models  
- Serial bottlenecks fundamentally limit scaling  
- Hybrid quantum workflows resemble HPC programs with shared resources  

This conceptual foundation prepares users to understand how HPC and quantum computing interact in practice.