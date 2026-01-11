# OpenMP Mini-Labs (C++)

This directory contains **minimal OpenMP C++ mini-labs** designed to build intuition about **shared-memory parallelism**.

These labs focus on **limits**, not tuning.

Key ideas:

- Parallelism does **not** guarantee speedup
- Some algorithms are fundamentally serial (**Amdahl’s Law**)
- Shared resources create **hard stalls**, not just overhead

All labs run on a **single machine** and are intentionally simple.

The goal is **architectural understanding**, not OpenMP expertise.

---

## Quickstart (try this first)

If your system already has a working OpenMP-capable compiler, this may be all you need.

From the `labs/openmp` directory:

```
make build
make run-warmup T=4
```

If you see multiple thread IDs printed, OpenMP is working.

---

## Platform setup

### Linux (native or WSL)

On most Linux systems, OpenMP works out of the box with GCC.

```
sudo apt-get update
sudo apt-get install -y g++ make
make build
```

Recommended thread counts for laptops: `1, 2, 4, 8`.

---

### macOS (recommended: GNU GCC)

Apple’s default `clang` **does not support OpenMP** without extra configuration.

The simplest and most reliable setup is **Homebrew GCC**:

```
brew install gcc
```

Then build normally:

```
make build
```

The Makefile automatically detects the newest available `g++-NN` under:

- `/opt/homebrew/bin` (Apple Silicon)
- `/usr/local/bin` (Intel Macs)

You can also force a specific version:

```
make build CXX=g++-15
```

---

### macOS (advanced: clang + libomp)

If you **prefer clang**, you can make it work by installing `libomp`:

```
brew install libomp
```

Then build with explicit flags:

```
make build CXX=clang++ \
  EXTRA_OMPFLAGS="-Xpreprocessor -fopenmp -lomp" \
  EXTRA_CXXFLAGS="-I$(brew --prefix libomp)/include"
```

This path is more fragile and **not recommended for beginners**.

---

### Windows

Native Windows OpenMP setups are inconsistent.

**Recommended:** use **WSL2 (Ubuntu)** and follow the Linux instructions.

---

## Lab overview

| Lab | File | Purpose | Core idea |
|---:|------|--------|-----------|
| 0 | `omp_00_warmup.cpp` | Sanity check | OpenMP threads are running |
| 1 | `omp_01_amdahl.cpp` | Algorithmic limit | Serial fraction caps speedup |
| 2 | `omp_02_mutex_bottleneck.cpp` | Resource limit | Contention creates queues |

---

## Using the Makefile

### Common targets

```
make build
make run-warmup T=4
make run-amdahl T=1
make run-amdahl T=4
make run-mutex T=8
```

### Inspect detected configuration

```
make print-vars
```

This prints:

- detected OS
- selected compiler
- compiler flags
- OpenMP flags

---

## Troubleshooting

### Error: `clang: unsupported option '-fopenmp'`

You are using Apple clang.

Fix:

```
brew install gcc
make build
```

Or explicitly:

```
make build CXX=g++-15
```

---

### Error: `<cstdio> not found`

This usually means **clang-style flags leaked into a GCC build** (often from Conda).

The provided Makefile **intentionally overrides** injected flags to avoid this.

If you see this error:

- Make sure you are using the provided Makefile
- Run `make print-vars` and confirm `CXX=g++-NN`

---

## Docker (optional)

If you don’t want to deal with platform-specific OpenMP compiler setup (especially on macOS), Docker can provide a clean Linux build environment.

### Install Docker

macOS / Windows:
- Install **Docker Desktop**
- On macOS, the first launch may be blocked by **Privacy & Security**. If Docker doesn’t start (no whale icon), go to:
  - System Settings → Privacy & Security → look for a “Docker was blocked” message → click **Open Anyway**.

Linux:
- Install Docker Engine using your distro’s instructions.

### Build and run

From your repo root:

```
docker build -t openmp-labs -f labs/openmp/Dockerfile .
```

Run an interactive shell with your repo mounted:

```
docker run --rm -it -v "$(pwd)":/repo openmp-labs
```

Inside the container:

```
cd /repo/labs/openmp
make clean
make build
make run-warmup T=4
```

### Notes

- Docker uses a Linux userland, so OpenMP with GCC is typically straightforward.
- Mounting the repo means your compiled binaries will appear in your local `labs/openmp/bin/` directory.

---

## Final note

These OpenMP labs intentionally overlap conceptually with the MPI labs.

This is **by design**:

- OpenMP shows **algorithmic and in-process limits**
- MPI shows **coordination and distributed limits**

If an algorithm does not scale **inside one process**, it will not scale when distributed.

Understand the limits first — then scale out.

