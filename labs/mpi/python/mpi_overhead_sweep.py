#!/usr/bin/env python3
"""
MPI Mini-Lab A (Simple): Broadcast + tiny work + gather

Goal
----
Show *overhead* effects: when the computation is tiny and communication dominates,
adding ranks does NOT speed things up. In many environments, it gets slower.

This is intentionally simple and easy to read.

What it does
------------
1) Rank 0 creates a payload array of size --mb.
2) All ranks participate in MPI_Bcast (broadcast the payload).
3) Each rank does a tiny computation (sum of the first --sum_n elements).
4) All ranks MPI_Gather the tiny scalar back to rank 0.
5) Repeat and report median/min timing (single runs are noisy).

Why timings can look "weird" on a laptop
-----------------------------------------
- OS scheduling jitter and background processes
- CPU frequency scaling / thermal throttling
- MPI implementation details (shared-memory fast paths)
- Allocation / paging effects (first-touch memory)

Run examples
------------
mpirun -np 1 python labs/mpi/python/mpi_overhead_sweep.py --mb 16 --reps 50
mpirun -np 2 python labs/mpi/python/mpi_overhead_sweep.py --mb 16 --reps 50
mpirun -np 4 python labs/mpi/python/mpi_overhead_sweep.py --mb 16 --reps 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from mpi4py import MPI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=16.0, help="Payload size in MB (float64 array)")
    ap.add_argument("--sum_n", type=int, default=1000, help="How many elements to sum (tiny work)")
    ap.add_argument("--reps", type=int, default=50, help="Measured repetitions")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup repetitions (not counted)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Convert MB -> number of float64 elements
    nbytes = int(args.mb * 1024 * 1024)
    n = max(nbytes // 8, 1)  # float64 is 8 bytes

    if rank == 0:
        payload = np.random.rand(n).astype(np.float64, copy=False)
    else:
        payload = np.empty(n, dtype=np.float64)

    # Clamp sum_n so we never slice past the end
    sum_n = min(max(args.sum_n, 1), payload.size)

    comm.Barrier()

    times = []

    # Warmup
    for _ in range(args.warmup):
        comm.Bcast(payload, root=0)
        _x = float(np.sum(payload[:sum_n]))
        _ = comm.gather(_x, root=0)

    comm.Barrier()

    # Timed reps
    for _ in range(args.reps):
        t0 = time.perf_counter()

        comm.Bcast(payload, root=0)
        x = float(np.sum(payload[:sum_n]))
        _ = comm.gather(x, root=0)

        comm.Barrier()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    all_times = comm.gather(np.array(times, dtype=np.float64), root=0)

    if rank == 0:
        flat = np.concatenate(all_times)
        med = float(np.median(flat))
        mn = float(np.min(flat))
        print(
            f"ranks={size} payload_MB={payload.nbytes/1e6:.3f} reps={args.reps} "
            f"median={med:.6f}s min={mn:.6f}s"
        )


if __name__ == "__main__":
    main()