#!/usr/bin/env python3
"""
MPI Mini-Lab 1: Overhead baseline (broadcast).

What it measures:
- The cost of coordinating ranks for a collective operation (Bcast).
- This is "overhead only": no useful parallel work.
Expected behavior:
- As ranks increase, time generally increases (more recipients + coordination).
- Single runs are noisy -> use median/min over many repetitions.

Run:
  mpirun -np 1 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
  mpirun -np 2 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
  mpirun -np 4 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
"""

from __future__ import annotations
import argparse
import time
import numpy as np
from mpi4py import MPI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=16.0, help="Payload size in MB")
    ap.add_argument("--reps", type=int, default=100, help="Measured repetitions")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup repetitions")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nbytes = int(args.mb * 1024 * 1024)
    # Use a byte buffer: avoids NumPy BLAS/threading and focuses on pure comm.
    buf = np.empty(nbytes, dtype=np.uint8)
    if rank == 0:
        buf[:] = np.random.randint(0, 256, size=nbytes, dtype=np.uint8)

    times = []

    for i in range(args.warmup + args.reps):
        comm.Barrier()
        t0 = time.perf_counter()
        comm.Bcast(buf, root=0)
        comm.Barrier()
        t1 = time.perf_counter()
        if i >= args.warmup:
            times.append(t1 - t0)

    times = np.array(times, dtype=float)
    # Gather per-rank timings so rank0 can summarize
    all_times = comm.gather(times, root=0)

    if rank == 0:
        flat = np.concatenate(all_times)
        med = float(np.median(flat))
        mn = float(np.min(flat))
        # Effective throughput for this collective (very rough, but intuitive)
        # One broadcast sends ~payload to (size-1) ranks; shared-memory may differ.
        eff = (args.mb * max(size - 1, 1)) / med
        print(
            f"ranks={size} payload_MB={args.mb:.1f} reps={args.reps} "
            f"median={med:.6f}s min={mn:.6f}s approx_MB_per_s={eff:.1f}"
        )


if __name__ == "__main__":
    main()