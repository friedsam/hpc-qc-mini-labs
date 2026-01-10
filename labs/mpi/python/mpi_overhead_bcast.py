#!/usr/bin/env python3
"""
MPI Mini-Lab 1: Overhead baseline (broadcast only)

Goal
----
Measure the *coordination cost* of an MPI collective (Bcast) as you add ranks.

What it does
------------
- Rank 0 allocates a payload buffer of size --mb.
- All ranks participate in comm.Bcast(payload) repeatedly.
- We time a synchronized “collective step”:
    Barrier -> Bcast -> Barrier

Why the barriers?
----------------
They make each repetition start and end in a synchronized state, so the timing
is a stable measure of “collective coordination” (not just raw Bcast kernel time).
This is especially helpful on laptops where OS scheduling noise is significant.

Expected behavior
-----------------
- Time generally increases with more ranks.
- Results can be noisy; use medians and mins over many repetitions.

Run examples
------------
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
    ap.add_argument("--warmup", type=int, default=10, help="Warmup repetitions (not counted)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nbytes = int(args.mb * 1024 * 1024)

    # Use uint8 to focus on communication rather than numeric kernels.
    payload = np.empty(nbytes, dtype=np.uint8)
    if rank == 0:
        payload[:] = np.random.randint(0, 256, size=nbytes, dtype=np.uint8)

    # Warmup (not timed)
    for _ in range(args.warmup):
        comm.Barrier()
        comm.Bcast(payload, root=0)
        comm.Barrier()

    times = np.empty(args.reps, dtype=np.float64)

    # Timed repetitions: Barrier -> Bcast -> Barrier
    for i in range(args.reps):
        comm.Barrier()
        t0 = time.perf_counter()
        comm.Bcast(payload, root=0)
        comm.Barrier()
        times[i] = time.perf_counter() - t0

    # Gather per-rank timings so rank 0 can summarize
    all_times = comm.gather(times, root=0)

    if rank == 0:
        flat = np.concatenate(all_times)
        med = float(np.median(flat))
        mn = float(np.min(flat))

        # Rough intuition: broadcast pushes ~payload to (size-1) recipients.
        eff = (args.mb * max(size - 1, 1)) / med if med > 0 else float("inf")

        print(
            f"ranks={size} payload_MB={args.mb:.1f} reps={args.reps} "
            f"median={med:.6f}s min={mn:.6f}s approx_MB_per_s={eff:.1f}"
        )


if __name__ == "__main__":
    main()