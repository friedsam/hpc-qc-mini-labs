#!/usr/bin/env python3
"""
MPI Mini-Lab 2: Granularity / crossover (task farm).

Goal
----
Show when adding MPI ranks helps vs hurts, depending on task granularity *and*
(optional) synchronization overhead.

This lab is designed to work on a laptop with <=8 ranks, where you may not see a
clear "knee" unless tasks are small or you introduce coordination overhead.

Model
-----
- There are NTASKS independent tasks total.
- Tasks are assigned round-robin: task_ids = range(rank, ntasks, size)
- Each task performs deterministic CPU work for WORK iterations (no sleep).

Optional coordination overhead
-----------------------------
Real MPI apps often synchronize periodically (reductions, halo exchange, etc.).
To mimic that, you can add a small synchronization cost:
- --sync-every K : every K tasks, perform an MPI allreduce on a scalar.

Expected behavior
-----------------
- With --sync-every 0 (default):
    * If WORK is large enough and size <= physical cores, throughput often rises
      with ranks (until you saturate cores).
    * If WORK is tiny, overhead and OS noise can dominate; more ranks can be slower.
- With --sync-every > 0:
    * As ranks increase, sync overhead grows and you usually see a knee even on a laptop.

Runs (examples)
---------------
# 1) Pure compute scaling (may be mostly monotone up to 8 ranks on a laptop)
mpirun -np 1 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 2000
mpirun -np 2 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 2000
mpirun -np 4 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 2000
mpirun -np 8 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 2000

# 2) Granularity crossover (tiny tasks => more ranks can be slower)
mpirun -np 1 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 20
mpirun -np 8 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 20

# 3) Force a clear knee on a laptop (periodic sync)
mpirun -np 1 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
mpirun -np 2 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
mpirun -np 4 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
mpirun -np 8 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2

Notes for laptops
-----------------
- If you run more ranks than physical cores, you may see slowdown from oversubscription.
- Single runs are noisy; this script uses warmup + reps and reports median makespan.
"""

from __future__ import annotations

import argparse
import time
import numpy as np
from mpi4py import MPI


def cpu_work(work: int, seed: int) -> int:
    """
    Deterministic CPU-only "work" that avoids sleep() and avoids BLAS threads.
    Uses a simple integer LCG update loop.

    Returns an integer accumulator (prevents the loop being optimized away).
    """
    x = seed & 0xFFFFFFFF
    for _ in range(work):
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntasks", type=int, default=2000, help="Total tasks (global), distributed across ranks")
    ap.add_argument("--work", type=int, default=2000, help="CPU work units per task (bigger => heavier tasks)")
    ap.add_argument("--sync-every", type=int, default=0, help="Every K tasks, do an MPI allreduce (0 disables)")
    ap.add_argument("--reps", type=int, default=25, help="Measured repetitions")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup repetitions")
    ap.add_argument("--seed", type=int, default=1234, help="Deterministic seed base")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Round-robin slice without broadcasting a Python list (keeps the lab about scaling).
    task_ids = range(rank, args.ntasks, size)
    my_ntasks = len(range(rank, args.ntasks, size))

    rep_times = []

    # Used for the optional sync (keep tiny payload).
    sync_val = np.array([rank], dtype=np.int64)
    sync_out = np.array([0], dtype=np.int64)

    for rep in range(args.warmup + args.reps):
        comm.Barrier()
        t0 = time.perf_counter()

        acc = 0
        since_sync = 0

        for tid in task_ids:
            acc ^= cpu_work(args.work, seed=args.seed + 100000 * rep + tid)
            if args.sync_every > 0:
                since_sync += 1
                if since_sync >= args.sync_every:
                    # Mimic periodic coordination/communication.
                    comm.Allreduce(sync_val, sync_out, op=MPI.SUM)
                    since_sync = 0

        # One last tiny sync so that rep end is comparable across ranks.
        comm.Barrier()
        t1 = time.perf_counter()

        _ = acc  # keep "used"
        if rep >= args.warmup:
            rep_times.append(t1 - t0)

    rep_times = np.array(rep_times, dtype=float)

    # Gather per-rank time series so rank 0 can compute makespan per repetition.
    all_rep_times = comm.gather(rep_times, root=0)

    if rank == 0:
        # makespan per rep = max over ranks (what wall time "feels like")
        makespans = np.max(np.vstack(all_rep_times), axis=0)
        med = float(np.median(makespans))
        mn = float(np.min(makespans))
        p90_wall = np.percentile(makespans, 90)

        tasks_per_sec = args.ntasks / med if med > 0 else 0.0

        print(
            f"ranks={size} ntasks={args.ntasks} work={args.work} sync_every={args.sync_every} "
            f"reps={args.reps} median_wall={med:.6f}s min_wall={mn:.6f}s p90_wall={p90_wall:.6f} "
            f"tasks_per_sec={tasks_per_sec:.1f}"
        )


if __name__ == "__main__":
    main()