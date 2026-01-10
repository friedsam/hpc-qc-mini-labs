#!/usr/bin/env python3
"""
MPI Mini-Lab 2: Granularity / crossover.

What it measures:
- Whether adding ranks increases throughput for a "bag of tasks".
- Tasks are distributed round-robin to ranks.
- Each task does controlled CPU work (not sleep) to avoid OS timing noise.

Expected behavior:
- For small WORK, MPI overhead dominates -> more ranks can be slower.
- For larger WORK, compute dominates -> more ranks can be faster.
- There is usually a "knee" (optimal rank count) for a given workload.

Run (default ntasks 2000, work 2000):
  mpirun -np 1 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 100 --work 20
  mpirun -np 2 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 100 --work 20
  mpirun -np 4 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 100 --work 20
  mpirun -np 8 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 100 --work 20
"""

from __future__ import annotations
import argparse
import time
from mpi4py import MPI


def do_work(work: int) -> None:
    # Deterministic CPU work; tune 'work' to change task cost.
    x = 0.0
    for _ in range(work):
        x = x * 1.0000001 + 0.0000001
    # Prevent optimization (Python won't optimize away anyway, but keep it explicit)
    if x < -1e300:
        raise RuntimeError("impossible")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntasks", type=int, default=20000)
    ap.add_argument("--work", type=int, default=2000)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 defines tasks; broadcast the task IDs to everyone.
    tasks = list(range(args.ntasks)) if rank == 0 else None
    tasks = comm.bcast(tasks, root=0)

    # Round-robin partitioning: each rank takes every 'size'-th task.
    my_tasks = tasks[rank::size]

    t0 = time.perf_counter()
    for _ in my_tasks:
        do_work(args.work)
    t1 = time.perf_counter()

    my_time = t1 - t0
    my_done = len(my_tasks)

    # Gather per-rank stats to rank 0
    all_times = comm.gather(my_time, root=0)
    all_done = comm.gather(my_done, root=0)

    if rank == 0:
        wall = max(all_times)  # slowest rank determines completion time
        done = sum(all_done)
        print(
            f"ranks={size} ntasks={args.ntasks} work={args.work} "
            f"wall={wall:.5f}s tasks_per_sec={done/wall:.1f}"
        )


if __name__ == "__main__":
    main()