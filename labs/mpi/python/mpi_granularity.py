#!/usr/bin/env python3
"""
MPI Mini-Lab B (Simple): Task granularity (sleep) scaling

Goal
----
Show why MPI parallelism helps only when each task is "big enough".

Model
-----
- Rank 0 creates a list of NTASKS identical tasks.
- Tasks are broadcast to all ranks.
- Each rank takes a round-robin slice of tasks (rank::size).
- Each task is simulated by time.sleep(--sleep).
- Results are gathered and a wall-time is reported.

Important
---------
This is *not* a communication benchmark; it is a simple demo of:
- embarrassingly parallel work distribution (parameter sweep style)
- overheads (startup, Python, MPI collectives) that dominate when tasks are tiny

Run examples
------------
mpirun -np 1 python labs/mpi/python/mpi_granularity.py --tasks 1000 --sleep 0.001
mpirun -np 2 python labs/mpi/python/mpi_granularity.py --tasks 1000 --sleep 0.001
mpirun -np 8 python labs/mpi/python/mpi_granularity.py --tasks 1000 --sleep 0.001
"""

from __future__ import annotations

import argparse
import time

from mpi4py import MPI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=1000, help="Total number of tasks")
    ap.add_argument("--sleep", type=float, default=1e-4, help="Seconds per task (simulated work)")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        tasks = list(range(args.tasks))
    else:
        tasks = None

    tasks = comm.bcast(tasks, root=0)
    my_tasks = tasks[rank::size]

    comm.Barrier()
    t0 = time.perf_counter()

    for _ in my_tasks:
        time.sleep(args.sleep)

    comm.Barrier()
    t1 = time.perf_counter()

    my_done = len(my_tasks)
    all_done = comm.gather(my_done, root=0)

    if rank == 0:
        done = sum(all_done)
        wall = t1 - t0
        print(f"ranks={size} tasks={done} sleep={args.sleep:.6f}s wall={wall:.6f}s tasks_per_s={done/wall:.1f}")


if __name__ == "__main__":
    main()