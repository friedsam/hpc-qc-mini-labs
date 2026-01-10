#!/usr/bin/env python3
"""
MPI Mini-Lab 3: Shared resource bottleneck (QPU/accelerator analog)

Goal
----
Show how a scarce shared resource (e.g., 1 QPU session, 2 GPU slots, 1 license token)
creates queueing and limits scaling even when you add more MPI worker ranks.

Model
-----
- Rank 0 is a token server representing the shared resource.
- Ranks 1..(N-1) are workers.
- Each worker executes TASKS_PER_RANK tasks.
- Each task has two phases:
    (A) CPU phase      : time.sleep(--cpu)  [parallel across workers]
    (B) RESOURCE phase : acquire token -> time.sleep(--res) -> release token
                         [serialized by number of --slots]

Protocol (simple + robust)
--------------------------
Workers send exactly these messages to rank 0:
- TAG_REQ      : request a token (payload ignored)
- TAG_RELEASE  : release a token (payload: token_id)
- TAG_DONE     : worker finished; payload is the worker stats dict

Rank 0 replies with:
- TAG_GRANT    : token granted (payload: token_id)

Run examples (minimum is 2 ranks: server + 1 worker)
----------------------------------------------------
mpirun -np 2 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 1
mpirun -np 4 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 1
mpirun -np 8 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 1

Then increase "accelerators" (slots)
------------------------------------
mpirun -np 8 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 2
mpirun -np 8 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 4

Try changing:
- --slots 1 vs 2 vs 4
- --cpu smaller/larger
- --res smaller/larger
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Deque, Dict, Any

from mpi4py import MPI


# Message tags (explicit protocol)
TAG_REQ = 10       # worker -> server: request token (payload ignored)
TAG_GRANT = 11     # server -> worker: token granted (payload: int token_id)
TAG_RELEASE = 12   # worker -> server: release token (payload: int token_id)
TAG_DONE = 13      # worker -> server: finished all tasks (payload: stats dict)


def server_loop(comm: MPI.Comm, slots: int, n_workers: int) -> Dict[str, Any]:
    """
    Rank 0: token server.

    Maintains:
    - available tokens (0..slots-1)
    - FIFO queue of waiting worker ranks
    - counts done workers
    - collects final worker stats dicts (sent with TAG_DONE)

    Returns:
      dict with server stats + worker_stats map.
    """
    available: Deque[int] = deque(range(slots))  # token IDs available now
    waiting: Deque[int] = deque()                # worker ranks waiting
    done_workers = 0

    # Server-side stats
    n_grants = 0
    n_releases = 0
    max_queue = 0

    worker_stats: Dict[int, Dict[str, float]] = {}

    status = MPI.Status()

    while done_workers < n_workers:
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_REQ:
            # Worker requests a token
            if available:
                token = available.popleft()
                comm.send(token, dest=src, tag=TAG_GRANT)
                n_grants += 1
            else:
                waiting.append(src)
                if len(waiting) > max_queue:
                    max_queue = len(waiting)

        elif tag == TAG_RELEASE:
            # Worker releases a token
            token = int(msg)
            n_releases += 1

            # Immediately hand token to next waiter if any; else return to pool.
            if waiting:
                nxt = waiting.popleft()
                comm.send(token, dest=nxt, tag=TAG_GRANT)
                n_grants += 1
            else:
                available.append(token)

        elif tag == TAG_DONE:
            # Worker finished; msg is the worker's final stats dict
            worker_stats[src] = msg
            done_workers += 1

        else:
            raise RuntimeError(f"Unknown tag received by server: {tag} from rank {src}")

    return {
        "n_grants": n_grants,
        "n_releases": n_releases,
        "max_queue": max_queue,
        "worker_stats": worker_stats,
    }


def worker_run(comm: MPI.Comm, tasks: int, cpu_s: float, res_s: float) -> None:
    """
    Worker ranks (rank != 0): execute tasks and send final stats to server.
    """
    rank = comm.Get_rank()

    total_wait = 0.0
    total_cpu = 0.0
    total_res = 0.0

    t_start = time.perf_counter()

    for _ in range(tasks):
        # (A) CPU phase (parallel)
        if cpu_s > 0:
            t0 = time.perf_counter()
            time.sleep(cpu_s)
            total_cpu += time.perf_counter() - t0

        # (B) RESOURCE phase (serialized by tokens)
        comm.send(None, dest=0, tag=TAG_REQ)

        w0 = time.perf_counter()
        token = comm.recv(source=0, tag=TAG_GRANT)  # blocks until granted
        total_wait += time.perf_counter() - w0

        r0 = time.perf_counter()
        if res_s > 0:
            time.sleep(res_s)
        total_res += time.perf_counter() - r0

        comm.send(int(token), dest=0, tag=TAG_RELEASE)

    wall = time.perf_counter() - t_start

    stats = {
        "rank": float(rank),
        "tasks": float(tasks),
        "wall": wall,
        "cpu_time": total_cpu,
        "res_time": total_res,
        "wait_time": total_wait,
    }

    # One message at the end: "I'm done and here are my stats"
    comm.send(stats, dest=0, tag=TAG_DONE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=50, help="tasks per worker rank")
    ap.add_argument("--cpu", type=float, default=0.01, help="seconds CPU phase per task")
    ap.add_argument("--res", type=float, default=0.05, help="seconds shared-resource phase per task")
    ap.add_argument("--slots", type=int, default=1, help="number of shared resource slots (tokens)")
    args = ap.parse_args()

    if args.tasks < 1:
        raise SystemExit("--tasks must be >= 1")
    if args.slots < 1:
        raise SystemExit("--slots must be >= 1")
    if args.cpu < 0 or args.res < 0:
        raise SystemExit("--cpu and --res must be >= 0")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        raise SystemExit("Run with at least 2 ranks (server rank 0 + >=1 worker).")

    if rank == 0:
        t0 = time.perf_counter()
        stats = server_loop(comm, slots=args.slots, n_workers=size - 1)
        server_wall = time.perf_counter() - t0

        workers = list(stats["worker_stats"].values())

        # Makespan is governed by the slowest worker (MPI-style completion time)
        makespan = max(w["wall"] for w in workers) if workers else 0.0

        total_wait = sum(w["wait_time"] for w in workers)
        total_res = sum(w["res_time"] for w in workers)

        # Utilization of the scarce resource (rough but intuitive):
        # total time spent "in resource" / (capacity * makespan)
        util = (total_res / (args.slots * makespan)) if makespan > 0 else 0.0

        print(
            f"ranks={size} workers={size-1} slots={args.slots} "
            f"tasks_per_rank={args.tasks} cpu={args.cpu:.3f}s res={args.res:.3f}s\n"
            f"makespan_worker_max={makespan:.4f}s  server_wall={server_wall:.4f}s\n"
            f"total_wait={total_wait:.4f}s  total_res={total_res:.4f}s  "
            f"resource_util~={util*100:.1f}%  max_queue={stats['max_queue']}"
        )

    else:
        worker_run(comm, tasks=args.tasks, cpu_s=args.cpu, res_s=args.res)


if __name__ == "__main__":
    main()