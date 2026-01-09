from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Change these two knobs
NTASKS = 1000
SLEEP  = 0.00001 # seconds per task (simulate latency)

def task(_):
    time.sleep(SLEEP)
    return 1

if rank == 0:
    tasks = list(range(NTASKS))
else:
    tasks = None

tasks = comm.bcast(tasks, root=0)

my_tasks = tasks[rank::size]

t0 = time.perf_counter()
my_count = sum(task(t) for t in my_tasks)
all_counts = comm.gather(my_count, root=0)
t1 = time.perf_counter()

if rank == 0:
    done = sum(all_counts)
    print(f"ranks={size} done={done} wall={t1-t0:.5f}s tasks/sec={done/(t1-t0):.1f}")