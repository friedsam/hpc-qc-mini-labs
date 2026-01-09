from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# payload size in floats (change to see crossover)
N = 2_000_000  # ~16MB if float64
payload = None

if rank == 0:
    payload = np.random.rand(N)
else:
    payload = np.empty(N)

comm.Barrier()
t0 = time.perf_counter()

# broadcast large array (cost grows with size and ranks)
comm.Bcast(payload, root=0)

# do tiny compute so comm dominates
x = float(np.sum(payload[:1000]))

# gather tiny results
allx = comm.gather(x, root=0)

comm.Barrier()
t1 = time.perf_counter()

if rank == 0:
    print(f"ranks={size}  wall={t1-t0:.5f}s  payload_MB={payload.nbytes/1e6:.3f}")