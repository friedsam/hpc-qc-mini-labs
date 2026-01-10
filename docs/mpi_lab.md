# MPI Mini-Labs: Performance Intuition for HPC (and Hybrid HPC–Quantum)

These mini-labs are short, runnable MPI scripts designed to build **performance intuition**:

- **MPI has overhead.** Coordination and communication cost time.
- **Parallelism only helps when there is enough work per rank.**
- **Shared resources create queues.** If everyone needs the same scarce accelerator, adding ranks increases waiting.

All labs are runnable on a laptop (e.g., `mpirun` + `mpi4py`) while illustrating the same **shape** you see on clusters.

---

## Quick start

### Requirements
- MPI runtime installed (Open MPI / MPICH)
- `mpi4py` installed in your Python environment

### General advice (important for laptops)
- **Expect noise.** Use multiple repetitions and look at **median** (and optionally p90), not a single timing.
- **Keep ranks modest.** On laptops, `-np 1,2,4,8` is usually enough.
- **Avoid oversubscription.** If results look chaotic, reduce `-np`, close background apps, re-run.

---

## Lab overview (1a, 1b, 2a, 2b, 3)

| Lab | Script | What it teaches | Primary knobs |
|---:|---|---|---|
| **1a** | `labs/mpi/python/mpi_overhead_bcast.py` | **Collective overhead baseline** (broadcast) | `--mb`, `--reps`, `--warmup` |
| **1b** | `labs/mpi/python/mpi_overhead_sweep.py` | **Overhead vs payload size** (sweep MB) | payload sweep args + `--reps` |
| **2a** | `labs/mpi/python/mpi_granularity.py` | **Granularity crossover** (sleep-based tasks; simple, can be noisy) | task count + sleep/task cost |
| **2b** | `labs/mpi/python/mpi_granularity_tasks.py` | **Granularity crossover** (CPU loop; more stable) + optional sync forcing | `--ntasks`, `--work`, `--sync-every`, `--reps` |
| **3** | `labs/mpi/python/mpi_shared_resource.py` | **Shared resource bottleneck** (token server; QPU/GPU analog) | `--tasks`, `--cpu`, `--res`, `--slots` |

---

# Lab 1a — Overhead baseline (broadcast)

## Purpose
Measure the **coordination cost** of an MPI collective (`Bcast`) as you add ranks.  
There is **no useful parallel work**; this is overhead-only.

## What the script does (conceptual)
- Rank 0 fills a payload buffer of size `--mb`.
- All ranks repeatedly call `comm.Bcast(...)`.
- Reports **median** (and min) walltime across ranks and repetitions.

## Run
```bash
mpirun -np 1 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
mpirun -np 2 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
mpirun -np 4 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
mpirun -np 8 python labs/mpi/python/mpi_overhead_bcast.py --mb 16 --reps 100
```
## Knobs

- `--mb`: payload size (MB)
- `--reps`: measured repetitions
- `--warmup`: warmup repetitions (not counted)

## Expected results / takeaway

- Often, walltime increases with more ranks (more recipients + coordination).
- On laptops, you may see small irregularities due to OS scheduling and shared-memory optimizations.

## Results placeholder

[Paste your console output here]

## Plot placeholders

- **Plot A:** Median walltime vs. number of ranks (fixed `--mb`)
- *(Optional)* Include **p90 walltime** to show noise and outliers


# Lab 1b — Overhead sweep (payload size)

## Purpose
Show that MPI “overhead” has **multiple components**:

- A **latency-like cost**: even very small messages take nonzero time.
- A **bandwidth-like cost**: larger messages take longer to move.

By **sweeping payload size**, these regimes become visible.

## What the script does (conceptual)
- Iterates over a range of payload sizes (in MB).
- For each size:
  - Repeats an MPI collective (typically `Bcast`) many times.
  - Records walltime statistics (median / min).
- No useful computation is performed — this is still **overhead-only**.

Compared to Lab 1a, the key difference is that **message size varies**, not rank count.

## Run (example)

Adjust flags to match the script’s CLI (the core idea is: sweep MB and repeat).

```bash
mpirun -np 4 python labs/mpi/python/mpi_overhead_sweep.py --reps 50
```
## Knobs

- Payload sweep arguments (min / max / steps) — depends on the script
- `--reps`: repetitions per payload size
- (optional) `--warmup`

## Expected results / takeaway

- **Small payloads:** time dominated by coordination / latency.
- **Large payloads:** time grows roughly with payload size (bandwidth-limited).
- **More ranks:** usually increases time due to more recipients and coordination.

## Results placeholder

[Paste your console output here]
[Optional: paste a small table of payload_MB vs median_walltime here]


## Plot placeholders

- **Plot B:** Median walltime vs. payload size (MB) (fixed ranks)
- *(Optional)* One curve per rank count (e.g., np=1/2/4/8) to compare scaling.

---

## Lab 2a — Granularity crossover (simple sleep-based tasks)

### Purpose

Teach **task granularity**:

- If tasks are too small, MPI overhead dominates and adding ranks doesn’t help.
- If tasks are larger, adding ranks can increase throughput.

This version is intentionally **simple** (often uses `sleep`), so it’s easy to read but can be **noisier on laptops**.

### Run (example)

```bash
mpirun -np 1 python labs/mpi/python/mpi_granularity.py
mpirun -np 2 python labs/mpi/python/mpi_granularity.py
mpirun -np 4 python labs/mpi/python/mpi_granularity.py
mpirun -np 8 python labs/mpi/python/mpi_granularity.py
```

### Knobs

- Task count (e.g., `NTASKS`)
- Task cost (e.g., `SLEEP` seconds)

### Expected results / takeaway

- With very small task cost, overhead dominates → more ranks can be slower.
- With larger task cost, throughput increases with ranks (until you hit a knee).

### Results placeholder

[Paste your console output here]
[Optional: paste a small table of payload_MB vs median_walltime here]

### Plot placeholders

- **Plot C:** Median walltime vs. payload size (MB) (fixed ranks)
- *(Optional)* One curve per rank count (e.g., np=1/2/4/8) to compare scaling.

---

## Lab 2b — Granularity crossover (CPU-work tasks; more stable)

### Purpose

Same lesson as Lab 2a, but more repeatable by using a controlled CPU loop instead of sleep.

### What the script does (conceptual)

- Rank 0 creates `--ntasks` task IDs and broadcasts them.
- Each rank takes a slice (`tasks[rank::size]`).
- Each task runs a CPU loop of length `--work`.
- Completion time is the slowest rank (makespan).

### Run (recommended laptop demo that tends to show a knee at np<=8)

```bash
mpirun -np 1 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
mpirun -np 2 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
mpirun -np 4 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
mpirun -np 8 python labs/mpi/python/mpi_granularity_tasks.py --ntasks 2000 --work 10 --sync-every 2
```
### Knobs

- `--ntasks`: number of tasks
- `--work`: CPU work per task (larger = “fatter” tasks)
- `--sync-every`: optional extra synchronization to make overhead visible  
  - higher value → less frequent synchronization (closer to real throughput runs)  
  - lower value → more frequent synchronization (forces coordination overhead to appear)
- `--reps`: repetitions (useful for median / p90 statistics)

### Expected results / takeaway

- With small `--work` (and/or more frequent synchronization), you can see a **knee** where speedup saturates or degrades.
- Increasing `--work` shifts the knee to higher ranks (more ranks help for longer).
- On a laptop, a clean demo often comes from a **moderate task count** (e.g., 2000) plus a **small amount of forced synchronization**.

### Results placeholder

[Paste results for at least two settings, e.g. work=10 and work=200]

### Plot placeholders

- **Plot D:** Tasks/sec vs. ranks for multiple `--work` values  
- *(Optional)* **p90 walltime vs. ranks** to show noise / outliers

---

# Lab 3 — Shared resource bottleneck (QPU / accelerator analog)

## Purpose

Model what happens when MPI ranks share a **scarce resource**:

- one QPU session  
- a small number of GPUs  
- limited license tokens  

Even if CPU work parallelizes, the scarce resource creates **queueing**, and scaling stalls or reverses.

## What the script does (conceptual)

- Rank 0 is a **token server**.
- Worker ranks repeatedly:
  1. do CPU phase (`--cpu`) in parallel  
  2. request token → do resource phase (`--res`) → release token  
- Only `--slots` tokens exist → only `--slots` workers can use the “accelerator” at once.

## Run (example)

```bash
# Fixed slots, increase ranks
mpirun -np 2 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 1
mpirun -np 4 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 1
mpirun -np 8 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 1

# Fixed ranks, increase slots
mpirun -np 8 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 2
mpirun -np 8 python labs/mpi/python/mpi_shared_resource.py --tasks 50 --cpu 0.01 --res 0.05 --slots 4
```
## Knobs

- `--tasks`: tasks per worker rank  
- `--cpu`: time per task in parallel CPU phase  
- `--res`: time per task holding the shared resource  
- `--slots`: number of concurrent “accelerator sessions” allowed  

## Output fields (interpretation)

- `makespan_worker_max`: runtime of the slowest worker (effective job runtime)  
- `server_wall`: how long the token server was active (should ~match makespan)  
- `total_res`: total time workers spent holding tokens (actual accelerator usage)  
- `resource_util`: approx utilization = `total_res / (slots * makespan)`  
- `total_wait`: total time workers spent blocked waiting for tokens (queueing cost)  
- `max_queue`: peak number of waiting workers  

## Expected results / takeaway

- With `--slots 1`, increasing ranks increases contention:
  - makespan rises  
  - `total_wait` can explode  
  - `resource_util` approaches ~100% (bottleneck saturated)  
- Increasing `--slots` reduces makespan and waiting.  
- Increasing `--cpu` can reduce queueing (requests arrive less frequently) and can lower `resource_util` (resource no longer always saturated).  

## Results placeholder

[Paste your runs here: slots=1 for np=2/4/8, then slots=2/4 for np=8]

## Plot placeholders

- **Plot E:** makespan vs ranks (fixed `--slots`)
- **Plot F:** total_wait vs ranks (fixed `--slots`)
- **Plot G:** makespan vs slots (fixed ranks)

---

## Suggested teaching flow (for newcomers)

1. **Lab 1a:** MPI costs something even when nothing useful happens (collective overhead).
2. **Lab 1b:** overhead depends on message size (latency vs bandwidth intuition).
3. **Lab 2b** (or **2a** first if you want simpler code): parallel speedup requires enough work per rank; show a knee/crossover.
4. **Lab 3:** the hybrid reality: shared accelerators create queueing; scaling requires more slots or workflow redesign.

---

## Notes for contributors

- Keep scripts short and runnable on laptops.
- Prefer reporting **median** (and optionally **p90**) over a single timing.
- Avoid hidden magic: if synchronization is used, make it explicit (e.g., `--sync-every`).