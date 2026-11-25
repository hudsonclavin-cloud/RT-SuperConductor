# Questions for the NCSU-Hackathon agent: sharing the laptop GPU

Hand this file to the agent that works in `C:\Users\hudso\NCSU-Hackathon`.
It asks that agent read-only questions. With the answers, the
RT-SuperConductor session decides whether both training pipelines can run
on the laptop at the same time, and with what limits.

Written 2026-11-25 by the RT-SuperConductor session.

---

**Goal:** answer the 12 questions below about your GPU training pipeline,
without changing anything, so that RT-SuperConductor's training and yours can
share the laptop's one GPU without breaking each other.

**TRIPWIRE DISCIPLINE:** run to completion. Stop only if:
- (a) a command would change a file, a package, a setting or a process;
- (b) a training run is going right now and a command would disturb it
  (reading `nvidia-smi` is fine);
- (c) an answer would have to be guessed. Write "unknown" and say what would
  measure it, then carry on with the rest.

## CONTEXT

**The laptop**, as RT-SuperConductor's run recorded it on 2026-11-23:
- NVIDIA GeForce RTX 5070 Laptop GPU, 7.96 GB, compute capability 12.0.
- Windows 11, Python 3.12.10, torch 2.15.0+cu132.

**The Python environment is shared.** RT-SuperConductor runs from
`C:\Users\hudso\RT-SuperConductor`, but it uses your environment's Python,
`C:\Users\hudso\NCSU-Hackathon\.venv-gpu`.

**What RT-SuperConductor runs on the GPU**

| Job | Measured alone | GPU memory |
| --- | --- | --- |
| `scripts/overnight_gpu.py` (one GPU job at a time; crystal-graph and tabular networks) | 26 jobs in 5.8 hours | [Certain] at most 0.77 GB allocated by PyTorch in any job. The process total that `nvidia-smi` shows is higher: CUDA context plus cache, not recorded |
| `scripts/local_gpu_train.py` (small tabular networks) | 16 to 18 minutes per 20-network bank | not recorded. [Likely] under 1 GB |
| `scripts/foundation_gpu_train.py --model tabicl` (next run, not yet run) | [Speculative] 10 to 30 minutes | unknown. [Speculative] 2 to 6 GB, the largest of all |

**How RT-SuperConductor behaves**
- To stop it, create a file named `STOP` in its output folder. It finishes
  the current job and starts no more.
- It keeps Windows awake while it runs.
- On a CUDA out-of-memory error, a crystal-graph job retries once with half
  the batch. That changes the recipe and spoils its comparison, so
  RT-SuperConductor must never run out of memory because of you.

**What a snapshot of your repository showed** (commit `65a6d4e`; it may be
out of date, so answer from the live code):
- `gpu/run_pipeline.py` runs one GPU job at a time, with CPU baselines beside
  it (`--cpu-workers 1`, `--cpu-threads 2`).
- It writes `telemetry.csv` (GPU use, memory, power, temperature) into each
  pipeline folder.
- Heat guard (`gpu/pipeline/orchestrator.py`):
  - before a GPU job, if the card is at 82 °C or more, it waits until 75 °C;
  - during a GPU job, three readings at 92 °C or more terminate the job.
- [Likely] Two pipelines heat the card more than one, so this guard would
  fire more often.

### CRITICAL: the shared environment

Do not install, upgrade or remove anything in `.venv-gpu` while answering.
RT-SuperConductor's next run needs `pip install tabicl`, which may pull in
other versions of packages you depend on. Question 9 decides whether that run
gets its own environment.

## STEPS (all read-only, Windows PowerShell)

**1. Confirm the repository state**

```powershell
cd C:\Users\hudso\NCSU-Hackathon
```
```powershell
git log --oneline -1
```
```powershell
git status --short
```

If HEAD is not `65a6d4e`, the snapshot notes above may be out of date. Say so,
and answer from the code as it is now.

**2. Environment versions**

```powershell
C:\Users\hudso\NCSU-Hackathon\.venv-gpu\Scripts\python.exe -m pip list --format=freeze --disable-pip-version-check
```

**3. The GPU right now**

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv
```
```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**4. The laptop: cores, memory and free disk**

```powershell
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
```
```powershell
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```
```powershell
Get-PSDrive C | Select-Object Used,Free
```

**5. Your last full GPU run's telemetry.** Use your last full run, not a smoke
rehearsal. If the newest folder is a rehearsal, set `$run` to the full run's
folder by hand.

```powershell
$run = (Get-ChildItem C:\Users\hudso\NCSU-Hackathon\gpu\runs\pipeline -Directory | Sort-Object LastWriteTime | Select-Object -Last 1).FullName
```
```powershell
$run
```
```powershell
C:\Users\hudso\NCSU-Hackathon\.venv-gpu\Scripts\python.exe -c "import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); g=lambda k:[float(x[k]) for x in r if x.get(k) not in (None,'','None')]; u=sorted(g('util_pct')); t=g('temp_c'); m=g('mem_mib'); print('samples',len(r),'| mean util %',round(sum(u)/len(u),1),'| p95 util %',u[round(0.95*(len(u)-1))],'| peak mem MiB',max(m),'| peak temp C',max(t),'| share >=82 C',round(sum(x>=82 for x in t)/len(t),3),'| share >=92 C',round(sum(x>=92 for x in t)/len(t),3))" "$run\telemetry.csv"
```

- This mean includes the gaps between GPU jobs, so it understates use during a
  job.
- If your records allow it, also give the mean GPU use during GPU jobs only.

**6. Your three largest GPU jobs.** From that run's records, give for each:
- its name and minutes;
- its peak GPU memory: what PyTorch allocated and, if recorded, what it
  reserved;
- its batch size, and whether mixed precision (`--amp`) was on.

## QUESTIONS

Answer each one, citing the `file:line` it rests on, or "measured" with the
number, or "unknown".

| # | Question | Why it matters |
| --- | --- | --- |
| 1 | When do you plan to run next, with which profile, and how long does it take alone? | Sets the schedule |
| 2 | What is the peak GPU memory of your largest job as `nvidia-smi` sees it (the process total, not only PyTorch's count)? Does any job reserve memory up front: a memory fraction, `torch.compile`, large cuDNN workspaces, or another framework? | The two pipelines together must fit in about 7 GB, leaving about 1 GB for Windows and the display ([Likely], not measured) |
| 3 | What does your pipeline do on a CUDA out-of-memory error: fail, retry the same, or retry with a smaller setting? Does a smaller setting change your results? | Either side running out of memory can silently change results |
| 4 | Running alone, how busy is the GPU during your GPU jobs (mean and 95th percentile)? | **The deciding number.** [Likely] if either pipeline keeps the GPU above about 80% busy, sharing makes each run at about half speed, with no time saved overall. Then running one after the other is better |
| 5 | Confirm the heat guard's thresholds and what it does. Would a job ended for heat be retried, and how much progress is lost (last checkpoint)? Is there a flag to wait instead of ending the job? | Two loads make the card hotter |
| 6 | CPU and memory: threads per GPU job (`torch.set_num_threads`, `OMP_NUM_THREADS`), `DataLoader` workers, threads of the CPU baselines, and peak RAM per job | The CPU can become the bottleneck before the GPU does |
| 7 | Does any job report a speed or timing as its result (for example `profile_batched_mlp`, the timed matmul in `check_env`, throughput figures in the report)? | Those numbers would be wrong while the GPU is shared |
| 8 | Does your pipeline change anything machine-wide: power plans, scheduled tasks to resume after a reboot, GPU clock or power settings, stopping other Python or GPU processes, or refusing to start when the GPU is busy? | Either pipeline could break the other |
| 9 | Which packages and versions in `.venv-gpu` must not change? Do you plan to install anything there? Would you rather RT-SuperConductor used its own copy of the environment? | RT-SuperConductor's next run needs `pip install tabicl` |
| 10 | Confirm how to stop and resume your run (`STOP`, `STOP_NOW`, `--resume`) and how much work a stop loses | So either side can yield the GPU safely |
| 11 | How much disk does a full run write, and how much is free? | Both runs write at once |
| 12 | Does anything else in your pipeline assume it has the GPU to itself? | Catches what the questions above missed |

## REPORT: reply in this form

```text
HEAD: <hash> (matches 65a6d4e: yes/no)
ENV: torch=<> cuda=<> numpy=<> scikit-learn=<> pandas=<>; other pins: <>
GPU NOW: <used> of <total> MiB; processes: <pid name MiB, or none>
LAPTOP: <cores> cores / <logical> logical; RAM <GB>; C: free <GB>
LAST RUN: <folder>; <hours> h; mean util <%>; p95 util <%>; mean util during GPU jobs <% or unknown>;
          peak mem <MiB>; peak temp <C>; share >=82 C <>; share >=92 C <>
LARGEST JOBS: <name, minutes, allocated GB, reserved GB, batch, amp> x3
Q1: ...
...
Q12: ...
ANOMALIES: <anything unexpected, or none>
```

## CONSTRAINTS

- Read-only:
  - no file edits, no commits, no pushes;
  - no package installs, no settings changes;
  - do not start or stop any run.
- Do not train or benchmark anything to answer. A new run would take the GPU,
  and a timing taken now could be distorted by another job. Use past records,
  and say "unknown" where none exist.
- Do not open or change anything in `C:\Users\hudso\RT-SuperConductor`.
  Everything you need about it is in this file.

---

## What happens next (for the owner)

Paste the agent's reply into the RT-SuperConductor session. It will then
propose, for its own side only:
- whether to share at all or run one after the other, decided by question 4;
- a memory cap per pipeline that keeps the two under the card's limit;
- turning off its half-batch retry while sharing, so a memory error stops a
  job instead of quietly changing it;
- its own Python environment for `tabicl`, if question 9 says so.

Any change on the NCSU-Hackathon side is for that project's agent to propose
and the owner to approve.

## Answers received, and the decision, 2026-11-25

The NCSU-Hackathon agent answered from its code and past run records (its
HEAD was `17c2a83`, not the snapshot's `65a6d4e`). The owner pasted the reply
into this session. The figures that decide the plan:

| | NCSU-Hackathon | RT-SuperConductor |
| --- | --- | --- |
| Next GPU work | Cycle 3: about 7 to 8 hours with 2 GPU jobs at once, about 11 hours with 1 (their forecast); nightly runs only if the owner registers them | One TabICLv2 run, about 10 to 30 minutes ([Speculative]) |
| GPU busy, alone | about 19% with one job (measured mean); about 66% with two (derived) | not measured |
| Peak GPU memory | 1,735 MiB for two jobs plus the desktop (measured); cycle 3's wider search unknown, under 4 GB per job ([Speculative]) | at most 0.77 GB per past job; TabICLv2 unknown |
| On out of memory | retries the same command once; the job can be lost, but its results never change | crystal-graph jobs retry with half the batch, which changes results |
| Heat guard | ends a GPU job after about 30 s at 92 °C or more; measured peak 63 °C | none |
| Python | must keep torch 2.15.0+cu132, numpy, scikit-learn and pandas as they are | needs `tabicl` added |

**Decision: run the two one after the other, not at once** [my
recommendation; the owner decides].
- RT-SuperConductor's only GPU work now is one short run. Sharing would cost
  the other pipeline far more than it saves here:
  - at 1 GPU slot, cycle 3 takes about 3 to 4 hours longer;
  - at 2 slots, the unknown TabICLv2 memory could push the card past 8 GB.
    The other side's jobs would then fail and be retried, and could be lost.
  - Waiting for the other pipeline costs this project about 30 minutes.
- So run TabICLv2 when no NCSU-Hackathon job is on the GPU: before cycle 3,
  or after it.
- **RT-SuperConductor gets its own Python environment**,
  `C:\Users\hudso\RT-SuperConductor\.venv`.
  - `tabicl` 2.2.0 asks only for `torch>=2.2` and `scikit-learn>=1.3.0`
    (checked on PyPI today). So installing it into `.venv-gpu` would probably
    not move the pinned packages ([Likely]).
  - A separate environment removes the coupling both ways for the cost of one
    download.
- The TabICLv2 run logs `nvidia-smi` every 5 seconds. That measures this
  project's GPU use and memory, the missing figures above.
- The shared lock file the other agent offered is not needed for one short
  run. Revisit it if this project ever gets a long or recurring GPU job.
- Commands: `docs/training-methods-review.md`, "Update, 2026-11-25".
