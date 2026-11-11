# Hazel runbook: running QE/EPW for RT-SuperConductor

**Main idea:** Hazel is used submit-and-wait. Set up and stage everything on the login node, submit small Slurm jobs that need no internet, copy results back to the repository, review them, then promote them into training. This runbook records what NC State's Research Facilitation and HPC staff explained on 2026-11-06, checked against the official HPC pages on 2026-11-09 ([Running Jobs](https://hpc.ncsu.edu/RunningJobs/RunningJobs.php), [Partitions and QOS](https://hpc.ncsu.edu/RunningJobs/Resources.php), [Connecting](https://hpc.ncsu.edu/GettingStarted/Connecting.php), [Acceptable Use Policy](https://hpc.ncsu.edu/About/AUP.php)). Access to the `libstudents` project was granted on 2026-11-09. Items marked *confirm* are still unverified, and the staff are still working out the Quantum ESPRESSO install and data storage.

## 1. The rules that shape everything

| Rule | What it means for us |
| --- | --- |
| Hazel uses **Slurm** | Use `scripts/hpc/job.slurm.sh.template`. The LSF template is obsolete. |
| Login nodes are shared | Install, compile and stage there. Never run calculations there. |
| Compute nodes have **no internet** | Inputs, pseudopotentials and structures must be in the run directory before `sbatch`. `scripts/hpc/stage_inputs.sh` checks this. |
| Put `--partition` and `--qos` together | Staff found QoS alone sometimes fails. Every partition has a default QOS, but naming both costs nothing. |
| Partner hardware gives way to its owners | Jobs over about 2 hours on scavenger QoS can be preempted (scavenger is a pilot, available on request). Keep QE steps under 2 hours, or use QE restarts (`max_seconds`, `restart_mode`). |
| Don't run Claude Code on Hazel | Staff: agents must not run on login nodes. The Acceptable Use Policy also says HPC-VCL nodes are for GUI applications only, although a staff demo showed the VCL image used as a personal shell. Until HPC support says in writing that agents are fine there, work with Claude on your own machine and use Hazel only to run jobs. |
| Request what the job needs | The policy asks for accurate resource requests, even if that means a longer wait. Use `seff <jobid>` after a job to see what it really used. |
| Repeated violations disable the account | Running real computation on a login node is the usual one. Installing software and submitting jobs there is fine. |

## 2. Storage

| Area | Size | Kept? | Use it for |
| --- | --- | --- | --- |
| `$HOME` | 15 GB, plus a file-count limit | Yes | Nothing large; never conda environments |
| User app space (`/usr/local/usrapps`) | 100 GB default, on GPFS; Derek is requesting it for the project | Yes | The conda environment with QE, EPW and Wannier90 |
| Work area (`/share/...`) | 10 TB default | **Wiped after ~30 days untouched; not backed up** | Every run directory. The policy says files here may be deleted at any time and the only copy of anything important must not live here. It also forbids techniques that protect files from the purge, so never write a script or workflow that touches files to keep them alive |
| Group share | About 20 TB shared by the group (*confirm*) | *Confirm* | Shared datasets later |
| Research storage | Allocated separately | Backed up | Long-term results, if Derek adds us |

Plan for about 1 TB now and up to 10 TB later. A result is safe only once it is copied into `results/<id>/` in this repository.

## 3. First login (10 minutes, no software yet)

Do this before anything else; if an account problem exists, finding it now leaves time for a ticket.

1. Log in. Either open the Open OnDemand link from Derek's email (the HPC pages list two hostnames, `servood` and `servood2`; use the one the email links to) and choose Clusters, then Hazel Shell Access, or from a terminal:
   ```bash
   ssh <unity_id>@login.hpc.ncsu.edu
   ```
2. Learn your real names. These are Hazel's own helper commands:
   ```bash
   sa
   ```
   ```bash
   sqos
   ```
   ```bash
   si
   ```
   `sa` shows the account, partitions and default QOS you hold; `sqos` shows the QOS you may use and their limits. Write the account, partition and QOS down; they replace the placeholders in the job templates. If `sa` shows nothing, open a ticket with help@ncsu.edu.
3. Look at storage: `echo $HOME`, `df -h $HOME`, and `ls /share` and `ls /usr/local/usrapps`.

Partitions on the official page: `compute` (default QOS `normal`, up to 4 days; `long` up to 10), `gpu` (QOS `gpu`), `compute_partners` and `gpu_partners` (idle partner nodes, QOS `short` for jobs under 2 hours at low priority), and `xfer` (data staging, 4 cores and 24 GB). We need only `compute` for now.

## 4. One-time setup (after the first login works)

1. Open a terminal on the login node.
2. Install Miniforge in user app space, not `$HOME`. Point conda's package cache there too (`conda config --add pkgs_dirs <app_space>/conda-pkgs`).
3. Create the environment. Derek found Quantum ESPRESSO installable through conda. Check whether `epw.x` and `wannier90.x` come with it (*confirm*):
   ```bash
   conda create -n rtsc-qe -c conda-forge qe
   conda activate rtsc-qe
   command -v pw.x ph.x epw.x wannier90.x
   ```
   If `epw.x` is missing, ask HPC support how existing QE users build EPW.
4. Create the work directory: `mkdir -p /share/<group>/<unity_id>/rtsc`.

## 5. First job: a hello world (5 minutes, 1 core)

This follows Derek's advice to test the smallest possible task first. Quantum ESPRESSO is not installed yet, so the first job only proves that submitting and collecting output works. Save this as `hello.sbatch` in your work area. The account defaults to your group, so add `#SBATCH --account=<name>` only if `sa` shows you must name one:

```bash
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=hello.%j.out
#SBATCH --error=hello.%j.err
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00

hostname
date
```

```bash
sbatch hello.sbatch
```

```bash
squeue -u $USER
```

When the job finishes, `hello.<jobid>.out` holds the compute node's name. That is the whole test. After QE is installed, repeat with `pw.x -h && command -v epw.x` in place of the two commands above.

## 6. Stage 0: the official EPW Pb example

Everything Stage 0 needs is in the repository: `data/benchmarks/epw-pb-tutorial/`. It holds the Quantum ESPRESSO `EPW/examples/pb` inputs and pseudopotential, copied unchanged with checksums in `PROVENANCE.md`. Nothing has to be downloaded on Hazel.

1. On the login node, activate the QE/EPW environment, then submit the whole chain:
   ```bash
   scripts/hpc/submit_stage0_pb.sh /share/<group>/<unity_id>/rtsc/pb-tutorial compute normal 8
   ```
   The script:
   - copies the package into the work area
   - runs `stage_inputs.sh` on both directories
   - submits six dependent jobs: `phonons/scf` → `ph` → `pp.py` → `epw/scf` → `nscf` → `epw`

   Add `DRY_RUN=1` in front to print the `sbatch` commands without submitting.
2. Watch with `squeue -u $USER`. Logs go to `<run_dir>/logs/`.
3. Copy `woSOC/epw/epw.out` and `pb-fcc.cif` to `results/pb-tutorial/` in the repository. Copy `review.template.json` to `results/pb-tutorial/review.json` and fill it in.
4. Run `python scripts/promote_dft_results.py --results-root results`.

**Stage 0 passes** when the report shows `tools_checks_passing: 1`. That means λ and ωlog were read from the real `epw.out`, and Tc recomputed from them matches the Tc EPW printed. The parser already handles the EPW Pb output format: it was tested on Quantum ESPRESSO's own Pb test (`test-suite/epw_metal`), where it reproduces EPW's Tc of 0.8067 K to within 0.03%.

The example's settings are EPW's tutorial settings, not converged production settings. Stage 0 tests the tools, not the physics.

## 7. Stage 1: tantalum benchmark

Prepare inputs with `scripts/prepare_tantalum_epw_benchmark.py`, adapting the EPW settings from the Pb tutorial. After the run, write `results/ta-jvasp-1014/review.json`:

```json
{
  "role": "benchmark",
  "formula": "Ta",
  "converged": true,
  "reviewer": "Hudson Clavin",
  "mu_star": 0.1,
  "reference": {"lambda": 0.7572, "omega_log_k": 164.99, "source": "JARVIS-SuperconDB JVASP-1014"},
  "notes": "cutoffs, k/q meshes, smearing and EPW windows recorded in logs/run-manifest.*.txt"
}
```

Set `converged` to `true` only after the convergence checks in `docs/epw-benchmark.md`. The promotion report then compares the result with JARVIS. A miss bigger than about 2 times means stop and fix the setup before Nb3Sn.

## 8. Candidates: closing the loop

A candidate's `review.json` uses `"role": "candidate"` and `"accept_for_training": true`. Then:

```bash
python scripts/ingest_dft_results.py --results-root results --output data/processed/dft-results/result-manifest.csv
python scripts/promote_dft_results.py --results-root results
python scripts/train_physics_targets.py --extra-records data/training/hazel-epw-records.json
```

The retrained models now include the new calculation. See `docs/pipeline-loop.md` for the full loop and its remaining gap.

## 9. Later: Globus Compute

Globus Compute lets a program outside Hazel, such as a GitHub Actions workflow, submit work to Hazel and collect the results. In outline:

- You run a small **endpoint** process under your account on Hazel. It is configured to turn incoming tasks into Slurm jobs.
- Elsewhere, Python code using the Globus Compute SDK sends a function or shell command to that endpoint, authenticated with a Globus client ID and secret. In GitHub these would be stored as repository secrets.
- The endpoint queues the work through Slurm, and the result comes back to the caller.

Points to settle before relying on it:

- Whether NC State lets a user keep an endpoint running on Hazel, and where. Ask Research Facilitation.
- Returned results are meant to be small. Large QE outputs should stay on Hazel or move by Globus Transfer, with only λ, ωlog and hashes sent back **[Likely]**.
- Staff said the documentation is thin and chatbots give poor advice on it. Get one manual run working first, then automate with their help.
