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

**Checked on 2026-11-11: the conda package has everything.** The conda-forge package `qe` version 7.6 for Linux was downloaded and its file list read. It contains `pw.x`, `ph.x`, `q2r.x`, `matdyn.x`, `epw.x`, `wannier90.x` and `pw2wannier90.x`, so EPW does not have to be built separately. The same package was then installed and ran the Stage 0 chain end to end (section 6). Steps 2 and 3 above are now one command:

```bash
scripts/hpc/setup_qe_env.sh /usr/local/usrapps/<group>/<unity_id>
```

The script:

- refuses to install under `$HOME`
- uses `conda`, `mamba` or `micromamba` if one is already available, and otherwise downloads micromamba, a single file, into the application space. On Hazel, try `module avail conda` first (*confirm* the module name with the HPC pages)
- creates `<app_space>/envs/rtsc-qe` with QE pinned to 7.6 and keeps the package cache in application space
- lists the seven programs and stops if one is missing

It only installs; it runs no calculation, so it is fine on a login node. The environment and its package cache take about 1.4 GB. Add `DRY_RUN=1` in front to print what it would do. If Derek's team provides its own QE install instead, use theirs and skip this script; the job scripts only need the programs on `PATH`.

To use the environment in a shell or job:

```bash
export PATH=/usr/local/usrapps/<group>/<unity_id>/envs/rtsc-qe/bin:$PATH
```

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

### Second job: check how MPI programs start (2 tasks, 5 minutes)

This is the most likely place for the first real problem. A conda-installed MPI does not always work with a cluster's `srun`. When it does not, `srun` starts several separate one-process copies of the program instead of one parallel run. Nothing crashes; the job is just several times slower and the copies overwrite each other's output. Check before Stage 0, with the environment on `PATH`:

```bash
sbatch scripts/hpc/check_qe_mpi.sbatch
```

Read `check_qe_mpi.<jobid>.out`. It has one line for `srun` and one for `mpirun`:

| The line reads | Meaning | For Stage 0 |
| --- | --- | --- |
| `1 x Number of MPI processes: 2` | One parallel run on 2 processes: correct | Use this launcher |
| `2 x Number of MPI processes: 1` | Two separate copies: wrong | Do not use this launcher |
| `no MPI line printed` | The launcher failed | Do not use this launcher |

If `srun` is correct, run Stage 0 as written. If only `mpirun` is correct, put `LAUNCHER="mpirun -np 8"` in front of the Stage 0 command. If neither is, send the output file to help@ncsu.edu and ask how QE users start MPI jobs. The check was tried without a cluster on 2026-11-11: `mpirun` gave the correct line, and a deliberately broken `srun` gave the "two separate copies" line.

## 6. Stage 0: the official EPW Pb example

Everything Stage 0 needs is in the repository: `data/benchmarks/epw-pb-tutorial/`. (Since 2026-11-11 `submit_stage0_pb.sh` is a thin wrapper around `scripts/hpc/submit_epw_chain.sh`, which Stage 1 shares; its commands and switches are unchanged.) It holds the Quantum ESPRESSO `EPW/examples/pb` inputs and pseudopotential, copied unchanged with checksums in `PROVENANCE.md`. Nothing has to be downloaded on Hazel.

1. On the login node, put the QE/EPW environment on `PATH` (section 4), then submit the whole chain:
   ```bash
   scripts/hpc/submit_stage0_pb.sh /share/<group>/<unity_id>/rtsc/pb-tutorial compute normal 8
   ```
   The script:
   - copies the package into the work area
   - runs `stage_inputs.sh` on both directories
   - submits six dependent jobs: `phonons/scf` → `ph` → `pp.py` → `epw/scf` → `nscf` → `epw`

   Add `DRY_RUN=1` in front to print the `sbatch` commands without submitting.

   The script has two sizes, chosen with `STAGE0_MODE`:

   | Mode | What changes | Cost |
   | --- | --- | --- |
   | `smoke` (default) | The copies in the run directory get three coarser meshes: phonon k-mesh 8×8×8 instead of 14×14×14, q-mesh 3×3×3 instead of 6×6×6, and EPW's fine q-mesh 8×8×8 instead of 20×20×20. The changes are listed in `<run_dir>/STAGE0_MODE.txt` | About 20 minutes on 4 cores in the rehearsal below |
   | `full` | Nothing: the official example as shipped | Heavy. Its phonon step covers 16 q-points on a 14×14×14 k-mesh and its EPW step 64 million k-q pairs. See the rehearsal note below |

   Stage 0 checks the tools, not the physics, so `smoke` is enough to pass it. The vendored files in the repository are never edited.

   Other switches, all put in front of the command:

   | Switch | Use |
   | --- | --- |
   | `STAGE0_MODE=full` | Run the official example unchanged. Give it many cores, for example 32 as the last argument. The phonon job then gets a 24-hour limit, which `compute` with QOS `normal` allows |
   | `LAUNCHER="mpirun -np 8"` | Start the programs with `mpirun` instead of `srun`, if the MPI check in section 5 says so |
   | `START_AT=<step>` | Resume an existing run directory at `ph-scf`, `ph`, `pp`, `epw-scf`, `nscf` or `epw`, without recopying inputs. A failed late step then does not cost the phonons again |
   | `PH_TIME=04:00:00` | Change the phonon job's time limit |
   | `LOCAL=1` | Run the six steps one after another with `mpirun`, with no scheduler. For a personal machine only, never a Hazel login node |
2. Watch with `squeue -u $USER`. Logs go to `<run_dir>/logs/`.
3. Copy `woSOC/epw/epw.out` and `pb-fcc.cif` to `results/pb-tutorial/` in the repository. Copy `review.template.json` to `results/pb-tutorial/review.json` and fill it in.
4. Run `python scripts/promote_dft_results.py --results-root results`.

**Stage 0 passes** when the report shows `tools_checks_passing: 1`. That means λ and ωlog were read from the real `epw.out`, and Tc recomputed from them matches the Tc EPW printed. The parser already handles the EPW Pb output format: it was tested on Quantum ESPRESSO's own Pb test (`test-suite/epw_metal`), where it reproduces EPW's Tc of 0.8067 K to within 0.03%.

The example's settings are EPW's tutorial settings, not converged production settings. Stage 0 tests the tools, not the physics.

### Rehearsal on 2026-11-11: what it found

Stage 0 was run for real before any Hazel job, on a 4-core cloud machine with no scheduler (`LOCAL=1`), using the conda-forge QE 7.6 environment that `setup_qe_env.sh` creates (EPW 6.1). The machine was doing other work during the phonon step, so read the times as upper bounds.

| Step | Wall time, 4 cores, smoke mode |
| --- | --- |
| Phonon ground state (`pw.x`) | 7 s |
| Phonons (`ph.x`, 4 q-points) | 1,010 s |
| Gather phonon files (`pp.py`) | 1 s |
| EPW ground state and band step (`pw.x` twice) | 23 s |
| EPW (`epw.x`, including Wannier90) | 130 s |

These times are from the final run: one command, from an empty directory, 19.5 minutes in total, exit status 0. (An earlier attempt had stopped at `pp.py` and was resumed; the problems it exposed are in the table below.)

The result passed the check: `promote_dft_results.py` read λ = 2.345 and ωlog = 63.2 K from the real `epw.out`, recomputed Tc = 10.049 K, and EPW had printed 10.050 K. The report showed `tools_checks_passing: 1`. Those values are not lead's real ones (measured Tc is 7.2 K, and λ is about 1.55); the coarse meshes overestimate the coupling, which is expected and is why smoke mode is a tools check only.

The rehearsal found four problems that would otherwise have surfaced on Hazel. All are fixed in the scripts.

| Problem | What would have happened on Hazel | Fix |
| --- | --- | --- |
| The official example is much heavier than "tutorial" suggests. Its second q-point alone needs 4,060 k-points; on 4 cores the phonon step was heading for roughly a day | The phonon job would have been killed at the old 1 h 50 min limit and the chain would have stopped | `smoke` mode is the default; `full` gets a 24-hour phonon limit |
| `pp.py` imports `numpy`, which the QE environment did not include | The third job would have failed after the phonons had been paid for | `setup_qe_env.sh` installs and checks `numpy`; the submit script checks it before submitting anything |
| With EPW's fine k-mesh coarsened to 8×8×8, EPW found zero electronic states at the Fermi level. Every λ became a division by zero, yet the job ended normally and printed "Tc = 16 K" | A meaningless number that looks like a result | Smoke mode leaves the fine k-mesh at 20×20×20. `promote_dft_results.py` now rejects such output by name (`zero_dos_at_fermi_level`, `lambda_overflow`); the real output is a test fixture |
| A failed late step meant starting over | Phonons recomputed for nothing | `START_AT` resumes at any step |

Two things the rehearsal could not test, because it had no Slurm: whether `srun` starts the conda-installed MPI correctly (section 5 checks that), and real queue behaviour. A rough guide for `full` mode, extrapolated from one q-point and so uncertain by a factor of two or more: on the order of 100 core-hours for the phonons.

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

### Update 2026-11-11: the inputs exist and have been rehearsed

The first sentence of this section is superseded. Stage 1 no longer starts from the generated template: a complete, tested set of inputs is in `data/benchmarks/epw-ta-jvasp-1014/`, with the pseudopotential, the Wannier settings the old template lacked, and the reasons for each choice in its `PROVENANCE.md`. A filled-in `review.template.json` is in the same folder.

Run it in two passes, after Stage 0 has passed:

```bash
scripts/hpc/submit_stage1_ta.sh /share/<group>/<unity_id>/rtsc/ta-smoke compute normal 8
```

```bash
CHAIN_MODE=full scripts/hpc/submit_stage1_ta.sh /share/<group>/<unity_id>/rtsc/ta-jvasp-1014 compute normal 16
```

The first is the smoke pass, a setup check with coarse meshes. The second uses the benchmark settings, and only its result is compared with the reference. Every switch from section 6 (`LAUNCHER`, `START_AT`, `PH_TIME`, `EPW_TIME`, `DRY_RUN`) works the same way, because both stages now run through one shared script, `scripts/hpc/submit_epw_chain.sh`. A new material needs only a new package folder.

Then copy `epw/epw.out` and `ta-bcc.cif` into `results/ta-jvasp-1014/`, write `review.json` from the template, and run `python scripts/promote_dft_results.py --results-root results`.

**Smoke rehearsal, 4 cores, no scheduler, QE 7.6:**

| Step | Wall time |
| --- | --- |
| Phonon ground state | 4 s |
| Phonons (3 q-points) | 47 s |
| EPW ground state and band step | 35 s |
| EPW, including Wannier90 | 208 s |

| Quantity | Smoke result | Reference (JARVIS JVASP-1014) | Difference |
| --- | --- | --- | --- |
| λ | 0.778 | 0.7572 | +2.8% |
| ωlog | 159.4 K | 164.99 K | −3.4% |
| Tc at μ* = 0.1 | 7.05 K | 6.9 K from the reference λ and ωlog | |

`promote_dft_results.py` sorted this as a benchmark, recomputed EPW's printed Tc to 0.01%, and reported it inside the suggested 25% tolerance. Treat the close agreement with care: these are the coarse smoke meshes (2×2×2 phonon grid), so part of it is luck, and it does not replace the full-settings run. Tantalum's measured Tc is 4.47 K; the 7 K here comes from using μ* = 0.1, the same convention as the reference.

The six Wannier functions came out compact (spreads 1.7 to 3.4 Å² for the five d-like ones and 7.2 Å² for the s-like one) and EPW found 0.67 states/eV per spin at the Fermi level.

**One failure found and fixed.** The first attempt died in `epw.x` with `free(): invalid pointer` and no other message. The cause was one line in the band-step input, `nosym = .true.`, which the lead example does not have. A second pseudopotential, without a core correction, crashed the same way, which ruled the pseudopotential out. Removing the line fixed it, and the chain script now refuses a band step that contains it.

The benchmark settings were started on the same 4-core machine on 2026-11-11. Their phonon step is much heavier (8 q-points, up to a few thousand k-points each, about 10 minutes per q-point there). The result will be recorded here when it finishes; on Hazel with 16 cores expect well under an hour for the phonons.

### Full-settings result, 2026-11-11: the benchmark does not pass yet

The full-settings run finished (exit status 0; phonons 55 minutes, EPW 25 minutes on 4 cores) and **missed the reference badly**. The smoke-mode agreement above was luck.

| Quantity | Full settings | Smoke | Reference (JARVIS JVASP-1014) | Full vs reference |
| --- | --- | --- | --- | --- |
| λ | 1.447 | 0.778 | 0.7572 | +91% |
| ωlog | 108.6 K | 159.4 K | 164.99 K | −34% |
| Tc at μ* = 0.1 | 11.9 K | 7.05 K | 6.9 K | |

`promote_dft_results.py` reports `benchmarks_passing_suggested_tolerance: 0`. That is the stage gate working: **Stage 1 has not passed, and no candidate should be run until it does.** Measured values for tantalum are λ of about 0.7 to 0.8 and Tc = 4.47 K, so the full-settings number is the wrong one, not the reference.

What is known about the cause:

| Check | Finding |
| --- | --- |
| Did the run itself go wrong? | No. Every step ended normally, the Wannier functions are compact (1.7 to 3.3 Å², 7.5 Å² for the s-like one), and EPW found 0.70 states/eV per spin at the Fermi level |
| Is it EPW's fine sampling or smearing? | Unlikely. EPW's last step was rerun with half the smearing (0.05 eV) and the double-delta approximation. It was cut off by a machine restart after 784 of 1,728 q-points; on those same points the average coupling changed by only +6% (1.32 to 1.41). The 244 negative, unphysical linewidths in the first run did disappear |
| Is the coupling strength itself off? | Probably not. λ × ωlog², which is roughly independent of how soft the phonons are, is 17,100 K² here, 19,700 K² in smoke mode and 20,600 K² for the reference: within 20% of each other |
| Where the two runs differ | The phonons. At the same q-points the full run's frequencies are 17% lower than the smoke run's, and λ goes as 1/ω². About 21% of the full run's λ comes from modes below 8 meV, where small errors are amplified most |

So the open problem is the phonon side: the frequencies and how much weight the low-frequency modes get. The directly calculated frequency at the H point is 158.6 cm⁻¹ with the 16×16×16 k-mesh and 181.2 cm⁻¹ with 8×8×8, a 13% change from the k-mesh alone, so the phonon calculation is not converged with respect to k-points and smearing. Which of the two is closer to tantalum's measured value has not been checked against a source in this work.

What to do next, in order. These are convergence tests and need more cores than the rehearsal machine, so they are the first real use of Hazel after Stage 0:

1. Compare the calculated phonon frequencies at H and N with published neutron measurements for tantalum. This says whether the full run's phonons are too soft or the reference's are too stiff.
2. Repeat the phonon ground state and the H-point phonon with denser k-meshes (20×20×20, 24×24×24) and a second smearing width, until the frequency stops moving.
3. Increase the phonon q-mesh from 4×4×4 to 6×6×6. Interpolating phonons and coupling from 8 q-points may be too coarse for a metal with sharp features in its phonon spectrum.
4. Only then rerun `CHAIN_MODE=full` and compare again.

Until then, treat `CHAIN_MODE=full` for tantalum as a known-failing configuration, useful for these tests but not as a benchmark.

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
