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
| Did the run itself go wrong? | No. Every step ended normally, the Wannier functions are compact (1.7 to 3.3 Å², 7.5 Å² for the s-like one), and EPW found 0.63 states/eV per spin at the Fermi level |
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

### Phonons compared with measurement, 2026-11-12

Step 1 of the list above is done. Source: S. Y. Savrasov and D. Y. Savrasov, "Electron-phonon interactions and related physical properties of metals from linear-response theory" ([arXiv:cond-mat/9604124](https://arxiv.org/abs/cond-mat/9604124), 1996), Table I, whose measured values are the neutron data compiled in Landolt-Börnstein, New Series III/13a (1981). Frequencies in THz; for the N point the two upper modes are both 4.35 THz in the measurement, so our modes are matched by size.

| Point and mode | Measured | Full run | Smoke run | Savrasov's calculation |
| --- | --- | --- | --- | --- |
| H | 5.03 | 4.75 (−5.5%) | 5.43 (+8.0%) | 5.13 (+2.0%) |
| N, T1 | 2.63 | 2.49 (−5.3%) | 2.94 (+11.8%) | 2.65 (+0.8%) |
| N, L and T2 (lower) | 4.35 | 3.79 (−12.9%) | 4.24 (−2.6%) | 4.18 (−3.9%) |
| N, L and T2 (upper) | 4.35 | 4.25 (−2.3%) | 4.52 (+4.0%) | 4.54 (+4.4%) |
| Average size of the miss | | 6.5% | 6.6% | 2.8% |

The same paper gives the electron-phonon numbers to compare with: λ = 0.78 from tunnelling experiments, λ = 0.86 and ωlog = 160 K from its own calculation, and measured Tc = 4.47 K (Tables II and III). The JARVIS reference (λ = 0.757, ωlog = 165 K) agrees with these, so **the reference is sound; the error is ours.**

What this changes in the diagnosis above:

1. **The full run's phonons are too soft, but only by about 6%.** The smoke run's are too stiff by about the same amount, so the true k-mesh answer lies between 8×8×8 and 16×16×16 and the convergence tests in step 2 are still needed. About a third of the softness is expected from the functional: the PBE lattice here is 3.320 Å against about 3.301 Å measured, a 1.7% larger volume, and phonons soften as a crystal expands **[Likely]**.
2. **6% softer phonons cannot explain λ being 91% too high.** Since λ goes as 1/ω², a uniform 6.5% softening raises λ by about 14%.
3. **The main problem is where the coupling sits, not the frequencies.** The plain log-average of the full run's 5,184 phonon frequencies is 147 K, close to the 160 K that a correct calculation gives for ωlog. Weighted by coupling, as ωlog is, it drops to 109 K: the run gives far too much coupling to low-frequency modes. Near Γ, 3% of the modes carry 12% of λ. The earlier statement that "the open problem is the phonon side" was only partly right.
4. **Most likely cause [Likely]: the interpolation from the coarse grids.** EPW computes the coupling on an 8×8×8 k-grid and a 4×4×4 q-grid and interpolates from there. EPW's own output advises `use_ws = .true.` ("Results may improve by using use_ws == .TRUE."), and the 244 negative linewidths point the same way.

Revised order of tests:

1. Rerun only EPW with `use_ws = .true.`, reusing the saved coupling files (`epbread = .true.`, `epbwrite = .false.`). About 25 minutes on 4 cores; no new phonons needed.
2. Increase the coarse grids, q to 6×6×6 and k to 12×12×12. This needs new phonons and is a Hazel job.
3. The phonon k-mesh and smearing convergence (step 2 of the earlier list), for the 6% frequency error.

### `use_ws` test, 2026-11-12: no improvement

Test 1 above was run on the 4-core rehearsal machine: EPW's last step only, from the saved coupling files, with `use_ws = .true.` and every other setting unchanged. It finished normally in about 25 minutes, and EPW no longer printed its "Results may improve" advice.

| Quantity | Without `use_ws` | With `use_ws` | Reference |
| --- | --- | --- | --- |
| λ | 1.447 | 1.487 | 0.757 |
| ωlog | 108.6 K | 105.3 K | 164.99 K |
| Tc at μ* = 0.1 | 11.9 K | 11.9 K | 6.9 K |
| Coupling-weighted log-average frequency | 109 K | 105 K | |
| Share of λ from modes near Γ (within 0.2 of the zone centre) | 11.7% | 17.6% | |

**The interpolation setting is not the cause.** If anything, more coupling moved to modes near Γ. The phonon frequencies are unchanged, as expected (147 K log-average in both).

What remains, in order of how likely it now looks:

1. **The coarse grids are too coarse [Likely].** The coupling is computed directly only on an 8×8×8 k-grid and a 4×4×4 q-grid and interpolated everywhere else. The lead smoke run, also on deliberately coarse grids, overestimated λ too (2.35). The test is test 2 above: q 6×6×6, k 12×12×12. On the rehearsal machine that is roughly 3 hours of computing (most of it new phonons); on Hazel with 16 cores, well under an hour.
2. **The electronic interpolation [Speculative].** One of the six Wannier functions is fairly spread out (7.5 Å²). Comparing Wannier-interpolated bands with directly calculated bands near the Fermi level is a 10-minute check.
3. **The pseudopotential [Speculative].** It reproduces the lattice constant and gives phonons 6% soft, which is ordinary, so it is the least likely cause.

### Band check, 2026-11-13: the electronic interpolation was a real cause

EPW computes everything on a coarse grid and fills in the rest by interpolating through Wannier functions, so the interpolated bands must match the true bands, especially near the Fermi level where the coupling is decided. They were checked against bands calculated directly with Quantum ESPRESSO at 308 k-points (a path through Γ, H, N and P plus 120 random points), none of them on the coarse grid.

| Wannier setup | Mean error within 0.4 eV of E_F | 95th percentile | Worst | DOS at E_F against direct calculation |
| --- | --- | --- | --- | --- |
| 6 functions (d, s), 8×8×8 grid: the failed run | 137 meV | 371 meV | 1,094 meV | +4% (0.1 eV smearing), +11% (0.05 eV) |
| 6 functions, 12×12×12 grid | 93 meV | 268 meV | 931 meV | +7%, +10% |
| **9 functions (d, s, p), 8×8×8 grid** | **36 meV** | **104 meV** | **149 meV** | **−1%, −1%** |

On the grid points themselves all setups are exact, as they should be. The six-function setup was too poor between them: its errors were larger than EPW's 0.1 eV smearing, so the wrong states counted as being on the Fermi surface. A denser grid helped only partly, and its s-like function spread to 15 Å². Adding the three p-like functions made the d-like ones compact (1.2 to 1.3 Å² against 1.7 to 3.4 Å² before).

EPW was then rerun with nine functions on the same phonons, everything else unchanged:

| Quantity | 6 functions | 9 functions | Reference |
| --- | --- | --- | --- |
| λ | 1.447 | **1.122** | 0.757 |
| ωlog | 108.6 K | **127.1 K** | 164.99 K |
| Tc at μ* = 0.1 | 11.9 K | 10.4 K | 6.9 K |
| Share of λ near Γ | 11.7% | 6.0% | |
| Negative, unphysical linewidths | 522 | 224 | |

**This is a real improvement but not a pass.** λ is still 48% too high and ωlog 23% too low. The package in `data/benchmarks/epw-ta-jvasp-1014/` now uses the nine-function setup (18 bands in the band step), exactly as tested here.

What is left, with a rough account of the remaining 48%:

1. **Phonons 6% too soft:** worth about +14% in λ, and they are soft at the coarse-grid points where nothing is interpolated.
2. **The 4×4×4 q-grid:** likely most of the rest **[Likely]**. The coupling-weighted frequency average (127 K) is still well below the plain one (147 K), so too much coupling still sits on soft modes. The test is a 6×6×6 q-grid with a 12×12×12 k-grid, best done on Hazel.

The band check took about 10 minutes on 4 cores and needs no phonons, so it is worth running on every new material before trusting its λ.

### Running the band check (added 2026-11-14)

`scripts/check_wannier_bands.py` does the check above for any finished EPW run folder (the one holding `scf.in`, `nscf.in`, `epw.in`, `<prefix>.save` and `<prefix>_hr.dat`). Three steps:

```bash
python scripts/check_wannier_bands.py prepare --epw-dir <run_dir>/epw --out <run_dir>/bandcheck
```

```bash
sbatch --partition=compute --qos=normal --ntasks=4 <run_dir>/bandcheck/run.sh
```

```bash
python scripts/check_wannier_bands.py analyze --epw-dir <run_dir>/epw --check-dir <run_dir>/bandcheck
```

- **prepare** copies only the charge density and pseudopotentials (a few MB). It writes a direct calculation at 300 random k-points plus 12 coarse-grid points, and a 24×24×24 calculation for the density of states.
- **run.sh** is two short `pw.x` runs, about a minute on 4 cores for tantalum. It is a calculation, so submit it as a job, not on the login node. Put `LAUNCHER="mpirun -np 4"` in front if the MPI check in section 5 said so. Off Hazel, run it with `LAUNCHER="mpirun -np 4" bash <run_dir>/bandcheck/run.sh`.
- **analyze** prints a verdict, writes `report.json` in the check folder, and exits 0 for pass, 1 for fail, 2 when files are missing.

**What it checks.** It takes every directly calculated state within EPW's own window around E_F (`fsthick`) and measures how far the nearest interpolated state is. It passes when:

- the mean error is at most 50 meV;
- 95% of errors are at most EPW's smearing (`degaussw`);
- the density of states at E_F agrees within 5% at two smearing widths.

It also checks that the coarse-grid points come out exact, which catches a `_hr.dat` from a different calculation. The thresholds come from one material and are a starting point, not a standard; each one is an option.

**Results on tantalum, with the script:**

| Setup | Mean error | 95th percentile | DOS ratio (0.1 / 0.05 eV) | Verdict |
| --- | --- | --- | --- | --- |
| 6 functions (the failed run) | 193 meV | 802 meV | 1.04 / 1.11 | fail |
| 9 functions (the package now) | 38 meV | 110 meV | 0.98 / 0.96 | fail, narrowly |

Both coarse-grid self-checks were exact (0.0 meV). The numbers differ a little from the hand check above because the script uses a different set of random points. The nine-function setup misses only the 95th-percentile limit (110 against 100 meV). Read it as "much better, still not clean at the 8×8×8 grid", which matches λ still being 48% high. The 12×12×12 grid planned for Hazel should be checked the same way.

The updated package was also run end to end in smoke mode with the chain script (7 minutes on 4 cores, exit 0). Smoke mode now gives λ = 1.04 and ωlog = 144 K. That it no longer lands near the reference, as the six-function smoke run did (0.778), confirms the earlier agreement was a coincidence of errors.

### 6×6×6 result, 2026-11-14: the formula was wrong, the grid mattered little, and the benchmark was held to the lowest published number

**Main idea:** with one input line added, this pipeline now gives λ = 1.08, ωlog = 140 K and Tc = 10.9 K for tantalum. The most careful comparable published calculation gives 0.96, 143 K and 11.2 K. The earlier "failures" came from three things: a formula that should not have been used, a reference at the low end of the published range, and Wannier functions that were too few (fixed on 2026-11-13).

**What was run.** The package inputs with a 6×6×6 phonon grid and a 12×12×12 electron grid (16 phonon points instead of 8, about 3.5 hours on 4 cores), nine Wannier functions.

**1. The formula.** The EPW input copied `temps = 0.075` from the lead tutorial but not `delta_approx = .true.`. Without that line EPW weights each electron pair by a difference of Fermi functions at 0.075 K, which on a finite grid is a difference of step functions and is extremely sensitive to errors of a few meV in the interpolated bands. Points that the crystal's symmetry maps onto each other must give the same coupling. In the 4×4×4 run they did not:

| Formula | Points with negative coupling | Disagreement inside a symmetry set (median) | Phonon frequencies inside a set |
| --- | --- | --- | --- |
| Without `delta_approx` (all runs before today) | 45 of 1,728, lowest −2.8 | 145% | identical |
| With `delta_approx = .true.` | none | 16% (four sets checked) | identical |

A negative coupling is not physically possible. `delta_approx = .true.` puts both electrons on the Fermi surface, which is the standard definition of λ and the one the published values use. It is now in the package's `epw.in`.

**2. The result, and what the grid did.**

| Run (double-delta, nine Wannier functions) | λ | ωlog | Tc at μ* = 0.10 |
| --- | --- | --- | --- |
| 4×4×4 phonon grid, 8×8×8 electron grid | 1.165 | 131.3 K | 11.3 K |
| 6×6×6 phonon grid, 12×12×12 electron grid | 1.079 | 139.9 K | 10.9 K |

The finer grid moved λ by 7%. The estimate above that the 4×4×4 grid was "likely most of the rest" was wrong.

**3. What to compare with.** Published values for tantalum:

| Source | Method | λ | ωlog | Tc calculated |
| --- | --- | --- | --- | --- |
| JARVIS-SuperconDB JVASP-1014 (arXiv:2205.00060) | high-throughput, coarse grids | 0.757 | 165 K | |
| Savrasov and Savrasov 1996 (arXiv:cond-mat/9604124) | LDA linear response | 0.86 | 160 K | 5.1 K with μ* fitted to 0.17 |
| Cerqueira et al. 2023 (arXiv:2307.10728) | PBEsol, double-delta, dense grids | 0.961 | 143.3 K | 11.2 K at μ* = 0.10 |
| Experiment | tunnelling; specific heat | 0.78; 0.83 | | 4.47 K measured |
| **This pipeline, 6×6×6** | PBE, double-delta | **1.079** | **139.9 K** | **10.9 K at μ* = 0.10** |

Against the 2023 calculation, which uses the same formula, this result is 12% high in λ, 2% low in ωlog and 3% low in Tc. Against JARVIS it is 42% high in λ. A calculated Tc near 11 K for a metal that superconducts at 4.5 K is normal for this kind of calculation with μ* = 0.10; every calculation in the table overshoots unless μ* is raised.

**4. Which half is off.** λ can be written as η / (M⟨ω²⟩): an electronic part η divided by the mass and the squared phonon frequency. `scripts/epw_diagnostics.py moments` reports η = 8.21 eV/Å² for this run, against about 8.6 for the 1996 calculation. So the electrons and their coupling agree with the literature to about 4%, and what differs between the published calculations is the phonon frequency average (143 to 165 K). This pipeline's phonons are 5 to 13% below the neutron measurements at the two points where data exist:

| Point | Calculated (THz) | Measured (THz) | Difference |
| --- | --- | --- | --- |
| H | 4.75 | 5.03 | −5.5% |
| N | 2.49 / 3.79 / 4.25 | 2.63 / 4.35 / 4.35 | −5.3% / −12.9% / −2.3% |

**5. A 24 times faster last step.** The final EPW loop ran over all 1,728 points of the fine phonon grid, but only 72 of them are different by symmetry. `mp_mesh_q = .true.` computes just those: 2 minutes instead of about 2 hours on 4 cores for the 4×4×4 data, 8 minutes for the 6×6×6 data. The per-point values are identical to the full run's. Because the interpolated coupling is not perfectly symmetric (the 16% above), the short cut is an estimate: in four symmetry sets the point it uses differed from the set's average by −2.3% to +2.9%. Use it for convergence tests; a full-grid check is in progress.

**Three new checks, none needing a new calculation** (`scripts/epw_diagnostics.py`):

```bash
python scripts/epw_diagnostics.py moments --a2f <run_dir>/epw/ta.a2f.01.0.075 --mass-amu 180.94788 --compare 0.961 143.3
```

```bash
python scripts/epw_diagnostics.py symmetry --epw-out <run_dir>/epw/epw.out --xml <run_dir>/epw/ta.xml
```

```bash
python scripts/epw_diagnostics.py stars --xml <run_dir>/epw/ta.xml --grid 12 12 12 --point 0 0 0.0833333333 --point 0.25 0.3333333333 0.0833333333 --out star.qf
```

- **moments** prints λ, the frequency averages and η, and with `--compare` an estimate of a published calculation's η for comparison.
- **symmetry** groups the phonon points of an EPW output into symmetry sets and reports the disagreement inside each. It exits 1 if any coupling is negative, if frequencies differ inside a set, or if the median disagreement is above 5%. It needs an output that contains whole sets: a full-grid run, or a run on a file written by **stars**.
- **stars** writes such a file for a few chosen points (set `filqf = 'star.qf'` in `epw.in`, remove the `nqf` lines). The check then takes a minute or two.

**What this means for Stage 1.** The one-number pass rule against JARVIS (0.757) should be replaced. A fair rule compares with the calculation that uses the same method (0.96 and 143 K) and accepts the published range. Whether 12% above that counts as a pass is the owner's decision; the remaining 12% is most likely the softer PBE phonons plus the 16% symmetry noise. Run the symmetry check on every new material before trusting its λ.

### Stage 1 pass rule, proposed 2026-11-14, written before the niobium result was read

**A correction to the comparison above.** Cerqueira et al.'s 11.2 K for
tantalum does not come from the formula this pipeline uses. Their λ = 0.961
and ωlog = 143.3 K put into this pipeline's Allen-Dynes formula at μ* = 0.10
give 9.4 K. Like for like, this pipeline's 10.9 K is 16% high, not 3% low;
that matches the +12% in λ. Their niobium and vanadium Tc values (17.0 and
24.9 K) are also above what the same formula gives from their own λ and ωlog
(13.8 and 20.6 K). So compare λ and ωlog, not Tc.

**Proposed rule (the owner decides; written down now so that it cannot be
fitted to the niobium result):**

1. Judge λ and ωlog, not Tc. Report Tc at μ* = 0.10 and 0.15 as information.
2. The reference for every element benchmark is the published calculation
   that uses the same double-delta definition on dense grids: Cerqueira et al.
   2023 (arXiv:2307.10728). Tantalum: λ 0.961, ωlog 143.3 K. **Niobium: λ
   1.248, ωlog 147.1 K.** Chosen before the niobium run finished, because
   published niobium values spread by about 30% (other calculations near
   0.90 to 0.97, and experimental estimates of 1.04 to 1.22, both as reported
   by the literature search and not checked here).
3. Pass only if all four hold: λ within ±15% of the reference (niobium: 1.061
   to 1.435), ωlog within ±15% (niobium: 125.0 to 169.2 K), the band check
   passes at its default limits (`scripts/check_wannier_bands.py`: mean error
   near E_F at most 50 meV, 95th percentile at most the smearing, density of
   states at E_F within 5%), and the symmetry check passes at its default
   limits (`scripts/epw_diagnostics.py symmetry`: no negative coupling, equal
   frequencies within a symmetry set, median disagreement within a set at
   most 5%).

**Tantalum under this rule, today: not yet a pass.** λ (+12%) and ωlog (−2%)
are inside. But the nine-function band check above has a 95th percentile of
104 meV against a 100 meV limit, and the 4×4×4 symmetry check gave a 16%
median disagreement against 5%. Both checks are close calls or depend on the
grid, and their 6×6×6 versions are queued.

**Open question for the owner:** are the two checks' default limits the right
ones? If they are loosened, decide the new limits before reading the niobium
and 6×6×6 checks, not after.

### Niobium on a 4×4×4 phonon grid, 2026-11-14: fails the rule, because of the phonon interpolation

Run on 4 cores with the tantalum package's settings adapted to niobium
(`celldm(1) = 6.2568` bohr, the PBE zero-pressure lattice at 90 Ry; 90 Ry
cutoff; nine Wannier functions; windows set from niobium's own bands, Fermi
level 17.96 eV; `delta_approx` and `mp_mesh_q` on). Phonons took 74 minutes,
EPW 8.

| Check (rule above) | Result | Limit | Verdict |
| --- | --- | --- | --- |
| λ | **3.35** | 1.061 to 1.435 (reference 1.248) | fail |
| ωlog | **28 K** | 125.0 to 169.2 K (reference 147.1 K) | fail |
| Band check, near E_F | mean 12.9 meV, 95th percentile 36.8 meV, worst 49.8 meV | 50 / 100 meV | pass |
| Band check, DOS at E_F | +1.0% at 0.1 eV smearing, +6.2% at 0.05 eV | 5% | fail (0.05 eV only) |
| Symmetry check | not run on this grid | | |

**Why.** The electrons are interpolated well, better than tantalum's (12.9
against 36 meV). The phonons are not. On the fine grid, 5 of the 72
symmetry-distinct phonon points have an imaginary mode, all near Γ, and two
points, q = (1/12, 1/12, 1/6) and (1/6, 1/6, 1/6) in crystal coordinates,
have acoustic modes at 0.7 and 1.1 meV with λ = 107 and 111. Since λ goes as
1/ω², those two points, 1.7% of the weight, carry 1.87 of the 3.35. Without
them λ would be 1.50. Real transverse acoustic modes there sit at several meV,
so this is the Fourier interpolation of the 4×4×4 force constants failing,
not niobium. Tantalum on the same grid had no imaginary mode.
[Likely] Niobium's force constants reach further than tantalum's (its phonons
have Kohn anomalies), so a 4×4×4 grid is too coarse for it.

**Next, already running:** the same calculation on a 6×6×6 phonon grid with a
12×12×12 electron grid. The point (1/6, 1/6, 1/6) is on that grid, so its
phonon is computed directly instead of interpolated. The rule above applies
unchanged.

### Tantalum 6×6×6: smearing and symmetry, 2026-11-14

Two checks on the 6×6×6 data, run after the rule above was written:

| Check | Result | Under the rule |
| --- | --- | --- |
| Electron smearing halved (`degaussw` 0.1 → 0.05 eV), same symmetry-reduced fine grid | λ 1.124 (+4.2% against 1.079), ωlog 139.2 K (−0.5%) | λ is +17% against the reference 0.961: outside ±15% |
| Symmetry check on four whole stars (90 points) | median disagreement inside a star 6.2%, largest 6.9%; no negative coupling; frequencies identical inside every star | fail, narrowly (limit 5%); it was 16% on the 4×4×4 grid |

- The denser phonon grid made the coupling much more symmetric (16% → 6%),
  so most of the earlier disagreement was the 4×4×4 interpolation.
- λ rises as the smearing falls. With the package's 0.1 eV, tantalum is inside
  the λ band (+12%); with 0.05 eV it is not (+17%). The rule does not say which
  smearing counts. [Likely] The converged value is at or above 1.12, so on λ
  this pipeline is about 15 to 20% above the reference calculation, which is
  what the softer PBE phonons (5 to 13% below neutron data) would predict.
- Not run, for lack of time on this machine: the phonon probes at H (lattice
  constant and electron mesh) and the full-grid check of `mp_mesh_q`. The
  cores went to niobium on a 6×6×6 grid instead.

### Niobium 6×6×6, before the λ result, 2026-11-14 11:40 UTC

Written while the phonons were at 8 of 16 q-points and before EPW had started,
so the prediction below can be checked against the result.

The 6×6×6 run computes directly the point that broke the 4×4×4 run.
Cartesian q = (0, 0, 1/3) (2π/a) is crystal (1/6, 1/6, 1/6):

| q (Cartesian, 2π/a) | Lowest modes, cm⁻¹ (6×6×6, computed) | 4×4×4 at the same point |
| --- | --- | --- |
| (0, 0, 1/3) = crystal (1/6, 1/6, 1/6) | 31.6, 31.6, 151.0 (3.9 meV) | 1.1 meV, interpolated, λ = 111 |
| (0, −1/6, 1/6) = crystal (0, 0, 1/6) | 28.7, 55.4, 95.8 | not on the grid |
| (0, −1/3, 1/3) | 109.3, 114.1, 142.6 | not on the grid |
| (0, 1/2, −1/2) = N | 130.7, 166.9, 171.1 | 130.7, 166.9, 171.1 (computed on both) |

- [Certain] None of the 8 computed q-points has an imaginary mode, and N agrees
  with the 4×4×4 run to 0.1 cm⁻¹. The 1.1 meV mode was the interpolation: DFT
  gives 3.9 meV there, so that point's coupling drops by about (3.9/1.1)² ≈ 13.
- [Certain] The transverse acoustic branch is still soft at small q. Read as
  sound speeds, the two lowest modes give ρv² = 12.4 GPa and 7.5 GPa. By
  q = (0, −1/3, 1/3) the branch is back to 45 GPa.
- [Likely, from memory, not looked up tonight] Niobium's measured C44 is about
  29 to 31 GPa. If so, DFT with this smearing (0.02 Ry Marzari–Vanderbilt,
  16×16×16 k-points) puts these modes at roughly half the measured frequency.
  Their coupling would then be about four times too large, since λ goes as 1/ω².

**Prediction** [Speculative], made before the result:

- λ below 1.061: 5%
- λ inside the pass band (1.061 to 1.435): 30%
- λ from 1.435 to 2.0: 45%
- λ above 2.0: 20%
- ωlog below 125 K: 50%

The most likely verdict is a fail on λ or ωlog. That would trace to the DFT
phonons near Γ, which points at the phonon smearing and k-grid, not the
q-grid.

### Niobium 6×6×6 result, 2026-11-14 14:05 UTC: fails the rule

Phonons took 3 h 15 min on 4 cores, EPW 63 min. Most of the EPW time (50
min) went to the coarse-grid matrix elements, because both the coarse q-grid
(4³ → 6³) and the coarse k-grid (8³ → 12³) grew. Fine grids as in the
package: k 24×24×24, q 12×12×12 with `mp_mesh_q`, `degaussw` 0.1 eV.

| Check (rule above) | Result | Limit | Verdict |
| --- | --- | --- | --- |
| λ | **1.767** (+42%) | 1.061 to 1.435 (reference 1.248) | fail |
| ωlog | **120.7 K** (−18%) | 125.0 to 169.2 K (reference 147.1 K) | fail |
| Band check | not run: λ and ωlog already decide the verdict | | |
| Symmetry check (96 points in 5 whole stars) | median spread inside a star 14%, largest 23%; no negative coupling; frequencies identical inside every star | median ≤ 5% | fail |
| Tc, this pipeline's formula (McMillan form, no f1·f2; information only) | 15.9 K at μ* = 0.10, 13.9 K at 0.15 (18.6 and 15.8 K with f1·f2); measured 9.25 K | | |

**Against the prediction.** λ fell in the 1.435 to 2.0 bin (given 45%, the
most likely bin) and ωlog below 125 K (given 50%). The predicted verdict, a
fail, was right. One prediction says little about how well calibrated the
odds were. The prediction also blamed the DFT phonons near Γ; the checks
below do not confirm that.

All numbers below were recomputed from the raw output by a second,
independent reader, who also argued against the first draft of this
section. The draft overstated what η shows and listed too few causes; this
version takes in those corrections.

**What the run shows.**

- [Certain] The 4×4×4 artifact is gone. No fine-grid point has an imaginary
  mode (72 symmetry-distinct points), and λ fell from 3.35 to 1.77, ωlog rose
  from 28 to 121 K.
- [Certain] 38% of this run's λ (0.67) comes from modes below 10 meV, which
  are 12% of the q-weighted modes; 22% (0.40) comes from modes below 6 meV
  (4.5%). Five of the six largest contributions are small-q points whose
  lowest mode is at 1.7 to 5.5 meV (the sixth is at 8.9 meV), for example
  crystal (0, 0, 1/12), λ_q = 9.5 at 1.7 meV, and (1/12, 1/12, 1/6), λ_q =
  6.5 at 3.8 meV.
- [Certain] The symmetry check fails worst nearest Γ: the spread inside a
  star is 23% at crystal (0, 0, 1/12), 16% at (1/12, 1/12, 1/6), 14% at
  (1/12, 1/12, 1/12), then 10% at (0, 1/12, 1/3) and 5% at (1/6, 1/6, 1/6).
  The frequencies agree exactly, so the phonons are symmetric and the
  coupling is not. [Likely] At small q the double delta picks out a thin set
  of electron pairs, which a 24×24×24 fine k-grid with 0.1 eV smearing
  samples unevenly. That is 10 to 20% noise on the points that carry the
  most coupling; it does not say which way the converged λ moves.
- [Certain] The electron density of states at E_F is 0.742 states per spin
  per eV, [Likely, from memory] a normal value for niobium, so the
  normalisation of λ is not suspect.

**Where the excess sits is not settled.**

- [Certain, arithmetic on this run's modes] Stiffening only the modes below
  10 meV cannot reproduce the reference pair: reaching λ = 1.248 that way
  needs a factor of 2.1 and pushes ωlog to 179 K, outside the band. A factor
  of 1.45 to 1.7 would put both λ (1.42 to 1.33) and ωlog (157 to 168 K)
  inside the band. Stiffening every mode by 1.2 instead gives λ = 1.227 and
  ωlog = 145 K, almost exactly the reference.
- [Speculative] So the data fit a lattice that is softer than the
  reference's everywhere as well as a problem confined to small q. The
  reference used PBEsol, which gives a stiffer lattice than PBE (used here);
  the same difference was the leading explanation for tantalum's +12%.
- η = λ M⟨ω²⟩ does not settle it. This run's 7.66 eV/Å² is 5% below the
  8.02 estimated for the reference, but that estimate borrows this run's
  own ratio ω₂/ωlog = 1.34, which the soft modes inflate. With the soft modes
  stiffened by 1.2 or 1.4, the ratio falls to 1.23 or 1.16 and the estimate
  to 6.7 or 6.0, which would make this run 14 to 27% high. η is also blind to
  the soft modes, frequencies and matrix elements alike (modes below 10 meV
  give 7% of it), so it cannot say which of the two is off there.
- The measured C44 (about 29 to 31 GPa, from memory) shows the small-q
  modes are softer than experiment (ρv² = 7.5 and 12.4 GPa at the two
  smallest computed q-points), not that they are softer than the reference
  calculation.
- Tc is not compared with Bercx et al. They report Migdal–Eliashberg Tc, not
  this formula, and even the broken 4×4×4 run gives 5.3 K, so "it still
  says superconductor" cannot tell a good run from a broken one.

**Candidate causes, and the test for each.** Only the first has been run.

1. **Fine q sampling.** [Likely] small. The whole first shell around Γ adds
   0.066 to λ, and every point with |q| ≤ 0.24 (2π/a) adds 0.20, while λ
   must fall by 0.33 to reach the band. Test: fine q 18×18×18 on the same
   data. **The second reader's prediction, written before the result:** λ
   moves by a few hundredths, probably upward.
   **Result (read 14:35 UTC, after the stop rule below was pushed):** 195
   symmetry-distinct points, no imaginary mode, λ = 1.710 (−0.057, −3.2%),
   ωlog = 120.5 K (unchanged), Tc 15.4 / 13.4 K at μ* = 0.10 / 0.15. Still a
   fail on both. The size of the change matched the prediction; the
   direction did not. [Likely] Fine q sampling is not the cause: 38% of λ
   still comes from modes below 10 meV.
2. **Electron fine grid.** 24×24×24 k with 0.1 eV smearing has never been
   tested on niobium. [Speculative, from memory of niobium's Fermi velocity]
   one k-step is 0.2 to 0.45 eV near E_F, two to four times the smearing.
   Tantalum's λ rose 4% when the smearing was halved. Test: nkf 36³ or 48³
   with `degaussw` 0.05 eV on the same data (no new phonons).
3. **Phonon scf k-grid and smearing** (16³, 0.02 Ry Marzari–Vanderbilt).
   Change both together, for example 24³ to 32³ at 0.01 Ry. The computed
   TA[00ζ] branch dips (3.9 meV at ζ = 1/3, 20.2 meV at 2/3, 25.7 meV at H),
   a Fermi-surface shape that smearing smooths out, so less smearing may
   soften these modes further and raise λ. This test can move either way.
4. **Functional.** PBE here against PBEsol in the reference. Test: lattice
   constant and phonons with a PBEsol pseudopotential. That is a change of
   method, not a convergence check.
5. **Coarse phonon grid.** 6³ against 8³ has not been checked. Going from
   4³ to 6³ halved λ.

**Stop rule, proposed before the 18×18×18 result was read** (the owner
decides; written now so that it cannot be fitted to the next results):

- "Converged grids" in the roadmap means: fine q 18³ or denser, fine k 36³
  or denser with `degaussw` 0.05 eV, and 6×6×6 phonons recomputed with an
  scf k-grid of 24³ or denser at 0.01 Ry smearing.
- If λ is still outside 1.061 to 1.435 or ωlog outside 125.0 to 169.2 K on
  those grids, the niobium benchmark counts as failed and the roadmap's stop
  condition is met: no candidate runs with this setup, and candidates are
  judged from published calculations instead (for example Bercx et al.,
  Materials Cloud 2025.39).
- Switching to PBEsol counts as a new method, which must pass tantalum and
  niobium again from the start.
- Candidates stay blocked until that result is in, not merely until a test
  has been run.

**What this means for Stage 1.** Not passed. Tantalum misses narrowly (band
check 104 against 100 meV; symmetry 6.2% against 5%). Niobium misses clearly
on λ and ωlog, and on the symmetry check.

**Open question for the owner:** do you accept the stop rule above, before
any of its tests is read?

### Stop rule accepted; PBEsol test for niobium, plan and prediction, 2026-11-14 18:01 UTC

**The owner accepted the stop rule above on 2026-11-14** (in the session,
after PR #41 was merged) and asked for the PBEsol test. The rule now applies
as written.

**Why this test.** Stiffening every niobium mode by 1.2 reproduces the
reference pair, and the reference used PBEsol where this pipeline uses PBE.
If PBEsol alone stiffens niobium's phonons enough, the miss is the method,
not the grids.

**How.**
- The package's PBE file cannot be regenerated with PBEsol here. Its
  generator, ONCVPSP, is not packaged on conda-forge, and building it from
  source was not allowed on this machine.
- Instead, Quantum ESPRESSO's own generator (`ld1.x` 7.6) makes two PAW
  datasets from pslibrary's niobium recipe, `Nb.$fct-spn-kjpaw_psl.0.3.0`.
  That is the version pslibrary does not list as broken with PBEsol; its
  1.0.0 niobium is listed as not working with PBEsol.
- The two generator inputs differ only in `dft` ('PBE' or 'PBESOL').
- Both runs use the same settings: 60 Ry and 480 Ry cutoffs (the datasets
  suggest at least 42 and 364), 16×16×16 k-points, Marzari–Vanderbilt
  smearing 0.02 Ry, `tr2_ph` 1e-14. Each runs at its own zero-pressure
  lattice constant.
- Phonons at three points (Cartesian, 2π/a): H = (0, 0, 1), (0, 0, 1/3) and
  (0, −1/6, 1/6).
- The PAW-PBE frequencies are also compared with the package's ONCV-PBE
  frequencies at the same points, to see how much the pseudopotential
  itself moves them.

**Decision rule, written before any result.**
- **Measure:** the PBEsol/PBE frequency ratio for each of the nine modes
  (three per point), both PAW. The summary is their geometric mean.
- **Threshold, from this run's own numbers:** stiffening every mode by 9.2%
  takes λ from 1.710 to 1.435, the band edge, and ωlog to 132 K.
- **Mean ratio 1.09 or more:** PBEsol may fix the miss. Run the full PBEsol
  benchmark (6×6×6 chain). Under the stop rule that is a new method, so
  tantalum and niobium must both pass from the start on converged grids.
- **1.04 to 1.09:** PBEsol explains part of the miss. The stop rule's
  remaining grid tests come first.
- **Below 1.04:** the functional is not the explanation, and the remaining
  stop-rule tests decide.
- **Caveat:** this measures phonons only. PBEsol also changes the coupling
  matrix elements, which these three phonons do not show.
- **Pseudopotential check:** if PAW-PBE and ONCV-PBE frequencies differ by
  more than 3% on average, the pseudopotential matters about as much as the
  functional, and that is reported.

**Prediction** [Speculative], made before any result:
- PBEsol lattice constant 1.0 to 1.5% below PBE (3.311 Å here with ONCV;
  measured 3.300 Å).
- Mean PBEsol/PBE frequency ratio below 1.04: 35%; 1.04 to 1.09: 45%; 1.09
  or more: 20%.
- The small-q transverse modes move more, in proportion, than the modes at H.

### PBEsol test for niobium: result, 2026-11-14 18:17 UTC

**PBEsol makes the soft modes worse, not better.** It gives imaginary
transverse acoustic modes, an instability of the lattice, at both small-q
points.

| q (Cartesian, 2π/a) | ONCV-PBE (package, cm⁻¹) | PAW-PBE (cm⁻¹) | PAW-PBEsol (cm⁻¹) | PBEsol / PBE |
| --- | --- | --- | --- | --- |
| H (0, 0, 1) | 207.5 ×3 | 208.2 ×3 | 216.5 ×3 | 1.040 |
| (0, 0, 1/3) | 31.6, 31.6, 151.0 | 31.0, 31.0, 151.4 | **−33.4, −33.4** (imaginary), 151.4 | imaginary, imaginary, 1.000 |
| (0, −1/6, 1/6) | 28.7, 55.4, 95.8 | 28.3, 55.2, 96.1 | **−15.6** (imaginary), 54.5, 90.1 | imaginary, 0.986, 0.938 |

- Zero-pressure lattice constants: PAW-PBE 6.2536 bohr (3.309 Å), ONCV-PBE
  6.2568 bohr (3.311 Å), PAW-PBEsol 6.1824 bohr (3.272 Å). PBEsol is 1.14%
  below PBE, inside the predicted 1.0 to 1.5%. The measured value is
  3.300 Å.
- Cutoff check: raising the cutoffs from 60/480 to 80/640 Ry moves the PBE
  pressure by 1 kbar.
- **[Certain] Pseudopotential check passes.** PAW-PBE and ONCV-PBE agree to
  within 2% at every mode; the geometric mean ratio is 0.996.
- **[Certain] Decision under the rule: "below 1.04".** The pre-registered
  geometric mean cannot be taken, because three of the nine PBEsol modes are
  imaginary. Over the six real modes it is 1.006. Only H stiffens, by 4.0%.
  The modes that carry niobium's excess coupling went the wrong way. So the
  functional is not the explanation, and the stop rule's remaining grid
  tests decide.
- **Against the prediction.** The lattice constant came out as predicted.
  The frequency ratio fell in "below 1.04" (given 35%). The prediction that
  the small-q modes move more than H was right in size but wrong in sign.
- [Speculative] This fits what is known of niobium under compression:
  PBEsol's 1.1% smaller lattice is like PBE under a few GPa. The transverse
  modes near Γ in niobium depend on the Fermi surface, and smearing and the
  k-grid move them strongly. If so, Cerqueira et al.'s PBEsol λ = 1.248 must
  rest on settings that keep these modes stable, most likely a larger
  smearing. That would make the reference itself depend on the smearing.

**Follow-up at (0, 0, 1/3), launched 18:18 UTC; prediction written before
reading it** [Speculative]:
- PBE at the PBEsol lattice constant: TA imaginary or below 15 cm⁻¹ (volume
  alone does it): 65%.
- PBEsol at the PBE lattice constant: TA real and within 10% of PBE: 60%.
- PBEsol with a 24×24×24 k-grid: TA still imaginary: 55%.
- PBEsol with 0.04 Ry smearing instead of 0.02: TA real: 70%.

### PBEsol follow-up: the imaginary mode is a k-grid artifact, 2026-11-14 18:32 UTC

All at q = (0, 0, 1/3), PAW, 60/480 Ry, Marzari–Vanderbilt smearing:

| Run | Lattice (bohr) | k-grid | Smearing (Ry) | Pressure (kbar) | Modes (cm⁻¹) |
| --- | --- | --- | --- | --- | --- |
| PBE (Stage A) | 6.2536 | 16³ | 0.02 | 0 | 31.0, 31.0, 151.4 |
| PBEsol (Stage A) | 6.1824 | 16³ | 0.02 | 0 | −33.4, −33.4, 151.4 |
| PBE at the PBEsol lattice | 6.1824 | 16³ | 0.02 | +63 | −25.4, −25.4, 155.3 |
| PBEsol at the PBE lattice | 6.2536 | 16³ | 0.02 | −60 | 22.0, 22.0, 147.5 |
| PBEsol, denser k | 6.1824 | **24³** | 0.02 | 0 | **41.2, 41.2, 162.4** |
| PBEsol, larger smearing | 6.1824 | 16³ | **0.04** | −1 | **47.4, 47.4, 163.1** |

- **[Certain] On the package's settings (16³ k, 0.02 Ry) these modes are
  not converged.** Going to 24³ moves PBEsol's transverse mode from −33.4 to
  +41.2 cm⁻¹, and even the longitudinal mode moves 7%.
- [Certain] At 16³, both a smaller volume (PBE compressed by 1.1%) and the
  functional at fixed volume (PBEsol at the PBE lattice: 22.0 against 31.0)
  soften the mode. At 16³, though, both effects are smaller than the
  k-grid's.
- **So the Stage A verdict ("below 1.04") stands as recorded, but its premise
  does not.** The comparison was made on phonon settings that are not
  converged for the modes that matter. It says nothing reliable about PBEsol
  until both functionals are compared on converged k-grids.
- **The package's PBE phonons have the same problem.** That makes cause 3
  (phonon scf k-grid and smearing) the leading suspect for niobium's excess
  λ. It is also the stop rule's test (b).
- Against the follow-up predictions:
  - PBE at the PBEsol lattice, imaginary: right (given 65%).
  - PBEsol at the PBE lattice, within 10% of PBE: wrong (given 60%); it is
    29% lower.
  - PBEsol at 24³ still imaginary: wrong (given 55%).
  - 0.04 Ry gives a real mode: right (given 70%).

**Next, launched 18:35 UTC: convergence of the PBE mode at (0, 0, 1/3)**,
at PBE's own lattice. The runs are k 24³ and 32³ at 0.02 Ry and at 0.01 Ry
(the stop rule's setting), plus PBEsol at 32³ and 0.01 Ry. Prediction,
written before the result [Speculative]:
- PBE at 24³ and 0.02 Ry: transverse mode at least 10% above the package's
  31.0 cm⁻¹: 65%.
- At 0.01 Ry, 32³ differs from 24³ by more than 10%: 50%.
- The best-converged PBE transverse mode is at least 1.3 times the package
  value (41 cm⁻¹ or more): 40%. If it is, a full 6×6×6 phonon rerun at
  that k-grid (about 11 h on 4 cores at 24³, so better on Hazel) is the next
  step, and the arithmetic above says λ could drop into the band.

### Corrections to the three PBEsol sections above, 2026-11-14 18:45 UTC

A second, independent reader recomputed every number in the three sections
above from the raw files and found all of them correct. It also confirmed
from the commit times that each prediction was written before its result.
It found the following faults of wording and reasoning. The sections are
kept as committed, because their timestamps are what make the predictions
checkable. These notes take precedence over them.

- **Heading "the imaginary mode is a k-grid artifact": too strong.** The
  data show the mode is not converged at 16³ k and 0.02 Ry. One denser
  grid (24³, real at 41.2 cm⁻¹) does not show that the converged mode is
  real. 24³ is also the only one of 16³, 24³ and 32³ that puts k+q for this
  q on the k-grid, so density and commensurability are mixed. [Likely] real
  at 0.02 Ry; unknown at 0.01 Ry.
- **"[Certain] … So the functional is not the explanation": only the bin is
  certain.** The bin "below 1.04" holds however the imaginary modes are
  treated, and its action (do not run the PBEsol chain; the stop rule's
  tests decide) stands. That the functional is not the explanation holds only
  at 16³ k and 0.02 Ry. This weakness was foreseeable: the plan used 16³ and
  0.02 Ry although the accepted stop rule already defined converged phonons
  as 24³ or denser at 0.01 Ry.
- **Most of PBEsol's softening at 16³ is the smaller lattice.** In ω² at
  (0, 0, 1/3), the volume effect is −1602 cm⁻², the functional at fixed
  volume −476, and the two add almost exactly to the observed change
  (−1119).
- **Stage A is not a reproduction of the reference.** It tests PBEsol with
  this PAW set, at its own relaxed lattice and the package's phonon
  settings. The reference's code, pseudopotentials, smearing, grids and
  lattice (relaxed or measured) are not recorded in this repository. The
  roadmap's readers did not read the paper, so even its use of PBEsol is
  second-hand (the runbook's tantalum table).
- **The [Speculative] paragraph on niobium under compression and the
  reference's smearing is superseded.** A denser k-grid at the same 0.02 Ry
  also stabilises the mode, so "most likely a larger smearing" no longer
  follows. "Like PBE under a few GPa" was +63 kbar (6.3 GPa). No source in
  the repository supports "what is known of niobium under compression".
- **"The package's PBE phonons have the same problem" is [Likely], not
  established.** It is inferred from the PBEsol runs. PBE at 16³ is real
  (31.0 PAW, 31.6 ONCV), and no PBE run on a denser grid had been read when
  this was written. "Leading suspect" assumes a direction that no data at
  0.01 Ry support: at 16³, raising the smearing from 0.02 to 0.04 Ry stiffened
  the mode more than 16³ → 24³ did, so halving it to 0.01 Ry should soften the
  mode and could undo the k-grid's stiffening.
- **"The stop rule's test (b)"** is defined nowhere. It means the stop rule's
  phonon requirement: 6×6×6 phonons at scf k 24³ or denser and 0.01 Ry.
- **The convergence plan, as actually run** (launched 18:33 UTC, not 18:35):
  PBE 24³ at 0.02 and 0.01 Ry, PBE 32³ at 0.01 Ry, PBEsol 32³ at 0.01 Ry.
  There is no 32³ run at 0.02 Ry. The baseline is the Stage A PAW-PBE value
  31.0 cm⁻¹, not "the package's" (ONCV, 31.6), so 1.3 times it is 40.2 cm⁻¹,
  not 41.
- **"λ could drop into the band" at 1.3×: wrong arithmetic.** Stiffening
  only the modes below 10 meV needs about 1.45× to reach λ = 1.435;
  1.3× gives about 1.445 to 1.49.
- **What the PBEsol 32³ run decides, written before it is read: nothing.**
  It is for information only. Under the stop rule, PBEsol is a new method
  that must pass tantalum and niobium from the start.
- **The next step does not depend on one mode's threshold.** The stop rule
  needs λ on 6×6×6 phonons at 24³ or denser and 0.01 Ry for any verdict, pass
  or fail. That rerun is required whatever the single-q runs show. They
  only choose its k-grid. It takes about 11 h on 4 cores at 24³ and about
  26 h at 32³, so it belongs on Hazel.
- **Cheaper required tests were missing from the plan.**
  - The stop rule's electron fine-grid test (fine k 36³ or denser at
    `degaussw` 0.05 eV, fine q 18³) needs no new phonons. It reuses the
    existing EPW data and takes about an hour here.
  - The second soft point, (0, −1/6, 1/6), is not in the convergence batch.
  - Both are queued after the batch, along with 32³ at 0.02 Ry.

### Queued 18:44 UTC: extra phonon checks and the electron fine-grid test; prediction

Queued after the convergence batch:
- PBE 32³ at 0.02 Ry at (0, 0, 1/3);
- PBE 24³ at 0.01 and 0.02 Ry at (0, −1/6, 1/6);
- the stop rule's electron fine-grid test on the existing ONCV-PBE 6×6×6 EPW
  data: fine k 36³, `degaussw` 0.05 eV, fine q 18³ with `mp_mesh_q`. That is
  a component test only: its phonons are still the 16³-scf ones.

First convergence result, read at 18:44 UTC: PBE 24³ at 0.02 Ry gives 44.3,
44.3 and 156.1 cm⁻¹ at (0, 0, 1/3), against 31.0, 31.0 and 151.4 at 16³.

Prediction for the fine-grid EPW test [Speculative], written before it
starts:
- λ stays above 1.435 (the phonons are unchanged): 80%.
- λ rises from 1.710 (the smaller smearing resolves more of the small-q
  coupling, as for tantalum's +4%): 60%.

### No Hazel; Cerqueira et al.'s methods; the stop rule's rerun runs here, 2026-11-14 19:30 UTC

**Correction: there is no Hazel account.** The owner confirmed on
2026-11-14 that the project has no access to Hazel. "Better on Hazel" and
"it belongs on Hazel" in the sections above assumed access that does not
exist; the assumption came from this runbook's name and its Slurm scripts,
and was never checked. Everything below runs on the project's 4-core cloud
machine. NC State HPC accounts and NSF ACCESS allocations usually need a
faculty sponsor [Likely; check with NC State's HPC office].

**PBE convergence at (0, 0, 1/3), all runs read.** PAW-PBE at its own
lattice; transverse (twice) and longitudinal modes, cm⁻¹:

| k-grid | Smearing (Ry) | Modes |
| --- | --- | --- |
| 16³ (package) | 0.02 | 31.0, 31.0, 151.4 |
| 24³ | 0.02 | 44.3, 44.3, 156.1 |
| 24³ | 0.01 | 41.4, 41.4, 157.1 |
| 32³ | 0.01 | 43.6, 43.6, 157.0 |

- [Certain] At the stop rule's smearing (0.01 Ry), 24³ is within 5% of 32³.
  The transverse mode at 32³ is 1.41 times the package's 16³ value.
- Against the predictions written at 18:32:
  - 24³ at least 10% above 16³: right (given 65%).
  - At 0.01 Ry, 32³ differs from 24³ by more than 10%: wrong (given 50%);
    it differs by 5%.
  - The best-converged mode at least 1.3 times the package value: right
    (given 40%).
- The first 32³ run crashed when the disk filled. It was rerun with
  `reduce_io`.

**Cerqueira et al.'s methods, read 2026-11-14** (arXiv:2307.10728v1,
section VI and Table I; the published *Advanced Materials* version was not
read):
- [Certain, quoted]
  - PBEsol, PseudoDojo "stringent" scalar-relativistic norm-conserving
    pseudopotentials, cutoff at PseudoDojo's high-precision hint.
  - Quantum ESPRESSO 6.8 and 7.1. The coupling comes from ph.x's own
    double-grid method, not EPW.
  - Lattice relaxed with PBEsol at 1500 k-points per reciprocal atom.
  - Coarse coupling k-grid: the relaxation grid; fine k-grid: four times
    denser in each direction; q-grid: half the k-grid.
  - Double-delta integration with Methfessel–Paxton smearing of 0.05 Ry
    (0.68 eV; this package's EPW uses 0.1 eV).
  - "Accurate" settings: 3000 k-points per reciprocal atom, coarse grid
    doubled.
  - Up to three imaginary frequencies at Γ below 35i cm⁻¹ are accepted.
    "Spurious soft modes due to insufficient k-point sampling" are
    acknowledged.
- [Certain] Table I, the source of niobium's λ = 1.248 and ωlog = 147.1 K,
  uses the accurate settings. Its Tc column is the isotropic Eliashberg
  solution at μ* = 0.10, not Allen–Dynes. It lists bcc niobium as 29 meV/atom
  above the convex hull, which the paper does not explain.
- [Speculative] For one-atom bcc niobium, 3000 k-points per atom is about
  14³, so the accurate coarse grid is about 28³. That is roughly twice as
  dense per direction as this package's 16³, and in the range where the
  transverse mode above has stiffened by 1.4 times.
- What this means:
  - The reference rests on denser phonon grids than this package's, so it
    stays a fair target.
  - It is still one high-throughput setting, not a convergence study, so a
    pass within ±15% is evidence, not proof.
  - [Likely] Much of the gap (λ 1.71 against 1.248) is this package's 16³
    phonon grid, not the functional.
  - The double-delta smearing differs by a factor of about 7, a difference
    in method that can move λ either way.

**The stop rule's rerun, queued 19:27 UTC to start after the electron
fine-grid test:**
- Phonons: 6×6×6 at scf k 24³ and 0.01 Ry.
- EPW ground state: scf on the same 24³ grid at 0.01 Ry (it was 12³ at
  0.02 Ry), so it matches the phonons.
- EPW fine grids: k 36³, `degaussw` 0.05 eV, q 18³ with `mp_mesh_q`.
- Everything else as in the package.
- Expected: about 11 to 14 h of phonons and 2 h of EPW on 4 cores.
- To make room, the 16³ run's wavefunction, coupling and dvscf files are
  deleted after the fine-grid test; its outputs, dynamical matrices and a2F
  files are kept. So the 16³ data cannot be re-run through EPW afterwards.

**Prediction for the rerun** [Speculative], written before it starts:
- λ inside 1.061 to 1.435: 45%; above 1.435: 45%; below 1.061: 10%.
- ωlog at least 125.0 K: 70%.
- λ and ωlog both inside: 40%. All four checks pass (adding band and
  symmetry): 25%.
- Under the accepted stop rule, a miss on these grids means niobium has
  failed and candidates are judged from published calculations.

### The electron fine-grid test failed (out of memory); the 16³ data were deleted by a script fault, 2026-11-14 19:40 UTC

- **The test did not run.** EPW with fine k 36³, fine q 18³ and 4 processes
  was killed by the kernel's out-of-memory handler about a minute in, after
  setting up the 195 q-points. Each process held about 3.6 GB against about
  15 GB for the whole machine. There is no result, and no prediction can be
  scored.
- **Script fault.** The queue then deleted the 16³ run's wavefunction,
  coupling and dvscf files, as planned, but it did so unconditionally. It
  should have deleted them only after the test succeeded. So this test can no
  longer be run on the 16³ data. The stop rule's fine-grid requirement is
  instead covered inside the 24³ rerun, whose EPW step uses the same fine
  grids.
- **Fix for the rerun.** The rerun's EPW step will hit the same limit with 4
  processes. It writes the Wannier-space coupling file before the fine-grid
  stage, so a follow-up queue redoes only that stage from the saved file
  with 2 processes (about 10 GB, about twice as slow) if the chain's EPW step
  fails.
- **Last convergence point**, PBE 32³ at 0.02 Ry, (0, 0, 1/3): 44.3, 44.3,
  155.9 cm⁻¹, the same as 24³ at 0.02 Ry to 0.1%. So 24³ is converged at
  0.02 Ry and within 5% at 0.01 Ry, which supports the rerun's choice of
  24³.
- The rerun started at 19:35 UTC; the phonon scf took 32 s.

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
