# Hazel runbook: running QE/EPW for RT-SuperConductor

**Main idea:** Hazel is used submit-and-wait. Set up and stage everything on the login node, submit small Slurm jobs that need no internet, copy results back to the repository, review them, then promote them into training. This runbook records what NC State's Research Facilitation and HPC staff explained on 2026-11-06. Items marked *confirm* are still unverified.

## 1. The rules that shape everything

| Rule | What it means for us |
| --- | --- |
| Hazel uses **Slurm** | Use `scripts/hpc/job.slurm.sh.template`. The LSF template is obsolete. |
| Login nodes are shared | Install, compile and stage there. Never run calculations there. |
| Compute nodes have **no internet** | Inputs, pseudopotentials and structures must be in the run directory before `sbatch`. `scripts/hpc/stage_inputs.sh` checks this. |
| Put `--partition` and `--qos` together | Staff found QoS alone sometimes fails. The HPC docs list which QoS goes with which partition. |
| Partner hardware gives way to its owners | Jobs over about 2 hours on scavenger QoS can be preempted. Keep QE steps under 2 hours, or use QE restarts (`max_seconds`, `restart_mode`). |
| Agents are allowed in the right place | Don't run Claude Code on the shared login node. The HPC VCL image works as a personal login node where agents are fine. |

## 2. Storage

| Area | Size | Kept? | Use it for |
| --- | --- | --- | --- |
| `$HOME` | 15 GB, plus a file-count limit | Yes | Nothing large; never conda environments |
| User app space | 100 GB default, on GPFS | Yes | The conda environment with QE, EPW and Wannier90 |
| Work area (`/share/...`) | 10 TB default | **Wiped after ~30 days untouched; not backed up** | Every run directory |
| Group share | About 20 TB shared by the group (*confirm*) | *Confirm* | Shared datasets later |
| Research storage | Allocated separately | Backed up | Long-term results, if Derek adds us |

Plan for about 1 TB now and up to 10 TB later. A result is safe only once it is copied into `results/<id>/` in this repository.

## 3. One-time setup (after access is granted)

1. Log in through Open OnDemand in a browser, or SSH to the login node.
2. Install Miniforge in user app space, not `$HOME`. Point conda's package cache there too (`conda config --add pkgs_dirs <app_space>/conda-pkgs`).
3. Create the environment. Derek found Quantum ESPRESSO installable through conda. Check whether `epw.x` and `wannier90.x` come with it (*confirm*):
   ```bash
   conda create -n rtsc-qe -c conda-forge qe
   conda activate rtsc-qe
   command -v pw.x ph.x epw.x wannier90.x
   ```
   If `epw.x` is missing, ask HPC support how existing QE users build EPW.
4. Create the work directory: `mkdir -p /share/<group>/<unity_id>/rtsc`.

## 4. First job: prove the environment (10 minutes, 1 core)

This follows Derek's advice to test the smallest possible task first.

```bash
#SBATCH --partition=<partition>
#SBATCH --qos=<qos>
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
pw.x -h && command -v epw.x
```

## 5. Stage 0: EPW tutorial 04 (Pb)

1. On the login node, download the Pb inputs and pseudopotential from [EPW tutorial 04](https://docs.epw-code.org/tutorials/tutorial_04/index.html) into `/share/.../rtsc/pb-tutorial/`.
2. Check staging: `scripts/hpc/stage_inputs.sh /share/.../rtsc/pb-tutorial scf ph q2r matdyn nscf epw`
3. Fill in the Slurm template once per step and chain the steps: `sbatch --dependency=afterok:<previous_job_id> ...`
4. Copy outputs to `results/pb-tutorial/` in the repository.
5. Run `python scripts/promote_dft_results.py --results-root results`. Stage 0 succeeds when the report shows a parsed λ and ωlog. If it shows `lambda_or_omega_log_not_parsed`, fix the parser patterns using the real `epw.out` before Stage 1.

## 6. Stage 1: tantalum benchmark

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

## 7. Candidates: closing the loop

A candidate's `review.json` uses `"role": "candidate"` and `"accept_for_training": true`. Then:

```bash
python scripts/ingest_dft_results.py --results-root results --output data/processed/dft-results/result-manifest.csv
python scripts/promote_dft_results.py --results-root results
python scripts/train_physics_targets.py --extra-records data/training/hazel-epw-records.json
```

The retrained models now include the new calculation. See `docs/pipeline-loop.md` for the full loop and its remaining gap.

## 8. Later: Globus Compute

Globus Compute lets a program outside Hazel, such as a GitHub Actions workflow, submit work to Hazel and collect the results. In outline:

- You run a small **endpoint** process under your account on Hazel. It is configured to turn incoming tasks into Slurm jobs.
- Elsewhere, Python code using the Globus Compute SDK sends a function or shell command to that endpoint, authenticated with a Globus client ID and secret. In GitHub these would be stored as repository secrets.
- The endpoint queues the work through Slurm, and the result comes back to the caller.

Points to settle before relying on it:

- Whether NC State lets a user keep an endpoint running on Hazel, and where. Ask Research Facilitation.
- Returned results are meant to be small. Large QE outputs should stay on Hazel or move by Globus Transfer, with only λ, ωlog and hashes sent back **[Likely]**.
- Staff said the documentation is thin and chatbots give poor advice on it. Get one manual run working first, then automate with their help.
