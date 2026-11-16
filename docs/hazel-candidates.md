# Hazel candidate packages: phonons and electron-phonon coupling for rung 4

**Main idea:** each candidate is a folder of ready Quantum ESPRESSO inputs. One
command on Hazel submits six chained jobs and ends with `report.json`, which
says whether the compound has an imaginary phonon mode and what its calculated
Tc is. Run the benchmark (Zr6Ga2Ni) first: it has a published answer, so it
tests the method before the method is trusted on a compound nobody has made.

Written 2026-11-16 10:23 UTC, before any full-settings result exists. Nothing here
has run on Hazel. The chain was rehearsed on this session's machine only
(results at the end).

## What is in the package

| Folder in `data/benchmarks/candidates/` | Compound | Role | Structure | Atoms | Status |
| --- | --- | --- | --- | --- | --- |
| `zr6ga2ni` | Zr6Ga2Ni | Benchmark: Bercx et al. calculated λ = 0.70, isotropic Tc 5.5 K (`docs/bercx-published-check.md`) | P-62m | 9 | see "Rehearsal" |
| `re3ir` | Re3Ir | Rung 4 candidate, pool rank 19 | Pm-3m (Cu3Au type) | 4 | ready |
| `sbte2` | SbTe2 | Rung 4 candidate, pool rank 39 | P-3m1 (CdI2 type) | 3 | ready |

Each folder holds:

- `run/scf_fit.in`, `run/scf.in`, `run/ph.in`, `run/q2r.in`, `run/matdyn.in`:
  the five inputs, at full settings;
- `smoke-edits.tsv`: the coarse meshes a smoke run swaps in;
- `chain.conf`: what the job script copies and the phonon job's time limit;
- `package.json`: the relaxed cell, the meshes and a checksum for every input;
- `<name>.cif`: the relaxed structure, for viewing.

The pseudopotentials are in `data/benchmarks/candidates/pp/` (norm-conserving
ONCV, PBE, scalar-relativistic). They are in the repository because compute
nodes have no internet.

## What was left out, and why

The literature check (`docs/rung4-plan.md`) left six shortlisted compounds
that nobody was found to have made. Two are packaged. The other four:

| Compound | Why it is not packaged |
| --- | --- |
| La3Pb | No lanthanum pseudopotential in the set I could download. It can be added if a La file from the same family is supplied. |
| NaYbPb2, CaYbPb2 | Ytterbium's 4f electrons. A standard calculation of this kind is not reliable for them, and the physics proxy already failed on f-element compounds. |
| AcYbTl2 | Actinium is radioactive, so no lab would make it, and it has the same f-electron problem. |

## Method, and what it cannot tell you

- **Programs:** `pw.x`, `ph.x`, `q2r.x` and `matdyn.x` from Quantum ESPRESSO
  7.6, the version `scripts/hpc/setup_qe_env.sh` installs.
- **Route:** `ph.x` with `electron_phonon = 'interpolated'`, then `q2r.x` and
  `matdyn.x` with `la2F`. This uses no Wannier functions.
  - The niobium and tantalum benchmarks failed their band and symmetry checks
    in the Wannier step (`docs/hazel-runbook.md`, section 7). This route has
    no such step to tune.
  - [Likely] It is also less precise than a converged EPW run, because the
    coupling is interpolated from a coarse phonon grid.
- **Settings:** plane-wave cutoff 90 Ry, Marzari-Vanderbilt smearing 0.02 Ry,
  PBE. Each structure was relaxed here with the same settings, to a residual
  pressure under 0.1 kbar.
- **Limits.**
  - [Certain] This route has not been benchmarked in this project. That is
    what Zr6Ga2Ni is for.
  - [Certain] Spin-orbit coupling is left out. [Likely] It matters for SbTe2,
    because Sb and Te are heavy and Sb2Te3's bands depend on it.
  - [Certain] The calculation assumes the ordered structure in the table.
    [Likely] A lab melting 75% Re with 25% Ir would get a disordered hexagonal
    alloy, not ordered Re3Ir (see "Second read"). The calculation says
    nothing about that alloy.
  - Calculated Tc ran 1.1 to 2.8 times above measured in the Bercx comparison,
    so read it as an upper bound.

## Rules, written before any full-settings result

1. **Benchmark rule (Zr6Ga2Ni).** The method passes if λ at the 0.020 Ry
   broadening is within ±20% of Bercx et al.'s 0.70, that is 0.56 to 0.84, and
   no phonon mode is imaginary.
   - The band is wider than Stage 1's ±15% because Bercx et al. used PBEsol
     and different pseudopotentials.
   - If the benchmark fails, the candidates' numbers are not used. They are
     reported as "method not validated".
2. **Candidate rule (rung 4, step 5, unchanged).** A candidate goes forward
   only with no imaginary mode (nothing below −5 cm⁻¹) and an Allen-Dynes Tc
   of at least 2 K at μ* = 0.13, at the 0.020 Ry broadening.
3. **Smoke runs give no verdict.** Their meshes are too coarse.

**Predictions [Speculative].**

| Outcome | Chance |
| --- | --- |
| Zr6Ga2Ni passes the benchmark rule | 55% |
| Re3Ir has an imaginary mode at full settings | 55% |
| SbTe2 has an imaginary mode at full settings | 45% |
| At least one of the two candidates passes the candidate rule | 25% |

What I had already seen when writing these: the Re3Ir rehearsal's first three
phonon points on smoke meshes, which show a threefold mode at −39 cm⁻¹ at the
zone centre. A coarse mesh produced a false imaginary mode for niobium before
(`docs/hazel-runbook.md`, PBEsol follow-up), so this moves the Re3Ir number
only a little above even. No Zr6Ga2Ni or SbTe2 phonon had been run.

## Running it on Hazel

Do sections 3 to 5 of `docs/hazel-runbook.md` first (first login, the conda
environment, the hello-world job). Then, on the login node:

1. Get the repository and activate the environment.
   ```bash
   cd /share/<group>/<unity_id> && git clone https://github.com/hudsonclavin-cloud/RT-SuperConductor.git
   ```
   ```bash
   cd RT-SuperConductor && conda activate rtsc-qe
   ```
2. See what would be submitted. Nothing runs.
   ```bash
   DRY_RUN=1 scripts/hpc/submit_phonon_chain.sh data/benchmarks/candidates/zr6ga2ni /share/<group>/<unity_id>/rtsc/zr6ga2ni-smoke compute normal 16
   ```
3. Smoke run: coarse meshes, to check that every program starts and the jobs
   chain. Use the partition and QOS that `sa` showed you.
   ```bash
   scripts/hpc/submit_phonon_chain.sh data/benchmarks/candidates/zr6ga2ni /share/<group>/<unity_id>/rtsc/zr6ga2ni-smoke compute normal 16
   ```
4. Watch it.
   ```bash
   squeue -u $USER
   ```
5. When the last job has gone from the queue, read the report.
   ```bash
   cat /share/<group>/<unity_id>/rtsc/zr6ga2ni-smoke/run/report.json
   ```
   A smoke report should say `"verdict": "smoke run: checks the tools only, no verdict"`
   and list ten rows under `lambda_by_broadening`. If a job failed, its
   messages are in `logs/` in the run directory.
6. Full run of the benchmark. A new run directory, and `CHAIN_MODE=full`.
   ```bash
   CHAIN_MODE=full scripts/hpc/submit_phonon_chain.sh data/benchmarks/candidates/zr6ga2ni /share/<group>/<unity_id>/rtsc/zr6ga2ni-full compute normal 16
   ```
7. If the benchmark passes rule 1, run the two candidates the same way, with
   `re3ir` and `sbte2` in place of `zr6ga2ni`.
8. Copy each `report.json` (and `run/lambda`, `run/*.dyn*`, `run/scf.out`) back
   into the repository under `results/<name>/`. The work area is wiped after
   about 30 days.

**If the phonon job hits its time limit**, resubmit from that step. It
continues where it stopped:

```bash
CHAIN_MODE=full START_AT=ph scripts/hpc/submit_phonon_chain.sh data/benchmarks/candidates/zr6ga2ni /share/<group>/<unity_id>/rtsc/zr6ga2ni-full compute normal 16
```

**Time.** [Speculative] On 16 cores a smoke run should take under an hour in
total, and a full run 5 to 20 hours, nearly all of it in the phonon job. These
are scaled from the rehearsal on 2 cores, not measured on Hazel.

## Reading `report.json`

| Field | Meaning |
| --- | --- |
| `lowest_frequency_cm1`, `imaginary_mode` | The lowest phonon frequency at any calculated point. Below −5 cm⁻¹ counts as imaginary: the structure is not stable as drawn. |
| `lambda_by_broadening` | λ and ωlog at ten broadenings. If λ changes a lot between rows, the result is not converged. |
| `headline` | The 0.020 Ry row, with Allen-Dynes Tc at μ* = 0.10 and 0.13. |
| `scf_pressure_kbar`, `scf_total_force_ry_bohr` | Should be near zero. They check the cell is relaxed. |
| `verdict` | The candidate rule applied. For the benchmark, apply rule 1 by hand to `headline.lambda`. |
