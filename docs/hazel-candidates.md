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
| `zr6ga2ni` | Zr6Ga2Ni | Benchmark: Bercx et al. calculated λ = 0.70, isotropic Tc 5.5 K (`docs/bercx-published-check.md`) | P-62m | 9 | ready |
| `re3ir` | Re3Ir | Rung 4 candidate, pool rank 19 | Pm-3m (Cu3Au type) | 4 | ready. **Update 2026-11-16 15:07 UTC: failed the pre-check (see the last section); do not submit a full run** |
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
   - **Update 2026-11-16 15:07 UTC:** run `sbte2` only. Re3Ir failed its
     pre-check, and the job script refuses a full run of it.
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

## Second read of the two candidates

One more low-tier reader each, independent of the first
(`data/benchmarks/rung4/second-read.csv`). Both read titles and abstracts
only.

| Compound | Found as made? | What the reader did find |
| --- | --- | --- |
| Re3Ir | No | Rhenium and iridium form disordered solid solutions, cubic on the iridium side and hexagonal on the rhenium side. No ordered compound near 75% rhenium was reported, and no superconductivity for any Re-Ir alloy. |
| SbTe2 | No | Every result was Sb2Te3. It superconducts only under pressure (about 7 K above 30 GPa, Zhu et al. 2013). |

- [Likely] Neither compound has been made. Two short searches finding nothing
  is weak evidence, and neither reader checked a phase-diagram database.
- [Likely] For Re3Ir the more important finding is the solid solution. The
  ordered structure the calculation assumes may not be what forms.

## Rehearsal on this machine, 2026-11-16 10:48 UTC

The Re3Ir package ran through all six steps in smoke mode on 2 cores, with
the same job script and `LOCAL=1` in place of the scheduler.

| Step | Result |
| --- | --- |
| Dense ground state, ground state | Finished in about 4 and 1 minutes. |
| Phonons (4 points, coarse meshes) | About 2 hours 10 minutes in total. The job was cut off at this machine's 2-hour limit and resumed with `START_AT=ph`; it continued at the fourth point and did not start over. |
| Force constants, a2F and λ, report | 2, 32 and 2 seconds. |

**What the rehearsal fixed.**

- **The report lost the package.** The job runs it from inside `run/` with the
  path `.`, and it then looked for `package.json` in the wrong folder, so the
  formula was missing and a smoke run was not labelled as one. Fixed, with a
  test.
- **The report's reading of the λ table** was written from the documentation.
  It matches what `matdyn.x` really writes. The test now uses real lines.
- **Zr6Ga2Ni had lost its symmetry.** The Materials Project's rounded
  coordinates gave 2 symmetry operations instead of 12, which makes every
  step several times more expensive. The generator now symmetrises a
  structure before relaxing it.

- **SbTe2's phonon step would have crashed on Hazel.** `ph.x` stopped with
  "divide_class: Wrong classes for D_3d" because the cell was drawn in the
  Materials Project's orientation, not Quantum ESPRESSO's. The pre-check
  below found it (11:00 UTC). The generator now writes hexagonal cells in
  Quantum ESPRESSO's orientation, and SbTe2's inputs were regenerated. The
  atoms and the physics are unchanged.

**What the smoke numbers show, and do not.** Re3Ir on the coarse meshes has
imaginary modes at two of its four points: −39 cm⁻¹ (threefold) at the zone
centre and −89 cm⁻¹ at the zone corner. Its λ of 2.6 comes from those soft
modes and means nothing. [Speculative] Two points agreeing makes a real
instability more likely than the single point I had seen when I wrote the
prediction, but rule 3 stands: no verdict from a smoke run.

**Rehearsed end to end: Re3Ir only.** For Zr6Ga2Ni and SbTe2 the ground
state and the start of the phonon step (where `ph.x` checks the symmetry) were
run here; the later steps were not. Zr6Ga2Ni's relaxation converged at 13:12
UTC with 12 symmetry operations, zero residual pressure and a total force of
0.00007 Ry/bohr. Its last step, saving the wavefunctions, failed because this
machine's disk was full; the relaxed cell and the final pressure were already
written, and the package needs nothing else from it.

**Not rehearsed:** the Slurm path itself (`sbatch`, job dependencies). The
EPW chain uses the same submit function and has not run on Hazel either. The
dry run in step 2 above is the first check of it.

## Stability pre-check at single points, planned 2026-11-16 10:48 UTC, before any result

**Why.** Hazel time is better spent on candidates that are not already known
to be unstable. One phonon point at full ground-state settings is cheap enough
to run here.

**What.** For Re3Ir and SbTe2: the package's own `scf.in` unchanged (full
k mesh), then `ph.x` at the zone centre only, with no electron-phonon step.
For Re3Ir also the zone corner (½ ½ ½), where the smoke run was softest.

**Rule.**

- A mode below −5 cm⁻¹ at full settings means the candidate fails the
  candidate rule now. It is still left in the package, marked as failed, and
  not submitted.
- No imaginary mode at these points proves nothing about the other points.
  The candidate stays in the queue for Hazel.

**Prediction [Speculative].** Re3Ir imaginary at one of its two points: 60%.
SbTe2 imaginary at the zone centre: 20% (instabilities in layered compounds
are more often away from the centre).

## Pre-check at the zone centre: results, and a correction to the rule, 2026-11-16 14:44 UTC

Written while Re3Ir's zone-corner run was still going, so that result is not
known here. The correction below cannot affect it: it applies at the zone
centre only.

**Results at full ground-state settings (cm⁻¹).**

| Compound | Raw output from `ph.x` | With the acoustic sum rule (`dynmat.x`) | Unstable at the centre? |
| --- | --- | --- | --- |
| Re3Ir | 11.8 (×3), 77.9 (×3), 114.4 (×3), 126.6 (×3) | 0 (×3), 76.9 (×3), 113.8 (×3), 126.1 (×3) | No |
| SbTe2 | −10.7 (×2), 3.8, 74.0 (×2), 92.0 (×2), 109.9, 141.2 | 0 (×3), 74.6 (×2), 92.7 (×2), 109.7, 141.3 | No |

- **Re3Ir.** The threefold mode that sat at −39 cm⁻¹ on the smoke meshes is at
  +78 cm⁻¹ on the full mesh. It was an artifact of the coarse mesh, as
  niobium's was.
- **SbTe2.** No optical mode is soft. Predicted 20% unstable here; it is not.

**Correction to the rule (made after seeing SbTe2's numbers; labelled so).**

- The rule said "a mode below −5 cm⁻¹ fails". Read literally, SbTe2's −10.7
  cm⁻¹ fails it. That reading is wrong, and the rule was badly written.
- [Certain] Those three modes are the whole crystal sliding in one direction.
  In the output file every atom moves the same way by the same amount (0.574
  to 0.579). Sliding a crystal costs no energy, so their true frequency is
  exactly zero. `ph.x` does not impose this; the later steps do.
- The rule now reads: **a mode below −5 cm⁻¹ fails, except the three
  whole-crystal translations at the zone centre.** A soft mode of any other
  kind at the centre still fails, and so does any soft mode away from it.
- `scripts/phonon_chain_report.py` had the same fault and would have marked
  SbTe2 as failed on Hazel. It now leaves those three modes out, lists them
  under `zone_centre_translations_cm1`, and has tests for both directions
  (a translation is excused; a soft optical mode at the centre is not).
- Changing a rule after a result is what the rules are meant to prevent. The
  owner can veto this one. The evidence is the atom displacements above and
  the `dynmat.x` column, both from files in this session's scratch space.

**What this does not show.** One point per compound. Neither result says the
compound is stable elsewhere in the zone.

## Pre-check at Re3Ir's zone corner: it fails, 2026-11-16 15:07 UTC

**Result.** At the zone corner (½ ½ ½), with the package's own ground-state
settings, Re3Ir has one mode at **−62.5 cm⁻¹**. The other eleven are between
84 and 148 cm⁻¹. All the numbers are in
`data/benchmarks/candidates/precheck.csv`.

| Point | Smoke meshes (k 6³) | Full settings (k 12³) |
| --- | --- | --- |
| Zone centre, lowest mode that counts | −39.0 | +77.9 |
| Zone corner, lowest mode | −88.8 | −62.5 |

**Verdict under the rule: Re3Ir fails the candidate rule.** The prediction
gave 60% to an imaginary mode at one of the two points.

- In the unstable mode the three rhenium atoms move and the iridium atom
  stays still. This is not a translation and it is not at the zone centre,
  so the correction in the section above does not touch it.
- [Likely] The instability is real at these settings. It means ordered Re3Ir
  in this structure would distort into a larger cell. Together with the second
  reader's finding (rhenium and iridium form disordered alloys), it says this
  structure is probably not what forms.
- [Speculative] It might not survive a denser k mesh. The zone-centre mode
  moved by 117 cm⁻¹ between the two meshes, and this one moved by 26 in the
  stable direction. I put about 25% on a denser mesh removing it. That was
  not tested: one more point at k 18³ is about five hours on this machine.

**What changed in the repository.**

- `data/benchmarks/candidates/re3ir/PRECHECK_FAILED.txt` records the result.
  The package itself is unchanged.
- `scripts/hpc/submit_phonon_chain.sh` refuses a full run of a package that
  carries that file, unless `ALLOW_FAILED=1` is set. A smoke run still
  works, because Re3Ir is the one package rehearsed end to end.

**Where rung 4 stands.**

- Of six never-made shortlist compounds, four could not be calculated, one
  has failed, and one (SbTe2) is still open after a single point.
- The prediction "at least one of the two candidates passes the candidate
  rule" was 25%. [Speculative] With Re3Ir out it is about 15%.
- [Likely] The next shortlist needs two filters before ranking, not after: a
  stability check like this one, and what a lab can make (no radioactive
  elements, and a phase diagram that shows a compound and not a solid
  solution).

## Update, 2026-11-16 19:11 UTC: Hazel storage granted

The HPC staff granted a Research Storage folder and sent their own Quantum
ESPRESSO test commands. What changes for the steps in "Running it on Hazel"
is in `docs/hazel-runbook.md` ("Update, 2026-11-16 19:11 UTC"). In short:

- Install the environment in Research Storage, run in `/share`, and copy
  each finished run directory to Research Storage as well as its report into
  the repository.
- `MEM=16G` in front of the submit command requests memory; the partner
  nodes need `PH_TIME=01:55:00` and resubmission with `START_AT=ph`.
- Access ends 2027-05-07.

## Update, 2026-11-17 17:00 UTC: the report now recognises translations for atoms of any mass

**What was wrong.** The rule sets aside the three whole-crystal translations
at the zone centre. The report recognised one only if every atom moved by
nearly the same amount. In NaMgPb2 and LiMgPb2 the atoms differ a lot in
mass and ph.x leaves the translations impure, so they were not set aside
(`docs/rung4-shortlist2.md`). Had they come out negative, the report would
have called a stable compound unstable.

**The fix.** The report reads each atom's mass from the file's header and
measures how much of a mode is the crystal sliding as a whole: the
mass-weighted overlap with a pure translation. An optical mode at the zone
centre carries no net momentum, whatever the masses, so its overlap is near
zero.

| File (zone centre) | Overlap of the three lowest modes | Largest overlap of any other mode |
| --- | --- | --- |
| NaMgPb2 | 0.996 | 0.07 |
| LiMgPb2 | 0.999 | 0.04 |
| CsIr2P2 | 0.996 to 1.000 | 0.09 |

- A mode counts as a translation at 0.9 or more, and only at the zone
  centre.
- The three real files are test fixtures (`tests/fixtures/phonon/`). The
  tests check that the translations are found, that a soft optical mode
  still counts, and that the same pattern away from the zone centre still
  counts.
- No verdict changes. With the fix the lowest counted modes at the zone
  centre are 60.4 (NaMgPb2), 72.3 (LiMgPb2) and 50.8 cm⁻¹ (CsIr2P2), the
  same as the sum-rule check gave.
- Files without a header fall back to the earlier test.

## Rehearsal of the new packages on this machine, 2026-11-17 19:31 UTC

**Main result:** the whole six-step chain now runs end to end, in smoke
mode, for NaMgPb2 and SbTe2 as well as Re3Ir. Nothing had to be fixed. The
same job script was used with `LOCAL=1` in place of the scheduler, on 4
cores.

| Step | NaMgPb2 | SbTe2 |
| --- | --- | --- |
| Dense ground state | 52 s | 188 s |
| Ground state | 12 s | 36 s |
| Phonons with electron-phonon coupling (coarse meshes) | 22 min (3 points) | 91 min (4 points) |
| Force constants | 2 s | 2 s |
| a2F and λ | 10 s | 14 s |
| Report | 2 s | under 1 s |
| **Whole chain** | **24 min** | **95 min** |

- SbTe2's phonon step shared the processors with other work for part of
  its run, so 91 minutes overstates it.
- Each report says `"verdict": "smoke run: checks the tools only, no
  verdict"`, as it should.

**The report's new translation test, on real chain output.**

| Compound | Translations set aside (cm⁻¹) | Lowest counted mode (cm⁻¹) | Imaginary? |
| --- | --- | --- | --- |
| NaMgPb2 | 21.8, 21.8, 21.8 | 38.0 | No |
| SbTe2 | −11.3, −11.3, 16.4 | 37.4 | No |

- [Certain] For SbTe2 two translations are at −11.3 cm⁻¹, below the −5
  limit. The report set them aside and did not call the compound unstable.
  That is the case the fix of the same day was written for.

**What the smoke numbers show, and do not.** λ came out at 0.85 for NaMgPb2
and 0.39 for SbTe2. NaMgPb2's ranges from 0.81 to 3.06 across the ten
broadenings, which means it is not converged on these coarse meshes. Rule 3
stands: no verdict from a smoke run.

**Not yet rehearsed:** CsIr2P2 (running), LiMgPb2 and Zr6Ga2Ni. Zr6Ga2Ni has
nine atoms; its smoke run may be too long for this machine's two-hour limit
on a job and will be continued from saved work as needed. The Slurm path
itself still cannot be rehearsed here.

## Rehearsal: CsIr2P2 also runs end to end, 2026-11-17 23:54 UTC

The CsIr2P2 package completed all six steps in smoke mode, with nothing to
fix. It is the first body-centred tetragonal cell to go through the last
three steps (force constants, λ, report).

| Step | CsIr2P2 |
| --- | --- |
| Dense ground state, ground state | 5 min and 84 s |
| Phonons with electron-phonon coupling (4 points, coarse meshes) | about 4.8 hours on 4 cores |
| Force constants, a2F and λ, report | 1 s, 9 s, under 1 s |

- The phonon step was cut three times by this machine's two-hour limit on a
  job and continued from saved work each time. On Hazel's ordinary nodes it
  would run through; on the partner nodes it needs the same resubmission
  with `START_AT=ph`.
- Report: translations set aside at −7.8, 11.7 and 11.7 cm⁻¹; lowest
  counted mode 28.0 cm⁻¹; no imaginary mode; verdict "smoke run: checks the
  tools only, no verdict".
- Smoke λ is 0.60 and ranges from 0.29 to 1.73 across broadenings: not
  converged, and no verdict.

**Rehearsed end to end so far:** Re3Ir, NaMgPb2, SbTe2, CsIr2P2. LiMgPb2
and the Zr6Ga2Ni benchmark are running.

**For planning Hazel time.** [Speculative] CsIr2P2's full run (finer
meshes and more phonon points; I have not counted them) is the most expensive of the candidates:
roughly 20 to 40 times this smoke run in processor time. On 16 cores that
is a day or more, so it belongs on the ordinary nodes, not the two-hour
partner queue.
