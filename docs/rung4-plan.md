# Rung 4: new compositions in known families

Written 2026-11-16 06:55 UTC, before any of the work below was run. The pool
sizes in "The pool" were counted while writing, from the committed candidate
table. No model has been trained or scored for this plan.

## Where this sits

The project's ladder of tests (owner's framing, 2026-11-16):

1. Retrospective: rank known materials in held-out chemistry. Done (AUC 0.79).
2. Out of time: predict later-published superconductors. Done once.
3. Prospective, made but unmeasured: call real metals before measurement.
   List ready, nothing measured.
4. **Prospective, new compounds: propose something never made that is
   stable, can be made, and superconducts.** This plan.

The owner chose the narrower version of rung 4: **new compositions in known
families**, not new families.

## What earlier work in this repository already showed

This plan is built around those results, not on top of a blank slate:
- **Structure-aware encoders did not beat the formula baseline** on
  experimental Tc. Frozen CHGNet, Tc-fine-tuned CHGNet and scratch ALIGNN
  were all worse on cross-family MAE (`docs/physics-features-deep-one-page.md`).
  So this plan does not start with a graph network.
- **The earlier substitution campaign** (`docs/deep-phase4-campaign.json`)
  generated element swaps on known superconductors. Most outputs were lightly
  doped versions of a known parent (for example GaTi0.072V2.928 from
  GaRe0.072V2.928). Those are new spellings of known materials, not new
  compounds. It also ran without Materials Project stability data.
- **The machine-learning phonon screen** (CHGNet + phonopy) flagged stable
  aluminium as unstable, so it is not a valid filter
  (`docs/phonon-screen-validation.md`). This plan does not use it.
- **The published calculations of Bercx et al.** (`docs/bercx-published-check.md`)
  cover 4,533 non-magnetic metals with a coarse calculated Tc screen (Table B
  against Table C), and 250 with full λ, ωlog and Tc. Their screen and this
  project's ranking agree more than chance on made metals (exploratory:
  30% against 16% with a coarse Tc above 1 K, p = 1×10⁻⁵).

## The pool: compounds calculated to be stable but never made

From `data/candidates/mp-candidates.csv.gz`: Materials Project entries marked
`theoretical` (no ICSD record), metallic, with no match in any
superconductivity database, energy above the hull of at most 0.025 eV/atom (the
table's own cap).

**"Known family" is defined as a structure prototype:** the same space group
and the same reduced formula pattern (sorted atomic fractions, for example
AB2 or A3B) as at least three experimentally reported compounds in the table
that are known superconductors (exact match, Tc above 0). This is crude: it
groups by shape, not by bonding.

| Step | Compounds |
| --- | --- |
| Never-made, untested metals (hull at most 0.025 eV/atom) | 8,780 |
| In a known-superconductor prototype (81 prototypes, holding 711 known superconductors) | 4,370 |
| Not magnetic in the calculation | 4,217 |
| No copper (the conventional lane, as for the made metals) | 4,014 |
| On the hull (0 eV/atom) | 2,399 |

**"Never made" is a claim about databases, not the literature.** A
`theoretical` entry has no ICSD structure. Some have been made and reported
elsewhere. The literature check in step 4 tests that.

## The work, in order

### Step 1: freeze the pool

A script writes the pool, one row per compound, with its prototype, hull
energy and the counts above, and commits the list of IDs. Nothing is scored
yet.

### Step 2: a physics proxy trained only on calculations

**Target:** for each of Bercx et al.'s 4,533 Table B metals, 1 if it reached
Table C (coarse Allen–Dynes Tc above 1 K), else 0. These are calculations on
experimentally known structures, not measurements.

**Inputs:** the same formula features as the ranking of record, plus the space
group number and crystal system from Bercx et al.'s own entry. No experimental
label is used anywhere in this model.

**Model:** round 2's boosting settings, unchanged. Grouped five-fold
cross-validation by element system.

**Test T1 (does it learn the calculation?).** Cross-validated AUC on Bercx
Table B. **Passes at 0.70 or more.**

**Test T2 (does what it learned carry over to experiments?).**
- Train on all of Table B, then score every experimentally labelled metal in
  the candidate table whose formula and space group are not in Table B.
- AUC against the experimental labels. **Passes at 0.60 or more.**
- Also reported: does adding the proxy to the ranking of record change
  cross-validated experimental AUC? Reported with its paired interval. It
  counts as a gain only if the interval excludes zero.

T2 is the closest thing in this plan to "does the model understand
physics": a model that never saw an experiment either ranks experimental
superconductors well or it does not.

### Step 3: shortlist from the pool

Every rule below is fixed now:
- Score each pool compound with the ranking of record: the bagged version if
  `docs/stable-ranking.md` supports it, otherwise the single model.
- Keep compounds on the hull (0 eV/atom).
- If T1 passes, also require the physics proxy to place the compound in its
  top quarter of the pool. If T1 fails, the proxy is not used.
- At most 2 compounds per element system and 3 per prototype, so one family
  cannot fill the list.
- Take the top 20.
- **Control group:** 20 compounds drawn at random from the same filtered pool
  (seed 2026), matched to the shortlist's prototype mix where possible.

### Step 4: literature check (shortlist and controls, blinded)

For each of the 40 compounds, mixed together so the readers cannot tell
shortlist from control:
- Has it been synthesized? If so, was it measured at low temperature?
- The same readers and verifier procedure as the label corrections. Results
  are recorded, and none enters training.

### Step 5: physics check on 5 finalists (Hazel, the owner's week)

- The five best-ranked shortlist compounds that the literature check finds
  never made, plus 1 control.
- DFPT phonons on the Materials Project structure, then EPW coupling. This
  uses the stop-rule settings, after the tantalum band and symmetry failures
  are understood, or Bercx et al.'s settings if the owner prefers.
- **A finalist goes to the lab list only if** it has no imaginary phonon modes
  and a calculated isotropic Tc of at least 2 K. Calculated Tc ran 1.1 to 2.8
  times above measured in the Bercx comparison, so it is read as an upper
  bound.

### Step 6: lab

A partner who can arc-melt or solid-state-synthesize intermetallics and
measure resistance or magnetization below 4 K. The lab, not this plan,
decides what is feasible.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| T1 passes (physics proxy AUC on Bercx Table B ≥ 0.70) | 60% |
| T2 passes (proxy AUC on experimental labels ≥ 0.60) | 45% |
| Adding the proxy to the ranking of record raises experimental AUC with an interval excluding zero | 20% |
| At least 5 of the 20 shortlisted compounds turn out to have been synthesized already | 40% |
| Shortlisted compounds are found synthesized more often than controls | 50% |
| At least 1 of the 5 finalists passes step 5 (no imaginary modes, calculated Tc ≥ 2 K) | 50% |

## What this plan cannot show

- **That a compound can be made.** Being on the calculated hull is necessary,
  not sufficient. Many hull compounds have never been synthesized for kinetic
  reasons.
- **That the model "understands physics" in general.** T2 tests one transfer,
  from one coarse calculation to experimental labels, in conventional metals.
- **Anything about unconventional superconductors.** Cuprates and iron-based
  compounds are out of this lane, and the calculations do not cover them.
- **Accuracy of calculated Tc.** It ran high in every matched comparison so
  far.

## Risks

- **Biggest:** the shortlist turns out to be variants of known superconductors
  (the same weakness as the earlier substitution campaign). The per-system and
  per-prototype caps and the literature check are there to catch it.
- **Most likely failure:** T2 fails. A proxy that learns the calculation does
  not, by itself, rank experiments. Then step 3 uses the ranking of record
  alone, and the physics check in step 5 carries the extrapolation.
- **Opportunity cost:** steps 1 to 4 take about a day of session time; step 5
  takes the Hazel week. Rung 3 (contacting a lab about the made metals) should
  not wait on this.

## What would change the plan

- If T2 passes strongly (AUC 0.70 or more), a structure-aware model trained on
  Bercx et al.'s 250 full calculations and the 1,057 JARVIS λ values becomes
  worth building.
- If the literature check finds most of the shortlist already made, the
  "never made" pool needs a second filter (OpenAlex or other synthesis
  literature) before scoring.

## Steps 1 and 2: results, 2026-11-16 07:01 UTC

The script (`scripts/rung4.py`, with its implementation choices in the
docstring) was committed at 06:57 UTC (`6aa4b67`), before either step was run.

**Step 1: the pool is frozen.** `data/benchmarks/rung4/pool.csv` holds 4,014
compounds; `docs/rung4/pool-summary.json` has the counts. They match the plan
exactly: 8,780, then 4,370, then 4,217, then 4,014, with 2,399 on the hull.
The pool spans 61 of the 81 prototypes and 3,426 element systems.

**Step 2: the physics proxy** (`docs/rung4/proxy.json`). The proxy is trained
only on Bercx et al.'s calculations: 4,366 distinct Table B entries, 1,549 of
them in Table C.

| Test | Result | Pass mark | |
| --- | --- | --- | --- |
| T1: learns the calculation (grouped CV AUC on Table B) | **0.825** (0.811 to 0.838) | 0.70 | passes |
| T2: carries over to experiments (AUC on 882 experimental metals not in Table B) | **0.531** (0.491 to 0.573) | 0.60 | **fails** |
| Adding the proxy to the ranking of record (739 of those rows) | 0.797 alone, 0.728 combined: **−0.069** (−0.099 to −0.037) | interval above 0 | **fails: it hurts** |

Against the predictions: T1 passing (60%) came true. T2 passing (45%) did
not. A gain from adding the proxy (20%) did not happen either: the proxy
made the ranking worse.

[Certain] **On the pre-registered test, the model trained on calculations does
not rank experimental superconductors.** Its AUC on the T2 rows is
indistinguishable from chance.

**Exploratory checks** (chosen after T2 failed, so read them as leads:
`scripts/rung4_explore.py`, `docs/rung4/proxy-exploratory.json`):
- **Inside Bercx et al.'s own set the transfer works.** On the 712
  experimental metals that are in Table B, the proxy's out-of-fold score
  (trained on other element systems) ranks the experimental labels at AUC
  0.763.
- **Bercx et al.'s coarse calculation itself predicts experiment.** At AUC
  0.713, 79% of compounds with a calculated Tc above 1 K are superconductors,
  against 35% of the rest. [Likely] So the calculation carries real physics
  signal, and the proxy learns it.
- **Outside that set it does not.** The failure is not explained by family
  (conventional lane only: 0.538), cell size or magnetism (12 sites or fewer
  and non-magnetic: 0.496; larger or magnetic: 0.604). The reason is not
  known.
  - [Speculative] The candidates are that Bercx et al. excluded some of these
    compounds for reasons that also matter to superconductivity, or that a
    structure in this table and theirs share a formula but differ in some way
    a space group alone does not capture.

**What it means for rung 4.**
- Every pool compound is calculated but never made, so all of them are
  outside Bercx et al.'s set of experimentally known structures. On the
  evidence above, the proxy should not be trusted on them.
- **The plan contradicts itself here.** Step 3 says to use the proxy as a
  filter if T1 passes, and T1 passed. But "Risks" says that if T2 fails,
  step 3 uses the ranking of record alone.
- I recommend the second reading: the proxy is **not** used in step 3,
  because T2 failed and combining it lowered experimental AUC by 0.069. Which
  reading holds is the owner's decision.

**Rung 4 status.** Pattern-matching on formulas (the ranking of record, AUC
0.797 on these rows) is still the only signal that ranks experiments outside
the calculated set. "Understanding physics" in the sense of T2 is not shown.
Step 5's direct calculation on finalists stays the physics check.

**Correction to the heading above (2026-11-16 07:02 UTC).** "Steps 1 and 2:
results, 2026-11-16 07:01 UTC" is later than the commit that added it
(`63db81d`, 07:00:55 UTC). The section was written at 07:00 UTC. Nothing else
in it changes.

**Correction to that correction (written 07:01 UTC).** Its stamp, "07:02 UTC",
is also later than its commit (`31a7e6f`, 07:01:14 UTC). It was written at
07:01 UTC.
