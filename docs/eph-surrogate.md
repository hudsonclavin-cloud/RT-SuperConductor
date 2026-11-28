# Can a network predict what a Hazel calculation would say? Plan, fixed before training

**Main idea:** train crystal-graph networks overnight on the owner's GPU to
reproduce the electron-phonon calculations of the Alexandria 2025 phonon
release. Then use them to rank the 27,000 Materials Project metals by
calculated Tc, before Hazel time is spent.

Written 2026-11-27, before any network has been trained on these targets.
Owner's request the same day: "a 6-hour training run, … create a script
for tonight". Script: `scripts/eph_gpu.py`.

## Why this and not a rerun

- **A rerun would repeat a negative result.** The 8-hour run of
  2026-11-22 (`docs/overnight-gpu.md`) trained structure networks on the
  measured labels. None beat cycle 6. Retraining the same models on the
  same rows would reproduce that.
- **This target is different.** It is the calculation Hazel runs
  (density-functional perturbation theory: λ, ωlog, Tc), not the
  literature's yes or no.
  - A model of the literature can only rank compounds like the ones
    already measured.
  - A model of the calculation can score any structure, including ones
    nobody has measured.
- **There are three times more rows than before.** 23,786 calculations,
  against the 7,000-odd of the release used in `docs/physics-proxy2.md`.

## Data

- **Source:** Alexandria `pbesol_ph` 2025.08.11 (CC BY 4.0), reduced by
  `scripts/alexandria_eph_2025.py`.
  - QE DFPT, PBEsol, harmonic phonons.
  - Tc is by Allen–Dynes at μ* = 0.10.
- **Rows:** dynamically stable entries with a Tc, without the
  `docs/alexandria-eph-2025.json` flags (λ > 5 or ωlog < 100 K).
- **Targets:** λ, ln ωlog and ln(1 + Tc).
- **Split, fixed at export:**
  - whole chemical systems (element sets) go to fit (75%), validation
    (10%) or test (15%), by a hash of the element set;
  - no test compound has a same-element sibling in training;
  - a random split would overstate accuracy, because polymorphs and
    near-identical cells would sit on both sides.
- **Candidates:** the two lists of the 2026-11-22 structure bundle. They
  are 26,911 made metals and the 571 compounds of the shortlist 3 pool,
  with their Materials Project structures.

## Training

- **Encoder:** the CGCNN of `scripts/structure_gpu_train.py`, with a
  three-output head.
- **Members:** networks are trained one after another until the time
  budget (6 h) is used.
  - The first half explores settings: depth, width, learning rate,
    batch, dropout and weight decay, with one seed each.
  - The second half trains more seeds of the three best.
- **Choosing members:** validation rows only, by the mean squared error
  of ln(1 + Tc). The five best are averaged.
- **Scoring:** the test rows are scored once, at the end, by the script
  itself.
- **A disclosure.** The first CPU rehearsal here, a 2-epoch model on
  400 rows, scored the test rows before this was caught: AUC 0.80 for
  Tc ≥ 10 K. Nothing was chosen from it. The thresholds below were
  written before it ran. Rehearsals now score validation rows instead.

## Reading, fixed now

**Measures, on the test rows:**
- Tc mean absolute error (K), against the error of always predicting the
  fit rows' median;
- Spearman correlation of Tc;
- AUC for calculated Tc ≥ 10 K and ≥ 20 K;
- λ and ln ωlog errors.

| Result | What follows |
| --- | --- |
| AUC (Tc ≥ 10 K) at least 0.85, **and** Tc error at most 0.7 × the median baseline's | The ensemble's calculated-Tc ranking is used to order Hazel candidates within the rung-4 and room-temperature tracks. Its top made metals not already in Alexandria are read and proposed as Hazel candidates |
| Anything else | Not used to order Hazel work. The result is recorded, and the next step is decided from what failed |

**In both cases:**
- The ensemble's outputs are **calculations predicted, not measurements**.
  They never become labels for the ranking of record.
- A candidate already in Alexandria (same composition) uses Alexandria's
  own value, not the network's.
- A candidate whose structure is unlike the training set (more sites than
  any training cell, or a large ensemble spread) is marked as such.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| AUC (Tc ≥ 10 K) at least 0.85 | 65% |
| Tc error at most 0.7 × baseline | 60% |
| Both, so the ensemble is used | 50% |
| Any of the 26,911 made metals predicted at 77 K or more, with a small spread | 15% |

**Why not higher.**
- [Likely] Calculated Tc depends on soft phonon modes, which small
  changes in structure move a lot. A graph network sees the structure,
  not the phonons.
- [Likely] The candidates are not like the training rows. Alexandria's
  phonon entries are small, mostly hypothetical cells. The made metals are
  known compounds, many with larger cells.
- [Certain] The test split holds out whole chemical systems, which is
  harder than the random splits most published results use.

## Caveats

- **Functional mismatch.** Alexandria is PBEsol, and the Hazel chain is
  PBE (`docs/hazel-candidates.md`). [Likely] λ differs by a few percent
  between them, more where a soft mode sits near an instability.
- **Not a room-temperature finder by itself.** Conventional
  (phonon-driven) superconductors predicted at ambient pressure top out
  well below room temperature in every large calculated set so far
  [Likely]. This tool spends Hazel time better. It does not change that
  physics.

## Amendments before training (2026-11-28)

Written after an independent review of the script. No network had been
trained on these targets yet, and nothing below was chosen from a result.

1. **The laptop is kept awake, and time is counted as training time.**
   - The review found that nothing stopped Windows from sleeping, and that
     the budget ran on the wall clock. One idle sleep would have used up
     the 6 h. The script would then have spent the one test scoring on a
     few early networks, and a rerun could not recover.
   - Now the script asks Windows not to sleep, as the 2026-11-22 driver
     did.
   - Any gap far longer than a usual epoch (more than 10 times, and more
     than 2 min) is charged as one usual epoch.
   - **Owner:** keep the lid open, or set "When I close the lid" to "Do
     nothing" while plugged in. The sleep request does not cover a closed
     lid.
2. **The test rows can only be scored once.** A finished run (with
   `share/result.json`) refuses to train or score again. A resume must
   have the same profile, bundle, seed, epochs and patience as the run it
   continues.
3. **A crash cannot break a resume.** Files are written whole or not at
   all, and an unreadable member is trained again.
4. **Settings are drawn without repeats.** Each of the 432 combinations is
   used once, in a seeded order. The three settings given more seeds are
   therefore always three different ones.
5. **Predicted Tc below 0 K counts as 0 K** in the Tc error. Tc is never
   negative. This can only lower the error, and the rule above is
   unchanged.
6. **Outputs for the morning:**
   - column names, and the spread named in its units (ln(1 + Tc));
   - each candidate's number of sites, and how many fit rows contain its
     rarest element;
   - the largest training cell's size, in `result.json`;
   - a log in `share/log.txt`.
7. **Reported only, read in the morning: a transfer check.** Some
   candidates' compositions are also in Alexandria's validation or test
   rows. For those, the prediction on the Materials Project structure is
   compared with Alexandria's own value. This shows how much is lost
   between PBEsol cells and Materials Project cells. It does not change
   the rule above.

Not changed: the 300-epoch cosine schedule with patience 30. A reviewer
flagged it, and the verifier measured no gain from shortening it. Each
member keeps its best-validation epoch either way.
