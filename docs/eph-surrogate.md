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

## Result, 2026-11-28 (run of 2026-11-27 to 28, scored once by the script)

**Main idea:** the network reproduces Alexandria's calculated Tc on whole
held-out chemical systems, so the plan's rule says to use it. Applied to the
26,911 made compounds, it finds **nothing that a calculation would put above
19 K** once insulators, magnetic compounds and compounds unlike the training
set are removed. It is a tool for spending Hazel time. It does not find a
room-temperature superconductor among known compounds.

Record: `data/benchmarks/local-gpu/eph-20261128/` (the run), the reading in
`docs/eph-surrogate.json` and `docs/eph-surrogate-candidates.csv`, by
`scripts/eph_read.py`. The run: 41 networks in 5.75 h on an RTX 5070 laptop
GPU, five averaged.

### The plan's rule

| Measure (3,303 test rows, 1,894 chemical systems) | Result | Needed |
| --- | --- | --- |
| AUC, calculated Tc ≥ 10 K | **0.951** (95% interval 0.940–0.959) | at least 0.85 |
| Tc error | **2.10 K**, 0.39 × the median baseline (interval 0.36–0.42) | at most 0.7 × 5.38 K = 3.76 K |
| Spearman correlation of Tc | 0.895 | not set |
| AUC, Tc ≥ 20 K | 0.961 | not set |
| λ error, ln ωlog error | 0.11, 0.12 | not set |

**Both conditions are met. The ensemble is used.** Intervals resample whole
chemical systems; they are reported only.

**Predictions against outcomes:**
- AUC at least 0.85: 65%; happened, by a wide margin.
- Error at most 0.7 × baseline: 60%; happened.
- Both: 50%; happened.
- Any made metal at 77 K or more with a small spread: 15%; did not happen.

### Reported only

- **The spread says when to trust a prediction.** Test AUC for Tc ≥ 10 K is
  0.994 in the third of rows where the five networks agree most, 0.975 in the
  middle third and 0.879 in the third where they disagree most. Errors are
  0.54, 1.80 and 3.94 K.
- **The top end.** Three test rows have calculated Tc of at least 77 K. All
  three were predicted at 40 K or more.
- **Transfer to Materials Project cells, the plan's check 7.** 549 candidate
  compositions are also in Alexandria's held-out systems. The prediction on
  the Materials Project cell against Alexandria's own value: Spearman 0.84,
  error 0.91 K against 1.80 K for the median, AUC 0.95 for Tc ≥ 10 K (only
  26 of the 549 are above 10 K). [Likely] The PBEsol and Materials Project
  cells do not matter much for ranking.
- **The ranking is coarse below a few K.** [Likely] The network tells 10 K
  from 1 K. It does not tell 0.1 K from 1 K, where its error is about the
  size of the values.

### The candidates: what the first list got wrong

The first list put 28 K to 47 K compounds at the top. They were lithium
oxides and Mg(BC)2, which have band gaps of 1.6 to 3.8 eV in Materials
Project. The "made metals" list is not limited to metallic compounds:
**8,759 of its 26,911 have a Materials Project band gap above zero.** The network has
only ever seen metals, so its output for an insulator is meaningless, and its
spread did not warn of it. 66 of the top 100 were insulators.

Two flags were therefore **added after seeing that list** (not metallic, and
magnetic in Materials Project, because Alexandria's calculations are not
spin-polarised [Likely]). The three flags fixed in advance are unchanged.

| Funnel | Compounds |
| --- | --- |
| Made compounds listed | 26,911 |
| Not metallic in Materials Project | 8,759 |
| Magnetic in Materials Project | 3,793 |
| Flagged by the three pre-set rules | 6,748 |
| Flagged by any rule (overlapping) | 14,070 |
| Already in Alexandria (its own value is used) | 3,445 |
| Matching a known superconductor | 1,892 |
| **New and unflagged** | **9,185** |
| of those: predicted at least 10 K / 20 K / 40 K | **12 / 0 / 0** |

- **The strongest new ones** are 8 to 19 K: La3(CrN3)2 (19.0 K),
  NaVO2 (16.2), SrRhF6 (13.8), CeB4 (13.0), Ce2CrN3 (12.4), AlV6Sn (12.1),
  Li2Ta3N5 (12.0), Cr2GeC (11.2), V6GaGe (11.2), LaCeOs4 (10.7), LiRuN
  (10.5) and Cr3IrN (10.3).
- **Read these with suspicion.** [Likely] Cr, V, Mn and Ce compounds are
  where a calculation without spin fluctuations is most too high, and the
  oxide and fluoride entries (NaVO2, SrRhF6) are probably Mott or magnetic
  insulators that Materials Project calls metals. None is a Hazel
  candidate until someone has read why.
- **Mg(BC)2** (47 K before the metallic flag) is a calculated semiconductor.
  It belongs to the LiBC family of the room-temperature track (C2, Li0.5BC),
  where doping is what would make it metallic. That is a reason to read the
  track, not evidence for it.

### The queue

| Compound | Network Tc (K) | Spread | Alexandria | Pool rank in the ranking of record |
| --- | --- | --- | --- | --- |
| ScTaN2 | 12.8 | 0.18 | entry flagged unreliable | 13 |
| LiRuN | 10.5 | 0.32 | none | 62 |
| Re3Ir | 4.1 | 0.11 | none | 4 |
| Nb2GeC | 2.4 | 0.14 | 1.4 | 12 |
| V2GaN | 1.6 | 0.41 (high) | 0.9 | 6 |
| NaMgPb2 | 1.3 | 0.21 | none | 2 |
| SbTe2 | 0.9 | 0.17 | imaginary mode | 1 |
| SrAl3Ag | 0.2 | 0.07 | 0.14 | 44 |
| LiGaIr | 0.1 | 0.08 | none | 35 |
| TaNbRe2 | 0.0 | 0.07 | 0.0 | 20 |

- **The two rankings answer different questions.** The ranking of record
  scores the chance of a *measured* superconductor. The network scores a
  *calculated* Tc. A compound can be a real superconductor with a small
  calculated Tc, which is what Hazel would then show.
- **What the network says about the queue** [Likely]: most of it is a
  one-to-few-kelvin calculation. Of the pool, ScTaN2 (12.8 K) and LiRuN
  (10.5 K) stand out. Nb2ReRu, TaTiRu2, Ta2ReRu and Ti3Pb already have
  Alexandria values (10.1, 8.7, 6.4 and 5.6 K).
- **No change to the queue is made here.** It is the owner's decision, and
  the Hazel runs test the calculation itself, which the network only
  imitates.

### Where calculated Tc above 77 K sits

Alexandria's own calculations: 24 stable, unflagged compositions with
Allen–Dynes Tc of at least 77 K. They are hydrides (H6KPt, H6NiRb, H6CsNi,
H6CuLi2, H3AlK, H6Mg2Pt, H4Pd and others) and boron–carbon frameworks (C8B2X,
CB, C7B5, B2Be4N2). KInH3 and SrAuH3 of the room-temperature track are
in the same perovskite-hydride family as H3AlK (87.6 K).
- [Certain] These are calculations on cells that are mostly hypothetical.
- [Likely] They are the right place to test how far a PBE Hazel run
  agrees with a published PBEsol calculation at the high end.
- **No made compound in the 26,911 is predicted anywhere near them.**

### What follows

1. The ensemble orders Hazel candidates within the rung-4 and
   room-temperature tracks, as the plan said.
2. A reading of the 12 new compounds at 10 K or more, for magnetism and
   for what is already known, comes before any of them is proposed.
3. The Alexandria high-Tc entries are the next thing to compare with a
   Hazel run, once the benchmark has run on Hazel.
