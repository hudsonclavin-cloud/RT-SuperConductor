# Research roadmap (14 November 2026)

> **How this was made.** Five literature strands (confirmed discoveries,
> calculated datasets, honest evaluation, electron-phonon benchmarks, cheaper
> physics screens) were each researched by one reader and checked by a second;
> three planners then drafted roadmaps from different starting points
> (physics first, data first, sceptic), and a final editor merged them. The
> run finished on 14 November 2026. Its claims about this repository were
> spot-checked against the repository before publishing: the six wrong labels
> (`docs/negative-labels.md`), 431 of 1,057 unstable rows in
> `data/training/physics-target-characteristics.csv`, the physics model's
> AUC of 0.53 on real outcomes (`docs/pipeline-loop.md`), PtPb3Bi at place 221
> of the made-metals order, and the aluminium false alarm of the phonon
> screen (`docs/next-experiments.md`). Claims about other papers rest on the
> readers' work; the "Chance" column is an estimate, not a measurement.

**Terms.** Tc: temperature below which a material superconducts. λ: electron-vibration coupling strength. μ*: an assumed electron-repulsion number that moves Tc a lot. EPW: the code calculating λ; its Wannier interpolation fills in electron energies between calculated points. Surrogate: an ML model imitating a calculation. Negative: a row labelled non-superconductor. Arc melting: fusing elements with an electric arc. AUC: chance a model ranks a superconductor above a non-superconductor (0.5 = coin flip). *Predicted* = model guess; *calculated* = quantum mechanics; *confirmed* = made and measured.

## 1. The short version

Every confirmed ML-led superconductor sits next to a known family and below 10 K, so a higher Tc is not a realistic goal; correctly calling a few unmeasured, already-made metals at a few kelvin is. Your models are not the bottleneck; wrong labels and a list sorted toward cuprate look-alikes are. Finish the EPW pipeline as a yes/no tool whose Tc runs two to three times high. Nothing becomes a discovery without someone to make and measure samples; look for them now.

## 2. What the literature shows

Full text read for every paper cited except the two marked under "Papers read" (supplements partly).

- **Certain: confirmed finds are few, all below 10 K.** Pogue et al. made under 100 compounds: one ~9 K signal, phase never identified (2212.11855). Gibson et al. made 2 of 741 candidates: 3.18 and 4.24 K measured, 14.9 and 7.5 K calculated (2503.20005). YRu3B2 and LuRu3B2 superconduct at 0.81 and 0.95 K (2512.16945). Dropped candidates go uncounted. *Speculative:* a near-family shortlist measured to 2 K might hit 1 in 5 to 1 in 2.
- **Certain: invented structures do not form.** 18 generated compounds were made; all 11 X-rayed came out as disordered alloys or known phases (2509.25186).
- **Likely: careful EPW gives a verdict, not a Tc.** With your toolchain, Bercx et al. found 74 of 82 measured top candidates superconduct, but calculated Tc exceeded measured in 26 of 27 known superconductors (median 2.6 times; rank correlation 0.09, recomputed by the reader), and poor Wannier interpolation lost two of 32 textbook cases (2503.10943). The kagome calculations got λ within 8–37% but Tc 2–3 times high (2512.16945).
- **Certain: formula-only models fail away from known chemistry.** They over-score cuprate-like insulators; held-out-cluster accuracy collapses (2212.11855).
- **Likely: structure-based surrogates are good filters.** Trained on 7,217 calculated metals, BEE-NET rejected 99.4% of calculated non-superconductors and kept 63% of superconductors (random split; judged against calculation, not experiment) (2503.20005, 2509.25186).
- **Likely: room temperature at ambient pressure is "extremely unlikely".** Every high-Tc compound among 20,000+ calculated metals is unstable (2502.18281). The predicted Mg2IrH6 (65–170 K) came out as the insulator Mg2IrH5 at every condition tried (2406.09538).

## 3. Roadmap

| Step | Why | Cost | Chance (estimate) | Success | First action |
|---|---|---|---|---|---|
| 1. Repair labels; freeze a verified test set | Six wrong negatives, two in the locked set; the correction file is empty ~~and the network builder ignores it~~ (correction, 14 Nov: the network builder does read it; see `docs/negative-labels.md`) | 1–2 days | 95% better labels; AUC barely moves | One correction file feeds both tables; nulls carry their lowest temperature | Task A |
| 2. Close the tantalum and niobium gate with a rule written first | Ta λ 1.08 vs 0.96 by the same method; the runbook's "10.9 vs 11.2 K" mixed formulas (9.4 K like-for-like; corrected in the runbook) | Half a day | 65% | Niobium inside the preset margin; band and symmetry checks pass | Task B |
| 3. Download calculated data; clean the λ model | 431 of 1,057 training rows unstable; physics model AUC 0.53 | 1–2 weeks; downloads need a normal network | 40–60% | Lookup before every run; AUC ≥ 0.65 on measured metals | Task C |
| 4. Calibrate EPW on ZrN, HfN, TiN, TaB2 | Bercx: ZrN 12.6 vs 9.3–9.6 K measured; TaB2 17.5 vs below 1.5 K | Hazel, 2–4 weeks | 50% | Your calculated/measured ratio at μ* 0.10 and 0.21 | Copy Bercx's settings (Materials Cloud 2025.39) |
| 5. Five made, never-measured metals | PtPb3Bi, a 2025 find, ranks 221st of 7,693 made metals | Over a month | 55% for one "Tc above 2 K" verdict | Per compound: λ, Tc range, temperature needed | Filter the made-metals list; search the literature |
| 6. Experimental partner | Every confirmation needed arc melting and 2 K measurements | Emails | Unknown | 3–5 compounds made | Show a low-temperature lab the step-5 table |

Steps 1–3 run in parallel this week.

**Where the drafts disagreed:**
- **Labels first:** cheap, certain, and every score depends on them.
- **Drop Nb3Sn and Nb12InSn3:** 2-atom cells are cheaper and published; Nb12InSn3 stands in for a disordered alloy, the kind that failed to form.
- **Your EPW runs are verdicts, not training data:** a handful against 7,217 public calculations.
- **Pressure labels wait:** nothing shows they change the ranking.

## 4. Stop doing

- Presenting the probability × Tc top 200 (139 fluorides, 149 non-metals) as candidates.
- Network work on formula inputs (locked AUC 0.815 vs 0.832).
- Ranking EPW candidates with the formula-only λ model.
- Grading tantalum against λ = 0.757 alone; quoting Tc without a μ* range.
- Chasing hydrides, generated or never-synthesized structures.
- Rejecting materials with the CHGNet phonon screen (it wrongly flags aluminium).

## 5. What would change this roadmap

- **Niobium or calibration compounds miss after converged grids:** stop candidate runs; use published calculations.
- **A surrogate reaches AUC 0.65 on measured metals:** it orders the EPW queue.
- **No partner by spring:** the output is a verdict table and clean data, not a discovery.
- **An ML prediction above 20 K is confirmed at ambient pressure:** revisit the few-kelvin ceiling.

## Papers read

- arXiv:2212.11855: Closed-loop machine learning for discovery of novel superconductors (Pogue et al.; text layer read in full plus the page-3 image; supplementary figure pages not viewed)
- arXiv:2503.20005: Developing a Complete AI-Accelerated Workflow for Superconductor Discovery (Gibson et al., BEE-NET; Supplemental Material is not on arXiv)
- arXiv:2509.25186: Guided Diffusion for the Discovery of New Superconductors (Prakash et al.)
- arXiv:2512.16945: Machine Learning-Guided Discovery of Kagome Superconductors YRu3B2 and LuRu3B2 (Albu Mustaf et al., v2)
- arXiv:2503.10943: Charting the landscape of Bardeen-Cooper-Schrieffer superconductors in experimentally known compounds (Bercx et al.; main text, methods and supplementary sections 2-3 read; per-material supplementary tables not read)
- arXiv:2502.18281: The Maximum Tc of Conventional Superconductors at Ambient Pressure (Gao et al.; main text, supplement not read)
- arXiv:2406.09538: Synthesis of Mg2IrH5: A potential pathway to high-Tc hydride superconductivity at ambient pressure (Hansen et al.; main text, supplement not read)
- arXiv:2307.10728: Sampling the Whole Materials Space for Conventional Superconducting Materials (Cerqueira, Sanna, Marques); not read by the roadmap's readers. Its tantalum, niobium and vanadium values were taken from the paper earlier in this project and are quoted in the runbook
- arXiv:2606.02730: Anharmonic lattice dynamics and superconductivity in strained bulk and surface niobium (Sahoo et al.); not read by the roadmap's readers. Its niobium λ range is a colleague's reading, quoted in Task B (not read in full)

## Status of the roadmap's first tasks (14 November 2026)

- **Task B, the Stage 1 gate:** the pass rule was written into
  `docs/hazel-runbook.md` (section 7, "Stage 1 pass rule") before the
  niobium result was read. Whether its check limits are right is an open
  question for the owner.
- **Task A, label repair:** the six rows are listed in
  `docs/negative-labels.md` for a person to check against the papers.
  `data/training/negative_evidence.csv` is meant to be filled in by a person
  who has read the paper, so no rows were added, and no label was changed.
- **Task C, the λ model's stability filter: measured, not yet applied.**
  `scripts/lambda_stability_check.py` (result in
  `docs/lambda-stability-check.json`) trains the λ and ωlog forests on the
  same grouped folds twice, on all 1,057 rows and on the 626 stable rows, and
  scores both on the same held-out stable rows. Training on stable rows only
  lowers the λ error from 0.271 to 0.220 (difference -0.051, interval -0.068
  to -0.035, resampling 517 element systems), the ωlog error from 105 K to
  63 K and the Tc error from 2.50 K to 2.18 K (-0.32 K, -0.53 to -0.15). It
  does not change how well the model orders materials (Tc rank correlation
  0.51 against 0.54, difference -0.03, -0.08 to +0.02). Trained on all rows,
  the model's λ error is worse than always guessing the median (0.242); on
  stable rows it is a little better. [Likely] So the filter should go in, but
  it fixes the size of the predictions, not the ranking, and the physics
  model should still not choose what EPW checks. The production retrain
  (`scripts/train_physics_targets.py`) downloads the JARVIS file, which this
  cloud machine cannot reach, so the one-line filter is left for a run on a
  normal network.

- **Task A, update (14 November 2026, later):** a blinded check of 80
  training negatives (40 that boosting rates highest, 40 at random) found one
  more confirmed wrong negative, Ti3V22 (V0.88Ti0.12, 6.4 K). Three more alloy
  rows are probably wrong but could not be verified from this machine. See
  "A blinded check of 80 training negatives" in `docs/negative-labels.md`.
  Again, no label was changed.

- **Task B, update (14 November 2026, 14:05 UTC):** niobium on the 6×6×6
  grid fails the pre-registered rule: λ 1.767 against 1.061 to 1.435, ωlog
  120.7 K against 125.0 to 169.2 K; the symmetry check also fails. 38% of
  λ comes from modes below 10 meV, but the run cannot yet say where the
  excess over the reference comes from. Stage 1 is not passed. A stop rule
  was proposed before the next tests were read: if niobium still misses on
  converged grids, stop candidate runs and use published calculations. See
  "Niobium 6×6×6 result" in `docs/hazel-runbook.md`.
