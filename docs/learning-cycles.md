# Learning cycles for the ranking of record

A learning cycle needs new evidence (`docs/active-learning.md`): retraining on
unchanged data only checks that the result reproduces. Here the new evidence is
**label corrections from the literature**: rows of the training table labelled
"not a superconductor" that published measurements show to be superconductors
(and, later, verified nulls). Each cycle adds a batch of corrections, retrains
the ranking of record (gradient boosting, `docs/scratchnet.md`), scores it, and
re-ranks the made metals.

## The owner's rule for corrections (decided 2026-11-16)

The owner chose this rule on 2026-11-16, in place of reviewing each row
before it enters training:

- A correction is applied only when two independent readers confirm it from
  the paper itself, with the DOI and the page, table or figure recorded.
- A third, sceptical reader then tries to refute it.
- Each correction is recorded in `data/training/negative_evidence.csv`, with
  the reviewer column saying it was applied under this rule.
- Each cycle's pull request lists every change, and the owner can veto any
  of them before it is merged.

What "confirm" means, fixed before the first batch was read:
- The paper's own text states superconductivity for that composition, within
  0.15 at.% on every element, in a bulk sample at ambient pressure, with a Tc.
- Or the composition lies inside a single-phase composition series whose
  measured samples bracket it within 2 at.% on both sides, all of them
  superconducting.
- A correction is recorded against a composition already in the table. The
  cycle script refuses a batch that adds, removes or moves a material between
  the development, locked and train-only rows.

The literature search uses the FastTrack Literature Open connector (OpenAlex
and Semantic Scholar, about 250 million records; abstracts, DOI checks against
Crossref and open-access PDF links) and arXiv full text.

## Procedure

```bash
python scripts/build_mp_candidates.py
```

```bash
python scripts/build_scratchnet_data.py
```

```bash
python scripts/run_learning_cycle.py --name cycle1 --previous cycle0
```

The cycle script (`scripts/run_learning_cycle.py`):
1. compares the table with the previous cycle's and lists every changed label;
2. runs grouped five-fold cross-validation of boosting on the development
   rows, with round 2's chosen settings;
3. scores the previous cycle's saved predictions against both answer keys,
   so a change splits into "the answer key changed" and "the model changed";
4. trains the final boosting models and ranks the made metals;
5. compares the order with the previous cycle's.

**The locked rows are neither trained on nor scored.** The locked test set
was used once (`docs/scratchnet-lockbox.json`) and stays that way. Two of the
first batch's corrections (W3Re2C and NbS) sit in it: they change the locked
labels on file, not any reported score.

Results: `docs/learning-cycles/<name>.json` and
`docs/learning-cycles/<name>-made-metals-top50.csv`. Predictions, models and
the full order are under `data/processed/learning-cycles/` and are not
committed. Each cycle takes about 2 minutes on four cores.

What a cycle does not do:
- It retrains boosting only. The network second opinion (`docs/scratchnet.md`)
  stays round 2's. Its saved models refuse a changed table, so
  `run_scratchnet.py rank --name round2` will not run after the first
  correction.
- It does not retune boosting. The settings stay round 2's, so a change comes
  from the labels and nothing else.

## Cycle 0: the baseline, 2026-11-16 02:30 UTC

Cycle 0 is the same procedure on the unchanged table. It reproduces round 2
exactly. [Certain]
- The out-of-fold boosting predictions are identical to
  `round2-oof.npz` on all 9,765 development rows (largest difference 0.0).
- The order of all 7,693 untested made metals is identical, position by
  position, to the one in `docs/bercx-published-check.md`. For example,
  Zr6Ga2Ni is 9th, Ti3SnH 511th and PtPb3Bi 221st.
- Cross-validated AUC on development rows: 0.790 (group-resampled interval
  0.769 to 0.810); non-cuprates 0.789.

## Cycle 1: plan and predictions, written before any cycle 1 result, 2026-11-16 02:40 UTC

**Batch:** the seven wrong negatives listed in `docs/negative-labels.md`
("Wrong negatives found in November 2026" and the blinded check):
- YRu3B2, LuRu3B2, ScS (development rows);
- W3Re2C and NbS (locked rows);
- WB (development, medium confidence);
- Ti3V22 (development; its known evidence is for V0.90Ti0.10, so it enters
  only if a measurement at or bracketing Ti 12 at.% is found).

Each goes through the rule above: two independent readers and a sceptical
verifier, run on 2026-11-16 before the table was rebuilt.

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| Cross-validated AUC on development rows changes by less than 0.005 | 85% |
| The paired difference between this model and cycle 0's, on today's labels, has an interval that includes zero | 85% |
| The made-metals top 50 keeps at least 45 of cycle 0's 50 | 85% |
| Zr6Ga2Ni stays in the top 20 | 90% |
| At least one made metal in a corrected chemistry (Ru–B, Sc–S, W–Re–C, W–B, Nb–S, V–Ti) moves up by 100 places or more | 50% |

Reasoning: five to seven corrected rows among about 9,400 labelled development
rows cannot move a pooled score much. Their effect should show near their own
chemistry.

## Cycle 1: result, 2026-11-16 03:10 UTC

**Evidence.** Two of the seven passed the rule. The full reader and verifier
records are in `data/benchmarks/learning-cycles/cycle1-evidence.json`.

| Row | Readers | Verifier | Why |
| --- | --- | --- | --- |
| YRu3B2 | both confirm | **apply** | Stoichiometric arc-melted sample, about 100% shielding, specific-heat jump; Tc 0.81 K (Phys. Rev. Research 8, 023308, 2026). Two other groups agree (0.63 to 0.73 K) |
| LuRu3B2 | both confirm | **apply** | Same paper; about 90% shielding, specific-heat jump; Tc 0.95 K. One group only |
| ScS | both confirm | do not apply | Weighed in sulfur-rich (1:1.2) and never analysed; the 1974 report is titled "ScS1+x"; the table holds S1.005Sc1 at 5.37 K beside S1Sc1 at 0, so the zero may be a real null at exact 1:1 |
| W3Re2C | split | do not apply | The measured sample was weighed in at C0.8 and EDX finds it about 1 at.% Re-poor: about 3 at.% off the named composition |
| WB | neither | do not apply | The 2012 paper could not be read beyond its title |
| NbS | split | do not apply | Abstract only; superconductivity near Nb:S = 1:1 depends on composition and structure (SuperCon: NbS0.899 3.8 K, NbS1.06 3.3 K, NbS 0) |
| Ti3V22 | neither | do not apply | The known measurement is V0.90Ti0.10, not V0.88Ti0.12; no bracketing series was found |

**Applied:**
- YRu3B2 at 0.81 K and LuRu3B2 at 0.95 K, both from DOI 10.1103/lpqj-7hyg.
- Both are recorded in `data/training/negative_evidence.csv`.
- Both are development rows (folds 3 and 2). Each changed from 0 to 1, and
  nothing else in the table moved.

**Against the predictions written before the result:**

| Prediction | Chance given | Result | |
| --- | --- | --- | --- |
| AUC changes by less than 0.005 | 85% | 0.790 to 0.793 | yes |
| Paired difference against cycle 0 includes zero | 85% | +0.003 (−0.001 to +0.008) | yes |
| Top 50 keeps at least 45 | 85% | **37** | **no** |
| Zr6Ga2Ni stays in the top 20 | 90% | 9th | yes |
| A made metal in a corrected chemistry moves up 100 places or more | 50% | yes (LaB2Ru3: 1,744th to 886th) | yes, but see below |

**A control, run after seeing the top-50 result (not pre-registered).** Why
did two Ru–B labels reshuffle unrelated chlorides by 300 to 500 places?
- Changing every model's random seed by 100, on the unchanged table, changes
  nothing at all. [Certain] These boosting models are deterministic: early
  stopping uses fixed validation rows and nothing is subsampled.
- So three placebos were run instead. In memory only, two random development
  non-cuprate negatives were marked as superconductors at 0.9 K. That is a
  change the size of cycle 1's, with no evidence behind it.

| Run (two labels changed) | AUC | Against cycle 0 | Top 10 kept | Top 50 kept | Top 200 kept | Rank correlation | Zr6Ga2Ni | Ti3SnH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Cycle 1 (two real corrections)** | **0.793** | **+0.003** | 7 | 37 | 161 | 0.990 | 9 | 1,031 |
| Placebo 1 | 0.790 | −0.000 | 7 | 37 | 166 | 0.991 | 6 | 484 |
| Placebo 2 | 0.789 | −0.001 | 5 | 40 | 170 | 0.993 | 3 | 657 |
| Placebo 3 | 0.786 | −0.004 | 6 | 37 | 165 | 0.991 | 17 | 879 |

(Cycle 0 had Zr6Ga2Ni 9th and Ti3SnH 511th.)

**What it shows.**
- **The made-metals order below the first few places is unstable.** [Certain,
  for this measurement]
  - Any two-label change swaps 10 to 13 of the top 50 and 30 to 39 of the top
    200, whatever the labels are.
  - It works by changing how many trees each model keeps (fold 3's Tc model
    went from 352 trees to 163) and so the whole fitted function.
  - Zr6Ga2Ni ranges from 3rd to 17th. Ti3SnH ranges from 484th to 1,031st, so
    its "511th" in `docs/bercx-published-check.md` carries no information
    beyond "around 500 to 1,000".
- **The corrections may still help.** Cycle 1 is the only run whose AUC went
  up; all three placebos went down. With three placebos that is suggestive,
  not shown. [Speculative]
- **The Ru–B moves are not distinguishable from noise.** The 20 Ru–B made
  metals moved up by a median of 40 places in cycle 1, against 19, 41 and
  −18 in the placebos. Prediction 5 came true, but it would have come true by
  chance too.
- **Ranks quoted to the place are not reliable.** The fix is to train many
  slightly perturbed models and report each candidate's rank as a range. It
  is not done here: it changes the ranking of record and needs its own rule
  written first.

**Open question for the owner (this pull request's one question).** ScS and
W3Re2C failed only on composition: each sample was a few at.% off the named
compound and was never analysed. Should a named compound count when its
measured sample was made a little off-ratio on purpose (to make up for loss
during melting) or was found slightly off by analysis? Under today's rule,
no.

## Cycle 2: result, 2026-11-16 03:36 UTC: no correction passed, so no retraining

**Batch:** the six alloy rows that the blinded check of 80 negatives
(`docs/negative-labels.md`) called "most likely also wrong": TlIn19, CdHg19,
Hf(Ti2Nb)8, Ta9Ti8, ReW9 and Zr3V. The procedure was the same as cycle 1's:
two independent readers and a sceptical verifier, with FastTrack, alphaXiv and
open-access PDFs. The records are in
`data/benchmarks/learning-cycles/cycle2-evidence.json`.

| Row | What was found | Why it does not pass |
| --- | --- | --- |
| TlIn19 (In0.95Tl0.05) | Stout and Guttman (Phys. Rev. 88, 703, 1952) measured a 5 at.% Tl single crystal and give its superconducting critical fields; Merriam, Hagen and Luo (Phys. Rev. 154, 424, 1967) measured over 75 In–Tl samples, all superconducting at 2.5 to 3.8 K. **The row superconducts.** | No text that could be read states its Tc, and the sample lists are in the APS full texts |
| CdHg19 (Hg0.95Cd0.05) | Claeson, Luo and Merriam (Phys. Rev. 141, 412, 1966) | Cd dissolves in Hg only up to 1.5 at.%, so 5 at.% Cd is a two-phase sample |
| Hf(Ti2Nb)8 (Ti0.64Nb0.32Hf0.04) | Inside the 42-alloy Ti–Nb–Hf grid of Katoh, Wada and Tachikawa (Denki-Seiko 53, 105, 1982); Nb–Ti–Hf alloys have Hc2(4.2 K) of 8.7 to 11 T (Hawksworth and Larbalestier 1981) | The Tc values are in a figure that could not be read; the nearest named composition is about 4 at.% away |
| Ta9Ti8 (Ta0.53Ti0.47) | Colling, Ralls and Wulff (J. Appl. Phys. 37, 4750, 1966): Tc varies smoothly across the Ta–Ti solid solution, peaking at 8.8 K | No sample list or per-composition Tc readable; the values are for cold-worked wire, which annealing lowers |
| ReW9 (W0.9Re0.1) | W–Re data found are for vapour-deposited samples, whose Tc the authors say is raised above equilibrium | Not an equilibrium bulk value |
| Zr3V (Zr0.75V0.25) | Zr–V solid solutions are reported non-superconducting down to 4.2 K; nearby splat-quenched samples superconduct through a ZrV2 impurity phase | No measurement of this composition; impurity-phase superconductivity |

**No retraining.** Retraining on an unchanged table only checks that the
result reproduces (`docs/active-learning.md`), and cycle 0 already showed
that it does.

**What blocked this batch: access, not evidence.**
- The deciding tables and figures are in *Physical Review* and *Journal of
  Applied Physics* full texts. journals.aps.org is refused by this
  environment's network, and the articles need a subscription anyway.
- TlIn19 in particular is shown to superconduct, but no readable text states
  its Tc.
- With library access, these PDFs would most likely settle TlIn19, Ta9Ti8 and
  Hf(Ti2Nb)8:
  - Stout and Guttman, Phys. Rev. 88, 703 (1952), doi:10.1103/physrev.88.703;
  - Merriam, Hagen and Luo, Phys. Rev. 154, 424 (1967),
    doi:10.1103/physrev.154.424;
  - Gubser, Mapother and Connelly, Phys. Rev. B 2, 2547 (1970),
    doi:10.1103/physrevb.2.2547;
  - Colling, Ralls and Wulff, J. Appl. Phys. 37, 4750 (1966),
    doi:10.1063/1.1708131;
  - Wada, Tachikawa and Kato, IEEE Trans. Magn. 17 (1981),
    doi:10.1109/tmag.1981.1060925.
- PDFs read for this purpose stay out of the repository. Only the evidence
  row (DOI, page, Tc, quote) is recorded.
