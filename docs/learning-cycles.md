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

## Cycle 3: plan and predictions, written before any cycle 3 result, 2026-11-16 04:55 UTC

**How the batch was chosen.**
- Triage: the 500 development non-cuprate negatives with the highest cycle 0
  out-of-fold probability were searched in OpenAlex, one query per element
  system (`scripts/openalex_triage.py`). 196 had a lead, meaning a work whose
  title or abstract mentions superconductivity and names every element
  (`data/benchmarks/learning-cycles/cycle3-triage-leads.json`). Seven of them
  were rows already handled in cycles 1 and 2, which leaves 189.
- Screen: each of the 189 was judged from its leads and a short literature
  search, without the model's score: 16 likely wrong, 52 needing a full text,
  121 unlikely to be wrong.
- The 16 likely-wrong rows went through the rule: two independent readers
  and a sceptical verifier. Records:
  `data/benchmarks/learning-cycles/cycle3-evidence.json`.

**Four pass the rule. All are development rows:**

| Row in the table | Paper | Sample | Tc used |
| --- | --- | --- | --- |
| Nb:0.189;Pb:0.216;Se:0.595 = (PbSe)1.14NbSe2 (fold 3) | Phys. Rev. Materials 7, L041801 (2023), doi:10.1103/PhysRevMaterials.7.L041801 | Bulk CVT single crystal; one misfit stage by XRD | 5.6 K (90% of normal resistance; zero near 4.7 K). Resistive only, and above earlier reports of 1.1 to 2.5 K |
| Fe:0.507;Se:0.222;Te:0.271 = Fe1.03Se0.45Te0.55 (fold 0) | Phys. Rev. B 80, 140511(R) (2009), doi:10.1103/physrevb.80.140511 | Bridgman single crystal, 97% one phase; full shielding | 13 K (onset of ac susceptibility, read from Fig. 4(g)). Composition is nominal |
| Hf:0.110;Nb:0.670;Ti:0.110;Zr:0.110 = [Nb]0.67(HfZrTi)0.33 (fold 4) | PNAS 113, E7144 (2016), doi:10.1073/pnas.1615926113 | Arc-melted bcc bulk alloy | 9.2 K (zero-field-cooled magnetization, Fig. 6) |
| As:0.400;Ca:0.200;Co:0.013;Fe:0.387 (fold 4) | Phys. Rev. B 79, 012504 (2009), doi:10.1103/physrevb.79.012504 | Sn-flux single crystal, CaFe1.94Co0.06As2 by microprobe (0.1 at.% from the row); zero resistance; 55 to 60% shielding | 17 K (onset). Processing-sensitive: some crystals near this composition do not superconduct in bulk |

**The twelve that do not pass:**

| Row | Why not |
| --- | --- |
| (V0.93Cr0.07)3Si | Abstract only; no Tc for x = 0.07 |
| Eu0.785K0.215Fe2As2 | Readers split. A crystal at x = 0.22 is judged not a bulk superconductor by its authors, and the lower bracket is only a partial resistive drop |
| In0.9Tl0.1 | Abstracts only (the APS full texts are blocked) |
| In0.83Tl0.17 | Readers split. One read a 1967 thesis table (Tc 3.29 K at 17.1 at.% Tl), but the thesis has no DOI and its host was blocked for the verifier |
| In0.75Tl0.25 | Nearest readable samples are 0.6 at.% below and 2.2 at.% above: the bracket misses the 2 at.% limit |
| Nb0.4Ti0.6 | Nearest ambient-pressure sample is Nb0.44Ti0.56, 4 at.% away |
| LaPt1.7Au0.3B2C | Abstract only; phase purity is not shown, and the phase is known to be non-stoichiometric |
| Nb3Ge | The bulk powder's composition and phases are unknown; stoichiometric Nb3Ge exists only as films |
| Pb0.925Bi0.075, Pb0.906Bi0.094 | Abstract only (the IOP host is blocked); the nearest readable series points are 2.5 and 0.5 at.% away |
| Ti0.60Nb0.36Ta0.04 | No Tc stated; the paper's own series is 4 at.% away |
| Lu2Ni3Si5 | Abstract only. It states Tc 2.1 K, zero resistance at 2.0 K and a specific-heat jump, but phase purity and figures cannot be checked without the APS full text |

[Likely] Most of the twelve are superconductors in fact (all except perhaps
Eu0.785K0.215Fe2As2 and Nb3Ge). They fail because the deciding page cannot
be read from here.

**Run order.**
1. Three placebos first, on the unchanged cycle 1 table. Each marks four
   random development non-cuprate negatives as superconductors at 0.9 K in
   memory, which is a change the size of this cycle's. Seeds 31, 32 and 33:
   `python scripts/run_learning_cycle.py --name cycle3-placebo-<n> --previous cycle1 --placebo-flips 4 --placebo-seed <30+n>`.
2. Then add the four rows to `data/training/negative_evidence.csv`, rebuild
   both tables, and run
   `python scripts/run_learning_cycle.py --name cycle3 --previous cycle1`.

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| Cross-validated AUC changes by less than 0.005 from cycle 1's 0.793 | 85% |
| The paired difference against cycle 1, on today's labels, has an interval that includes zero | 80% |
| The made-metals top 50 keeps between 30 and 45 of cycle 1's 50 | 75% |
| Zr6Ga2Ni stays in the top 20 | 85% |
| Cycle 3's AUC change is larger than that of each of the three placebos | 45% |

Reasoning:
- Four corrected rows among about 9,400 cannot move a pooled score much.
- Cycle 1 showed that any small label change reshuffles the order below the
  first few places, so the top-50 band is set from the placebos (37 to 40 kept
  with two flips) and widened for four flips.
- With no real effect, cycle 3 would beat all three placebos one time in four.
  I put it a little higher because the corrections are real.

## Cycle 3: result, 2026-11-16 05:05 UTC

**Applied:** the four rows in the plan above, recorded in
`data/training/negative_evidence.csv`. Each changed from 0 to 1 against an
existing development composition (folds 0, 3, 4 and 4). Nothing moved between
splits. The table now has 11,825 superconductors and 3,702 negatives.

**Runs.** All four runs below change four development labels from the cycle 1
table and are scored against cycle 1's saved predictions. The paired
difference is this model minus cycle 1's, both scored on the run's own labels,
with a 300-resample interval.

| Run (four labels changed) | AUC | Paired difference against cycle 1 | Top 10 kept | Top 50 kept | Top 200 kept | Rank correlation | Zr6Ga2Ni | Ti3SnH |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Cycle 3 (four real corrections)** | **0.790** | **−0.0031 (−0.0074 to +0.0003)** | 7 | 36 | 167 | 0.992 | 18 | 494 |
| Placebo 1 (seed 31) | 0.787 | −0.0055 (−0.0102 to −0.0009) | 8 | 37 | 148 | 0.991 | 21 | 731 |
| Placebo 2 (seed 32) | 0.787 | −0.0056 (−0.0101 to −0.0016) | 7 | 38 | 157 | 0.990 | 12 | 708 |
| Placebo 3 (seed 33) | 0.792 | −0.0012 (−0.0060 to +0.0032) | 8 | 39 | 163 | 0.991 | 8 | 763 |

(Cycle 1 had AUC 0.793, Zr6Ga2Ni 9th and Ti3SnH 1,031st.)

**Against the predictions written before the result:**

| Prediction | Chance given | Result | |
| --- | --- | --- | --- |
| AUC changes by less than 0.005 | 85% | 0.793 to 0.790 | yes |
| Paired difference against cycle 1 includes zero | 80% | −0.0031 (−0.0074 to +0.0003) | yes, by 0.0003 |
| Top 50 keeps between 30 and 45 | 75% | 36 | yes |
| Zr6Ga2Ni stays in the top 20 | 85% | 18th | yes |
| Cycle 3's AUC change is larger than each placebo's | 45% | larger than placebos 1 and 2; placebo 3's was larger (−0.0012 against −0.0031) | **no** |

**What it shows.**
- **The real corrections cannot be told apart from placebos.** [Certain, for
  these runs]
  - On the paired difference, the real batch ranked first of four runs in
    cycle 1 (+0.0031 against −0.0004, −0.0013 and −0.0043) and second of four
    in cycle 3.
  - If the corrections were no better than four random flips, the two ranks
    would add up to 3 or less (first and first, first and second, or second
    and first) 3 times in 16 (19%).
  - That assumes the four runs in each cycle are interchangeable. They may
    not be: the real rows came from the 500 negatives the model rated most
    likely to superconduct, while the placebo rows were drawn at random.
  - The test has almost no power. Even first place in both cycles would
    happen 1 time in 16 (6%) by chance, so this design can only rule out a
    large effect. Reaching 5% in one cycle would take about 19 placebos.
- **Any small label change, real or random, moves AUC by up to about 0.006.**
  - Refitting on an unchanged table moves it by exactly 0
    (`docs/learning-cycles/control-seed100.json`). The models are
    deterministic.
  - The four-flip placebos moved it by −0.0012 to −0.0056, and the two-flip
    placebos of cycle 1 by −0.0004 to −0.0043.
  - [Likely] A batch's effect on pooled AUC has to be well above that spread
    to be seen. A handful of corrections among about 9,400 labelled rows is
    not.
  - The sign says little. All six placebos came out below zero, and so did
    the real cycle 3 batch (−0.0031). [Likely] Part of that is regression to
    the mean: cycle 1's 0.793 was the highest of the five runs in cycles 0
    and 1, and round 2 chose these settings as the best of its grid. So any later refit
    is expected to score a little below it.
- **The ranking of the made metals is as unstable as in cycle 1.**
  - Zr6Ga2Ni ranges from 8th to 21st across these four runs, and from 3rd to
    21st across all eight runs so far.
  - Ti3SnH ranges from 494th to 763rd here, and from 484th to 1,031st overall.
  - Ranks quoted to the place still carry no information beyond their range.
- **Two of the four corrections are in the iron-based family**
  (Fe1.03Se0.45Te0.55 and Ca(Fe,Co)2As2). The made-metals list leaves that
  family out, so these two act on the list only through what the models
  learn from shared features.

**What the triage measured: wrong negatives in the 500 most suspicious rows.**

| Step | Rows | Share of the 500 |
| --- | --- | --- |
| Most suspicious development non-cuprate negatives searched | 500 | 100% |
| With an OpenAlex lead | 196 | 39% |
| New (not handled in cycles 1 and 2) and screened | 189 | 38% |
| Screened as likely wrong | 16 | 3.2% |
| Passed the rule and applied | 4 | 0.8% |
| Screened as needing a full text | 52 | 10% |

- [Likely] Shown wrong: at least 4 of the 500 (0.8%).
- [Likely] Wrong on the physics: about 14 (2.8%). That is the 16 minus
  Eu0.785K0.215Fe2As2 and Nb3Ge.
- [Speculative] If a third of the 52 full-text rows also turn out wrong, the
  rate in this slice is about 6%.
- The 304 rows with no lead are not verified nulls.
- [Speculative] Among the 1,976 development non-cuprate negatives, that
  suggests tens to about a hundred wrong labels, if the less suspicious rows
  have fewer errors.
  - That premise is untested. Cycle 1's two confirmed errors had
    out-of-fold p 0.60 and 0.67, below this triage's cutoff of 0.77.
  - The 1,372 cuprate negatives and the 358 locked or train-only non-cuprate
    negatives were not sampled at all.
- Cost:
  - 387 OpenAlex searches for 367 element systems (a system whose rows
    include both compounds and alloys got one query of each kind), about
    0.39 USD of a 1 USD daily budget;
  - screening and checking took 29 agents and about 80 minutes.

**The bottleneck is mostly reading, not finding.** Of the twelve rows that
failed:
- seven because the deciding text could not be read from here (APS, IOP and a
  thesis host are blocked, or only the abstract was read):
  (V0.93Cr0.07)3Si, In0.9Tl0.1, In0.83Tl0.17, the two Pb–Bi rows,
  LaPt1.7Au0.3B2C and Lu2Ni3Si5;
- three because no readable sample was close enough, or none had a stated Tc:
  In0.75Tl0.25, Nb0.4Ti0.6 and Ti0.60Nb0.36Ta0.04;
- two on the evidence itself:
  - Eu0.785K0.215Fe2As2 was read in full. A crystal 0.1 at.% away is judged
    by its own authors not to be a bulk superconductor.
  - For Nb3Ge, the bulk A15 phase does not reach 25 at.% Ge, and
    stoichiometric Nb3Ge exists only as films.

`data/benchmarks/learning-cycles/cycle3-library-requests.csv` lists the papers
that could settle the 52 full-text rows and ten of the twelve failed rows,
ordered by how many rows each could settle. The first five would cover 22
rows:
- Superconductivity and lattice parameters in the Zr–Mo, Zr–W, Hf–Mo and Hf–W
  systems, doi:10.1016/0022-5088(70)90162-1 (7 rows);
- Savitskii, Baron and Efimov, on vanadium in Nb–Zr alloys, Izv. Akad. Nauk
  SSSR, Metally 3 (1966). It is in Russian, has no DOI, and OSTI 7297509 holds
  only its bibliographic record (5 rows);
- Merriam, Hagen and Luo, Phys. Rev. 154, 424 (1967), In–Tl (4 rows,
  including cycle 2's TlIn19);
- Low-temperature specific heats of Zr–Ti, Zr–Hf and Zr–Sc alloys, Phys. Rev.
  168, 715, doi:10.1103/physrev.168.715 (3 rows);
- Superconductivity in Cr3Si-type ternary phases with niobium and group VIII
  metals, Phys. Rev. 137, A1438, doi:10.1103/physrev.137.a1438 (3 rows).

**Corrections to the plan section above (found by an independent review
before this pull request; the plan's text is left as written):**
- "one query per element system": it was one query per element system and
  kind, 387 queries for 367 systems.
- "They fail because the deciding page cannot be read from here": true of
  seven of the twelve. See the breakdown above.
- Pb–Bi row: "the nearest readable series points are 2.5 and 0.5 at.% away".
  The 0.5 at.% point (Gandhi et al., 2017) is on the same blocked IOP host and
  was not read. The nearest point that was read (5 at.% Bi, Aegerter et al.)
  is 2.5 at.% from Pb0.925Bi0.075 and 4.4 at.% from Pb0.906Bi0.094.
- LaPt1.7Au0.3B2C and Lu2Ni3Si5 rows: **both readers confirmed both rows from
  the abstract.** Only the verifier refused them, because the instructions I
  gave the verifier listed an abstract-only reading as grounds to refuse.
  - The owner's rule does not contain that condition.
  - The verifier did not refute either row on the physics.
  - Phase purity, given as a reason in the plan, is not a condition of the
    rule's first route.
  - See the open question below.
- Fe1.03Se0.45Te0.55 row: the text states bulk superconductivity for x = 0.45
  but prints no Tc for it. The 13 K is the verifier's reading of Fig. 4(g),
  checked against the value printed for x = 0.5. The rule allows a figure.
- Ca(Fe,Co)2As2 row:
  - The table's zero comes from the source row Ca1Fe1.936Co0.064As2
    (SuperCon, Tc 0). The paper's crystal, CaFe1.94Co0.06As2, is 0.08 at.%
    from it, and is already a separate superconducting row in the table at
    15.4 K.
  - A Sn-flux crystal at this row's own composition (x_EDX = 0.032,
    Abdel-Hafiez et al.) keeps the SDW. So the zero may be a measured null
    for that sample.
  - The correction passes the rule. It is also the one most open to a veto.
  - The evidence file now records the source formula rather than an
    equivalent spelling of it.

**Open question for the owner (this pull request's one question).**
- The rule asks two readers to confirm "from the paper itself, with the DOI
  and the page, table or figure recorded". It does not exclude abstracts. My
  instructions to the verifier did.
- Two rows hang on that difference. In both, both readers confirmed from the
  abstract, the DOI is registered, the composition is exact, and the
  verifier found no physical refutation:
  - **Lu2Ni3Si5** (Phys. Rev. B 50, 13879, 1994, p. 13879): Tc 2.1 K, zero
    resistance at 2.0 K, and a specific-heat jump "which indicates bulk
    superconducting ordering".
  - **LaPt1.7Au0.3B2C** (Phys. Rev. B 50, 4216, 1994, p. 4216): Tc 10.2 K
    from heat capacity, with no resistivity or shielding in the abstract.
    Related papers describe the phase as non-stoichiometric.
- **The question:** should the verifier's refusal of abstract-only readings
  stand?
  - If not, both rows enter the next cycle.
  - If you want a middle line (for example, an abstract counts only with a
    bulk signature), note that this cycle's four full-text corrections were
    each accepted on one or two kinds of evidence:
    - (PbSe)1.14NbSe2: resistive only;
    - Nb0.67(HfZrTi)0.33: magnetization only;
    - Fe1.03Se0.45Te0.55: susceptibility and muon data;
    - Ca(Fe,Co)2As2: zero resistance and partial shielding.
  - The other abstract-only failures ((V0.93Cr0.07)3Si, In0.9Tl0.1 and the
    two Pb–Bi rows) fail either way: their abstracts give no Tc for the
    composition.

## Abstract-only rows: the owner's answer and what follows, written 2026-11-16 07:51 UTC

**The owner's answer to cycle 3's open question (2026-11-16): option B.** The
verifier no longer refuses a row only because it was read from an abstract.
It refuses on substance: a concrete reason to doubt that the composition, as a
bulk sample at ambient pressure, superconducts, or that the sample matches the
row.

**Re-check under that instruction** (one sceptical verifier, 2026-11-16; it
could not read either full text, and re-read both abstracts):

| Row | Verifier | My decision | Why |
| --- | --- | --- | --- |
| Lu2Ni3Si5 (Phys. Rev. B 50, 13879, 1994) | apply | **apply, with the next batch** | The abstract states Tc 2.1 K, zero resistance at 2.0 K and a specific-heat jump of ΔC/γTc = 1.1 for the exact composition. No contrary report was found. Not replicated by another group |
| LaPt1.7Au0.3B2C (Phys. Rev. B 50, 4216, 1994) | apply | **hold** | The abstract gives Tc 10.2 K for the nominal formula. Related papers describe the phase as Ln(Pt,Au)2+δB2C, off-stoichiometric. A shift of 0.1 in Pt + Au moves the composition by about 1 at.%, past the rule's 0.15 at.%, and an abstract cannot settle the measured composition. That is a reason of substance under option B. It waits for the full text |

The owner left these two decisions to my discretion on 2026-11-16.

**Why Lu2Ni3Si5 is not applied today.** One changed label cannot be measured
(cycles 1 and 3), and every table change invalidates the saved comparisons
that the stability test, rung 4 and the five-children experiment use. It is
recorded here as approved and goes into `data/training/negative_evidence.csv`
with the next batch of corrections.

## Cycle 4: plan and predictions, written before any cycle 4 result, 2026-11-16 18:48 UTC

**Main idea:** cycles 1 to 3 corrected labels. This cycle adds rows: the
compositions that the NIMS SuperCon file has and the training table lacks.
The owner asked for it on 2026-11-16, after the training-data test
(`docs/training-data-test.md`) passed its rule.

**The assumption I am challenging.** That test's +0.018 does not carry over.
It trained on 3,035 extra NIMS compositions; only 851 can be added here
(1,754 are this table's locked rows, 430 are tied to locked or several-fold
groups). [Likely] 6% more rows of the same kind of data will change the
ranking's scores by less than can be measured.

**What is added.** New evidence here is a different source's records, not
readings of papers, so the owner's two-reader rule for corrections does not
apply and no existing label changes.

- A NIMS composition is added only if it is absent from the whole table.
- It is tied to the table's groups the way the table ties its own rows: by
  major elements (5 at.% or more) and by near twins (within 0.02 on every
  element), chained through other added rows.
- Tied to a locked group: not added. Tied to groups in more than one fold or
  split: not added. Tied to one development group: it takes that group's
  fold. Tied to nothing: it forms a new group and takes a fold from a hash of
  the group's name.
- Label: 1 if any NIMS record reports superconductivity, else 0. Tc: the
  median of the distinct positive values.
- The 55 shared compositions where the two sources disagree are listed in
  `docs/learning-cycles/nims-disagreements.csv` for a later correction batch
  under the two-reader rule. They are not changed here.

**Guards.**

- The table is not rebuilt. Every existing row keeps its label, split and
  fold; the cycle script's layout check still runs.
- The added rows are a separate training-only block
  (`data/processed/learning-cycles/nims-additions.csv.gz`, not committed; its
  checksum and counts are in `docs/learning-cycles/nims-additions.json`).
- An added row is fitted only in rounds where its fold is not held out, and
  never when its group is in that round's early-stopping rows.
- The locked rows are neither trained on nor scored.

**Two measurements.**

| | Question | Rows scored | How |
| --- | --- | --- | --- |
| E1 | Does adding the rows help on new compounds of the kind NIMS has and the table lacks? | The 851 added rows, each scored only by the round that held its fold out | AUC with the other folds' added rows in training, minus AUC without any; 1,000 resamples of groups |
| E2 | Does it hurt at home? | The development rows, as in every cycle | Cycle 4 minus cycle 3, paired, non-cuprates |

- [Certain] Neither measurement is a blind test. E1's rows come from the
  same source as what is added, so it is the favourable case.

**Placebos.** The models are deterministic, so a changed seed is no control.
Three placebos each add 851 randomly chosen development rows a second time
(seeds 41, 42, 43): a change of the same size that carries no new
information. They show how much the made-metals order moves from re-weighting
alone.

**Rule, fixed now.**

| Outcome | Reading |
| --- | --- |
| E1's 95% interval is above zero **and** E2's median difference is not below −0.005 | Cycle 4 becomes the ranking of record. Shortlist 2 was ranked with cycle 3 and is not re-ranked in this pull request. |
| Anything else | Cycle 3 stays the ranking of record. The added-rows file is kept. |

**Run order.**
1. `python scripts/build_nims_additions.py`
2. `python scripts/run_learning_cycle.py --name cycle4-placebo-<n> --previous cycle3 --duplicate-rows 851 --placebo-seed <40+n>` for n = 1, 2, 3
3. `python scripts/run_learning_cycle.py --name cycle4 --previous cycle3 --additions data/processed/learning-cycles/nims-additions.csv.gz`

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| E1's 95% interval is above zero | 40% |
| E2: non-cuprate AUC changes by less than 0.005 from cycle 3 | 85% |
| Cycle 4 is adopted under the rule | 35% |
| The made-metals top 50 keeps between 25 and 45 of cycle 3's 50 | 70% |
| Zr6Ga2Ni stays in the top 20 | 80% |
| Cycle 4 keeps fewer of cycle 3's top 50 than each of the three placebos does | 55% |

## Cycle 4: result, 2026-11-16 19:34 UTC

**Main result:** the rule's two conditions are met, so cycle 4 is the
ranking of record **if the owner does not veto it**. The gain is on new
compounds of the kind NIMS has; at home the scores are slightly lower, inside
the range the placebos show. It passed the home condition narrowly.

Numbers are from `docs/learning-cycles/cycle4.json` and the three
`cycle4-placebo-<n>.json` files. The table itself did not change: no row
moved, appeared, disappeared or changed label.

**E1: the 851 added rows, each scored by the round that held its fold out.**

| Rows | AUC without any added rows | AUC with the other folds' added rows | Difference | 95% interval |
| --- | --- | --- | --- | --- |
| All 851 (185 nulls) | 0.6790 | 0.7643 | +0.0852 | +0.0414 to +0.1317 |
| Non-cuprates, 812 (183 nulls) | 0.6884 | 0.7734 | +0.0851 | +0.0434 to +0.1292 |

**E2: the development rows, cycle 4 against cycle 3 (non-cuprates).**

| | Cycle 3 | Cycle 4 | Paired difference (median, 95% interval) |
| --- | --- | --- | --- |
| AUC | 0.7891 | 0.7856 | −0.0037 (−0.0102 to +0.0023) |
| Tc error, log scale (lower is better) | 0.3871 | 0.3964 | +0.0086 (−0.0025 to +0.0212) |

**Placebos: 851 development rows duplicated, against cycle 3.**

| Run | Non-cuprate AUC | Paired AUC difference (median) | Top 50 kept | Zr6Ga2Ni's place (was 18) |
| --- | --- | --- | --- | --- |
| Placebo 1 | 0.7857 | −0.0030 | 36 | 6 |
| Placebo 2 | 0.7917 | +0.0027 | 34 | 14 |
| Placebo 3 | 0.7879 | −0.0010 | 37 | 8 |
| **Cycle 4** | **0.7856** | **−0.0037** | **29** | **63** |

**Against the rule.**

| Condition | Result | Met? |
| --- | --- | --- |
| E1's 95% interval above zero | +0.0434 to +0.1292 | Yes |
| E2's median difference not below −0.005 | −0.0037 | Yes, by 0.0013 |

**The strongest argument against adopting it.**

- [Certain] At home every measure moved the wrong way: AUC −0.0037, Tc error
  +0.0086, Brier score 0.1407 against 0.1392. None is outside its interval,
  and the AUC change is lower than all three placebos'.
- [Certain] E1 is the favourable case, as the plan said: its rows come from
  the same source as what was added. It shows the model learned NIMS's extra
  chemistry, not that it ranks unseen candidates better.
- [Likely] The honest summary is a trade: clearly better on about 850
  compounds of a kind the table lacked, and no measurable change, or a slight
  loss, on the 8,598 it already had.

**What it changes.**

- [Certain] The made-metals order moved more than re-weighting alone moves
  it: 29 of cycle 3's top 50 kept, against 34 to 37 for the placebos; 5 of the
  top 10 kept.
- [Certain] Zr6Ga2Ni fell from 18th to 63rd and Zr6Ga2Co from 19th to 70th.
  In the placebos they rose. So the added rows did this, not noise.
  [Speculative] 11 added rows are at least half zirconium and 6 of them are
  nulls (Zr-Rh-O), which would pull zirconium-rich compounds down.
- [Certain] Zr6Ga2Ni stays the Hazel benchmark. It was chosen because a
  published calculation exists to compare with, not for its place in the order.
- Not changed here: shortlist 2 and the rung 4 scripts still use cycle 3's
  models. Switching them belongs to the next shortlist.

**Checks.**

- The models fitted without added rows inside this run reproduce cycle 3's
  saved predictions exactly (largest gap 0.0).
- Added rows fitted per round: 599 to 677 of 851 in cross-validation (the
  rest were of the held-out fold or tied to that round's early-stopping
  groups); final models were fitted on 12,558 to 13,007 rows.

**Against the predictions written before the result:**

| Prediction | Chance given | Result |
| --- | --- | --- |
| E1's 95% interval is above zero | 40% | Yes (+0.043 to +0.129) |
| E2: non-cuprate AUC changes by less than 0.005 | 85% | Yes (−0.0035) |
| Cycle 4 is adopted under the rule | 35% | Yes |
| Top 50 keeps between 25 and 45 of cycle 3's | 70% | Yes (29) |
| Zr6Ga2Ni stays in the top 20 | 80% | No (63rd) |
| Cycle 4 keeps fewer of the top 50 than each placebo | 55% | Yes |

- I expected the added rows to matter less than they did. E1's gain is four
  times what I guessed, and I did not foresee the benchmark compound moving.

**Open question for the owner (this pull request's one question).** The rule
says adopt. Veto it if a slight loss at home is not worth a gain that is only
shown on NIMS's own kind of compound. Merging this pull request accepts
cycle 4 as the ranking of record.

## Cycle 4 accepted by the owner, written 2026-11-16 20:08 UTC

The owner merged the cycle 4 pull request on 2026-11-16 at 20:07 UTC
(https://github.com/hudsonclavin-cloud/RT-SuperConductor/pull/59). **Cycle 4
is the ranking of record.**

What that means in files:

- The models of record are `data/processed/learning-cycles/models/cycle4/`
  (five, one per development fold held back), trained on the table plus the
  851 added rows in `data/processed/learning-cycles/nims-additions.csv.gz`
  (sha256 `e7507a3f…`, full value in `docs/learning-cycles/nims-additions.json`).
- The order of record for made metals is
  `docs/learning-cycles/cycle4-made-metals-top50.csv`.
- The next cycle compares with `--previous cycle4` and must pass
  `--additions` again, or it would silently drop the added rows.

Still on cycle 3, on purpose:

- Shortlist 2 (`docs/rung4-shortlist2.md`) and the Hazel packages were chosen
  with cycle 3 and are not re-ranked. `scripts/rung4.py` reads cycle 3's
  models and predictions; the next shortlist switches it.
- The blind test and the children experiments never used the ranking of
  record's models for selection, so nothing there changes.

## Cycle 5: plan and predictions, written before any cycle 5 result, 2026-11-21 01:54 UTC

**Why now.** On 2026-11-21 the owner asked for training to continue without
stopping while tokens last. [Certain] Retraining on an unchanged table adds
nothing: the models are deterministic and a cycle takes two minutes. A cycle
needs new evidence, so the tokens go to readers who check labels against
the papers. This is cycles 1 to 3 again, on the rows not yet checked.

**Where the batch comes from.**

| Lane | Rows | How |
| --- | --- | --- |
| 1. Rows never checked | The 1,476 development non-cuprate negatives that cycle 3's triage did not reach, most suspicious first by cycle 4's out-of-fold probability | `scripts/openalex_triage.py --previous cycle4`, then a screen of every row with a lead, blind to the model's score |
| 2. Cycle 3's backlog | The 39 papers in `data/benchmarks/learning-cycles/cycle3-library-requests.csv`, which could settle 52 rows that needed a full text and the rows that failed for lack of access | Tried again with connectors that cycle 3 did not have (a page-fetch service, Wiley's full-text search) |

- Lane 1 starts with the 628 rows at 0.5 or above and goes on down the
  list while tokens last.
- Lu2Ni3Si5, approved on 2026-11-16, is applied with this batch.
  LaPt1.7Au0.3B2C stays held unless its full text is read.

**The rule is unchanged:** the owner's rule of 2026-11-16, with option B for
abstracts. Two independent readers confirm from the paper; a third,
sceptical reader tries to refute; the owner can veto any row before merging.

**Two changes from cycle 3, stated before any result.**

- Rows the screen calls "needs a full text" go to readers now, not to a
  waiting list, because readers have more ways to reach a text.
- When a paper states that a composition was measured and did not
  superconduct, readers record that null with its lowest temperature. Those
  records are kept in the cycle's evidence file. They change no label and
  are not added to `data/training/negative_evidence.csv` in this cycle.

**What cannot be read from here.** Only arXiv and OpenAlex answer directly.
Other hosts go through the page-fetch service, which returns text but not
scanned pages. Papers behind a paywall are not read; nothing is taken from
sites that copy papers without permission.

**Run order.**
1. Three placebos on the cycle 4 table: as many random development
   non-cuprate negatives flipped as there are corrections (seeds 51 to 53),
   each with `--additions`.
2. Add the corrections to `data/training/negative_evidence.csv`, rebuild
   both tables, and run
   `python scripts/run_learning_cycle.py --name cycle5 --previous cycle4 --additions data/processed/learning-cycles/nims-additions.csv.gz`.
3. Corrections found after that run wait for cycle 6.

**What becomes the ranking of record.** Cycle 5, if the owner merges its
pull request. A correction is a fact about a label, not a modelling choice,
so no score decides it; the scores below only describe the change.

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| Corrections that pass the rule, both lanes together: between 5 and 20 | 60% |
| Fewer than 5 | 25% |
| Lane 2 gives at least 2 of them | 50% |
| Cross-validated non-cuprate AUC changes by less than 0.005 from cycle 4's 0.7856 | 85% |
| The paired difference against cycle 4 has an interval that includes zero | 80% |
| The made-metals top 50 keeps between 30 and 45 of cycle 4's 50 | 70% |
| Cycle 5's AUC change is larger than that of each of the three placebos | 40% |

Reasoning:
- Cycle 3 searched the 500 most suspicious rows and four passed. These rows
  are less suspicious, but there are three times as many and the
  needs-full-text rows are now read.
- A dozen corrected rows among about 9,400 cannot move a pooled score much.
  The gain to expect is a cleaner top of the order, not a higher AUC.

### Cycle 5: a note on the readers, 2026-11-21 02:41 UTC (before any correction was applied)

- [Certain] The reader agents started on one model (Claude Fable 5.2). At
  about 02:40 UTC that model's usage limit stopped roughly 30 of them
  mid-task. They are being rerun on Claude Opus 5.5 from the saved results:
  agents that had finished are kept, and only the stopped ones run again.
- So one row's readers may run on different models. The rule asks for two
  independent readings and a sceptic; it does not ask for one model. Each
  reader still works only from the task file and the literature.
- Nothing here changes the rule, the batch, or the predictions above.

## Cycle 5: result, 2026-11-21 05:10 UTC

**Main result:** seven labels corrected. As predicted, the overall score does
not move beyond noise: the three placebos (seven random flips each) moved it
a little more. The corrections do move the made metals of their own
chemistries up, more than the placebos do.

**The check, by numbers.**

| Stage | Rows |
| --- | --- |
| Development non-cuprate negatives cycle 3 did not reach | 1,476 |
| Searched in OpenAlex (the rest hit the free daily budget; least suspicious first) | 1,129 |
| With at least one lead | 457 |
| Screen: unlikely to be wrong / needs a full text / likely wrong | 412 / 31 / 14 |
| Backlog papers from cycle 3 retried (lane 2) | 39 papers, about 65 rows |
| Reader A confirmed / verified null / not confirmed | 16 / 5 / 89 |
| Reader B confirmed (of reader A's confirmations) | 15 |
| Sceptic: apply / do not apply | **7** / 8 |

Records: `data/benchmarks/learning-cycles/cycle5-evidence.json` (screen,
both readers and the sceptic for every row; short quotes only),
`cycle5-triage-leads.json` (lead titles and DOIs) and
`cycle5-library-requests.csv` (43 papers that could settle 65 rows the
readers could not open from here).

**Applied** (recorded in `data/training/negative_evidence.csv`; all are
development rows, each changed from 0 to 1, nothing moved between splits):

| Row | Paper | Sample | Tc used (K) | Read from | Where the zero probably came from |
| --- | --- | --- | --- | --- | --- |
| BaGa2 (fold 4) | Phys. Rev. B 105, 214517 (2022) | Self-flux single crystals; zero resistance, diamagnetism, field-suppressed specific-heat jump | 0.36 | full text (arXiv) | Specific heat and transport that stopped at 1.8 K |
| WBe2 (fold 4) | J. Alloys Compd. 1010, 177756 (2025) | Arc-melted bulk, 90-95% WBe2 with W metal; resistive onset, bulk specific-heat jump | 1.05 | full text (arXiv) | A 1963 search that stopped at 1.68 K |
| NbGe2 (fold 4) | Phys. Rev. B 102, 064507 (2020) | Single crystals, single-phase C40; >100% shielding, specific-heat jump. Also arc-melted 1978 (2.09 K) and 2024 polycrystals | 2.0 | full text (arXiv) | Hardy and Hulm 1954, "normal to 1.2 K" (contradicted by every later bulk report) |
| NbSi2 (fold 2) | J. Low Temp. Phys. 92, 335 (1993) | C40 single crystals, specific heat | 0.13 | **abstract only** (option B) | Searches that stopped at 1.2 K |
| FeS (fold 4) | Phys. Rev. B 93, 094522 (2016) | Single crystals of **tetragonal (mackinawite) FeS**, full site occupancy; shielding 60-90%, specific-heat jump. Replicated by several groups | 3.9 | full text (arXiv) | NIMS 2014 papers on Fe1.08S, 1.9 at.% off, measured to 2 K |
| Nb0.4Hf0.2Zr0.2Ti0.2 (fold 4) | Acta Mater. 261, 119420 (2023) | Arc-melted bulk; specific-heat jump 3.3 | 7.99 | **abstract only** (option B) | Probably Raevskii et al. 1969/1971 (not readable) |
| Lu2Ni3Si5 (fold 4) | Phys. Rev. B 50, 13879 (1994) | Polycrystal; zero resistance at 2.0 K, specific-heat jump | 2.1 | **abstract only** (option B) | Not identified. Approved on 2026-11-16; confirmed again by this cycle's readers and sceptic |

**Refused by the sceptic although both readers confirmed** (reasons of
substance, as option B requires):

| Row | Reason |
| --- | --- |
| MoB2 | The one positive sample's composition was not determined and boron-rich Mo-B superconducts in the same range; four other MoB2 samples (two papers plus the same paper's arc-melted one) found nothing |
| CeFeAsO0.94F0.06 | On the edge of the dome; two primary papers measured samples written this way and found no superconductivity; the positive sample's F content is uncertain by more than the 0.15 at.% limit |
| Ir3Te8 | The measured crystal refines to about Ir0.80Te2, 1.4 at.% off; incomplete transition; not reproduced |
| NbB2.24 | Bracket from a series described only as ">94% NbB2"; two other groups find only trace superconductivity at this boron content |
| EuFe2(As0.64P0.36)2 | Single crystals bracketing the row (EDX compositions) do not superconduct; the polycrystal bracket shows partial drops only |
| Ir0.75Pt0.25Te2 | The only direct measurement at this composition finds nothing above 0.3 K, below what the bracket implies |
| Fe1.03Se | McQueen et al. report this exact composition as a bulk non-superconductor to 0.6 K in two papers; the positive match rests on a refined occupancy at the limit |
| Bi2OS2 | Superconducts only after treatment at 2 GPa; made at ambient pressure it does not. Cycle 3 flagged pressure-processed samples as a rule question for the owner, still open |

Verified nulls found by readers (recorded in the evidence file; no label
changes): Al3Nb to 0.012 K, HfW2 to 0.03 K, LaB2C2 to 1.8 K, Ti2Rh to 1.2 K,
and one Eu-Fe-As-P row to 2.0 K.

**Scores on the development rows, non-cuprates** (paired difference: this
model minus cycle 4's, both on today's labels, 300 resamples):

| Run (seven labels changed) | AUC | Paired AUC difference (median, 95% interval) | Tc error, log scale | Top 10 / 50 / 200 kept | Rank correlation | Zr6Ga2Ni (was 63) |
| --- | --- | --- | --- | --- | --- | --- |
| **Cycle 5 (seven corrections)** | **0.7852** | **+0.0005 (−0.0058 to +0.0057)** | 0.3904 (was 0.3964) | 9 / 38 / 165 | 0.991 | 32 |
| Placebo 1 (seed 51) | 0.7868 | +0.0009 | 0.3888 | 9 / 38 / 166 | 0.991 | 56 |
| Placebo 2 (seed 52) | 0.7884 | +0.0033 | 0.3900 | 8 / 38 / 164 | 0.991 | 47 |
| Placebo 3 (seed 53) | 0.7876 | +0.0023 | 0.3904 | 8 / 39 / 167 | 0.990 | 75 |

**Made metals in the corrected chemistries** (sharing at least two elements
with a corrected row; 175 compounds): median rise **102 places** in cycle 5,
against −29, +17 and +3 in the three placebos. The four in exactly a
corrected element system rose a median of 514 places (placebos 51, 254, 27):
LuNiSi2 1,218th to 843rd, BaGa4 2,209th to 1,557th, Nb3Si2 2,962nd to
2,703rd, LuNi2Si2 4,139th to 3,357th.

**Against the predictions written before the result:**

| Prediction | Chance given | Result |
| --- | --- | --- |
| Corrections that pass, both lanes: 5 to 20 | 60% | Yes (7) |
| Fewer than 5 | 25% | No |
| Lane 2 gives at least 2 | 50% | Yes, but only by counting Lu2Ni3Si5, approved before; 1 new |
| AUC changes by less than 0.005 from cycle 4 | 85% | Yes (−0.0004) |
| The paired difference includes zero | 80% | Yes |
| Top 50 keeps 30 to 45 of cycle 4's | 70% | Yes (38) |
| Cycle 5's AUC change is larger than each placebo's | 40% | **No**: smaller than all three |

**What it shows.**

- [Certain] Seven corrected rows among 9,399 do not change a pooled score.
  The placebos moved it more. As the plan said, no score decides a label
  correction.
- [Likely] The corrections do what they should locally: made metals sharing
  chemistry with a corrected row rose about 100 places, where random flips
  of the same size moved them by −29 to +17.
- [Certain] In three of seven (BaGa2, WBe2, NbSi2) the zero comes from a
  search that stopped above the Tc found later (floors of 1.2 to 1.8 K
  against Tc of 0.13 to 1.05 K). For FeS it comes from a sample 1.9 at.%
  off; for NbGe2 from a 1954 null that every later bulk report contradicts.
  For the other two the source was not found. SuperCon's "not a
  superconductor" often means "not above the lowest temperature reached".
- [Speculative] Six of the seven are in fold 4. Readers never saw the fold,
  and fold 4 supplied only 79 of the 457 rows sent to the screen. By chance
  this is about a 0.5% event. Fold 4's negatives also carry the highest mean
  out-of-fold suspicion of the five folds (0.60 against 0.50 to 0.55). One
  reading is that fold 4 happens to hold more wrong zeros. Cycle 6 could test
  that by screening fold 4's remaining negatives first.
- Zr6Ga2Ni rose from 63rd to 32nd. Two placebos moved it less, one moved it
  down; it stays the benchmark for its published calculation, not its rank.

**For the owner's veto** (each row can be vetoed before merging):

- **FeS.** Only the metastable tetragonal polymorph superconducts; the common
  hexagonal one does not. The table works by composition, so the rule
  corrects the row. Veto it if a composition label should follow the stable
  polymorph.
- **WBe2.** The paper names the phase WBe2 without a chemical analysis and
  places it on the W-rich side of a range whose edge is 0.7 at.% away. The
  excess tungsten is seen as a separate phase. This is close to the cycle 1
  question about off-ratio samples (ScS, W3Re2C), which those rows failed.
- **NbSi2, Nb0.4Hf0.2Zr0.2Ti0.2, Lu2Ni3Si5** were read from abstracts only,
  under option B. Each abstract states the exact composition, a Tc and a
  bulk signature (specific heat).

**What becomes the ranking of record.** Cycle 5, if the owner merges this
pull request. Its models are `data/processed/learning-cycles/models/cycle5/`
(not committed), trained on the corrected table plus the 851 NIMS rows.
Shortlist 2 and the Hazel packages stay as they are (chosen with cycle 3).

## Cycle 6: plan and predictions, written before any cycle 6 result, 2026-11-21 05:14 UTC

(Correction to the heading above: the cycle 5 result was written at about
05:10 UTC, not 05:30 as first typed.)

**Why.** The owner's request of 2026-11-21 stands: keep going while tokens
last. Cycle 5 used the rows that had an automatic lead. This cycle takes the
rest.

**Batch.** The 1,019 development non-cuprate negatives that neither cycle 3
nor cycle 5 sent to a screen: 672 whose OpenAlex search found no lead and 347
the search did not reach (its free daily budget ran out). They are taken
most suspicious first by cycle 5's out-of-fold probability, starting with
the 381 at 0.5 or above, and continued down the list while tokens last.

**What differs from cycle 5.** These rows have no automatic lead, so the
screen runs its own searches (a few per row, through the literature
connectors). Everything after the screen is cycle 5's: two independent
readers, then a sceptic, under the owner's rule of 2026-11-16 with option B.
Screens and readers are not shown the model's score or the row's fold.

**Not in this cycle:** development cuprate negatives (318; their labels hang
on oxygen content, which needs its own rule), locked-row negatives (439;
never scored again) and train-only negatives (973).

**Run order.** As in cycle 5: three placebos with as many random flips as
there are corrections (seeds 61 to 63, with `--additions`), then the
corrections, a rebuild of both tables, and
`python scripts/run_learning_cycle.py --name cycle6 --previous cycle5 --additions data/processed/learning-cycles/nims-additions.csv.gz`.
If cycle 5 is vetoed in part before then, cycle 6 is rerun on the table that
results.

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| Corrections that pass: 0 to 4 | 65% |
| 5 or more | 35% |
| Pass rate per row screened is below cycle 5's (7 of 457 with leads, 1.5%) | 80% |
| Cross-validated non-cuprate AUC changes by less than 0.005 from cycle 5's 0.7852 | 90% |
| If there are at least two corrections, fold 4 supplies at least half of them (it holds 174 of the 1,019 rows, 17%) | 30% |

Reasoning:
- A row with no lead has had the most obvious search done already and
  found nothing, so fewer of these zeros should be wrong.
- The fold-4 line tests cycle 5's unexplained pattern (six of seven
  corrections in fold 4). With no real effect, fold 4 would supply half of
  two or more corrections well under one time in five.

## Cycle 6: result, 2026-11-21 07:05 UTC

**Main result:** nine labels corrected, one held for the owner. The score
moves a little more than in cycle 5 but stays within noise, and the
corrections' effect on their own chemistries is weaker than in cycle 5.

**What was screened.** The 620 rows at cycle 5 suspicion 0.3 or above that
had no automatic lead (381 at 0.5 or above, 239 from 0.3 to 0.5). The 399
rows below 0.3 were not reached in this cycle.

| Stage | Rows |
| --- | --- |
| Screen: unlikely / needs a full text / likely wrong | 570 / 32 / 18 |
| Reader A confirmed / verified null / not confirmed | 15 / 4 / 31 |
| Reader B confirmed | 14 |
| Sceptic: apply / do not apply | 10 / 4 |
| Applied (one apply held by me, below) | **9** |

All nine came from the 0.5-and-above band; the 0.3 to 0.5 band gave two
(PrNiAsO, ScSe) and the held row. Records:
`data/benchmarks/learning-cycles/cycle6-evidence.json`.

**Applied** (all development rows, 0 to 1, nothing moved between splits):

| Row (fold) | Paper | Tc used (K) | Read from | Note |
| --- | --- | --- | --- | --- |
| Ir0.9Pt0.1Te2 (0) | Adv. Mater. 30, 1801556 (2018) | 1.7 | full text (arXiv) | Self-flux single crystals, zero resistance; resistive only, nominal x. The zero is from polycrystals measured to 2.0 K |
| Ir0.8Pt0.2Te2 (0) | same | 0.65 | full text | Stated; zero resistance |
| Ir0.85Pt0.15Te2 (0) | same | 1.18 | full text | **Bracketed** by x = 0.1 and 0.2 (each 1.67 at.% away); the Tc is an interpolation, not a measurement |
| CeOs2 (1) | J. Appl. Phys. 57, 3137 (1985) | 1.1 | full text (author manuscript) | Hexagonal C14 form, made by annealing; the cubic C15 form does not superconduct above 70 mK. A later group agrees |
| Pd0.1HoTe3 (4) | arXiv:1301.3032 (Supercond. Sci. Technol. 29, 065018, 2016; journal version not read) | 2.6 | full text (arXiv) | One member of a measured series; zero resistance, strong diamagnetism |
| (Li0.8Fe0.2)OH·Fe(S0.08Se0.92) (1) | Chem. Commun. 51, 4689 (2015) | 36 | full text (arXiv) | Exact member of a single-phase series, composition by EDX, ICP and elemental analysis |
| BaFe1.5Ru0.5As2 (3) | Phys. Rev. B 82, 014534 (2010) | 14 | full text (arXiv) | **Bracketed** by WDS-analysed crystals 0.4 and 1.6 at.% away; a 2012 crystal at exactly this composition superconducts. One nominal polycrystal (2010) found nothing to 4 K |
| PrNiAsO (2) | J. Phys.: Condens. Matter 26, 425701 (2014) | 0.93 | **abstract only** | Specific heat reported; a second group reports it at 1.6 K |
| ScSe (0) | J. Low Temp. Phys. 33, 175 (1978) | 3.7 | first two pages only (publisher preview) | Stated as a discovery of this work; composition analysis and the source of the table's three zeros not seen |

**Held by me, not applied: DyRh4B4.** Both readers and the sceptic say it
superconducts (4.7 K, Acta Phys. Pol. A 114, 83, 2008), but only in a
metastable body-centred form made at about 8 GPa; ordinary arc-melted
DyRh4B4 is a 12 K ferromagnet. That is the open rule question that cycle 5's
sceptic cited when refusing Bi2OS2 (do samples made under high pressure and
measured at ambient count?). Applying one and not the other would be
inconsistent, so both wait for the owner's answer. Precedent: cycle 3's
LaPt1.7Au0.3B2C was held by me after its verifier said apply.

**Refused by the sceptic although both readers confirmed:**

| Row | Reason |
| --- | --- |
| LaFeAsO0.96F0.04 | Two primary papers on the same nominal composition (Dresden) find full SDW order and no superconductivity; x = 0.04 is the SDW/superconductor boundary and the groups' F scales differ by about 0.02 |
| Sr0.6La0.4FBiS2 | Sakai et al. 2014 measured the same nominal composition and found nothing to 1.8 K; steep insulator-superconductor boundary |
| Ba(Fe0.965Co0.035)2As2 | Hardy et al. 2010 report a crystal at exactly this composition with no specific-heat anomaly to 0.4 K; the underdoped edge of the dome |
| Ba(Fe0.85Co0.15)2As2 | Hardy et al. 2010: EDX-analysed crystal at this composition, nothing to 0.4 K; the overdoped edge |

Verified nulls (evidence file only; no label changes): LaH1.96 to 0.33 K,
hydrated Na0.3CoO2 at this water content to 2.0 K, one (Ce,Pr)Pt4Ge12 row
and one Ca(Fe,Co)AsH row to 2.0 K.

**Scores, non-cuprates** (paired difference: this model minus cycle 5's,
both on today's labels):

| Run (nine labels changed) | AUC | Paired AUC difference (median, 95% interval) | Tc error, log scale | Top 10 / 50 / 200 kept | Zr6Ga2Ni (was 32) |
| --- | --- | --- | --- | --- | --- |
| **Cycle 6** | **0.7882** | **+0.0034 (−0.0021 to +0.0094)** | 0.3868 | 9 / 40 / 166 | 102 |
| Placebo 1 (seed 61) | 0.7829 | −0.0023 | 0.3848 | 7 / 40 / 162 | 53 |
| Placebo 2 (seed 62) | 0.7852 | +0.0009 | 0.3869 | 8 / 41 / 166 | 52 |
| Placebo 3 (seed 63) | 0.7879 | +0.0028 | 0.3919 | 8 / 37 / 154 | 41 |

Made metals sharing at least two elements with a corrected row (285
compounds): median rise 42 places, against 41, 5 and 15 in the placebos.

**Against the predictions written before the result:**

| Prediction | Chance given | Result |
| --- | --- | --- |
| 0 to 4 corrections | 65% | **No** (9) |
| 5 or more | 35% | Yes |
| Pass rate per row screened below cycle 5's (1.53%) | 80% | Yes, barely: 9 of 620 = 1.45% (1.61% if the held row counted) |
| AUC changes by less than 0.005 from cycle 5's 0.7852 | 90% | Yes (+0.0030) |
| Fold 4 supplies at least half of two or more corrections | 30% | No (1 of 9) |

**What it shows.**

- [Certain] Rows with no automatic lead were not cleaner than rows with
  one: the pass rate was about the same (1.45% against 1.53%). The search
  missed them for reasons of naming (doping series, intercalates,
  polymorphs), not because they were right. I expected fewer.
- [Certain] Cycle 5's fold-4 pattern did not repeat (1 of 9). Taken with
  cycle 5, 7 of 16 corrections are in fold 4, which held about 17 to 27% of
  the rows screened. [Speculative] That is still somewhat high, but most
  likely chance.
- [Likely] Cycle 6's AUC change beat all three placebos, as cycle 5's did
  not. With three placebos this is weak evidence either way.
- [Certain] The local effect is not distinguishable from placebo 1. Several
  corrections are in families the table already knows well (iron arsenides,
  Ir-Te), so there was less for the model to learn.
- Four of the five refusals in this cycle, and three of cycle 5's eight, sit
  on the edge of a doping dome, where groups' composition scales disagree
  by more than the rule's 0.15 at.%. The rule is doing its job there.
- Zr6Ga2Ni fell from 32nd to 102nd (placebos: 41st to 53rd). It is the
  benchmark for its published calculation, not its rank.

**For the owner's veto**, in addition to cycle 5's list: CeOs2 (polymorph,
as FeS), ScSe (two pages read), BaFe1.5Ru0.5As2 (one nominal null at the
exact composition), Ir0.85Pt0.15Te2 (interpolated Tc), PrNiAsO (abstract
only).

**What becomes the ranking of record.** Cycle 6, if the owner merges the
pull request holding cycles 5 and 6.

### Cycle 7: plan, written before any cycle 7 result, 2026-11-21 07:05 UTC

The 399 development non-cuprate negatives below 0.3 that no cycle has
screened, with cycle 6's method, most suspicious first, while tokens last.
DyRh4B4 and Bi2OS2 stay held until the owner rules on pressure-synthesised
samples. Predictions [Speculative]: 0 to 4 corrections, 70%; pass rate below
cycle 6's 1.45%, 75%; AUC changes by less than 0.005 from cycle 6's 0.7882,
90%. Placebos, rebuild and retraining as in cycle 6, if anything passes.

## Cycle 7: stopped by the usage limit, 2026-11-21 18:01 UTC

**Main result:** no corrections. The model usage limit stopped the readers
partway through, so cycle 7 is incomplete and the table is unchanged; no
placebos or retraining were run, and cycle 6 stays the latest cycle.

| | Rows |
| --- | --- |
| In the batch (development non-cuprate negatives below 0.3, never screened) | 399 |
| Screened | 240 |
| Screen: unlikely / needs a full text / likely wrong | 237 / 2 / 1 |
| Not screened (the limit stopped those agents) | 159 |
| Sent to readers but not read (limit) | 3: Y0.75Ce0.25Ni2B2C (likely wrong), and two Be-Fe alloys, Be0.95Fe0.05 and Be0.977Fe0.023 (needs a full text) |

- [Certain] Against the predictions: 0 corrections falls in the 0-to-4 band
  (70%); the AUC prediction does not apply because nothing changed. The
  pass-rate prediction cannot be judged on an incomplete batch.
- [Likely] The low band is mostly real nulls: the screen called 237 of 240
  unlikely to be wrong, a higher share than in cycle 6 (570 of 620).
- Records: `data/benchmarks/learning-cycles/cycle7-evidence.json` (the 240
  screens). The 159 unscreened rows and the 3 unread ones are the place to
  resume.

## The ranking of record from 2026-11-27: the blend of cycle 6 and TabICLv2

The owner accepted the 50/50 blend of cycle 6 and TabICLv2 as the ranking of
record, after it beat cycle 6 on the development rows and on its one scoring
on the locked rows (`docs/lockbox-blend.md`).
- Cycle 6 stays the boosting half of the blend, and the reference for future
  learning cycles.
- Rankings: `python scripts/blend_ranking.py`.

## Cycle 7: plan and predictions, written before any cycle 7 result, 2026-11-27

**Main idea:** add the usable part of the NEMAD superconductor database to
cycle 6's training, and fix three nulls that a paper check found wrong.
The owner approved it on 2026-11-27 (`docs/nemad-superconductor.md`, "What
would come next").

**The assumption I am challenging.** By the learning curve
(`docs/learning-curve.md`), a quarter more rows is worth about +0.01 AUC.
That assumes the new rows are as good as the old ones. NEMAD's usable subset
was right about 10 times in 11 in a small check, so part of the gain may
not appear.

**What changes**
- **Added rows.**
  - Cycle 6's 851 NIMS rows stay, unchanged.
  - Added to them: 2,083 NEMAD compositions (1,572 superconductors, 511
    non-superconductors), from records flagged experimental at normal or
    unstated pressure.
  - Each is new to the project: not in the table (locked rows included),
    not in the NIMS file, and not a near twin of either.
  - Placed by cycle 4's rule, with ties chained through the NIMS rows:
    - 726 more were left out because they tie to a locked group;
    - 24 were left out because they tie to more than one fold.
  - Command: `python scripts/build_nemad_additions.py`, written to
    `data/processed/learning-cycles/cycle7-additions.csv.gz`, with counts
    in `docs/learning-cycles/cycle7-additions.json`.
- **Corrections.** Three nulls (Nb3Ge, YBa2Cu3O6.32,
  SmFeAs0.47P0.53O0.85H0.15) are being checked under the owner's rule: two
  independent readers and a sceptical verifier.
  - Only those that pass are recorded in
    `data/training/negative_evidence.csv` and applied.
  - The table is then rebuilt, and the added rows rebuilt against it,
    before the run.
  - A correction that fails is left out and listed.
- **Nothing else.** Same boosting settings, folds and seeds as cycle 6.

**Runs** (after the table rebuild):

```bash
python scripts/run_learning_cycle.py --name cycle7 --previous cycle6 --additions data/processed/learning-cycles/cycle7-additions.csv.gz
```

Three placebos use the same rows with the NEMAD labels shuffled among
themselves, which keeps the rows, the features and the label balance:

```bash
python scripts/run_learning_cycle.py --name cycle7-placebo-1 --previous cycle6 --additions data/processed/learning-cycles/cycle7-additions.csv.gz --permute-added-labels nemad --placebo-seed 71
```

The other two use `--placebo-seed 72` and `73`.

**Measures**

| | Question | Rows | Measure |
| --- | --- | --- | --- |
| E1 | Do the NEMAD rows hold learnable information? | The NEMAD rows, each predicted by the round that held its fold out | AUC with all added rows minus without any, non-cuprates, 95% interval over groups |
| E2 | Does it help at home? | The development rows, scored against today's labels | Cycle 7 minus cycle 6, paired, non-cuprates: median and 95% interval |

**Rule, fixed now (cycle 4's)**

| Result | What happens |
| --- | --- |
| E1's 95% interval is above zero **and** E2's median difference is not below −0.005 | Cycle 7 becomes the boosting half of the ranking of record. The bundle is rebuilt, and the owner's laptop reruns TabICLv2 on it. The new blend replaces the current one if its paired non-cuprate median difference against the current blend, on the development rows, is not below −0.005 |
| Anything else | Cycle 6 stays, and so does the current blend. The added-rows file is kept |

- The placebos are reported and do not decide.
- The locked rows are not used: they have been scored twice already.

**Predictions [Speculative]**

| Outcome | Chance |
| --- | --- |
| E1's interval is above zero (NEMAD rows, non-cuprates) | 80% |
| E2's median difference is at least 0 | 60% |
| E2's median difference is at least +0.005 | 30% |
| E2's interval lies above zero | 15% |
| Cycle 7 is adopted under the rule | 75% |
| Cycle 7's E2 is larger than each placebo's | 55% |
| The made-metals top 50 keeps between 25 and 45 of cycle 6's | 70% |
| All three corrections pass the owner's rule | 50% |
