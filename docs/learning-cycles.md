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
