# Training methods: what the literature supports, and one planned test

**Main idea:** the owner asked for a literature search for a training
method proven better than the current one (2026-11-23).
- One family has independent evidence at this project's data size:
  **tabular foundation models**, which predict by in-context learning
  with no training.
- Nothing else found is shown to beat gradient boosting in this regime.
- One test of one such model (TabICLv2) is set up below, with its rule
  and predictions fixed before any result.

Written 2026-11-23, before any result. No new model was run here.

## How the search was done

- **Tools:** the FastTrack Literature connector (OpenAlex and Semantic
  Scholar records), OpenAlex, and alphaXiv for full-text pages.
- **Read in full text (selected pages):** TabArena, the TabPFN-2.5 and
  TabPFN-3 reports, and Li et al. 2026. Everything else was read as an
  abstract or record only.
- **Two helper searches** covered label noise (section 3) and selection
  bias (section 5).
- **Not done:** no new model was run. Model weights cannot be downloaded
  in this session (huggingface.co is unreachable from it), and nothing
  was installed.

## 1. Tabular foundation models

A tabular foundation model is one network pretrained on millions of
synthetic tables. Given a new table, it reads the training rows as
context and predicts the test rows in one pass. There is no fitting and
no tuning, so there is nothing to select between.

| Paper | Status | Tested on | Finding |
| --- | --- | --- | --- |
| Hollmann et al. 2025, *Nature* 637, 319 (TabPFN v2), 10.1038/s41586-024-08328-6 | peer reviewed | tables up to 10,000 rows and 500 features | "outperforms all previous methods ... by a wide margin"; one 2.8-second pass beats an ensemble of the strongest baselines tuned for 4 hours (classification) |
| Erickson et al. 2025, NeurIPS Datasets and Benchmarks (TabArena), 10.52202/085713-0519 | peer reviewed; authors include the makers of AutoGluon and TabPFN | 51 curated tables, up to 100,000 rows | with ordinary tuning CatBoost ranks first. With ensembling of settings, RealMLP, TabM and LightGBM lead. On the 33 tables with 10,000 rows or fewer, TabPFN v2 leads "by a large margin" |
| Qu et al. 2025, ICML (TabICL), arXiv 2502.05564 | peer reviewed | 200 classification tables | level with TabPFN v2; on the 53 tables above 10,000 rows it beats both TabPFN v2 and CatBoost |
| Qu et al. 2026, ICML, PMLR 306 (TabICLv2), arXiv 2602.11139 | peer reviewed | TabArena and TALENT | untuned, it beats RealTabPFN-2.5 after tuning, ensembling and fine-tuning. Open code and weights |
| Prior Labs 2025, TabPFN-2.5 report, arXiv 2511.08667 | vendor report, not peer reviewed | up to 50,000 rows and 2,000 features | default beats tuned tree models and matches AutoGluon 1.4 tuned for 4 hours |
| Prior Labs 2026, TabPFN-3 report, arXiv 2605.13986 | vendor report, not peer reviewed | TabArena's 15 tables with 10,000 to 100,000 rows | Elo: TabPFN-3 default 1835, TabICLv2 default 1712, CatBoost tuned and ensembled 1625, LightGBM tuned and ensembled 1604. Win rate over CatBoost tuned and ensembled: TabPFN-3 75%, TabICLv2 67% |
| Li et al. 2026, *npj Comput. Mater.* 12, 222, 10.1038/s41524-026-02089-8 | peer reviewed | six MatBench composition tasks, 636 to 18,928 rows, Magpie features | TabPFN on Magpie features beats the best composition-only models on 5 of 6 tasks, by 0.2% to 9.9% in error. It gains 4% to 6% on the two 10,987-row tasks and loses 4% on perovskites. With graph-network embeddings it does not beat the best structure models |
| McElfresh et al. 2023, NeurIPS, 10.52202/075280-3337 | peer reviewed | 176 tables, 19 methods | the trees-against-networks question matters less than light tuning; TabPFN v1 was best on average |
| Grinsztajn et al. 2022, arXiv 2207.08815 | preprint of a NeurIPS paper | 45 tables of about 10,000 rows | trees were the state of the art at this size, before foundation models |

**How big the gains are**
- [Certain, from TabArena's per-table results] On binary tasks scored by
  AUC, the gap between tuned CatBoost and the best foundation model is
  mostly 0.005 or less, and at most about 0.02. Examples: 0.961 to 0.981
  on `polish_companies_bankruptcy`, and 0.749 to 0.773 on `jm1`.
- [Certain] TabArena includes the UCI superconductivity table (Tc
  regression, random split).
  - Tuned and ensembled LightGBM reaches 9.26 K RMSE.
  - The best models reach 9.08 to 9.24 K.
  - That is a gap of 2% or less.

**Why this may not carry over to this project**
- [Certain] Every benchmark above splits rows at random. This project
  holds out whole chemical groups and then tests on later papers. No
  study was found of these models under that kind of shift.
- [Certain] This project's negatives are noisy; the benchmark tables are
  cleaner. One 2026 preprint on TabPFN's noise robustness exists
  (arXiv 2604.04868). It was not read.
- [Certain] The two strongest claims for 10,000 to 100,000 rows come
  from the vendor's own reports.
- [Likely] The realistic gain here is between 0 and +0.015 AUC. Cycle 6's
  paired interval is about ±0.01 wide, so only the upper half of that
  range would show.

**Licences**
- TabICLv2: BSD-3 for code and weights.
- TabPFN v2: Apache 2.0 with an attribution requirement.
- TabPFN 2.5 and later: the weights carry a **non-commercial** licence
  and need a login on first use.
  - This project's candidate results are before NC State's Office of
    Research Commercialization.
  - So TabPFN 2.5 or later should not be used to choose candidates until
    that office says the licence is acceptable.
- For that reason the planned test uses TabICLv2.

## 2. Stronger tree libraries and ensembles

- [Certain, TabArena table A.1] CatBoost with default settings (Elo
  1427) matches tuned LightGBM (1446) and is far above default LightGBM
  (1197).
  - This project's HistGradientBoosting is of the LightGBM family and is
    already tuned over six settings.
  - [Likely] Little is left to gain from changing the tree library.
- [Certain] In TabArena, ensembling many settings of one model, and
  ensembling across models, both raise scores. The paper also warns that
  cross-model ensembles overfit the validation data.
- [Certain, this project] A 50/50 blend of cycle 6 with a network gave
  +0.001 to +0.002 AUC, inside its interval
  (`docs/local-gpu.md`, `docs/overnight-gpu.md`).

## 3. Label noise and missing negatives

**Main finding:** no method was found that is shown to beat plain
gradient boosting in this regime. No paper tests any of these methods on
superconductor classification.

| Method | Best evidence found | Why it does not settle the question |
| --- | --- | --- |
| Positive-unlabeled learning | Jang et al. 2020, *JACS*, 10.1021/jacs.0c07384 (synthesizability, graph network); Frey et al. 2019, *ACS Nano*, 10.1021/acsnano.8b08014; Aydın et al. 2026, 10.1002/adts.70395 (nine strategies with tree learners on about 130,000 crystals) | they report true-positive rates or candidate lists, with no supervised baseline or effect size in the abstract |
| Confident learning (finding label errors) | Northcutt et al. 2021, *JAIR*, 10.1613/jair.1.12125 | tested on images and text; the gain is described as moderate |
| Label-error detectors on tables | Nazaretyan et al. 2025, *Artif. Intell. Rev.*, 10.1007/s10462-025-11293-9 (20 detectors) | mean precision 0.16 to 0.55: many flagged rows are clean |
| Noise-robust boosting | Morampudi et al. 2026, *Algorithms*, 10.3390/a19090730 (15 tables, three libraries) | tested on symmetric synthetic noise only; this project's noise sits in one class |

- [Likely] Plain training may already be near the best for ranking.
  Jain et al. 2017 (AAAI, 10.1609/aaai.v31i1.10937) state that training
  positives against unlabeled rows gives an optimal classifier under
  mild conditions. Only the performance estimates are biased.
- [Certain] The one superconductor-specific result favours buying
  verified labels over any training method: Pogue et al. 2023
  (*npj Comput. Mater.*, 10.1038/s41524-023-01131-3) more than doubled
  the discovery success rate over four rounds of experimental feedback.
- This project's own result agrees: 22 corrected labels are 0.15% of the
  rows, and the overall score moved less than its noise.

No test of these methods is planned.

## 4. Learning the physics instead of the labels

- Gibson et al. 2026 (*npj Comput. Mater.* 12, 95,
  10.1038/s41524-026-01964-8) trained an ensemble of graph networks
  (BEE-NET) to predict the calculated electron-phonon spectral function
  and Tc from structure.
  - Its mean error is 0.87 K against the calculations, and its
    true-negative rate is 99.4%.
  - The workflow cut 1.3 million structures to 741 stable compounds with
    calculated Tc above 5 K.
  - Two new compounds were then made and confirmed as superconductors.
- [Likely] This is the strongest published route to new conventional
  superconductors, and it is a different route: the model learns from
  calculations, not from literature labels.
  - It needs a large set of electron-phonon calculations to train on.
  - This project's Hazel runs produce exactly that kind of data, one
    compound at a time.
- It is not a drop-in training method, so no test is planned here. It is
  the direction to weigh if the Hazel queue shows the project's own
  calculations can be trusted (the Zr6Ga2Ni benchmark).

## 5. Choosing between many variants without fooling yourself

**Main finding:** the literature describes the "lucky child" exactly,
and gives usable rules. The project already follows most of them. Three
are new here.

| Paper | Finding |
| --- | --- |
| Cawley and Talbot 2010, *JMLR* 11 | over-fitting the selection criterion has effects "often of comparable magnitude to differences in performance between learning algorithms" |
| Varma and Simon 2006, *BMC Bioinformatics*, 10.1186/1471-2105-7-91 | on data with no signal at all, the tuned cross-validation error looked good (below 30%) in 18.5% to 38% of datasets. Nested cross-validation was nearly unbiased |
| Wainer and Cawley 2021, *Expert Syst. Appl.*, 10.1016/j.eswa.2021.115222 | over 115 datasets, plain cross-validation picks as good a model as nested cross-validation when few settings are tuned. Its score for the winner stays biased |
| Tsamardinos et al. 2018, *Machine Learning*, 10.1007/s10994-018-5714-4 | BBC-CV: bootstrap the selection step over the stored out-of-fold predictions. No retraining, works for AUC, lower bias and variance than nested cross-validation |
| Schneider et al. 2025, PMLR 293, arXiv 2506.19540 | in about 10% of tuning runs the validation-best setting generalises worse than the default. It is worst on small data |
| Dwork et al. 2015, *Science*, 10.1126/science.aaa9375 | a holdout reused for adaptively chosen analyses loses its validity unless access to it is limited |
| Meredig et al. 2018, 10.1039/C8ME00012C; Sheridan 2013, 10.1021/ci400084k | random splits are too optimistic for discovery; grouped and time-split validation are closer to prospective results |

**How much luck to expect** (the helper's simulation, not from a paper;
it assumes independent children, so it is an upper guide)
- If every child's gain were pure noise with spread s, the best of 36
  children would average about 2.1 s.
- The best of 360 would average about 2.9 s.
- So ten times more children buys more luck, and no skill.

**Evidence the other way**
- Roelofs et al. 2019 (NeurIPS) found "little evidence of substantial
  overfitting" from holdout reuse across more than 100 Kaggle
  competitions.
- Search does pay when it is judged on fresh or large data.
- [Likely] The conditions for that are large holdouts and low-noise
  scores. This project has neither: 1,950 locked rows and blind tables
  of 65 to 82 compounds.

**Rules already followed here:** grouped and time-split validation; rules
and predictions written first; the lockbox scored once; the winner's
score never quoted from the data that chose it.

**Rules to add** (the sizes are the helper's calculations from standard
AUC error formulas, under stated assumptions; treat them as estimates)
1. **Blind tables need to be bigger.**
   - With 65 to 82 compounds, the 95% interval for a paired AUC
     difference is about ±0.04 to ±0.08. A true gain of 0.03 cannot be
     seen.
   - Resolving 0.03 needs roughly 130 to 260 compounds.
   - So pool the blind tables as they grow, and aim for 200 or more.
2. **Budget the lockbox.** Keep it to one look per pre-registered
   question and about five looks in all. Its worst-case bias grows with
   the square root of the number of looks.
3. **If children are ever selected again,** report the winner with the
   BBC-CV correction over all children's stored predictions, not its raw
   score.

## Verdict

| Candidate | Independent evidence at this size | Expected gain here [Speculative] | Planned? |
| --- | --- | --- | --- |
| TabICLv2 (open foundation model) | yes: two peer-reviewed papers and one benchmark | 0 to +0.015 AUC | **yes, the one primary test** |
| TabPFN 3 or 3.5 | yes, mostly vendor reports | 0 to +0.02 AUC | only as a secondary, and only after the licence is cleared |
| CatBoost defaults | weak: level with tuned LightGBM | 0 to +0.005 AUC | secondary, optional |
| Positive-unlabeled or noise-robust training | none in this regime | within ±0.01 | no |
| Learning calculated spectral functions | strong, but a different project | not comparable | no |

**The honest summary:** no method is *proven* superior for this
project's task. One is proven superior on similar-sized tables with
random splits, which is enough to earn one clean test.

## The planned test (fixed before any result)

**Model:** TabICLv2 at its default settings (8 ensemble members,
classifier and regressor checkpoints as shipped by `tabicl` 2.2.0). No
tuning of any kind, so there is nothing to select and no lucky child.

**Data:** the formula bundle already on the owner's laptop
(`bundle.npz`, content checksum `daed58cb…`).
- For each of the 5 folds, the model is given the fold's fit rows and
  validation rows together. It has no early stopping, so it needs no
  validation set.
- It predicts the fold's test rows.
- Locked rows are not in the bundle.

**Runner:** `scripts/foundation_gpu_train.py --model tabicl`.
- [Certain] The runner was checked here with cycle 6's own boosting
  settings (`--model hgb`): it reproduced cycle 6's saved predictions
  exactly, on all 9,765 development rows.
- So any difference TabICLv2 shows comes from the model.
- If the GPU cannot hold every row, the fallback is fixed now:
  `--max-context 8000`.

**Rule:** the standing one (`docs/local-gpu.md`), with its lockbox step
written out. TabICLv2 joins or replaces the ranking of record only if all
of these hold:
1. Its paired non-cuprate AUC interval against cycle 6 lies above zero on
   the development rows (`python scripts/local_gpu.py compare <folder>`).
2. Only if step 1 passes, both models are scored once on the 1,950 locked
   rows, and TabICLv2's non-cuprate AUC there is at least cycle 6's.
   - The laptop would be sent the locked rows' features without their
     labels. The predictions are scored here.
3. The owner accepts it.

**Secondary results**, reported but unable to trigger adoption alone:
- the 50/50 blend with cycle 6;
- the Tc error;
- CatBoost defaults (`--model catboost`), if the owner installs it.

### Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| TabICLv2 beats cycle 6 under step 1 of the rule (interval above zero) | 25% |
| TabICLv2's non-cuprate AUC is within ±0.01 of cycle 6's 0.788 | 55% |
| The 50/50 blend with cycle 6 beats cycle 6 (interval above zero) | 40% |
| TabICLv2 beats the 20-network tabular bank (0.770) | 75% |
| Its Tc error on the log scale is lower than cycle 6's | 35% |
| It runs on the 8 GB GPU without the context cap | 70% |

These chances are higher than any given to the networks (10%), because
the evidence is independent and at this data size. They are still below
even odds, because of the grouped splits and the noisy negatives.

### Commands for the owner's laptop (Windows PowerShell)

```powershell
cd C:\Users\hudso\RT-SuperConductor
```
```powershell
C:\Users\hudso\NCSU-Hackathon\.venv-gpu\Scripts\Activate.ps1
```
```powershell
git checkout claude/practical-tesla-ctf6oq
```
```powershell
git pull origin claude/practical-tesla-ctf6oq
```
```powershell
pip install tabicl
```
```powershell
python scripts\foundation_gpu_train.py data\processed\local-gpu\bundle.npz --model tabicl --out runs\fm-20261123\tabicl --device cuda
```

The first run downloads the two TabICLv2 checkpoints from Hugging Face.
Then send the results back as before:

```powershell
git checkout -b fm-results-20261123
```
```powershell
New-Item -ItemType Directory -Force data\benchmarks\local-gpu | Out-Null
```
```powershell
Copy-Item runs\fm-20261123\tabicl data\benchmarks\local-gpu\fm-tabicl-20261123 -Recurse
```
```powershell
git add data/benchmarks/local-gpu/fm-tabicl-20261123
```
```powershell
git commit -m "TabICLv2 on cycle 6's folds: predictions and run record"
```
```powershell
git push -u origin fm-results-20261123
```
