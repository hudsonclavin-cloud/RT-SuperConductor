# Which model best spots superconductors? A controlled benchmark

**Main idea:** on the 2,244 labelled Materials Project materials, gradient boosting is the best model (AUC 0.817). A small neural network comes fifth (0.769). The production classifier now uses gradient boosting. Neural networks should be retried once there are more labels or richer structure inputs. Use this benchmark to decide when one genuinely wins.

## Setup

Every model gets the same inputs and the same test:

- **Task:** superconductor (reported Tc > 0, 1,140 materials) vs tested non-superconductor (3DSC Tc = 0, 1,104 materials).
- **Features:** formula features (Magpie, shell, Lewis-dot) plus Materials Project properties (band gap, metallicity, magnetism, density, formation energy, symmetry, volume per site).
- **Test:** StratifiedGroupKFold over element systems, 5 folds, repeated with 3 shuffles. A test material never shares its element set with any training material.
- **Metrics:** ROC AUC (0.5 = chance), average precision, and the superconductor share among each fold's top-scored 10%. The base rate is 51%.

Reproduce with `python scripts/benchmark_sc_models.py`. Results are in `docs/sc-model-benchmark.json`, which also records the git commit, library versions, seeds and a feature-matrix hash. Every out-of-fold prediction (model, shuffle, fold, label, score) is saved in `data/candidates/sc-benchmark-oof.csv.gz`.

Protocol details, following the reproducibility checklist from the 2026-11-06 external review:

- Imputation, scaling and early stopping are fitted inside each training fold. The outer test fold is never used.
- Positives are reported Tc > 0. Negatives are explicit Tc = 0 reports. Materials with no report are never treated as negatives.
- Paired differences: each shuffle's AUC is compared with the random forest on the same folds.

## Results

| Model | AUC (± spread across shuffles) | Paired gain vs random forest | Average precision | Brier (lower is better) | Superconductors in top 10% |
| --- | --- | --- | --- | --- | --- |
| Gradient boosting (HistGradientBoosting) | **0.817 ± 0.009** | +0.017 (wins 3 of 3) | 0.822 | 0.179 | 94% |
| Ensemble: forest + boosting + neural net | 0.812 ± 0.006 | +0.012 | 0.813 | 0.178 | 93% |
| Random forest (previous classifier) | 0.800 ± 0.005 | reference | 0.799 | 0.183 | 92% |
| Extra trees | 0.800 ± 0.006 | 0.000 | 0.796 | 0.183 | 91% |
| Neural network (MLP, 256-64, early stopping) | 0.769 ± 0.002 | −0.031 | 0.760 | 0.223 | 89% |
| Logistic regression | 0.739 ± 0.010 | −0.061 | 0.713 | 0.213 | 80% |

## What this means

- **Trees win at this size.** With about 2,200 labelled materials and about 180 tabular features, boosted trees beat the neural network by about 0.05 AUC. That is the usual pattern for small materials datasets **[Likely]**.
- **The gain over the old forest is small but consistent:** +0.017 AUC, about two shuffle spreads.
- **The repository's earlier deep-learning experiments agree.** Pretrained CHGNet embeddings (`docs/deep-chgnet-jarvis.json`) and a 5-epoch scratch ALIGNN (`docs/deep-scratch-alignn.json`) did not beat the random-forest baselines on their tasks.
- **When to try neural networks again:**
  - after more labels are added
  - with structure-aware models such as graph networks, once there are thousands more labelled structures
  - as a member of an ensemble, which already nearly ties the best single model

  Rerun this benchmark each time. The neural network earns its place when it beats gradient boosting beyond the spread.
