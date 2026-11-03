# Modeling workflows

## Reproducible active-learning cycle

One command downloads the pinned UCI snapshot and the pinned 3DSC Materials
Project table, fetches a deterministic CIF subset, bridges reviewed evidence,
ingests 3DSC, trains both models, evaluates optional structure features, and
writes a reviewable candidate shortlist:

```bash
python -m pip install -e '.[ml]'
python scripts/run_active_learning.py
```

Raw data are kept under `data/raw/` and ignored by git. Use
`--all-cifs` for a complete local structure run; CI uses a stratified CIF
subset to keep the weekly job bounded. The run directory contains the dataset
hashes, JSON metrics, blocked reports, model artifacts, and
`candidates/top_k.csv`.

## Formula-level electronic descriptors

The formula descriptor artifact records neutral-atom electron configurations,
shell fill percentages, and Lewis-dot outer-electron proxies. Sigma and pi
bond counts, oxidation states, and charge transfer remain explicit unknowns
for formula-only records; those require a crystal structure and electronic
density analysis. These descriptors are available as an auditable feature
source, but should not be described as trained model inputs unless a run
manifest confirms that backend and feature set.

## Trust tiers and training boundary

The model does not treat every row as an observation:

| tier | schema values | production use |
| --- | --- | --- |
| demonstration | `provenance=synthetic_demo`, `is_synthetic=1` | fixtures and smoke tests only |
| reviewed literature | `provenance=literature`, `review_status=confirmed`, `is_synthetic=0` | eligible after human review |
| curated database | `provenance=curated`, `review_status=curated_database`, `is_synthetic=0` | eligible database evidence |
| automatic SuperCon2 extraction | pending or non-literature provenance | never model training |

`select_training_records` enforces this policy. The five reviewed Y-cuprate
records are literature evidence, not synthetic demonstrations; automatic
SuperCon2 extraction remains outside training even though the reviewed rows
retain their source identifier.

## Initial AI baseline

The composition-only `RandomForestRegressor` is trained on UCI `unique_m.csv`.
Because UCI does not provide chemical families, the script records a
deterministic family heuristic from the elemental columns and uses
`GroupKFold` to hold out each family. Each report contains MAE, RMSE, and R²
per held-out family, overall metrics, a mean predictor comparison, and ROC-AUC
for `Tc>0 K` and `Tc>77 K`. These family labels are a modeling convention, not
new source annotations.

## Pressure-aware surrogate

`data/training/harmonized_supercon.csv` is the training boundary for
`scripts/train_pressure_aware.py`. Every row carries `provenance` and
`is_synthetic`; production training accepts `review_status` of `confirmed` or
`curated_database` when `is_synthetic == 0`, while automatic SuperCon2
extractions are excluded. Synthetic demonstration rows remain available for
fixtures, candidate-screen smoke tests, and gate plumbing, but never enter
model fitting or cross-validation metrics.

The pipeline uses Matminer's `ElementProperty.from_preset("magpie")` elemental
composition statistics, `applied_pressure_gpa`, and Materials Project
`energy_above_hull`. It fits a random-forest ensemble and reports strict
`GroupKFold` evaluation by `chemical_family`, with per-family regression
metrics, mean-predictor comparison, and the two classification ROC-AUC tasks.
All model artifacts carry the harmonized dataset SHA-256 in both the joblib
bundle and the JSON report.

```bash
python -m pip install -e '.[ml,mp]'
PYTHONPATH=src python scripts/train_pressure_aware.py
```

The five confirmed Y-cuprate records in
`data/training/reviewed/reviewed_records.csv` are bridged automatically by
`scripts/bridge_reviewed_records.py` with `provenance="literature"` and
`is_synthetic=0`. The 3DSC importer assigns `review_status=curated_database`,
keeps all `Tc=0` negatives, and records that its pressure is an explicit
ambient assumption. If an empirical set does not contain two independent
families, the trainer fails closed; the cycle writes a blocked report without
contaminating model weights with synthetic rows.

The output includes `pressure-aware-model.joblib`,
`grouped-oof-predictions.csv`, `candidate_predictions.csv`, and
`promotion_predictions.csv`, and `pressure-aware-report.json`. The full-fit
predictions are for queue screening; the promotion file combines
family-held-out predictions for eligible rows with fitted-model predictions
for excluded negative controls.

### Within-family elemental-system evaluation

Alongside the cross-family GroupKFold estimate, the report now evaluates each
chemical family containing at least 100 eligible empirical rows. Within a
family, rows are grouped by a sorted elemental-system signature (the set of
elements in the formula), and `GroupKFold` holds out entire signatures so
related compositions cannot appear on both sides of a fold. Each family entry
in `within_family_evaluation.families` reports row and system counts, `mae_k`, `mean_predictor_mae_k`, Spearman
rank correlation (`spearman_rho`), and fold-level held-out
system names and metrics. Families with too few distinct systems are marked
blocked; families below 100 rows are omitted. This estimates interpolation
within a known family and complements, rather than replaces, the stricter
family-held-out result.

The promotion gate treats pressures up to 0.000101325 GPa (1 atm) as ambient
and includes both `confirmed` literature records and non-synthetic
`curated_database` records, subject to the existing provenance exclusions.

### Real family-holdout run

The following fallback metrics are retained as an earlier reproducibility
record. They were generated by `train_pressure_aware.py` from the locally
available 3DSC MP table, not from a synthetic fixture, before Matminer was
installed. The current Magpie-backed run is reported below and is the headline
result. The 3DSC source does not supply measurement pressure; its importer
assigns atmospheric pressure as a screening assumption. Thus neither run
measures pressure dependence, and neither establishes ambient-pressure
performance by itself. The current checkout has one local CIF; the earlier
`cif_linked_records` value in this historical JSON should not be interpreted as
current structure coverage.

```json
{
  "run": "train_pressure_aware.py; random_state=42; n_splits=5",
  "source": "3DSC_MP.csv",
  "source_sha256": "353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24",
  "harmonized_dataset_sha256": "9000678ded8098e2c48c9d5d0abbf9c68d530b839831062cb167b4ec4d6f29fc",
  "records": 5773,
  "zero_tc_records": 1778,
  "cif_linked_records": 5773,
  "chemical_family_groups": 9,
  "feature_backend": "fraction_fallback",
  "cross_validation": "GroupKFold by chemical_family",
  "grouped_cv": {
    "mae_k": 9.7336548618,
    "rmse_k": 20.8382500980,
    "r2": -0.1976816187
  },
  "mean_predictor": {
    "mae_k": 16.0524879254,
    "rmse_k": 22.1364901460,
    "r2": -0.3515633822
  },
  "folds": [
    {"fold": 1, "held_out_families": ["Other"], "test_rows": 3529, "mae_k": 4.2626773481, "rmse_k": 7.4423341716, "r2": -0.0752954607},
    {"fold": 2, "held_out_families": ["Cuprates"], "test_rows": 874, "mae_k": 34.1692123276, "rmse_k": 48.5123448484, "r2": -0.8895585342},
    {"fold": 3, "held_out_families": ["Ferrites"], "test_rows": 517, "mae_k": 12.9978802602, "rmse_k": 19.1260627077, "r2": -0.6589759746},
    {"fold": 4, "held_out_families": ["Heavy-fermion", "Heavy-fermion/Chevrel"], "test_rows": 425, "mae_k": 1.1533843132, "rmse_k": 2.0217573223, "r2": -0.0633273508},
    {"fold": 5, "held_out_families": ["Carbon-based", "Chevrel", "Oxides", "Oxides/Heavy-fermion"], "test_rows": 428, "mae_k": 9.5220026948, "rmse_k": 12.1888858432, "r2": -2.0816000536}
  ],
  "interpretation": "Exploratory baseline only: aggregate MAE beats the mean predictor, but aggregate R2 is negative and the held-out cuprate fold performs poorly. This is not evidence of room-temperature discovery or validated pressure prediction."
}
```

The weak extrapolation result is useful: the current surrogate does not yet
generalize reliably to an unseen superconducting family, especially cuprates.
The metrics are a reproducibility checkpoint, not a promotion result. The run
artifacts are temporary and are not committed; the source and harmonized input
hashes above identify the exact data used.

### Historical Magpie-only rerun

After installing Matminer 0.10.1, the same pinned 3DSC input and harmonized
dataset were rerun with `feature_backend="matminer.magpie"`, `random_state=42`,
and five family-held-out folds. This is a historical Magpie-only snapshot; the
enhanced Magpie plus formula-shell/orbital run below is the current headline.
The pressure column remains the 3DSC ambient-pressure assumption, not a
measured pressure feature. Of 5,773 rows, 175 have `Tc > 77 K`;
all are in the Cuprate family. Because Cuprates are wholly held out in one fold,
the model cannot train on any `Tc > 77 K` examples in that fold, and the pooled
out-of-fold ROC-AUC for that threshold is not estimable. Its value is recorded
as JSON `null`, not as a fabricated score.

```json
{
  "run": "train_pressure_aware.py; matminer=0.10.1; random_state=42; n_splits=5",
  "source_sha256": "353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24",
  "harmonized_dataset_sha256": "9000678ded8098e2c48c9d5d0abbf9c68d530b839831062cb167b4ec4d6f29fc",
  "records": 5773,
  "zero_tc_records": 1778,
  "chemical_family_groups": 9,
  "feature_backend": "matminer.magpie",
  "grouped_cv": {"mae_k": 10.4617443549, "rmse_k": 21.0354380924, "r2": -0.2204556849},
  "mean_predictor": {"mae_k": 16.0524879254, "rmse_k": 22.1364901460, "r2": -0.3515633822},
  "roc_auc_tc_gt_0k": 0.5809187103,
  "roc_auc_tc_gt_77k": null,
  "roc_auc_tc_gt_77k_status": "not estimable: all positive records are in the held-out Cuprate family; no positive class is available to train that fold",
  "mean_predictor_roc_auc_tc_gt_0k": 0.5,
  "mean_predictor_roc_auc_tc_gt_77k": 0.5,
  "folds": [
    {"fold": 1, "held_out_families": ["Other"], "test_rows": 3529, "mae_k": 4.8810086485, "rmse_k": 7.8772670691, "r2": -0.2046492628, "roc_auc_tc_gt_0k": 0.5689222704},
    {"fold": 2, "held_out_families": ["Cuprates"], "test_rows": 874, "mae_k": 33.8837070569, "rmse_k": 47.6690893589, "r2": -0.8244397562, "roc_auc_tc_gt_0k": 0.6326289849},
    {"fold": 3, "held_out_families": ["Ferrites"], "test_rows": 517, "mae_k": 12.5494139690, "rmse_k": 18.1359170405, "r2": -0.4916536218, "roc_auc_tc_gt_0k": 0.5551857276},
    {"fold": 4, "held_out_families": ["Heavy-fermion", "Heavy-fermion/Chevrel"], "test_rows": 425, "mae_k": 1.6183434106, "rmse_k": 2.2446631672, "r2": -0.3107241340, "roc_auc_tc_gt_0k": 0.5905472410},
    {"fold": 5, "held_out_families": ["Carbon-based", "Chevrel", "Oxides", "Oxides/Heavy-fermion"], "test_rows": 428, "mae_k": 14.9074011697, "rmse_k": 20.3530768374, "r2": -7.5922795445, "roc_auc_tc_gt_0k": 0.6666214137}
  ],
  "interpretation": "Magpie improves aggregate MAE relative to the mean predictor, but R2 remains negative, the held-out cuprate MAE is 33.88 K, and the >77 K ROC-AUC cannot be estimated. This is an exploratory benchmark, not a useful room-temperature predictor or pressure validation."
}
```

## Structure-aware comparison

`scripts/evaluate_structure_features.py` adds Matminer's
`DensityFeatures` and `GlobalSymmetryFeatures` to the composition features and
repeats the same family-held-out split on CIF-backed records. Its report says
whether the combined model beats composition-only by MAE. If the downloaded
CIF subset does not cover two empirical families, it writes
`blocked_insufficient_structures`; this is a coverage limitation, not a claim
that structure features help. ALIGNN/M3GNet is reserved as an optional second
stage and is reported as not attempted in the baseline cycle.

## 3DSC spin, bond, and dimensionality ablation

`scripts/evaluate_3dsc_structure_spin.py` joins ingested 3DSC records to their
source-table rows and CIFs. The table's `is_magnetic_2`, `ordering_2`,
`total_magnetization_2`, and `num_magnetic_sites_2` values come from the
Materials Project calculations included in 3DSC. They are DFT-computed
features, not measured magnetic properties. Total magnetization is reported in
μB/atom and magnetic-site count per atom by dividing by the source table's
`nsites_2`; ordering is retained as
NM/FM/AFM/FiM one-hot fields while its original categorical label is preserved.

For each matched CIF, pymatgen computes one periodic nearest-neighbor distance
per site (minimum, mean, and standard deviation), the mean metal/nonmetal bond
distance from CrystalNN neighbors, and mean CrystalNN coordination number.
The metal/nonmetal split follows pymatgen's element classification (metalloids
fall in the nonmetal category).
Dimensionality is computed with pymatgen's Larsen algorithm on a CrystalNN
structure graph and represented as 0D/1D/2D/3D one-hot values; `layers_present`
is true for 2D structures. For partially occupied CIFs the feature pipeline
uses the highest-occupancy species at each site and records that proxy choice.

The ablation uses one shared CIF-backed cohort and identical folds for the
composition baseline, spin only, bonds only, patterns only, and all three
groups together. It reports cross-family GroupKFold regression and both
`Tc > 0 K` and `Tc > 77 K` ROC-AUC classifiers, plus within-family
elemental-system GroupKFold metrics for families with at least 100 records.
Missing values remain missing and are imputed from each training fold only.
Full-CIF run (3 November 2026): 5,773/5,773 training records had a matched source-table row and CIF; no structure-feature failures occurred. The cross-family split held out chemical families; within-family splits held out elemental-system signatures for families with at least 100 rows. Each ablation uses identical folds and the same 158-feature composition baseline. Random forests use 150 regression trees and 100 classification trees; all missing feature values are imputed from the training fold only. Values below are out-of-fold; MAE is K, and AUC labels are strict `Tc > threshold`. A dash means AUC could not be estimated because one or more held-out examples had no score from a fold whose training data contained both classes.

| Evaluation | Feature set | Rows | MAE | Mean-predictor MAE | Spearman ρ | ROC-AUC Tc > 0 K | ROC-AUC Tc > 77 K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-family (GroupKFold by family) | Baseline | 5773 | 10.3887 | 16.0525 | 0.1458 | 0.5693 | — |
| Cross-family (GroupKFold by family) | + spin | 5773 | 10.2636 | 16.0525 | 0.1337 | 0.5823 | — |
| Cross-family (GroupKFold by family) | + bonds | 5773 | 10.3982 | 16.0525 | 0.1059 | 0.5780 | — |
| Cross-family (GroupKFold by family) | + patterns | 5773 | 10.3652 | 16.0525 | 0.1452 | 0.5675 | — |
| Cross-family (GroupKFold by family) | + spin + bonds + patterns | 5773 | 10.3286 | 16.0525 | 0.1055 | 0.5844 | — |
| Within Cuprates (elemental-system groups) | Baseline | 874 | 16.9719 | 31.1061 | 0.6861 | 0.7965 | 0.9294 |
| Within Cuprates (elemental-system groups) | + spin | 874 | 16.9560 | 31.1061 | 0.6836 | 0.7951 | 0.9351 |
| Within Cuprates (elemental-system groups) | + bonds | 874 | 16.5728 | 31.1061 | 0.7020 | 0.7977 | 0.9289 |
| Within Cuprates (elemental-system groups) | + patterns | 874 | 16.9702 | 31.1061 | 0.6850 | 0.7989 | 0.9314 |
| Within Cuprates (elemental-system groups) | + spin + bonds + patterns | 874 | 16.7934 | 31.1061 | 0.6925 | 0.8062 | 0.9310 |
| Within Ferrites (elemental-system groups) | Baseline | 517 | 7.7684 | 12.1965 | 0.6402 | 0.8089 | — |
| Within Ferrites (elemental-system groups) | + spin | 517 | 7.7830 | 12.1965 | 0.6358 | 0.8098 | — |
| Within Ferrites (elemental-system groups) | + bonds | 517 | 7.7832 | 12.1965 | 0.6382 | 0.8057 | — |
| Within Ferrites (elemental-system groups) | + patterns | 517 | 7.7562 | 12.1965 | 0.6407 | 0.7991 | — |
| Within Ferrites (elemental-system groups) | + spin + bonds + patterns | 517 | 7.7750 | 12.1965 | 0.6399 | 0.8021 | — |
| Within Heavy-fermion (elemental-system groups) | Baseline | 418 | 0.8314 | 1.1735 | 0.5479 | 0.8196 | — |
| Within Heavy-fermion (elemental-system groups) | + spin | 418 | 0.7638 | 1.1735 | 0.5494 | 0.8213 | — |
| Within Heavy-fermion (elemental-system groups) | + bonds | 418 | 0.8178 | 1.1735 | 0.5696 | 0.8190 | — |
| Within Heavy-fermion (elemental-system groups) | + patterns | 418 | 0.8329 | 1.1735 | 0.5524 | 0.8203 | — |
| Within Heavy-fermion (elemental-system groups) | + spin + bonds + patterns | 418 | 0.7627 | 1.1735 | 0.5603 | 0.8176 | — |
| Within Other (elemental-system groups) | Baseline | 3529 | 1.9543 | 4.8770 | 0.7358 | 0.8486 | — |
| Within Other (elemental-system groups) | + spin | 3529 | 1.9445 | 4.8770 | 0.7378 | 0.8465 | — |
| Within Other (elemental-system groups) | + bonds | 3529 | 1.9302 | 4.8770 | 0.7357 | 0.8512 | — |
| Within Other (elemental-system groups) | + patterns | 3529 | 1.9506 | 4.8770 | 0.7345 | 0.8484 | — |
| Within Other (elemental-system groups) | + spin + bonds + patterns | 3529 | 1.9320 | 4.8770 | 0.7359 | 0.8518 | — |
| Within Oxides (elemental-system groups) | Baseline | 320 | 3.6513 | 5.6932 | 0.6334 | 0.7884 | — |
| Within Oxides (elemental-system groups) | + spin | 320 | 3.6811 | 5.6932 | 0.6292 | 0.8053 | — |
| Within Oxides (elemental-system groups) | + bonds | 320 | 3.6809 | 5.6932 | 0.6308 | 0.8049 | — |
| Within Oxides (elemental-system groups) | + patterns | 320 | 3.7018 | 5.6932 | 0.6311 | 0.7996 | — |
| Within Oxides (elemental-system groups) | + spin + bonds + patterns | 320 | 3.7392 | 5.6932 | 0.6362 | 0.8164 | — |

The source spin fields were populated for all 5,773 records. The metal/nonmetal mean distance was available for 3,964 CIFs; it remains missing for structures with no CrystalNN metal/nonmetal neighbor pair. The `Tc > 77 K` classifier is estimable only for the Cuprates within-family evaluation. In cross-family evaluation, the held-out family can contain all high-Tc examples, leaving no positive examples in training; the corresponding out-of-family classifier score is therefore unavailable rather than imputed.

The JARVIS electron-phonon model also computes the five CIF-like bond features
for each structure and includes them in its fitted lambda/omega_log models.
The bond-feature ablation reports held-out `omega_log` MAE change against the
composition/Lewis-shell baseline; results are reported in the JARVIS ablation
table below.

## Candidate generation and acquisition

`scripts/rank_candidates.py` uses pymatgen to make same-family elemental
substitutions, removes observed formulas, queries Materials Project when
`MP_API_KEY` is available, and rejects known candidates at or above
`0.05 eV/atom`. Without a key it degrades explicitly: hull energy remains
unknown, plausibility is zero, and every shortlist row carries a stability
review reason. The acquisition function caps the uncertainty reward and
multiplies it by plausibility, so uncertainty alone cannot promote an
implausible row.

`data/processed/candidates/top_k.csv` is a shortlist for human review. Its
prediction is a hypothesis from the full-fit ensemble, not a measurement or a
claim that any candidate superconducts.

`scripts/enrich_materials_project.py` and the low-budget screen use the shared
auditor in `rt_superconductor.materials_project`. Before a query, it accepts
only an unambiguous fixed elemental composition: variables, mixed phases,
sample labels, charge/oxidation annotations, grouping syntax, and malformed
tokens are recorded as `ambiguous_input` and are not sent to the API. For a
returned document, the reduced element/amount signature must exactly equal the
query signature; an API search hit alone is not accepted. Every query is
retained in `materials_project_audit.json` with source record IDs, original and
normalized formula, query status, validation status, returned candidates, and
sanitized error detail. Multiple validated polymorphs remain separate in the
audit; enrichment selects the lowest-hull validated alternative only for its
legacy one-row CSV projection. The low-budget artifact reports queried,
matched, no-match, rejected, ambiguous, and API-error counts separately.

The `KD 2 PO 4` → `KP(HO2)2` example is therefore unresolved before querying
because `D` is not silently interpreted as another element; if a fuzzy API
response is returned for a parseable formula, the same composition check rejects
it. The screen establishes only a database-reference hypothesis. It is not AI
training, DFT, experimental validation, or a superconductivity prediction.

The promotion workflow then applies the ambient MAE, hydride pressure-trend,
and excluded-reference false-positive thresholds in `scripts/promotion_gate.py`.

The model is a prioritization tool, not evidence of superconductivity. A
candidate must still receive structure-aware relaxation and experimental or
first-principles validation.

## Current Ferrite-focused validation run

The reproducible Ferrite cycle was run on 2026-11-02 from the pinned 3DSC MP
table (`353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24`).
It ingested 5,773 rows, including 1,778 zero-Tc negatives, and found nine
chemical-family labels. The feature backend was
`matminer.magpie+formula_shell_orbital_descriptors`; no fallback metrics are
used as the headline. Five-fold GroupKFold was used inside each sufficiently
large family, grouping by elemental-system signature so substituted members of
the same system do not cross a fold.

| Family | Rows | Elemental systems | MAE (K) | Mean MAE (K) | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ferrites | 517 | 162 | 7.019 | 12.144 | 0.6803 |
| Cuprates | 874 | 214 | 17.264 | 31.011 | 0.6828 |
| Heavy-fermion | 418 | 222 | 0.876 | 1.181 | 0.5283 |
| Other | 3,529 | 1,653 | 1.908 | 4.882 | 0.7338 |
| Oxides | 320 | 119 | 3.388 | 5.763 | 0.6636 |

The pooled family-held-out result was MAE 10.435 K, RMSE 20.914 K, and R2
-0.206, compared with mean-predictor MAE 16.052 K. The superconducts-at-all
classifier had ROC-AUC 0.5811; the Tc > 77 K ROC-AUC was not estimable because
the positive examples are concentrated in the held-out Cuprate family. These
numbers show useful within-family ranking for Ferrites, not reliable
cross-family discovery.

The exact artifacts are written by `scripts/train_pressure_aware.py`; the
dataset hash for this run is
`ebe808a22db39c12abff28a189c83e0d0540801323320fd43b71105b109a2c01`.
`python scripts/run_ferrite_shortlist.py` regenerates ingestion, training,
promotion evaluation, ranking, literature review intake, and the DFT hand-off
report in one cycle.

The promotion gate result for this run was `fail`. Aggregate ambient MAE was
10.435 K (passing the 12 K ceiling), but aggregate mean-predictor improvement
was 11.77%, below the required 25%. The per-family gate also requires every
family to meet that threshold; Ferrites had promotion-gate MAE 12.225 K versus
12.121 K for its mean predictor. No H3S or LaH10 benchmark rows and no excluded
reference rows were present, so the hydride-trend and false-positive checks
were not satisfied. A failed gate is retained as a useful honest result, not
silently converted into a deployment decision.

Materials Project lookup was `degraded_no_mp_api_key`. Consequently, hull
energy, band gap, metallicity, MP existence, and novelty are unknown rather
than negative. The Ferrite ranker enumerated 272 substitution candidates and
produced 25 rows; `rejected_nonmetal_count` was 0 because no MP electronic
properties were available. Family mode ranks predicted Tc and capped,
plausibility-weighted uncertainty; it does not use the 273 K target window.
The top 25 are grouped into known and unknown database status, and every row
is a hypothesis requiring human and structure review.

## Electron-phonon target model and prototype search

The physics-target workflow is implemented in `scripts/train_physics_targets.py`
and `scripts/run_physics_search.py`. It trains separate random-forest regressors
for electron-phonon coupling `lambda` and `omega_log` (stored in kelvin), then
maps their held-out predictions through the modified McMillan/Allen-Dynes
expression with `mu*=0.1`:

```text
Tc = (omega_log / 1.2) * exp[-1.04(1+lambda) /
     (lambda - mu*(1 + 0.62*lambda))]
```

The strong-coupling `f1`/`f2` factors are omitted, so this is an explicitly
specified screening approximation, not a full Eliashberg calculation. The
training labels are real published DFT results from JARVIS-SuperconDB, not
values generated by this repository. Its NIST/Figshare record contains the
3D table with 1,058 compounds and columns `lamb`, `wlog`, `Tc`, `a2F`, structure,
stability, and pressure. We verified the published Figshare license as CC BY
4.0 and checked the downloaded file against both the publisher MD5 and the
recorded SHA-256. See
[`data/training/physics-target-source.json`](../data/training/physics-target-source.json)
for exact URLs and hashes. Cite K. Choudhary and K. Garrity, “Designing high-TC
superconductors with BCS-inspired screening, density functional theory, and
deep-learning,” *npj Computational Materials* 8, 244 (2022),
[doi:10.1038/s41524-022-00933-1](https://doi.org/10.1038/s41524-022-00933-1),
and the dataset DOI
[10.6084/m9.figshare.21370572.v4](https://doi.org/10.6084/m9.figshare.21370572.v4).

Run the reproducible download, feature build, grouped ablation, and model fit
with:

```bash
python -m pip install -e '.[ml,mp]'
python scripts/train_physics_targets.py
python scripts/enrich_physics_mp.py  # optional; requires working MP API access
python scripts/train_physics_targets.py  # rerun to consume the MP cache
python scripts/run_physics_search.py
python scripts/enrich_candidate_stability.py  # optional exact-composition MP checks
python scripts/run_physics_search.py  # include candidate MP status/hull fields
```

The characteristics table includes one JARVIS compound per row, Magpie
composition statistics, the existing neutral-atom shell and Lewis-dot proxy
descriptors, structure-derived density, hydrogen fraction, and light-element
fraction (atomic number `Z <= 10`). Materials Project DOS at the Fermi level,
band gap, metallicity, and density have dedicated, separate fields. The
`scripts/enrich_physics_mp.py` step only fills these from an exact formula with
a unique MP summary match and records ambiguous phases rather than selecting
one silently. The local feature-table run returned HTTP 403, so its MP fields
were blank and its ablation is reported as blocked; no MP values were inferred
from JARVIS or another database. The later secret-backed Actions run is
recorded below. The feature table
committed with this change is
[`data/training/physics-target-characteristics.csv`](../data/training/physics-target-characteristics.csv)
and records that status per compound.

### Held-out results and feature ablation

The run used 1,057 of the source's 1,058 records; one row with an invalid or
nonpositive EPC target was excluded. Five-fold `GroupKFold` held out complete
elemental-system signatures (834 systems total). Metrics below are the actual
run from this code and the checksum-pinned dataset. Direct-Tc and
physics-derived Tc models share the same held-out `Tc` recomputed from the
source `lambda` and `omega_log` labels using the equation above.

| Feature set | Direct RF Tc MAE (K) | Physics-derived Tc MAE (K) | Δ physics Tc MAE (K) | Lambda MAE | omega_log MAE (K) | Δ omega_log MAE (K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Magpie + Lewis/shell | 2.8602 | 5.8582 | 0.0000 | 0.8643 | 93.6672 | 0.0000 |
| + structure density, H/light fraction, site count | 2.8487 | 5.8715 | +0.0133 | 0.8594 | 93.4718 | -0.1955 |
| + five bond features | 2.8598 | 5.8918 | +0.0336 | 0.8695 | 92.6127 | -1.0546 |
| + structure features and five bond features | 2.8470 | 5.9636 | +0.1054 | 0.8605 | 92.1587 | -1.5085 |
| Frozen pretrained ALIGNN graph embeddings + fold-wise RF probes | 2.8790 | 5.5879 | -0.2703* | 0.8312 | 93.8780 | +0.2108* |
| + Materials Project DOS/gap/metallicity/density | blocked | blocked | n/a | n/a | n/a | n/a |

For the MP ablation, 211 fully populated rows were required by the predefined
coverage rule; actual complete rows were zero in the local run. The bond-only
features reduced held-out `omega_log` MAE by 1.0546 K, while physics-derived
Tc MAE increased by 0.0336 K. Combining all structure and bond features
reduced `omega_log` MAE by 1.5085 K but increased physics-derived Tc MAE by
0.1054 K. These are small changes beside the large absolute `omega_log` error;
they do not validate chemistry extrapolation to room temperature. The `*`
ALIGNN deltas compare against its separate earlier run's baseline metrics and
are not a paired ablation against this newly recomputed JARVIS baseline.

The ALIGNN row uses a frozen 256-dimensional pooled structure representation
from the published `jv_formation_energy_peratom_alignn` checkpoint, followed by
fold-wise random-forest probes for lambda and omega_log. The checkpoint is
pretrained for formation energy, not EPC, so none of the evaluation EPC labels
are used by the pretrained network. The same 5-fold elemental-system
GroupKFold produced 1,057 usable structures and zero graph failures. Source
weights are [ALIGNN Figshare file 31458679](https://ndownloader.figshare.com/files/31458679),
SHA-256 `d80f5e067c0b0b0f569b8158ecfeeeb68472bebe2ceb27ff787dcfcc68948e20`.
The pretrained formation-energy corpus may include held-out structures, so
this is target-label-safe transfer learning with a transductive pretraining
caveat, not a fully independent structure benchmark. It improves
physics-derived Tc MAE by 0.2663 K versus the composition baseline, while
omega_log error is slightly worse; this is a small benchmark change, not proof
of discovery-scale extrapolation.

The existing project random-forest Tc model remains the baseline. A random
forest regression prediction is bounded by the range of target values in its
training leaves; it cannot predict above the training maximum. On this EPC
benchmark the direct-RF target maximum was 31.9108 K after applying the stated
`mu*=0.1` formula (the source's reported Tc maximum is 33.4702 K). The two-target
model can produce a derived Tc above the Tc-label maximum because it combines
separate bounded lambda and omega_log estimates nonlinearly; that does not
make it a validated high-Tc extrapolator. The tree-to-tree standard deviation
is only a rough ensemble-spread indicator and is not calibrated uncertainty.

### Prototype-substitution shortlist

`run_physics_search.py` changes one element at a time within the same periodic
table group, using elements present in the published source structures, and
retains the parent structure as an explicitly unrelaxed prototype. Hydrogen is
never substituted. Metal, metalloid, and nonmetal categories are not crossed;
candidate compositions also need a pymatgen common-oxidation-state charge
balance. On the pinned run this considered 1,257 proposals: 22 were dropped
because a substitution involved hydrogen, 323 for element-class mismatch, and
498 for lacking a common-oxidation-state charge balance; 145 were already in
the source and 38 duplicated another proposal, leaving 231 eligible candidates.
The filter counts are emitted in `physics-search-summary.json`.

Rows with `tc_tree_sd_k > predicted_tc_k` are assigned to the separate
`low_confidence` group and get independent `group_rank` values. The top-20
export reserves up to 10 places per group and backfills if one group is short.
In the actual screened pool all 20 output rows were low-confidence, so there
were zero standard-confidence entries in that export. The current result is
[`physics-target-top20.csv`](physics-target-top20.csv). Candidate stability is
unknown without an exact candidate MP phase calculation; the source prototype
stability is shown separately as a proxy. The latest local MP attempt was
blocked by HTTP 403; candidate-query results from the later successful Actions
run are listed below. Every row is marked `DFT
verification=pending`.

The leading row, CrC on the MoC prototype, has predicted lambda 2.442,
omega_log 250.80 K, and Tc 31.96 K. Its tree-spread Tc SD is 37.63 K, larger
than the mean, and its candidate stability is unknown. This is a model-generated
hypothesis with poor confidence, not a discovery or recommendation to
synthesize. No structure relaxation, phonon calculation, or experimental
check has been performed for this shortlist.

### Secret-backed Materials Project workflow run

The pushed GitHub Actions workflow completed successfully on 2026-11-03:
[run 41689587893](https://github.com/hudsonclavin-cloud/RT-SuperConductor/actions/runs/41689587893).
Its artifact contains the source-compound enrichment CSV, candidate stability
CSV, retrained model report, frozen-ALIGNN report, and final candidate table.
The 1,057 source rows were classified as 347 ambiguous multi-phase formulas,
272 with no exact MP formula match, and 438 with a unique exact-formula match
whose DOS request failed validation. For those 438 rows, MP band gap,
metallicity, and density were retrieved; the Fermi-level DOS remains missing.
Consequently the predefined MP-feature ablation still has zero rows with the
complete feature group and remains blocked. The Action's API requests did
complete with the repository secret; the DOS validation issue remains
unresolved.

The same run queried all 20 shortlisted candidate formulas: 3 returned a
single MP composition match requiring structure review, 8 had multiple
possible MP phases, and 9 had no exact composition match. None of these
statuses establishes that the unrelaxed substituted prototype is stable.
The workflow's run summary and machine-readable outputs are retained in the
Actions artifact rather than committed as raw generated model files. On that
run, the baseline direct-RF Tc MAE was 2.8197 K, the composition-model
physics-derived Tc MAE was 5.9467 K, and the frozen-ALIGNN
physics-derived Tc MAE was 5.2886 K (lambda MAE 0.8469 and omega_log MAE
94.1365 K). These are the metrics from the workflow runtime; the table above
records the separate local benchmark run.

## Deep physical-feature study: Phase 1 — leakage audit

The new `deep_physics_features.py` uses an explicit physical-feature allowlist.
`ingest_3dsc.py --features data/processed/structure_features.csv` writes the
record-keyed table; the feature builder extends it with CIF-derived quantities.
Magnetism and electronic properties in the `_2` columns are Materials Project
DFT results, not measured magnetic properties of the matched superconducting
sample. `run_type_2` is an audit flag, excluded from the predictor matrix.
Point groups and crystal systems are categorical, not ordinal values.

Excluded columns and their leakage risks:

| Excluded field | Reason |
| --- | --- |
| synth_doped | Records the matching/artificial-doping construction, not independently measured structure physics. |
| totreldiff | Composition matching residual can encode which literature label was assigned. |
| formula_similarity | Quality of the label-to-structure matching, not an independent materials property. |
| correct_formula_frac | Matching/cleaning decision depends on how the record was assembled. |
| formula_frac | Matching fraction can identify repeated samples and database construction. |
| weight | Dataset weighting/duplication convention can encode labels or family frequency. |
| created_at / updated columns | Database processing chronology is not experimental publication chronology and may encode ingestion batches. |

All matrix entry points reject these names. CIF results are cached by content
SHA-256, and failed calculations retain a reason. Layer spacing is the largest
empty periodic slab along a cell reciprocal-axis normal for a 2D graph; this
is a cell-dependent proxy, not a symmetry-invariant interlayer binding energy.
Ewald energy uses a common formal oxidation-state assignment when available;
it is an electrostatic proxy, not a measured bond strength or formation energy.
Pure metals retain missing Ewald energy. Disordered CIFs use a recorded
highest-occupancy proxy, so neither geometry nor formal charges reproduce
fractional-site electronic disorder.

Phase 1 measured result (2026-11-03): **5,773 rows, 5,773 CIFs, 0 geometry failures**. Nonmissing counts: `band_gap_2` 5773, `efermi_2` 5773, `ewald_ev_per_atom` 2796, `layer_spacing_proxy_angstrom` 358, `coord_mean` 5773. Source SHA-256: `353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24`. Feature CSV SHA-256: `15d29e07cd9d1a3247d5e6bc6806ae1f5aedd7153a13b799ebfca38dd765cb25`. The generated table is 2.9 MB and is an artifact; raw CIFs and caches are not committed. All **87 unit tests passed**.

## Deep physical-feature study: Phase 2 — paired ablation

All **5,773** records remain in every comparison. Baseline = Magpie plus neutral-atom shell/Lewis descriptors (**155 columns**); pressure and hull columns appended by the older feature builder are removed. Each outer split also purges training rows sharing an MP ID with the held-out rows. Thus these numbers supersede older comparisons for this study; changed baseline inputs and stricter structure separation explain why earlier within-family scores differ. Trees: 150 regression / 100 per binary classifier; seeds fixed by fold. All imputers are fitted on training rows only. AUC uses pooled out-of-fold scores and is unavailable when a fold lacks both training classes or the evaluation has only one class. The mean-class predictor uses each training fold’s positive prevalence, so its pooled AUC need not equal 0.5.

| Scope | Inputs | N | MAE K | Mean MAE K | Spearman | AUC >0 | Mean AUC >0 | AUC >77 | Mean AUC >77 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cross-family | Formula baseline | 5773 | 10.4523 | 16.1169 | 0.1007 | 0.5792 | 0.4575 | — | 0.0624 |
| Ferrites | Formula baseline | 517 | 9.3870 | 11.7727 | 0.5090 | 0.7208 | 0.4730 | — | — |
| Cuprates | Formula baseline | 874 | 21.3573 | 32.6682 | 0.5624 | 0.6717 | 0.4793 | 0.8275 | 0.3187 |
| Other | Formula baseline | 3529 | 2.8606 | 4.3630 | 0.6913 | 0.7932 | 0.4885 | — | — |
| Oxides | Formula baseline | 320 | 3.9724 | 5.8215 | 0.5659 | 0.7344 | 0.4644 | — | — |
| cross-family | + electronic | 5773 | 10.5306 | 16.1169 | 0.1107 | 0.5742 | 0.4575 | — | 0.0624 |
| Ferrites | + electronic | 517 | 9.4925 | 11.7727 | 0.5099 | 0.7331 | 0.4730 | — | — |
| Cuprates | + electronic | 874 | 21.6690 | 32.6682 | 0.5495 | 0.6733 | 0.4793 | 0.8450 | 0.3187 |
| Other | + electronic | 3529 | 2.8746 | 4.3630 | 0.6938 | 0.8018 | 0.4885 | — | — |
| Oxides | + electronic | 320 | 4.0669 | 5.8215 | 0.5608 | 0.7591 | 0.4644 | — | — |
| cross-family | + lattice | 5773 | 10.3884 | 16.1169 | 0.1440 | 0.5811 | 0.4575 | — | 0.0624 |
| Ferrites | + lattice | 517 | 9.2700 | 11.7727 | 0.5135 | 0.7181 | 0.4730 | — | — |
| Cuprates | + lattice | 874 | 21.5255 | 32.6682 | 0.5553 | 0.6735 | 0.4793 | 0.8417 | 0.3187 |
| Other | + lattice | 3529 | 3.0330 | 4.3630 | 0.6726 | 0.7952 | 0.4885 | — | — |
| Oxides | + lattice | 320 | 3.9662 | 5.8215 | 0.5389 | 0.7514 | 0.4644 | — | — |
| cross-family | + bonding | 5773 | 10.4227 | 16.1169 | 0.1236 | 0.5934 | 0.4575 | — | 0.0624 |
| Ferrites | + bonding | 517 | 9.4317 | 11.7727 | 0.5041 | 0.7209 | 0.4730 | — | — |
| Cuprates | + bonding | 874 | 21.2465 | 32.6682 | 0.5626 | 0.6856 | 0.4793 | 0.8262 | 0.3187 |
| Other | + bonding | 3529 | 2.9307 | 4.3630 | 0.6858 | 0.8026 | 0.4885 | — | — |
| Oxides | + bonding | 320 | 4.0443 | 5.8215 | 0.5209 | 0.7530 | 0.4644 | — | — |
| cross-family | + spin | 5773 | 10.3908 | 16.1169 | 0.1038 | 0.5780 | 0.4575 | — | 0.0624 |
| Ferrites | + spin | 517 | 9.3485 | 11.7727 | 0.5058 | 0.7054 | 0.4730 | — | — |
| Cuprates | + spin | 874 | 21.3103 | 32.6682 | 0.5638 | 0.6669 | 0.4793 | 0.8403 | 0.3187 |
| Other | + spin | 3529 | 2.9015 | 4.3630 | 0.6903 | 0.7980 | 0.4885 | — | — |
| Oxides | + spin | 320 | 3.9945 | 5.8215 | 0.5582 | 0.7465 | 0.4644 | — | — |
| cross-family | All physical groups | 5773 | 10.4250 | 16.1169 | 0.1257 | 0.5634 | 0.4575 | — | 0.0624 |
| Ferrites | All physical groups | 517 | 9.3524 | 11.7727 | 0.5211 | 0.7172 | 0.4730 | — | — |
| Cuprates | All physical groups | 874 | 21.2622 | 32.6682 | 0.5621 | 0.6697 | 0.4793 | 0.8194 | 0.3187 |
| Other | All physical groups | 3529 | 3.0830 | 4.3630 | 0.6749 | 0.8051 | 0.4885 | — | — |
| Oxides | All physical groups | 320 | 3.9825 | 5.8215 | 0.5227 | 0.7619 | 0.4644 | — | — |

The professor’s five factors are tested through grouped proxies, not isolated causal interventions. Symmetry and lattice patterns share the lattice group; interatomic distances and electrostatic/formation-energy bond-strength proxies share the bonding group. The electronic group changes cross-family MAE by 0.0782 K; lattice by -0.0639 K; bonding by -0.0297 K; spin by -0.0615 K; all together by -0.0273 K. Negative changes mean lower error. Improvements vary by family and are small; no statistical significance or causal mechanism is established.

**Cuprate spin limitation:** directly counted in the source table: NM 743/874, FM 69, FiM 48, AFM 14. Real undoped cuprate parents are antiferromagnetic; standard reference-cell DFT and its magnetic initialization/correlation approximation can miss that state, and matched/artificially doped references are not necessarily the experimental sample. These labels cannot substitute for measured spin order. See Lee, Nagaosa & Wen, [Rev. Mod. Phys. 78, 17 (2006)](https://doi.org/10.1103/RevModPhys.78.17).

### Top 20 cross-family feature importances

Impurity importances averaged across outer folds are model diagnostics, not causal importance. The complete top-20 lists for every group and scope, plus MP-overlap purge counts, are in [deep-phase2-ablation.json](deep-phase2-ablation.json).

| Inputs | Feature | Importance |
|---|---|---:|
| Formula baseline | MagpieData range Electronegativity | 0.115359 |
| Formula baseline | MagpieData mean NUnfilled | 0.115272 |
| Formula baseline | MagpieData avg_dev GSvolume_pa | 0.068974 |
| Formula baseline | MagpieData avg_dev NdValence | 0.061713 |
| Formula baseline | MagpieData avg_dev NUnfilled | 0.052427 |
| Formula baseline | MagpieData avg_dev GSmagmom | 0.035682 |
| Formula baseline | descriptor_total_atoms | 0.031690 |
| Formula baseline | descriptor_shell_3_fill_pct | 0.029744 |
| Formula baseline | MagpieData avg_dev Electronegativity | 0.028035 |
| Formula baseline | descriptor_shell_3_electrons | 0.027225 |
| Formula baseline | MagpieData avg_dev Column | 0.020912 |
| Formula baseline | MagpieData avg_dev SpaceGroupNumber | 0.018014 |
| Formula baseline | descriptor_avg_p_electrons | 0.017914 |
| Formula baseline | MagpieData mean SpaceGroupNumber | 0.016375 |
| Formula baseline | MagpieData mean NdValence | 0.015491 |
| Formula baseline | MagpieData mean MeltingT | 0.012474 |
| Formula baseline | MagpieData avg_dev NValence | 0.012161 |
| Formula baseline | MagpieData avg_dev MendeleevNumber | 0.012020 |
| Formula baseline | MagpieData mean Electronegativity | 0.011849 |
| Formula baseline | MagpieData mean MendeleevNumber | 0.011356 |
| + electronic | MagpieData range Electronegativity | 0.115416 |
| + electronic | MagpieData mean NUnfilled | 0.114699 |
| + electronic | MagpieData avg_dev GSvolume_pa | 0.068334 |
| + electronic | MagpieData avg_dev NdValence | 0.061836 |
| + electronic | MagpieData avg_dev NUnfilled | 0.052114 |
| + electronic | MagpieData avg_dev GSmagmom | 0.035541 |
| + electronic | descriptor_total_atoms | 0.031538 |
| + electronic | descriptor_shell_3_electrons | 0.030768 |
| + electronic | MagpieData avg_dev Electronegativity | 0.027094 |
| + electronic | descriptor_shell_3_fill_pct | 0.025910 |
| + electronic | MagpieData avg_dev Column | 0.019537 |
| + electronic | MagpieData avg_dev SpaceGroupNumber | 0.017401 |
| + electronic | descriptor_avg_p_electrons | 0.017195 |
| + electronic | MagpieData mean SpaceGroupNumber | 0.015984 |
| + electronic | MagpieData mean NdValence | 0.015065 |
| + electronic | MagpieData mean MeltingT | 0.012614 |
| + electronic | descriptor_shell_6_fill_pct | 0.011784 |
| + electronic | MagpieData mean MendeleevNumber | 0.011528 |
| + electronic | MagpieData avg_dev MendeleevNumber | 0.011483 |
| + electronic | MagpieData avg_dev NValence | 0.011477 |
| + lattice | MagpieData range Electronegativity | 0.115190 |
| + lattice | MagpieData mean NUnfilled | 0.105718 |
| + lattice | MagpieData avg_dev GSvolume_pa | 0.067467 |
| + lattice | MagpieData avg_dev NUnfilled | 0.055863 |
| + lattice | c_over_a | 0.054646 |
| + lattice | MagpieData avg_dev GSmagmom | 0.035501 |
| + lattice | latc_2 | 0.032531 |
| + lattice | MagpieData avg_dev NdValence | 0.031917 |
| + lattice | descriptor_shell_3_fill_pct | 0.028662 |
| + lattice | descriptor_shell_3_electrons | 0.027413 |
| + lattice | MagpieData avg_dev Electronegativity | 0.023083 |
| + lattice | MagpieData avg_dev Column | 0.019310 |
| + lattice | descriptor_avg_p_electrons | 0.017726 |
| + lattice | MagpieData avg_dev SpaceGroupNumber | 0.015309 |
| + lattice | MagpieData mean NdValence | 0.013418 |
| + lattice | MagpieData mean MeltingT | 0.012560 |
| + lattice | MagpieData mean Electronegativity | 0.011183 |
| + lattice | MagpieData mean SpaceGroupNumber | 0.011055 |
| + lattice | MagpieData avg_dev MendeleevNumber | 0.010353 |
| + lattice | MagpieData avg_dev NValence | 0.010202 |
| + bonding | MagpieData mean NUnfilled | 0.101535 |
| + bonding | shortest_metal_anion | 0.092217 |
| + bonding | MagpieData range Electronegativity | 0.076029 |
| + bonding | MagpieData avg_dev NUnfilled | 0.046160 |
| + bonding | ewald_ev_per_atom | 0.042913 |
| + bonding | MagpieData avg_dev GSmagmom | 0.036031 |
| + bonding | MagpieData avg_dev NdValence | 0.029273 |
| + bonding | descriptor_shell_3_electrons | 0.028242 |
| + bonding | MagpieData avg_dev GSvolume_pa | 0.028108 |
| + bonding | descriptor_shell_3_fill_pct | 0.027398 |
| + bonding | coord_fraction_8 | 0.019502 |
| + bonding | MagpieData avg_dev Column | 0.019485 |
| + bonding | MagpieData avg_dev SpaceGroupNumber | 0.017734 |
| + bonding | descriptor_avg_p_electrons | 0.017027 |
| + bonding | MagpieData avg_dev Electronegativity | 0.015937 |
| + bonding | formation_energy_per_atom_2 | 0.014078 |
| + bonding | MagpieData mean MeltingT | 0.013785 |
| + bonding | descriptor_shell_6_electrons | 0.013261 |
| + bonding | MagpieData mean SpaceGroupNumber | 0.011639 |
| + bonding | descriptor_shell_6_fill_pct | 0.011344 |
| + spin | MagpieData range Electronegativity | 0.115263 |
| + spin | MagpieData mean NUnfilled | 0.114591 |
| + spin | MagpieData avg_dev GSvolume_pa | 0.068666 |
| + spin | MagpieData avg_dev NdValence | 0.061072 |
| + spin | MagpieData avg_dev NUnfilled | 0.051934 |
| + spin | MagpieData avg_dev GSmagmom | 0.034978 |
| + spin | descriptor_total_atoms | 0.031475 |
| + spin | descriptor_shell_3_fill_pct | 0.029498 |
| + spin | MagpieData avg_dev Electronegativity | 0.027577 |
| + spin | descriptor_shell_3_electrons | 0.027291 |
| + spin | MagpieData avg_dev Column | 0.020136 |
| + spin | descriptor_avg_p_electrons | 0.018141 |
| + spin | MagpieData avg_dev SpaceGroupNumber | 0.017038 |
| + spin | MagpieData mean SpaceGroupNumber | 0.015895 |
| + spin | MagpieData mean NdValence | 0.014840 |
| + spin | MagpieData mean MeltingT | 0.012146 |
| + spin | descriptor_shell_6_electrons | 0.011824 |
| + spin | total_magnetization_normalized_formula_units_2 | 0.011581 |
| + spin | MagpieData avg_dev MendeleevNumber | 0.011498 |
| + spin | MagpieData avg_dev NValence | 0.011385 |
| All physical groups | MagpieData mean NUnfilled | 0.099908 |
| All physical groups | shortest_metal_anion | 0.091748 |
| All physical groups | MagpieData range Electronegativity | 0.075692 |
| All physical groups | MagpieData avg_dev NUnfilled | 0.045214 |
| All physical groups | MagpieData avg_dev GSmagmom | 0.035253 |
| All physical groups | ewald_ev_per_atom | 0.034455 |
| All physical groups | c_over_a | 0.030322 |
| All physical groups | descriptor_shell_3_fill_pct | 0.028762 |
| All physical groups | latc_2 | 0.027238 |
| All physical groups | MagpieData avg_dev GSvolume_pa | 0.027103 |
| All physical groups | descriptor_shell_3_electrons | 0.026441 |
| All physical groups | MagpieData avg_dev Column | 0.017698 |
| All physical groups | descriptor_avg_p_electrons | 0.016920 |
| All physical groups | MagpieData avg_dev SpaceGroupNumber | 0.014659 |
| All physical groups | MagpieData avg_dev NdValence | 0.014017 |
| All physical groups | MagpieData avg_dev Electronegativity | 0.014001 |
| All physical groups | MagpieData mean MeltingT | 0.012809 |
| All physical groups | formation_energy_per_atom_2 | 0.012510 |
| All physical groups | descriptor_shell_6_electrons | 0.011265 |
| All physical groups | MagpieData mean Electronegativity | 0.009854 |

## Deep physical-feature study: Phase 3 — frozen structure encoders

Measured frozen CHGNet 0.3.0 results on **5,773 structures, 0 graph failures**, using the exact Phase 2 outer partitions and shared-MP training embargo. The 64-dimensional encoder is frozen; only fold-local random-forest regressors and binary classifiers are fitted. Formula descriptors are not included in this row; a combined formula+encoder control is being evaluated separately. Fine-tuning (five epochs per outer fold) and scratch ALIGNN (compact two ALIGNN/two GCN layers, five epochs) are still in progress and are not yet assigned performance numbers.

| Scope | MAE K | Mean MAE K | Spearman | AUC >0 | Mean AUC >0 | AUC >77 | Mean AUC >77 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cross-family | 10.5140 | 16.1169 | -0.0347 | 0.5155 | 0.4575 | — | 0.0624 |
| Ferrites | 11.2309 | 11.7727 | 0.2475 | 0.5987 | 0.4730 | — | — |
| Cuprates | 22.0833 | 32.6682 | 0.5282 | 0.6684 | 0.4793 | 0.8416 | 0.3187 |
| Other | 3.4535 | 4.3630 | 0.5232 | 0.7501 | 0.4885 | — | — |
| Oxides | 4.6464 | 5.8215 | 0.3191 | 0.7342 | 0.4644 | — | — |

The frozen representation does not improve the formula baseline: cross-family MAE 10.5140 K versus 10.4523 K, and AUC 0.5155 versus 0.5792. All **3,435 partially occupied CIFs** use a highest-occupancy ordered proxy; that proxy cannot reproduce experimental dopant/vacancy disorder. Pretraining used Materials Project trajectories, which may contain the held-out reference structures. This is target-label-safe transfer learning, not an independent test of structural pretraining generalization.

**JARVIS frozen encoder:** 1,057 usable structures; lambda MAE 0.8505 and omega_log MAE 93.7629 K. The paired formula baseline is 93.6672 K, so the change is **0.0957 K**, slightly worse even though the result is below the rounded 94 K reference. This comparison uses the same five elemental-system folds, 180 trees, leaf size two, and target-specific seeds as the physics-target benchmark. Source labels are computed EPC properties, not experiments. The raw source has 1,058 rows, all labeled 0 GPa, and source stability labels `stable=626`, `unstable=432`; one row fails the positive-EPC-target filter. The source stability label is retained as metadata, never substituted for candidate stability.

**Methods and provenance:** installed CHGNet 0.4.2, its 0.3.0 checkpoint SHA-256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`; [CHGNet source](https://github.com/CederGroupHub/chgnet), [published model](https://doi.org/10.1038/s42256-023-00716-3). ALIGNN 2026.11.1 / DGL 2.1.0 runs in an isolated Torch 2.2.1 CPU environment; CHGNet uses Torch 2.15.0 on CPU. The `physical` and `graph` extras are intended for separate environments because their Torch requirements differ. NumPy 1.26.4 / SciPy 1.16.3 are retained for binary compatibility. See [frozen metrics](deep-phase3-frozen-report.json) and [EPC metrics](deep-chgnet-jarvis.json).

An environment reconnect interrupted the initial training attempts. Completed frozen/EPC results were retained, while unfinished models received no invented metrics. [Attempt audit](deep-phase3-attempts.json) records measured process CPU time. Recovery now persists graphs, model/optimizer/RNG state at minibatch/epoch boundaries, and completed evaluation scopes; CPU consumption carries across restarts into the six-hour cap. No raw structures, weights, model files, or large caches are committed.

Phase 3 progress validation: **93 unit tests passed** after adding persistent
training/evaluation checkpoints and cumulative CPU-budget tests. This is an
intermediate checkpoint; supervised encoder/scratch comparisons remain open.
