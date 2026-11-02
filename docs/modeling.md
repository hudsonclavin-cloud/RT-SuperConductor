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

## Candidate generation and acquisition

`scripts/rank_candidates.py` uses pymatgen to make same-family elemental
substitutions, removes observed formulas, queries Materials Project when
`MP_API_KEY` is available, and rejects known candidates at or above
`0.05 eV/atom`. Without a key it degrades explicitly: hull energy remains
unknown, plausibility is zero, and every shortlist row carries a stability
review reason. Fractional candidates with no exact MP composition match may
include `parent_mp_id` and `parent_energy_above_hull`; these columns are marked
`parent_proxy_status` and refer only to the integer lineage parent. They never
populate the candidate's `mp_id` or `energy_above_hull` and are never used as
candidate stability features. Rows with `uncertainty_k > predicted_tc_kelvin`
are flagged `high_uncertainty` and placed in their own lower-priority ranking
group. The acquisition function caps the uncertainty reward and multiplies it
by plausibility, so uncertainty alone cannot promote an implausible row.

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
