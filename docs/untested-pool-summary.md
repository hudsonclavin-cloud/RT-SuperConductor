# Existing-but-untested Materials Project shortlist (v2)

This is a computational screening list, not evidence that any entry superconducts or is globally untested. The source comparison covers 3DSC, UCI, Stanev, SuperCon2, and SuperMat. The provisional metallicity gate accepts Materials Project summary `is_metal=true` or `band_gap=0` and records `metallicity_source=mp_summary_provisional`; these are screening labels, not experimental validation. DOS(Ef) above 0.1 states/eV/atom from a Static or Uniform calculation is optional enrichment when its curve is available. Missing curves and DOS at or below the threshold never block pool inclusion.

Run status: **ready**. Shortlist size: **120**; eligible families: **4**.

## Counts per filter

Counts are cumulative at each stage unless the stage is explicitly a rejection counter.

| Stage | Count |
|---|---:|
| `mp_summary_returned` | 30358 |
| `energy_above_hull_le_0p025` | 30358 |
| `site_count_le_20` | 30358 |
| `formula_unparseable` | 0 |
| `formula_matches_known_superconductor_source` | 2764 |
| `ambiguous_formula_source_system_overlap` | 718 |
| `composition_policy_rejected` | 3630 |
| `unmatched_reduced_formula` | 23964 |
| `mp_summary_is_metal_true` | 15500 |
| `mp_summary_band_gap_zero` | 15493 |
| `mp_summary_metallicity_accepted` | 15500 |
| `mp_summary_metallicity_rejected` | 8464 |
| `structure_material_ids_queried` | 15500 |
| `structure_documents_returned` | 15500 |
| `missing_structure_records` | 0 |
| `ordered_structures` | 15500 |
| `ordered_and_source_unmatched` | 15500 |
| `dos_formula_fallback_materials_missing_refs` | 15500 |
| `dos_formula_fallback_materials_recovered` | 154 |
| `dos_formula_fallback_task_refs_recovered` | 806 |
| `dos_formula_fallback_query_errors` | 0 |
| `dos_formula_fallback_task_documents_returned` | 3358 |
| `dos_formula_fallback_non_static_uniform_tasks_rejected` | 2416 |
| `dos_formula_fallback_structure_comparisons` | 1112 |
| `dos_formula_fallback_structure_matches` | 806 |
| `dos_metadata_returned` | 15654 |
| `dos_task_references` | 154 |
| `dos_static_or_uniform_tasks` | 115 |
| `dos_non_static_uniform_tasks_rejected` | 0 |
| `dos_curves_retrieved` | 0 |
| `dos_missing_task_or_curve` | 15500 |
| `dos_below_or_equal_0p1_states_ev_atom` | 0 |
| `dos_qualified_gt_0p1_states_ev_atom` | 0 |
| `dos_ef_gt_0p1_optional_enrichment` | 0 |
| `dos_material_ids_queried` | 88 |
| `dos_task_ids_queried` | 13284 |
| `task_documents_returned` | 13284 |
| `dos_payload_fetch_failures` | 88 |
| `dos_curve_validation_failures` | 88 |
| `source_formula_unmatched` | 23964 |
| `summary_metallicity_qualified` | 15500 |

## Optional DOS(Ef) enrichment diagnostics

Material-associated DOS retrievals attempted: 88; curves returned: 0; fetch errors by type: `{"OSError": 88}`.
Retrieval path: `optional: candidate material_id -> eligible static/uniform task_id -> parsed DOS`.
Curve-validation errors by type: `{"OSError": 88}`.
Formula-based task lookup recovered 806 task references for 154 exact-structure matches from 15500 candidates missing DOS references; task documents returned: 3358; non-static/uniform tasks rejected: 2416; structure comparisons/matches: 1112/806; lookup errors: `{}`.
- Example `mp-1025306`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaicryh.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-12073`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaibdzy.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-755117`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaicdxr.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-1103999`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaifnbi.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-335`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaidnto.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-1025306`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaicryh.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-12073`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaibdzy.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-755117`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaicdxr.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-1103999`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaifnbi.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-335`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaidnto.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.

## Family gate and shortlist

| Family | Main within-family Spearman | Assigned candidates | Scored | Top 30 |
|---|---:|---:|---:|---:|
| Cuprates | 0.5322 | 95 | 95 | 30 |
| Ferrites | 0.5153 | 61 | 61 | 30 |
| Other | 0.6392 | 4827 | 4827 | 30 |
| Oxides | 0.5576 | 439 | 439 | 30 |

### Assignment/scoring outcomes

| Outcome | Count |
|---|---:|
| `family_assigned_within_distance_and_definers` | 6013 |
| `family_gate_passed` | 5422 |
| `family_gate_skipped` | 591 |
| `no_nearest_training_record` | 0 |
| `family_unassigned` | 9487 |
| `family_assignment_distance_rejected` | 8822 |
| `family_definer_rejected` | 665 |
| `family_assigned:Carbon-based` | 7 |
| `family_assigned:Chevrel` | 52 |
| `family_assigned:Cuprates` | 95 |
| `family_assigned:Ferrites` | 61 |
| `family_assigned:Heavy-fermion` | 532 |
| `family_assigned:Other` | 4827 |
| `family_assigned:Oxides` | 439 |

Rejected candidate counts are keyed by MP ID in the report JSON under `candidate_assignment_outcomes`; unassigned records carry a reason and are never scored.


The family gate uses `within_family_evaluation.families[*].spearman_rho` from `train_pressure_aware.py`. That evaluator keeps shared Materials Project IDs in the same validation fold. Candidates are assigned only when the nearest training record is within distance 0.3; Ferrites require Fe, and Cuprates require both Cu and O.

## Previous shortlist: before and after

Previous rows: 0; retained in this top-30 output: 0; removed: 0.

| Old rank | MP ID | Formula | Old family | V2 disposition | Reason |
|---:|---|---|---|---|---|
| — | — | — | — | baseline unavailable | Prior shortlist CSV was not provided to the CI run. |

## Limits

Materials Project formation energies, ordering, and DOS are DFT outputs. The random-forest Tc values are family-specific screening predictions with tree-spread uncertainty, not calibrated probabilities or discoveries. The term ‘untested’ only means absent by reduced formula from the five indexed exports; a literature review is still required. Candidate ranking is pending independent DFT and experimental validation.
