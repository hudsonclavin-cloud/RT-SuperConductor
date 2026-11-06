# Existing-but-untested Materials Project shortlist (v2)

This is a computational screening list, not evidence that any entry superconducts or is globally untested. The source comparison covers 3DSC, UCI, Stanev, SuperCon2, and SuperMat. The provisional metallicity gate accepts Materials Project summary `is_metal=true` or `band_gap=0` and records `metallicity_source=mp_summary_provisional`; these are screening labels, not experimental validation. DOS(Ef) above 0.1 states/eV/atom from a Static or Uniform calculation is optional enrichment when its curve is available. Missing curves and DOS at or below the threshold never block pool inclusion.

Run status: **ready**. Main shortlist size: **120**; EPW-eligible list size: **3370**; eligible families: **4**.

## Counts per filter

Counts are cumulative at each stage unless the stage is explicitly a rejection counter.

| Stage | Count |
|---|---:|
| `mp_summary_returned` | 30358 |
| `energy_above_hull_le_0p025` | 30358 |
| `site_count_le_20` | 30358 |
| `formula_unparseable` | 0 |
| `formula_matches_known_superconductor_source` | 2764 |
| `oxygen_variant_of_known` | 138 |
| `ambiguous_formula_source_system_overlap` | 678 |
| `composition_policy_rejected` | 3574 |
| `unmatched_reduced_formula` | 23882 |
| `mp_summary_is_metal_true` | 15472 |
| `mp_summary_band_gap_zero` | 15465 |
| `mp_summary_metallicity_accepted` | 15472 |
| `mp_summary_metallicity_rejected` | 8410 |
| `structure_material_ids_queried` | 15472 |
| `structure_documents_returned` | 15472 |
| `missing_structure_records` | 0 |
| `ordered_structures` | 15472 |
| `ordered_and_source_unmatched` | 15472 |
| `dos_formula_fallback_materials_missing_refs` | 15472 |
| `dos_formula_fallback_materials_recovered` | 157 |
| `dos_formula_fallback_task_refs_recovered` | 802 |
| `dos_formula_fallback_query_errors` | 0 |
| `dos_formula_fallback_task_documents_returned` | 3224 |
| `dos_formula_fallback_non_static_uniform_tasks_rejected` | 2341 |
| `dos_formula_fallback_structure_comparisons` | 1145 |
| `dos_formula_fallback_structure_matches` | 802 |
| `dos_metadata_returned` | 15629 |
| `dos_task_references` | 157 |
| `dos_static_or_uniform_tasks` | 124 |
| `dos_non_static_uniform_tasks_rejected` | 0 |
| `dos_curves_retrieved` | 0 |
| `dos_missing_task_or_curve` | 15472 |
| `dos_below_or_equal_0p1_states_ev_atom` | 0 |
| `dos_qualified_gt_0p1_states_ev_atom` | 0 |
| `dos_ef_gt_0p1_optional_enrichment` | 0 |
| `dos_material_ids_queried` | 92 |
| `dos_task_ids_queried` | 13237 |
| `task_documents_returned` | 13237 |
| `dos_payload_fetch_failures` | 13237 |
| `dos_curve_validation_failures` | 92 |
| `source_formula_unmatched` | 23882 |
| `summary_metallicity_qualified` | 15472 |

## Label quarantine and training comparison

Active quarantined labels excluded from empirical training: **3**. Source observations are preserved. The three `mp-510625` 3DSC entries are suspected label errors because their high transition labels share one MP structure; this is an audit flag, not a declaration that the source values are false. The pinned 3DSC scan found **0** other empirical records above 77 K without Cu (the rule remains enabled for future rows). Synthetic demo fixtures are marked not applicable, not label errors.
Previous shortlist nearest-neighbor references directly affected by these quarantined rows: **0**.

| Family | Before rows / MAE / Spearman | After quarantine rows / MAE / Spearman |
|---|---|---|
| Cuprates | 879 / 21.508 K / 0.5322 | 876 / 21.340 K / 0.5442 |
| Ferrites | 517 / 9.039 K / 0.5153 | 517 / 9.039 K / 0.5153 |
| Heavy-fermion | 418 / 1.150 K / 0.2761 | 418 / 1.150 K / 0.2761 |
| Other | 3529 / 3.036 K / 0.6392 | 3529 / 3.036 K / 0.6392 |
| Oxides | 320 / 3.810 K / 0.5576 | 320 / 3.810 K / 0.5576 |

## Optional DOS(Ef) enrichment diagnostics

Material-associated DOS retrievals attempted: 92; curves returned: 0; fetch errors by type: `{"OSError": 92}`.
Retrieval path: `optional: candidate material_id -> eligible static/uniform task_id -> parsed DOS`.
Curve-validation errors by type: `{"ValueError": 92}`.
Formula-based task lookup recovered 802 task references for 157 exact-structure matches from 15472 candidates missing DOS references; task documents returned: 3224; non-static/uniform tasks rejected: 2341; structure comparisons/matches: 1145/802; lookup errors: `{}`.
- Example `mp-1025306`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaicryh.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-12073`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaibdzy.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-1227433`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaiiwmo.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-1227663`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaijmkz.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-15889`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaigraw.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-cgisw`: ValueError  — No parsed DOS curve returned for the material's eligible tasks
- Example `mp-rwj`: ValueError  — No parsed DOS curve returned for the material's eligible tasks
- Example `mp-crvsz`: ValueError  — No parsed DOS curve returned for the material's eligible tasks
- Example `mp-crwbv`: ValueError  — No parsed DOS curve returned for the material's eligible tasks
- Example `mp-xnd`: ValueError  — No parsed DOS curve returned for the material's eligible tasks

## Family gate and shortlist

| Family | Main within-family Spearman | Assigned candidates | Scored | Top 30 |
|---|---:|---:|---:|---:|
| Cuprates | 0.5442 | 84 | 84 | 30 |
| Ferrites | 0.5153 | 61 | 61 | 30 |
| Other | 0.6392 | 4827 | 4827 | 30 |
| Oxides | 0.5576 | 428 | 428 | 30 |

## Magnetic exclusions for the EPW route

MP summary magnetic order and total magnetization are carried into each scored row. Any MP `is_magnetic=true`, FM/AFM/FiM ordering, or nonzero total magnetization is excluded from the EPW-eligible list. Unknown magnetic state is also excluded rather than assumed non-magnetic. The separate EPW audit CSV retains each candidate and its reason.

| EPW decision | Count |
|---|---:|
| `eligible_non_magnetic_other_max_20_sites` | 3370 |
| `family_not_other` | 573 |
| `magnetic_excluded_from_epw` | 1457 |

## FeCuO2 follow-up

Previous-shortlist outcome for `FeCuO2` (`mp-505110`): **removed** — `not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query`. Its oxygen-insensitive key is `FeCu`; matching source formulas: `[]`. With no matching source formula, the oxygen-variant rule does not remove it. Current run screening values: predicted Tc 30.171184 K, tree spread 29.25503076708252 K; nearest measured reference YBa2Cu4O8 at 80.0 K. It is assigned to Cuprates and is therefore outside the Other-only EPW list.

The separate EPW list is `data/processed/continuous-training/epw-eligible-shortlist.csv` and contains only Other-family, explicitly non-magnetic entries with at most 20 sites, ranked by predicted Tc minus tree-spread SD. Its nearest known compound and measured Tc are included for context.

### Assignment/scoring outcomes

| Outcome | Count |
|---|---:|
| `family_assigned_within_distance_and_definers` | 5991 |
| `family_gate_passed` | 5400 |
| `family_gate_skipped` | 591 |
| `no_nearest_training_record` | 0 |
| `family_unassigned` | 9481 |
| `family_assignment_distance_rejected` | 8830 |
| `family_definer_rejected` | 651 |
| `family_assigned:Carbon-based` | 7 |
| `family_assigned:Chevrel` | 52 |
| `family_assigned:Cuprates` | 84 |
| `family_assigned:Ferrites` | 61 |
| `family_assigned:Heavy-fermion` | 532 |
| `family_assigned:Other` | 4827 |
| `family_assigned:Oxides` | 428 |

Rejected candidate counts are keyed by MP ID in the report JSON under `candidate_assignment_outcomes`; unassigned records carry a reason and are never scored.


The family gate uses `within_family_evaluation.families[*].spearman_rho` from `train_pressure_aware.py`. That evaluator keeps shared Materials Project IDs in the same validation fold. Candidates are assigned only when the nearest training record is within distance 0.3; Ferrites require Fe, and Cuprates require both Cu and O.

## Previous shortlist: before and after

Previous rows: 120; retained in this top-30 output: 0; removed: 120.

| Old rank | MP ID | Formula | Old family | V2 disposition | Reason |
|---:|---|---|---|---|---|
| 1 | mp-1228298 | Ba3SrTl2Cu2HgO10 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 2 | mp-561182 | Ba4Tl2Cu2HgO10 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 3 | mp-1147633 | LiCu2BiO4 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 4 | mp-1223089 | La3Nd(CuO4)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 5 | mp-1099249 | Mg3CuO4 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 6 | mp-10335 | Ba2CuTeO6 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 7 | mp-1227234 | CaMn2Cu3(RuO6)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 8 | mp-9417 | Sm(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 9 | mp-4371 | Nd(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 10 | mp-1045844 | YCu3(NiO3)4 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 11 | mp-505110 | FeCuO2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 12 | mp-1045774 | YCo4(CuO4)3 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 13 | mp-1218759 | Sr2NdTlCu2O7 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 14 | mp-1045770 | YFe4(CuO4)3 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 15 | mp-754674 | Lu(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 16 | mp-22618 | Y(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 17 | mp-755805 | Sc(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 18 | mp-1216344 | VGa(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 19 | mp-9418 | Eu(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 20 | mp-20707 | Gd(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 21 | mp-1218849 | Sr2GdCu2IrO8 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 22 | mp-9416 | La(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 23 | mp-555332 | BaCu(SiO3)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 24 | mp-559633 | Cu6PbO8 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 25 | mp-1186515 | PrCuO3 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 26 | mp-1101476 | Pr(CuO2)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 27 | mp-722515 | NaCu2H2(SeO5)2 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 28 | mp-1214598 | Ba2NdDyTi2Cu2O11 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 29 | mp-1228257 | Ba2YCu3PbO7 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 30 | mp-554109 | CuSeO4 | Cuprates | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 1 | mp-1219594 | RbTl(FeSe)8 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 2 | mp-1220955 | NaSr2(FeAs)6 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 3 | mp-616481 | NbFeTe2 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 4 | mp-9884 | KFeSe2 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 5 | mp-15120 | RbFe2Se3 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 6 | mp-22434 | HfFeP | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 7 | mp-662563 | CsFe2Se3 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 8 | mp-1102843 | FeMoP | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 9 | mp-1216176 | Y(CrFe3)3 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 10 | mp-1189647 | Ba(FeSb3)4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 11 | mp-1188542 | Sr(FeSb3)4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 12 | mp-1219022 | SmTiFe11 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 13 | mp-1218432 | Sr3Sm(FeO4)2 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 14 | mp-13464 | Ca(FeSb3)4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 15 | mp-1209892 | Nd(FeAs3)4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 16 | mp-1218069 | SrPr(FeO3)2 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 17 | mp-1224771 | Fe3NiP4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 18 | mp-1207125 | Cs(FeP)2 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 19 | mp-1189397 | Pr(FeAs3)4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 20 | mp-1224768 | Fe3NiAs8 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 21 | mp-1219069 | SmFe11Mo | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 22 | mp-1218464 | Sr4TlFe2O9 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 23 | mp-20856 | Eu(FeP3)4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 24 | mp-1217545 | TbFe11Mo | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 25 | mp-1226290 | CrFe3As4 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 26 | mp-1217868 | Tb(Fe3Mo)3 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 27 | mp-1225727 | DyTiFe11 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 28 | mp-1101988 | MnFeP | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 29 | mp-1217506 | TbTiFe11 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 30 | mp-505312 | Sr2FeCuSO3 | Ferrites | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 1 | mp-1221007 | Nb12InSn3 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 2 | mp-1220795 | Nb12Sn3Sb | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 3 | mp-1215994 | YLu(BRh)8 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 4 | mp-1229062 | Al(V4Si)3 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 5 | mp-1220758 | Nb12IrAu3 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 6 | mp-1216436 | V6GaGe | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 7 | mp-1229060 | Al(V4Ge)3 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 8 | mp-1228829 | AlV6Ge | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 9 | mp-1228831 | AlV6Sb | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 10 | mp-1228822 | AlV6Sn | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 11 | mp-1229114 | Al(V4Sn)3 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 12 | mp-1224108 | HgPb4 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 13 | mp-1216450 | V6GaSb | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 14 | mp-1221653 | MnNb4(CS2)2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 15 | mp-1217435 | TbY2Ni6(B2C)3 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 16 | mp-1224285 | HfTaC2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 17 | mp-1219462 | SbTe2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 18 | mp-1223890 | In4Pb | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 19 | mp-1215219 | ZrTaC2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 20 | mp-9459 | Y4C5 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 21 | mp-867883 | La3Pb | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 22 | mp-1224247 | HfZrN2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 23 | mp-1215233 | ZrNb3C4 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 24 | mp-1220391 | Nb6PdPt | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 25 | mp-1216692 | TiNbB4 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 26 | mp-1215222 | ZrNbC2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 27 | mp-11473 | TlHg | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 28 | mp-1220596 | Nb4AgS8 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 29 | mp-11028 | K3Hg11 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 30 | mp-1189574 | YWC2 | Other | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 1 | mp-1223105 | La4Bi4S8O3F | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 2 | mp-1278550 | LiV3O4 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 3 | mp-754378 | LiV3O4 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 4 | mp-11906 | LiPr2IrO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 5 | mp-11905 | LiNd2IrO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 6 | mp-10321 | La2ZnRhO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 7 | mp-1041214 | CaHo(FeO3)2 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 8 | mp-1043900 | CaLa(FeO3)2 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 9 | mp-1044733 | CaPr(NiO3)2 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 10 | mp-1217787 | SrSmMn2O6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 11 | mp-1218059 | SrPrMn2O6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 12 | mp-1218091 | SrPr(CoO3)2 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 13 | mp-1218102 | SrNdMn2O6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 14 | mp-1218109 | SrPrFeCoO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 15 | mp-1218237 | SrLa(FeO3)2 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 16 | mp-1218831 | Sr2FeRuO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 17 | mp-1220568 | Nd2CoNiO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 18 | mp-1223335 | La2FeNiO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 19 | mp-1272890 | La2ZnCoO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 20 | mp-1274023 | La2ZnFeO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 21 | mp-1279341 | Sr2MnMoO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 22 | mp-1295386 | La2MgFeO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 23 | mp-18847 | V2CdO4 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 24 | mp-19170 | PrNiO3 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 25 | mp-19208 | La2MnCoO6 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 26 | mp-22280 | PrNiO3 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 27 | mp-22495 | SmNiO3 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 28 | mp-755865 | TbNiO3 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 29 | mp-1208510 | Sr2YFe3O8 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |
| 30 | mp-19102 | Sr2FeO4 | Oxides | removed | not returned by the current stable (<=0.025 eV/atom), <=20-site MP summary query |

## Limits

Materials Project formation energies, ordering, and DOS are DFT outputs. The random-forest Tc values are family-specific screening predictions with tree-spread uncertainty, not calibrated probabilities or discoveries. The term ‘untested’ only means absent by reduced formula from the five indexed exports; a literature review is still required. Candidate ranking is pending independent DFT and experimental validation.


## Archived report before the provisional summary-metallicity gate

The following report is retained unchanged for historical comparison. Its empty result reflected the former mandatory DOS-curve gate; the current run above supersedes that screening logic.

# Existing-but-untested Materials Project shortlist (v2)

This is a computational screening list, not evidence that any entry superconducts or is globally untested. The source comparison covers 3DSC, UCI, Stanev, SuperCon2, and SuperMat. Metallic acceptance requires a Materials Project DOS(Ef) above 0.1 states/eV/atom from a Static or Uniform calculation. DOS curves are resolved from task IDs linked to the candidate material and checked against task metadata; the selected task ID is recorded for any qualified candidate. Missing parsed DOS objects remain blocked. Relaxation calculations and band gaps are not substitutes.

Run status: **blocked_dos_curve_retrieval**. Shortlist size: **0**; eligible families: **4**.

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
| `structure_material_ids_queried` | 23964 |
| `structure_documents_returned` | 23964 |
| `missing_structure_records` | 0 |
| `ordered_structures` | 23964 |
| `ordered_and_source_unmatched` | 23964 |
| `dos_formula_fallback_materials_missing_refs` | 23964 |
| `dos_formula_fallback_materials_recovered` | 230 |
| `dos_formula_fallback_task_refs_recovered` | 1221 |
| `dos_formula_fallback_query_errors` | 0 |
| `dos_formula_fallback_task_documents_returned` | 5575 |
| `dos_formula_fallback_non_static_uniform_tasks_rejected` | 4057 |
| `dos_formula_fallback_structure_comparisons` | 2162 |
| `dos_formula_fallback_structure_matches` | 1221 |
| `dos_metadata_returned` | 24194 |
| `dos_task_references` | 230 |
| `dos_static_or_uniform_tasks` | 196 |
| `dos_non_static_uniform_tasks_rejected` | 0 |
| `dos_curves_retrieved` | 0 |
| `dos_missing_task_or_curve` | 23964 |
| `dos_below_or_equal_0p1_states_ev_atom` | 0 |
| `dos_qualified_gt_0p1_states_ev_atom` | 0 |
| `dos_material_ids_queried` | 150 |
| `dos_task_ids_queried` | 19973 |
| `task_documents_returned` | 19973 |
| `dos_payload_fetch_failures` | 150 |
| `dos_curve_validation_failures` | 150 |
| `source_formula_unmatched` | 23964 |
| `metallic_dos_qualified` | 0 |

## DOS payload retrieval diagnostics

Material-associated DOS retrievals attempted: 150; curves returned: 0; fetch errors by type: `{"OSError": 150}`.
Retrieval path: `candidate material_id -> eligible static/uniform task_id -> parsed DOS`.
Curve-validation errors by type: `{"OSError": 150}`.
Formula-based task lookup recovered 1221 task references for 230 exact-structure matches from 23964 candidates missing DOS references; task documents returned: 5575; non-static/uniform tasks rejected: 4057; structure comparisons/matches: 2162/1221; lookup errors: `{}`.
- Example `mp-1214889`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaahmwlb.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-2172`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaiclfo.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-8881`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaiebhz.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-3104`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaibdnk.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-861940`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaidaff.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-1214889`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaahmwlb.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-2172`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaiclfo.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-8881`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaiebhz.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-3104`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaibdnk.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.
- Example `mp-861940`: OSError  — unable to access bucket: 'materialsproject-parsed' key: 'dos/aaaidaff.json.gz' version: None error: An error occurred (NoSuchKey) when calling the GetObject operation: The specified key does not exist.

## Family gate and shortlist

| Family | Main within-family Spearman | Assigned candidates | Scored | Top 30 |
|---|---:|---:|---:|---:|
| Cuprates | 0.5322 | 0 | 0 | 0 |
| Ferrites | 0.5153 | 0 | 0 | 0 |
| Other | 0.6392 | 0 | 0 | 0 |
| Oxides | 0.5576 | 0 | 0 | 0 |

### Assignment/scoring outcomes

| Outcome | Count |
|---|---:|
| `family_assigned_within_distance_and_definers` | 0 |
| `family_gate_passed` | 0 |
| `family_gate_skipped` | 0 |
| `no_nearest_training_record` | 0 |
| `family_unassigned` | 0 |
| `family_assignment_distance_rejected` | 0 |
| `family_definer_rejected` | 0 |

Rejected candidate counts are keyed by MP ID in the report JSON under `candidate_assignment_outcomes`; unassigned records carry a reason and are never scored.


The family gate uses `within_family_evaluation.families[*].spearman_rho` from `train_pressure_aware.py`. That evaluator keeps shared Materials Project IDs in the same validation fold. Candidates are assigned only when the nearest training record is within distance 0.3; Ferrites require Fe, and Cuprates require both Cu and O.

## Previous shortlist: before and after

Previous rows: 150; retained in this top-30 output: 0; removed: 150.

| Old rank | MP ID | Formula | Old family | V2 disposition | Reason |
|---:|---|---|---|---|---|
| 1 | mp-1063607 | CrPdO2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 2 | mp-505110 | FeCuO2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 3 | mp-996990 | BaAgO2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 4 | mp-997034 | BaCuO2 | Cuprates | removed | undoped_bacuO2_parent_composition |
| 5 | mp-1113808 | Rb2MnCuF6 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 6 | mp-1025177 | Rb2CuF4 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 7 | mp-20982 | K3Cu2F7 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 8 | mp-1206438 | Rb2CuF6 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 9 | mp-1147665 | K4MgCu3F12 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 10 | mp-5699 | K2CuF4 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 11 | mp-1221799 | Mn3Cu2NiO8 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 12 | mp-1217677 | Tb2CuAg | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 13 | mp-867283 | Ho2CuRu | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 14 | mp-1206453 | Sm2CdCu2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 15 | mp-1216344 | VGa(CuO2)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 16 | mp-865220 | Dy2CuRu | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 17 | mp-1105690 | PrFe4(CuO4)3 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 18 | mp-755584 | CrPtO2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 19 | mp-1079700 | Dy6CoTe2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 20 | mp-755805 | Sc(CuO2)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 21 | mp-555101 | K3Co2F7 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 22 | mp-1217672 | Tb2ZnCu | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 23 | mp-1105642 | TbFe4(CuO4)3 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 24 | mp-9417 | Sm(CuO2)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 25 | mp-13649 | Cs2AgF4 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 26 | mp-22618 | Y(CuO2)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 27 | mp-1105138 | SrFe4(CuO4)3 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 28 | mp-1207159 | Cs3EuF6 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 29 | mp-1229021 | CsVCuF6 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 30 | mp-976404 | NaF3 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 31 | mp-754674 | Lu(CuO2)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 32 | mp-864745 | Na3ZrF8 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 33 | mp-1227234 | CaMn2Cu3(RuO6)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 34 | mp-1045774 | YCo4(CuO4)3 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 35 | mp-9416 | La(CuO2)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 36 | mp-1045844 | YCu3(NiO3)4 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 37 | mp-866100 | Er2CuOs | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 38 | mp-1205871 | Dy6FeBi2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 39 | mp-1184585 | Ho2CuOs | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 40 | mp-16751 | ErCuPb | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 41 | mp-1078184 | Eu4Bi2O | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 42 | mp-864959 | Dy2CuOs | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 43 | mp-16750 | HoCuPb | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 44 | mp-27496 | YUO4 | Cuprates | removed | radioactive_element:U |
| 45 | mp-1223089 | La3Nd(CuO4)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 46 | mp-504573 | Cr2CuO4 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 47 | mp-12105 | RbO2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 48 | mp-22085 | DyCuPb | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 49 | mp-1272781 | Cr2CuO4 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 50 | mp-1103973 | Cr2CuO4 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 1 | mp-1087511 | UFeAs2 | Ferrites | removed | radioactive_element:U |
| 2 | mp-20803 | USbAs | Ferrites | removed | radioactive_element:U |
| 3 | mp-1077953 | Eu4As2O | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 4 | mp-1226520 | CeTh2S3 | Ferrites | removed | radioactive_element:Th |
| 5 | mp-1077201 | UAsP | Ferrites | removed | radioactive_element:U |
| 6 | mp-1232230 | EuSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 7 | mp-787 | U3P4 | Ferrites | removed | radioactive_element:U |
| 8 | mp-4374 | UAsS | Ferrites | removed | radioactive_element:U |
| 9 | mp-1219900 | Pr2UP3 | Ferrites | removed | radioactive_element:U |
| 10 | mp-15691 | NpAsS | Ferrites | removed | radioactive_element:Np |
| 11 | mp-567978 | Pa3P4 | Ferrites | removed | radioactive_element:Pa |
| 12 | mp-1207101 | KMnSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 13 | mp-862869 | Pa2S3 | Ferrites | removed | radioactive_element:Pa |
| 14 | mp-1219176 | Sm2AsSe | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 15 | mp-1018954 | PuSe2 | Ferrites | removed | radioactive_element:Pu |
| 16 | mp-10516 | Dy6FeTe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 17 | mp-1018665 | CeSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 18 | mp-550454 | Ba2Mn2As2O | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 19 | mp-1018707 | GdSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 20 | mp-1079205 | Tb2NiAs2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 21 | mp-1078556 | Dy2NiAs2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 22 | mp-20995 | Eu4As3 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 23 | mp-9884 | KFeSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 24 | mp-1216654 | U2NiSb4 | Ferrites | removed | radioactive_element:U |
| 25 | mp-8787 | Ho2NiAs2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 26 | mp-1217712 | Tb2NiSb4 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 27 | mp-1087228 | Er2NiAs2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 28 | mp-1225313 | Dy2NiSb4 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 29 | mp-1215868 | Yb2USe3O2 | Ferrites | removed | radioactive_element:U |
| 30 | mp-1219144 | Sm2AsSe | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 31 | mp-38593 | Sm2AsSe | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 32 | mp-22434 | HfFeP | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 33 | mp-1025077 | TbSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 34 | mp-662563 | CsFe2Se3 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 35 | mp-1223952 | Ho2NiSb4 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 36 | mp-7086 | RbFeSe2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 37 | mp-1078445 | Nd2BiN | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 38 | mp-1095352 | LiEuAs | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 39 | mp-7080 | Li5(NiN)3 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 40 | mp-1102843 | FeMoP | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 41 | mp-1219596 | Rb5Bi4 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 42 | mp-1095668 | TaFeAs | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 43 | mp-4989 | CrNiAs | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 44 | mp-1219226 | SiNi3W2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 45 | mp-1227043 | CaSm2S3 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 46 | mp-1216694 | TmAlFe | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 47 | mp-1219280 | SiNi3Mo2 | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 48 | mp-13074 | LuNiPb | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 49 | mp-10300 | ZrMoP | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 50 | mp-505016 | CsMnAs | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 1 | mp-1245396 | Mn3SiN4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 2 | mp-975106 | Rb2Bi2O5 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 3 | mp-1206874 | CoH | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 4 | mp-759968 | Na2BiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 5 | mp-769558 | LiMn(CO3)2 | Oxides | removed | carbonate_salt |
| 6 | mp-753103 | LiCo(CO3)2 | Oxides | removed | carbonate_salt |
| 7 | mp-21434 | CoCO3 | Oxides | removed | carbonate_salt |
| 8 | mp-1180084 | Ni(HO)2 | Oxides | removed | hydroxide_salt |
| 9 | mp-1278550 | LiV3O4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 10 | mp-754378 | LiV3O4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 11 | mp-643570 | NiH2SO5 | Oxides | removed | sulfate_salt |
| 12 | mp-18969 | FeCO3 | Oxides | removed | carbonate_salt |
| 13 | mp-30147 | ZrTiAs | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 14 | mp-13920 | Ba(ZnSn)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 15 | mp-1041214 | CaHo(FeO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 16 | mp-1043874 | CaLaFeNiO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 17 | mp-1043900 | CaLa(FeO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 18 | mp-1044733 | CaPr(NiO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 19 | mp-1217787 | SrSmMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 20 | mp-1218059 | SrPrMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 21 | mp-1218091 | SrPr(CoO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 22 | mp-1218102 | SrNdMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 23 | mp-1218109 | SrPrFeCoO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 24 | mp-1218237 | SrLa(FeO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 25 | mp-1218831 | Sr2FeRuO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 26 | mp-1220568 | Nd2CoNiO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 27 | mp-1223335 | La2FeNiO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 28 | mp-1272890 | La2ZnCoO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 29 | mp-1274023 | La2ZnFeO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 30 | mp-1279341 | Sr2MnMoO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 31 | mp-1295386 | La2MgFeO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 32 | mp-18720 | Mn2CdO4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 33 | mp-18847 | V2CdO4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 34 | mp-19170 | PrNiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 35 | mp-19208 | La2MnCoO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 36 | mp-22280 | PrNiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 37 | mp-22495 | SmNiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 38 | mp-556857 | VCdO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 39 | mp-755865 | TbNiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 40 | mp-1208510 | Sr2YFe3O8 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 41 | mp-19102 | Sr2FeO4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 42 | mp-1094054 | Sr2FeCoO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 43 | mp-1275986 | YFeO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 44 | mp-19242 | YNiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 45 | mp-22668 | YCoO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 46 | mp-510624 | SrFeO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 47 | mp-1184362 | Gd2CdHg | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 48 | mp-1209118 | RbYb(WO4)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 49 | mp-1227144 | CaNdMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 50 | mp-1227151 | CaSmMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |

## Limits

Materials Project formation energies, ordering, and DOS are DFT outputs. The random-forest Tc values are family-specific screening predictions with tree-spread uncertainty, not calibrated probabilities or discoveries. The term ‘untested’ only means absent by reduced formula from the five indexed exports; a literature review is still required. Candidate ranking is pending independent DFT and experimental validation.

## Snapshot replay follow-up (2026-11-06)

GitHub Actions run [42250629328](https://github.com/hudsonclavin-cloud/RT-SuperConductor/actions/runs/42250629328) replayed the saved Materials Project snapshot and completed successfully in **7m25s**. The corrected queue has **3 active suspected label-error rows** for `mp-510625`; six synthetic H3S/LaH10 fixtures remain in the queue as `not_applicable` with `excluded_from_training=false`. The pinned empirical 3DSC scan found **0** other `Tc > 77 K` records without Cu, and **0** old shortlist nearest-neighbor references depend on the quarantined rows. The empirical training table contains 5,775 rows, so the replay leaves the previously reported per-family before/after metrics unchanged.

The published EPW list has **3,370** entries, all `Other` family, explicitly non-magnetic, and at most 20 sites. It is sorted by predicted Tc minus tree-spread SD; every row has a nearest measured-compound formula and Tc. This only defines a screening queue: DFT relaxation, phonons, and EPW calculations remain pending.

`FeCuO2` was not rejected as an oxygen-only variant: the oxygen-insensitive key `FeCu` has no matching source formula. The old shortlist row used MP ID `mp-505110`, which the refreshed stable-material query did not return. The formula is present in the current shortlist under a different returned ID, `mp-bctfi` (Cuprates rank 11). That current row is marked magnetic (`FM`, total magnetization 9.9977027), so it is excluded from the EPW list both because it is magnetic and because the EPW list is `Other`-only. Its nearest measured reference is YBa2Cu4O8 at 80 K; its predicted Tc is 30.1712 K with 29.2550 K tree spread. These are predictions, not experimental evidence for FeCuO2.
