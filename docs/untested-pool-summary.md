# Existing-but-untested Materials Project shortlist (v2)

This is a computational screening list, not evidence that any entry superconducts or is globally untested. The source comparison covers 3DSC, UCI, Stanev, SuperCon2, and SuperMat. Metallic acceptance requires a retrieved Materials Project DOS(Ef) above 0.1 states/eV/atom from a task whose task_type or calc_type is Static or Uniform; the functional-only run_type is not used to infer calculation type. Relaxation DOS and band gaps are not substitutes.

Run status: **ready**. Shortlist size: **90**; eligible families: **3**.

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
| `dos_metadata_returned` | 23964 |
| `dos_task_references` | 18987 |
| `dos_static_or_uniform_tasks` | 18987 |
| `dos_non_static_uniform_tasks_rejected` | 0 |
| `dos_curves_retrieved` | 18984 |
| `dos_missing_task_or_curve` | 4980 |
| `dos_below_or_equal_0p1_states_ev_atom` | 6515 |
| `dos_qualified_gt_0p1_states_ev_atom` | 12469 |
| `dos_task_ids_queried` | 18987 |
| `task_documents_returned` | 18987 |
| `dos_payload_fetch_failures` | 0 |
| `dos_curve_validation_failures` | 3 |
| `source_formula_unmatched` | 23964 |
| `metallic_dos_qualified` | 12469 |

## DOS payload retrieval diagnostics

Eligible static/uniform task IDs: 18987; curves returned: 18987; fetch errors by type: `{}`.
Curve-validation errors by type: `{"ValueError": 3}`.
- Example `aaadrblu`: ValueError  — DOS grid, Fermi level, or site count is invalid
- Example `aaadrbev`: ValueError  — DOS grid, Fermi level, or site count is invalid
- Example `aaadrkvg`: ValueError  — DOS grid, Fermi level, or site count is invalid

## Family gate and shortlist

| Family | Main within-family Spearman | Assigned candidates | Scored | Top 30 |
|---|---:|---:|---:|---:|
| Cuprates | 0.5401 | 84 | 84 | 30 |
| Ferrites | 0.5783 | 37 | 37 | 30 |
| Oxides | 0.5128 | 348 | 348 | 30 |

### Assignment/scoring outcomes

| Outcome | Count |
|---|---:|
| `family_assigned_within_distance_and_definers` | 4805 |
| `family_gate_passed` | 469 |
| `family_gate_skipped` | 4336 |
| `no_nearest_training_record` | 0 |
| `family_unassigned` | 7664 |
| `family_assignment_distance_rejected` | 7094 |
| `family_definer_rejected` | 570 |
| `family_assigned:Carbon-based` | 6 |
| `family_assigned:Chevrel` | 30 |
| `family_assigned:Cuprates` | 84 |
| `family_assigned:Ferrites` | 37 |
| `family_assigned:Heavy-fermion` | 467 |
| `family_assigned:Other` | 3833 |
| `family_assigned:Oxides` | 348 |

Rejected candidate counts are keyed by MP ID in the report JSON under `candidate_assignment_outcomes`; unassigned records carry a reason and are never scored.


The family gate uses `within_family_evaluation.families[*].spearman_rho` from `train_pressure_aware.py`. That evaluator keeps shared Materials Project IDs in the same validation fold. Candidates are assigned only when the nearest training record is within distance 0.3; Ferrites require Fe, and Cuprates require both Cu and O.

## Previous shortlist: before and after

Previous rows: 150; retained in this top-30 output: 31; removed: 119.

| Old rank | MP ID | Formula | Old family | V2 disposition | Reason |
|---:|---|---|---|---|---|
| 1 | mp-1063607 | CrPdO2 | Cuprates | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 2 | mp-505110 | FeCuO2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 3 | mp-996990 | BaAgO2 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 4 | mp-997034 | BaCuO2 | Cuprates | removed | undoped_bacuO2_parent_composition |
| 5 | mp-1113808 | Rb2MnCuF6 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 6 | mp-1025177 | Rb2CuF4 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 7 | mp-20982 | K3Cu2F7 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 8 | mp-1206438 | Rb2CuF6 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 9 | mp-1147665 | K4MgCu3F12 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 10 | mp-5699 | K2CuF4 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 11 | mp-1221799 | Mn3Cu2NiO8 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 12 | mp-1217677 | Tb2CuAg | Cuprates | removed | family_definer_missing:Cu_and_O |
| 13 | mp-867283 | Ho2CuRu | Cuprates | removed | family_definer_missing:Cu_and_O |
| 14 | mp-1206453 | Sm2CdCu2 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 15 | mp-1216344 | VGa(CuO2)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 16 | mp-865220 | Dy2CuRu | Cuprates | removed | family_definer_missing:Cu_and_O |
| 17 | mp-1105690 | PrFe4(CuO4)3 | Cuprates | removed | unassigned_family_distance_gt_0.3 |
| 18 | mp-755584 | CrPtO2 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 19 | mp-1079700 | Dy6CoTe2 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 20 | mp-755805 | Sc(CuO2)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 21 | mp-555101 | K3Co2F7 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 22 | mp-1217672 | Tb2ZnCu | Cuprates | removed | family_definer_missing:Cu_and_O |
| 23 | mp-1105642 | TbFe4(CuO4)3 | Cuprates | removed | unassigned_family_distance_gt_0.3 |
| 24 | mp-9417 | Sm(CuO2)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 25 | mp-13649 | Cs2AgF4 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 26 | mp-22618 | Y(CuO2)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 27 | mp-1105138 | SrFe4(CuO4)3 | Cuprates | removed | unassigned_family_distance_gt_0.3 |
| 28 | mp-1207159 | Cs3EuF6 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 29 | mp-1229021 | CsVCuF6 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 30 | mp-976404 | NaF3 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 31 | mp-754674 | Lu(CuO2)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 32 | mp-864745 | Na3ZrF8 | Cuprates | removed | alkali_alkaline_earth_transition_metal_fluoride |
| 33 | mp-1227234 | CaMn2Cu3(RuO6)2 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 34 | mp-1045774 | YCo4(CuO4)3 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 35 | mp-9416 | La(CuO2)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 36 | mp-1045844 | YCu3(NiO3)4 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 37 | mp-866100 | Er2CuOs | Cuprates | removed | family_definer_missing:Cu_and_O |
| 38 | mp-1205871 | Dy6FeBi2 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 39 | mp-1184585 | Ho2CuOs | Cuprates | removed | family_definer_missing:Cu_and_O |
| 40 | mp-16751 | ErCuPb | Cuprates | removed | family_definer_missing:Cu_and_O |
| 41 | mp-1078184 | Eu4Bi2O | Cuprates | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 42 | mp-864959 | Dy2CuOs | Cuprates | removed | family_definer_missing:Cu_and_O |
| 43 | mp-16750 | HoCuPb | Cuprates | removed | family_definer_missing:Cu_and_O |
| 44 | mp-27496 | YUO4 | Cuprates | removed | radioactive_element:U |
| 45 | mp-1223089 | La3Nd(CuO4)2 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 46 | mp-504573 | Cr2CuO4 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 47 | mp-12105 | RbO2 | Cuprates | removed | family_definer_missing:Cu_and_O |
| 48 | mp-22085 | DyCuPb | Cuprates | removed | family_definer_missing:Cu_and_O |
| 49 | mp-1272781 | Cr2CuO4 | Cuprates | removed | no_static_or_uniform_dos_task_reference |
| 50 | mp-1103973 | Cr2CuO4 | Cuprates | retained | still ranks within the v2 top 30 for its family |
| 1 | mp-1087511 | UFeAs2 | Ferrites | removed | radioactive_element:U |
| 2 | mp-20803 | USbAs | Ferrites | removed | radioactive_element:U |
| 3 | mp-1077953 | Eu4As2O | Ferrites | removed | family_definer_missing:Fe |
| 4 | mp-1226520 | CeTh2S3 | Ferrites | removed | radioactive_element:Th |
| 5 | mp-1077201 | UAsP | Ferrites | removed | radioactive_element:U |
| 6 | mp-1232230 | EuSe2 | Ferrites | removed | family_definer_missing:Fe |
| 7 | mp-787 | U3P4 | Ferrites | removed | radioactive_element:U |
| 8 | mp-4374 | UAsS | Ferrites | removed | radioactive_element:U |
| 9 | mp-1219900 | Pr2UP3 | Ferrites | removed | radioactive_element:U |
| 10 | mp-15691 | NpAsS | Ferrites | removed | radioactive_element:Np |
| 11 | mp-567978 | Pa3P4 | Ferrites | removed | radioactive_element:Pa |
| 12 | mp-1207101 | KMnSe2 | Ferrites | removed | family_definer_missing:Fe |
| 13 | mp-862869 | Pa2S3 | Ferrites | removed | radioactive_element:Pa |
| 14 | mp-1219176 | Sm2AsSe | Ferrites | removed | family_definer_missing:Fe |
| 15 | mp-1018954 | PuSe2 | Ferrites | removed | radioactive_element:Pu |
| 16 | mp-10516 | Dy6FeTe2 | Ferrites | removed | unassigned_family_distance_gt_0.3 |
| 17 | mp-1018665 | CeSe2 | Ferrites | removed | family_definer_missing:Fe |
| 18 | mp-550454 | Ba2Mn2As2O | Ferrites | removed | family_definer_missing:Fe |
| 19 | mp-1018707 | GdSe2 | Ferrites | removed | family_definer_missing:Fe |
| 20 | mp-1079205 | Tb2NiAs2 | Ferrites | removed | family_definer_missing:Fe |
| 21 | mp-1078556 | Dy2NiAs2 | Ferrites | removed | family_definer_missing:Fe |
| 22 | mp-20995 | Eu4As3 | Ferrites | removed | family_definer_missing:Fe |
| 23 | mp-9884 | KFeSe2 | Ferrites | retained | still ranks within the v2 top 30 for its family |
| 24 | mp-1216654 | U2NiSb4 | Ferrites | removed | radioactive_element:U |
| 25 | mp-8787 | Ho2NiAs2 | Ferrites | removed | family_definer_missing:Fe |
| 26 | mp-1217712 | Tb2NiSb4 | Ferrites | removed | family_definer_missing:Fe |
| 27 | mp-1087228 | Er2NiAs2 | Ferrites | removed | family_definer_missing:Fe |
| 28 | mp-1225313 | Dy2NiSb4 | Ferrites | removed | family_definer_missing:Fe |
| 29 | mp-1215868 | Yb2USe3O2 | Ferrites | removed | radioactive_element:U |
| 30 | mp-1219144 | Sm2AsSe | Ferrites | removed | family_definer_missing:Fe |
| 31 | mp-38593 | Sm2AsSe | Ferrites | removed | family_definer_missing:Fe |
| 32 | mp-22434 | HfFeP | Ferrites | retained | still ranks within the v2 top 30 for its family |
| 33 | mp-1025077 | TbSe2 | Ferrites | removed | family_definer_missing:Fe |
| 34 | mp-662563 | CsFe2Se3 | Ferrites | retained | still ranks within the v2 top 30 for its family |
| 35 | mp-1223952 | Ho2NiSb4 | Ferrites | removed | family_definer_missing:Fe |
| 36 | mp-7086 | RbFeSe2 | Ferrites | removed | unassigned_family_distance_gt_0.3 |
| 37 | mp-1078445 | Nd2BiN | Ferrites | removed | family_definer_missing:Fe |
| 38 | mp-1095352 | LiEuAs | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 39 | mp-7080 | Li5(NiN)3 | Ferrites | removed | family_definer_missing:Fe |
| 40 | mp-1102843 | FeMoP | Ferrites | retained | still ranks within the v2 top 30 for its family |
| 41 | mp-1219596 | Rb5Bi4 | Ferrites | removed | family_definer_missing:Fe |
| 42 | mp-1095668 | TaFeAs | Ferrites | removed | unassigned_family_distance_gt_0.3 |
| 43 | mp-4989 | CrNiAs | Ferrites | removed | family_definer_missing:Fe |
| 44 | mp-1219226 | SiNi3W2 | Ferrites | removed | family_definer_missing:Fe |
| 45 | mp-1227043 | CaSm2S3 | Ferrites | removed | family_definer_missing:Fe |
| 46 | mp-1216694 | TmAlFe | Ferrites | removed | no_static_or_uniform_dos_task_reference |
| 47 | mp-1219280 | SiNi3Mo2 | Ferrites | removed | family_definer_missing:Fe |
| 48 | mp-13074 | LuNiPb | Ferrites | removed | family_definer_missing:Fe |
| 49 | mp-10300 | ZrMoP | Ferrites | removed | family_definer_missing:Fe |
| 50 | mp-505016 | CsMnAs | Ferrites | removed | family_definer_missing:Fe |
| 1 | mp-1245396 | Mn3SiN4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 2 | mp-975106 | Rb2Bi2O5 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 3 | mp-1206874 | CoH | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 4 | mp-759968 | Na2BiO3 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 5 | mp-769558 | LiMn(CO3)2 | Oxides | removed | carbonate_salt |
| 6 | mp-753103 | LiCo(CO3)2 | Oxides | removed | carbonate_salt |
| 7 | mp-21434 | CoCO3 | Oxides | removed | carbonate_salt |
| 8 | mp-1180084 | Ni(HO)2 | Oxides | removed | hydroxide_salt |
| 9 | mp-1278550 | LiV3O4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 10 | mp-754378 | LiV3O4 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 11 | mp-643570 | NiH2SO5 | Oxides | removed | sulfate_salt |
| 12 | mp-18969 | FeCO3 | Oxides | removed | carbonate_salt |
| 13 | mp-30147 | ZrTiAs | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 14 | mp-13920 | Ba(ZnSn)2 | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 15 | mp-1041214 | CaHo(FeO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 16 | mp-1043874 | CaLaFeNiO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 17 | mp-1043900 | CaLa(FeO3)2 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 18 | mp-1044733 | CaPr(NiO3)2 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 19 | mp-1217787 | SrSmMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 20 | mp-1218059 | SrPrMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 21 | mp-1218091 | SrPr(CoO3)2 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 22 | mp-1218102 | SrNdMn2O6 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 23 | mp-1218109 | SrPrFeCoO6 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 24 | mp-1218237 | SrLa(FeO3)2 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 25 | mp-1218831 | Sr2FeRuO6 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 26 | mp-1220568 | Nd2CoNiO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 27 | mp-1223335 | La2FeNiO6 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 28 | mp-1272890 | La2ZnCoO6 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 29 | mp-1274023 | La2ZnFeO6 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 30 | mp-1279341 | Sr2MnMoO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 31 | mp-1295386 | La2MgFeO6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 32 | mp-18720 | Mn2CdO4 | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 33 | mp-18847 | V2CdO4 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 34 | mp-19170 | PrNiO3 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 35 | mp-19208 | La2MnCoO6 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 36 | mp-22280 | PrNiO3 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 37 | mp-22495 | SmNiO3 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 38 | mp-556857 | VCdO3 | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 39 | mp-755865 | TbNiO3 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 40 | mp-1208510 | Sr2YFe3O8 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 41 | mp-19102 | Sr2FeO4 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 42 | mp-1094054 | Sr2FeCoO6 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 43 | mp-1275986 | YFeO3 | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 44 | mp-19242 | YNiO3 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 45 | mp-22668 | YCoO3 | Oxides | removed | dos_ef_le_0.1_states_per_ev_per_atom |
| 46 | mp-510624 | SrFeO3 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 47 | mp-1184362 | Gd2CdHg | Oxides | removed | unassigned_family_distance_gt_0.3 |
| 48 | mp-1209118 | RbYb(WO4)2 | Oxides | retained | still ranks within the v2 top 30 for its family |
| 49 | mp-1227144 | CaNdMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |
| 50 | mp-1227151 | CaSmMn2O6 | Oxides | removed | no_static_or_uniform_dos_task_reference |

## Limits

Materials Project formation energies, ordering, and DOS are DFT outputs. The random-forest Tc values are family-specific screening predictions with tree-spread uncertainty, not calibrated probabilities or discoveries. The term ‘untested’ only means absent by reduced formula from the five indexed exports; a literature review is still required. Candidate ranking is pending independent DFT and experimental validation.
