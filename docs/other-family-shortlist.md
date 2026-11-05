# `Other` family within-family shortlist

Run date: 2026-11-02. Workflow: [Continuous AI training run 41536858466](https://github.com/hudsonclavin-cloud/RT-SuperConductor/actions/runs/41536858466), dispatched with `family=Other` from commit `d4ccf6ae2a3e2b2bc6199311b5e8ebd55ff9a1b2`.

This is a model-generated hypothesis list, not evidence that any candidate superconducts. The screen produced no exact known-compound match for any of these ten formulas. Because none had a composition-validated Materials Project entry, their hull energies and band gaps are **unknown**; no values were borrowed from a chemically different compound.

## Run and model context

The 3DSC MP ingestion contained 5,773 records, including 3,529 rows assigned to `Other`; its imported family has 1,653 distinct elemental-system signatures. The model used Matminer Magpie descriptors plus formula-shell/orbital descriptors. Within-family evaluation used five-fold GroupKFold, holding out elemental-system signatures within `Other`:

| Metric | `Other` result |
|---|---:|
| Rows | 3,529 |
| Elemental-system groups | 1,653 |
| Within-family MAE | 1.908 K |
| Mean-predictor MAE | 4.882 K |
| Spearman correlation | 0.734 |

**Superseded within-family estimate:** the 1.908 K / 0.734 headline predates
the shared-MP embargo and can benefit from the same structure reference in both
train and test folds. With duplicate Materials Project IDs purged, the current
family-only result is **3.0781 K MAE / 0.6776 Spearman**; the global model on
the same `Other` rows is **3.0798 K / 0.6623**. The older run remains recorded
above for provenance, not as the current estimate.

These scores measure ranking/regression on held-out elemental-system groups drawn from the same broad `Other` category. They do not validate room-temperature extrapolation. The model's family score combines predicted Tc (weight 0.8) and capped uncertainty (weight 0.2); it does not reward closeness to the 273.15–298.15 K target interval. The top prediction is only 25.33 K.

Materials Project screened 215 fixed-composition queries, skipped 31 ambiguous inputs, returned 50 composition-validated structure matches overall, rejected zero mismatched returned structures, and had zero API errors. **None of the top ten formulas matched an MP composition**, so the table reports no MP ID, hull energy, or band gap for them. All ten were also absent from the available exact-formula checks against 3DSC, the Stanev table, UCI, SuperCon2, and SuperMat; they are marked `new` by this snapshot-based check, not proven to be novel in the full literature.

## Top 10 hypotheses

| Rank | Formula | Predicted Tc (K) | Model uncertainty (K) | MP match | Hull energy (eV/atom) | Band gap (eV) | Exact known-compound match in screened databases |
|---:|---|---:|---:|---|---:|---:|---|
| 1 | BiFSrTe2 | 25.334 | 26.620 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 2 | BiFSe2Sr | 24.052 | 26.926 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 3 | Al0.9Nb3Tl0.1 | 17.214 | 1.583 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 4 | GaNb2.928Re0.072 | 16.990 | 1.935 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 5 | Nb3Sb0.15Sn0.85 | 16.984 | 1.932 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 6 | B0.1Ga0.9Nb3 | 16.169 | 4.118 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 7 | B0.1In0.9Nb3 | 15.136 | 3.975 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 8 | B0.1Nb3Tl0.9 | 14.618 | 4.938 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 9 | CB2Lu0.772Ni2Y0.228 | 14.299 | 2.218 | None | Unknown | Unknown | None found; `new` in available snapshots |
| 10 | BaBiFS2 | 14.169 | 22.451 | None | Unknown | Unknown | None found; `new` in available snapshots |

“None found” is an exact-formula result for the database snapshots available to this run. It is not a claim that a composition is impossible to synthesize or absent from every publication. Fractional stoichiometries also need expert review: candidate generation is arithmetic substitution from records in a broad family and does not establish phase stability, charge balance, synthesis feasibility, or structure.

## Pressure, evidence, and next checks

3DSC MP does not include measurement pressure. The ingestion therefore assigned 0.000101325 GPa as an explicit ambient-pressure **screening assumption** to its records; this is not a measured pressure label. Synthesis pressure is a separate condition and is not inferred by this run. The top-ten rows are ranked formula hypotheses, not compounds with known synthesis or Tc measurements.

The promotion gate for `Other` failed overall because exclusion controls and the hydride trend benchmark were unavailable for this family; its checks for within-family ambient MAE and improvement over the mean predictor passed. The runner's Ferrites-specific DFT preparation step was blocked with no selected rows, as expected for `Other`; no DFT was performed.

Before spending experimental or DFT resources, the next checks are to normalize each fractional formula with a materials chemist, verify charge balance and likely prototypes, retrieve or construct defensible crystal structures, and manually search the literature for close compositions. A candidate should move forward only after those checks and a successful stability/metallicity screen. The current top ten do not include a room-temperature prediction and provide no basis for a claim of ambient-pressure superconductivity.

## Reproducibility files

The workflow artifacts are available from the linked Actions run as `family-shortlist-41536858466`. The principal files are:

- `family-other/other_top25.csv` — ranked family shortlist and per-row MP/database fields.
- `family-other/candidate-ranking-report.json` — candidate counts, MP audit summary, and score mode.
- `family-other/model/pressure-aware-report.json` — grouped and within-family metrics.
- `family-other/model/promotion-gate.json` — promotion checks and failed criteria.
- `family-other/ingestion-report.json` — source row counts and explicit pressure assumption.

