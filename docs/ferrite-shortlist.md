# Ferrite shortlist: review and DFT hand-off

## Executive result

The current cycle is a within-family optimization experiment, not a claim of a
new superconductor. It trains on the real 3DSC MP table, uses Matminer Magpie
composition features plus formula-shell/orbital descriptors, and evaluates
Ferrites with elemental-system-held-out GroupKFold. The Ferrite result was
MAE **7.019 K** versus **12.144 K** for a mean predictor, with within-family
Spearman **0.6803** over **517 rows and 162 elemental systems**. The pooled
family-held-out model had MAE **10.435 K** versus **16.052 K** for the mean
predictor; cross-family generalization remains weak.

Regenerate the complete cycle from the repository root with:

```bash
python scripts/run_ferrite_shortlist.py
```

The cycle writes the harmonized 3DSC table, Magpie model and metrics, the
promotion-gate report, `data/processed/candidates/ferrites_top25.csv`, a
paper-review queue, and the QE hand-off report. The 3DSC importer explicitly
records atmospheric pressure as `0.000101325 GPa` because that source does not
provide measurement pressure. The run used source SHA256
`353b86f9c60505d11a3e25df8312da02ff88fd8e27af05b632dec4739d52eb24` and
harmonized dataset SHA256
`ebe808a22db39c12abff28a189c83e0d0540801323320fd43b71105b109a2c01`.

## Gate and ranking status

The promotion gate was **fail**, with real checks as follows:

| Check | Result |
| --- | --- |
| Ambient MAE \(≤ 12 K\) | pass, 10.435 K |
| Beat mean predictor by \(≥ 25\%\) | fail, 11.77% aggregate improvement |
| Ferrite family gate | fail, 12.225 K model MAE vs 12.121 K mean MAE |
| H3S / LaH10 trend \(r ≥ 0.80\) | not satisfied; 0 benchmark rows |
| Excluded-reference false-positive rate | not satisfied; 0 reference rows |

Materials Project was **`degraded_no_mp_api_key`**. The ranker enumerated 272
Ferrite substitutions, screened 272, and emitted a shortlist of 25.
`rejected_nonmetal_count` was **0**, which means “no MP electronic data were
available,” not “all candidates were proven metallic.” There were 11 known
source-database candidates and 261 candidates with unknown database status; no
candidate could honestly be called new while MP was unavailable. Hull energies,
band gaps, MP IDs, and metallicity are blank/unknown in the CSV for the same
reason.

Family mode scores predicted Tc and capped uncertainty weighted by
plausibility. It deliberately does not score distance to 273 K. Known and
unknown database groups are ranked separately, and every row includes a reason
string for review.

## Provisional top five for review

The rows below are the first five model-ranked Ferrite hypotheses from the
25-row CSV. They are **not DFT-ready**: MP structures and hull/electronic
checks were unavailable.

| Rank | Formula | Predicted Tc (K) | Uncertainty (K) | Known flag | Reason boundary |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | AsF0.11FeLaO0.89 | 28.378 | 17.100 | known in UCI | MP hull, gap, and metallicity unknown |
| 2 | AsF0.1FeLaO0.9 | 27.135 | 16.934 | known in UCI | MP hull, gap, and metallicity unknown |
| 3 | AsF0.08FeLaO0.92 | 20.613 | 14.993 | known in UCI | MP hull, gap, and metallicity unknown |
| 4 | AsF0.07FeLaO0.93 | 19.137 | 15.666 | known in UCI | MP hull, gap, and metallicity unknown |
| 5 | As2Ba0.8Fe2K0.2 | 14.075 | 4.610 | known in UCI | MP hull, gap, and metallicity unknown |

These formulas are review leads, not measured values. In particular, the
presence of a formula in UCI is not confirmation of the model ranking or of a
new result.

## Literature validation

The top ten rows already had a known-source flag, so
`top10_known_compound_count` was **10**. SuperCon2 and SuperMat exports were
not present in this environment; therefore the validator reported
`blocked_missing_supercon2_and_supermat_exports`, with **0** matched candidates,
**0** known superconductors, **0** literature match rows, and ordering
Spearman **null**. The generated
`data/training/review_queue_ferrite_shortlist.csv` contains only the review
queue header and no fabricated Tc or DOI. Any future matches must remain
`pending_paper_review` until a human checks the paper.

## DFT hand-off

The requested top-five filter is: MP structure available, MP hull energy below
0.05 eV/atom, and metallic or otherwise reviewable electronic status. It
selected **0** candidates and generated **0** QE jobs because
`MP_API_KEY` was absent. The report is explicitly
`degraded_no_mp_api_key`; no relax, SCF, or phonon calculation was run.
Pseudopotential UPF availability and Hazel access are also external blockers.

Once MP access, real CIFs, a verified pseudopotential catalog, and Hazel
credentials exist, `scripts/prepare_ferrite_dft.py` uses the existing QE
generator to emit relax, SCF, phonon, and EPW starter inputs. DFT would test
structural relaxation, metallicity, phonons, and electron-phonon coupling
lambda; it would not by itself establish experimental superconductivity.

No result in this report claims a room-temperature superconductor. Every
candidate is a hypothesis requiring structure checks, first-principles work,
and literature or experimental review.
