# Three proposed next tests

These are proposed acceptance tests, not completed results or promises of discovery. Obtain faculty review of the methods and pressure/phase interpretation before spending laboratory or HPC time. A candidate score or ML-potential phonon screen does not establish superconductivity.

## 1. Audit labels against the sample and measurement conditions

Select 30 records before looking at model residuals: 10 Ferrites, 10 Other and 10 Cuprates, including zero-Tc entries and records with fractional stoichiometry. Trace each to its primary source. Record the exact material/sample, phase, Tc definition (onset, zero resistance, susceptibility or thermodynamic anomaly), measurement pressure, synthesis pressure and source passage. Check whether the associated CIF describes the measured sample or only a reference parent. Have a second reader review ambiguous matches.

**Expected useful outcome:** a small source-audited test set, plus an explicit list of unknowns, mismatched structures and untested-zero conventions. Unknown pressure remains unknown; high-pressure synthesis does not imply high-pressure measurement. An unresolvable formula or oxygen ordering does not become a unique simulation cell.

**Acceptance and failure decision:** every retained condition-specific label needs source support for its composition, phase and measurement conditions. Quarantine a mismatched record rather than relabeling a magnetic transition or an assumed pressure as superconducting evidence. If the audit exposes systematic errors, repair the affected source/family and repeat its evaluation before expanding candidate generation. This test can change the dataset and evaluation target; it does not require new synthesis.

## 2. Reproduce one conventional electron–phonon reference with converged DFPT

Ask the mentor to approve an ordered, affordable reference and matching functional, pseudopotentials, magnetic/SOC treatment, k/q meshes and smearing. A documented computational control is **Ta, JARVIS JVASP-1014**, in the [JARVIS dataset v4](https://doi.org/10.6084/m9.figshare.21370572.v4), described by the [source paper](https://doi.org/10.1038/s41524-022-00933-1). Its source targets are **λ = 0.7572** and **ω_log = 164.9922 K**. This repository recomputes **Tc = 6.8822088869 K** at μ* = 0.1 using its stated modified McMillan/Allen–Dynes expression. The dataset's stored Tc is **7.6395091414 K**; reconcile the source's assumptions before comparing those Tc values. Neither number is an experimental measurement in this test.

The current CHGNet/phonopy screen falsely flags stable fcc Al at the requested threshold across tested supercell/mesh settings. Include that control when validating the phonon method; do not reject experimental candidates solely on these ML signals.

Run relaxation, electronic and phonon convergence checks, then electron–phonon calculations using the approved external compute environment. Preserve input/output hashes, executable versions, pseudopotential checksums, pressure target, residual force/stress, grids, failures and convergence history. GitHub Actions may prepare and ingest these jobs; it does not claim to perform the DFT calculation.

**Expected useful outcome:** a reproducible reference calculation and an error budget for the method transfer. As a proposed criterion for mentor review, require λ and ω_log to change by no more than 5% between the final two mesh/cutoff refinements. Agreement with a published target must be assessed under matching assumptions; it is not guaranteed by that numerical convergence criterion.

**Acceptance and failure decision:** if the calculation is not converged, or its structure/method cannot be matched, keep the benchmark blocked and refine the setup. Do not change μ* merely to reproduce a desired Tc. If λ/ω_log are reproducible but Tc definitions differ, resolve the formula and parameter choices before treating the Tc discrepancy as a model error. A successful reference unlocks prospective screening; it does not certify a new superconductor.

## 3. Freeze a small prospective candidate/control comparison

After the reference succeeds, have the mentor select a small, affordable panel of ordered candidate/control pairs from one conventional family. Prefer cells and compositions with clear lineage and a defensible pressure target. Freeze the formula baseline and physics-target model predictions, uncertainty diagnostics, ranking reasons and exclusion rules before returning DFT results. Retain candidates rejected by chemical, structural or dynamic checks in the audit trail.

Calculate stability and the same converged λ/ω_log targets for the panel. Compare predictions with the returned calculations, including failures, and report paired errors and ranking behavior against the simple baseline. Use identical Tc conventions when deriving Tc from λ/ω_log. A small pilot can falsify an extreme prediction; it cannot establish broad calibration or discovery performance.

**Expected useful outcome:** the first prospective feedback that was unavailable during training, including negative computational results. It should show whether a shortlist contains informative, calculable materials rather than only confident-looking scores.

**Acceptance and failure decision:** if proposed structures are unstable, insulating under the selected method, or fail convergence, repair the proposal/filtering rules before enlarging the search. If physics-target ranking does not improve decisions over the baseline, report that result and investigate the missing mechanism or data coverage. Add qualified calculations to the DFT-target dataset with their provenance; do not turn a failed calculation or an ML-stability flag into an experimental Tc = 0 label. Laboratory synthesis is a later, separately reviewed test.
