# Active research task: physical features and honest evaluation

Branch: `codex/physics-features-deep`. Preserve all earlier documents, reviewed
records, manifests and tests. Commit and push after each phase, with passing
`python -m unittest discover -s tests` and measured results in `docs/modeling.md`.
No raw datasets, model files, large caches or API keys enter git. Every blocked
calculation gets a reason and a count. Check the diff against the start commit
`eb3a931` before every push. No pure documentation deletion is acceptable.

1. **Leakage audit and feature table:** allowlisted 3DSC electronic, lattice,
   bonding and DFT magnetic fields, content-hashed CIF geometry caches; exclude
   matching/cleaning artifacts, weights and timestamps from model matrices.
2. **Ablation:** identical cross-family and within-family grouped splits, MAE,
   Spearman, both ROC-AUC thresholds and mean-predictor comparisons; top 20
   feature importances and honest interpretation of nonmagnetic DFT cuprates.
3. **Structure models:** frozen CHGNet encoder with RF probes, supervised
   fine-tuning pilot, scratch ALIGNN pilot, and the frozen encoder on JARVIS
   lambda/omega_log. Keep target labels out of held-out folds. Record attempts
   that hit six CPU hours rather than silently omitting them.

## Phase 3.5 — new physics checks (user amendment)

- **Dynamic stability:** for every shortlisted candidate with a structure,
  relax it with CHGNet or M3GNet, compute phonons with phonopy at Gamma and on
  a coarse q-mesh, and flag imaginary modes below −0.5 THz. Report how many
  shortlist rows fail, pass, or are blocked. These are ML-potential screening
  results, not DFT-confirmed dynamic stability. Prepare the interface here;
  apply it to the newly generated candidates after Phase 4.
- **Orbital character:** in GitHub Actions using `MP_API_KEY`, fetch element-
  and orbital-projected DOS for available 3DSC reference structures. Integrate
  s-, p-, and d-weights within ±0.5 eV of Ef and add an orbital ablation group.
  Keep missing projections missing; record the energy window, units and
  normalization, including any f contribution.
- **Bond angles:** compute mean/minimum metal–oxygen–metal angles and oxide
  buckling from CIFs, state the geometric definition, and add them to lattice.
- **Correlation proxy:** add the 3DSC GGA+U flag and formula-weighted neutral-
  atom d-electron count as a separate group. Neither is an experimental measure
  of correlation strength or solid-state orbital occupancy.
- **Mechanism split:** compare global, family-specific, and joint family/Tc
  multi-task models on the same held-out rows. A family-specific model cannot
  predict an entirely held-out family without training evidence; report that
  limitation rather than leaking its family labels or samples.

4. **Chemistry-constrained generation:** preserve H and metal/nonmetal class;
   require periodic-group or common-oxidation overlap; balance ionic parents;
   reject known insulating parent structures. Regenerate Ferrites, Other and
   physics-target shortlists, recording rule counts and full reasons. Only
   confident rows receive review ranks; passing screens does not establish
   synthesizability. Run the Phase 3.5 phonon screen on available structures.
5. **Uncertainty and extrapolation:** held-out error versus tree SD with a PNG,
   calibration check, a forest range-bound proof/test and per-family maxima;
   count and audit physics-derived Tc exceedances of fold training maxima.
6. **Reporting and CI:** dispatchable ablation plus secret-backed DOS/elastic
   enrichment; artifact tables; additive one-page README/faculty report;
   `docs/next-experiments.md` with three tests, expected outcomes and failure
   decisions. Final completion reports include actual numbers per phase.
