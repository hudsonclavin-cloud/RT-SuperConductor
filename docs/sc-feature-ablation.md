# Do structure features help? Feature sets × models on identical folds

**Main idea:** chemistry carries most of the signal. Structure features give the random forest and the neural network a small, consistent gain, but the best model (gradient boosting) gets the same answer from formula and MP features alone. The production classifier therefore stays on formula + MP features. The structure table is committed for the ensemble and the neural-network work.

## What was added

`scripts/fetch_mp_structures.py` downloads the relaxed structures of all 30,358 snapshot materials and the elastic moduli of the 7,376 with elastic calculations (kept in `data/raw`, not committed). `scripts/structure_features.py` turns them into 65 features per material in `data/candidates/mp-structure-features.csv.gz`:

| Group | Features |
| --- | --- |
| Symmetry (categorical) | Point group one-hot (32), lattice centering one-hot (P, I, F, C, A, R), centrosymmetric flag, point-group order |
| Lattice | Volume per atom, packing fraction, reduced-cell length ratios, angle departure from 90° |
| Local geometry | Nearest-neighbour distances, bond length relative to summed atomic radii, coordination counts, unlike-element and metal–metal contact shares |
| Connectivity | Bonded dimensionality (Larsen method): 307 molecular (0D), 270 chain (1D), 1,362 layered (2D), 28,419 framework (3D) |
| Stiffness | Bulk and shear moduli, Pugh ratio, Debye temperature (Anderson's method: Al 407 K vs measured 428 K, Cu 340 K vs 343 K) |

These come from computed, ordered Materials Project structures, not measured sample structures.

## Results

AUC on 2,244 labelled materials, StratifiedGroupKFold by element system, 5 folds × 3 shuffles (`python scripts/ablation_sc_features.py`, full output in `docs/sc-feature-ablation.json`):

| Feature set | Random forest | Gradient boosting | Neural network (MLP) |
| --- | --- | --- | --- |
| Formula only | 0.793 | 0.813 | 0.758 |
| Formula + MP properties | 0.800 | **0.817** | 0.769 |
| + structure | 0.806 | **0.817** | 0.774 |
| + structure + stiffness | 0.807 | **0.817** | 0.773 |
| MP + structure + stiffness, no formula | 0.769 | 0.756 | 0.712 |

Paired against formula + MP on the same shuffles:

| Added | Random forest | Gradient boosting | MLP |
| --- | --- | --- | --- |
| Structure | +0.006 (3 of 3) | +0.0003 (2 of 3) | +0.005 (2 of 3) |
| Structure + stiffness | +0.007 (3 of 3) | 0.000 (2 of 3) | +0.005 (2 of 3) |

## What this means

- **Composition dominates this task.** Formula features alone give 0.813, and everything else adds at most 0.004 for the best model.
- **Structure is informative but mostly redundant.** Without formula features it still reaches 0.77, well above chance. It helps the weaker models, so it can still contribute inside an ensemble.
- **Stiffness adds nothing for this yes-or-no task.** Only 24% of materials have elastic data. The Debye temperature is more likely to matter for predicting Tc itself, where the phonon frequency scale enters directly.
- **The production model is unchanged.** It stays gradient boosting on formula + MP features, because +0.0003 is noise. `train_mp_sc_classifier.py --extra-features data/candidates/mp-structure-features.csv.gz` switches the structure table on.

## Cleanup in this step

`scripts/evaluate_structure_features.py` (the older 3DSC structure evaluator) fitted its structure imputer on all rows before cross-validation. Its aggregate mean-predictor baseline also used every target, including test rows. Both are now fold-local, and a test checks them. Reports produced by the old version slightly flattered the structure model and the baseline. Rerun it before comparing old and new numbers.
