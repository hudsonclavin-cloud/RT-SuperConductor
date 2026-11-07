# Model roadmap: from random forest toward neural networks

> **Update 2026-11-07.** Negatives are now graded by evidence and only tiers A–C train the classifier (`docs/negative-labels.md`). On that basis gradient boosting scores 0.820 and the forest 0.818; removing suspect insulator negatives raised AUC on metallic test rows by about 0.02. The next-step list below stands, with one addition ahead of all of them: verified null results near the top candidates, recorded in `data/training/negative_evidence.csv`.

**Main idea:** the progression random forest → boosted trees → direct neural network has been run on identical folds. Boosted trees win (AUC 0.817). Neural networks trail by about 0.05, even when tuned and when given more labels. The next neural steps should change the inputs (composition-attention or pretrained crystal encoders), not just the network on top of them.

| Step | Status | Result (AUC, same folds) | Where |
| --- | --- | --- | --- |
| Random forest | Done | 0.800 | `docs/sc-model-benchmark.md` |
| Boosted trees (production) | Done | **0.817**; 0.819 with weighted extra labels | `docs/sc-model-benchmark.md`, `docs/sc-label-expansion.md` |
| Structure, symmetry, stiffness features | Done | +0.006 forest, +0.000 boosting | `docs/sc-feature-ablation.md` |
| Direct neural network (scikit-learn MLP) | Done | 0.769 | `docs/sc-model-benchmark.md` |
| Direct neural network (PyTorch: LayerNorm, SiLU, AdamW, grouped early stopping; multitask; tuned) | Done | 0.760–0.762 | `scripts/torch_models.py` |
| More labels with polymorph weighting | Done | boosting +0.004 (2 of 3), forest +0.007 (3 of 3) | `docs/sc-label-expansion.md` |
| Composition-attention network (CrabNet-style) | Next | — | Learns element representations from stoichiometry |
| Frozen crystal encoder (CHGNet/ALIGNN) + boosting and MLP heads | Next | — | The earlier encoder runs used RF readouts and 5-epoch training only |
| Fine-tuned graph network | Later | — | Only after a frozen encoder beats its paired control |
| Family specialists / mixture of experts | Later | — | After the global baseline is reliable |

## Rules every new model must follow

These are already built into `scripts/benchmark_sc_models.py`:

1. Same records, features and outer folds as the table above (StratifiedGroupKFold by element system, 5 × 3).
2. Preprocessing, tuning and early stopping inside the training fold only.
3. Save every out-of-fold prediction; report AUC, average precision, Brier score and paired gains per shuffle.
4. Promote a model only when it beats the current one on most shuffles by more than the shuffle spread.
5. Record the git commit, library versions (including PyTorch), seeds and data hashes.

PyTorch is optional. Install the CPU build with `pip install torch --index-url https://download.pytorch.org/whl/cpu`; without it, the torch models are skipped.
