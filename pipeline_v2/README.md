# pipeline_v2: a rebuild of the train → evaluate → select loop

**Main idea:** this folder rebuilds the loop as one small package (`rtsc2`, about 850 lines) with one command. It is faster to rerun and it measures two things the existing scripts do not: whether the selection agent's choice rule works, and how reliable each model's probabilities are. It does **not** deliver a more accurate classifier. Boosted trees still win, and the new neural network still trails them.

Nothing outside this folder was changed. The existing scripts, data and docs are untouched, and every number here is measured on the same records, features and folds as `docs/sc-model-benchmark.md`.

## Run it

```bash
pip install '.[ml,test]'                                             # from the repository root
pip install torch --index-url https://download.pytorch.org/whl/cpu   # needed for the network only
cd pipeline_v2
python -m rtsc2 run        # benchmark -> agent replay -> next batch
python -m pytest -q        # 22 tests
```

`run` writes to `pipeline_v2/reports/`. Single stages: `benchmark`, `replay`, `propose`. Add `--force` to recompute.

| File | Contents |
| --- | --- |
| `reports/benchmark.json` | Every model on the standard folds, with paired gains per shuffle |
| `reports/benchmark-oof.csv.gz` | Every out-of-fold score from the new models |
| `reports/agent-replay.json` | Each selection policy against random picking, standard and strict |
| `reports/next-batch.csv` | The next 25 untested materials the agent would send for checking |

## What changed, and what was measured

Measured on 2026-11-11, 4 CPU cores, 2,153 labelled materials (1,192 superconductors), 177 features, StratifiedGroupKFold by element system, 5 folds × 3 shuffles.

### 1. Speed

| Step | Existing scripts | `rtsc2` |
| --- | --- | --- |
| Build the feature matrix (30,358 materials) | 41.5 s on every run | 41.5 s once, then 0.4 s from cache |
| Rerun with nothing changed | Recomputes everything | 0.35 s; each stage is skipped when its inputs, settings and code are unchanged |
| Whole loop from a cold start | Several scripts run by hand | One command, 8 min 36 s (benchmark 265 s, standard replay 244 s, proposal 6 s; the strict replay adds 218 s) |

The saving is in reruns and bookkeeping. Model fitting itself is **not** faster: the tree bag costs about six times the single boosting model (64 s against 10 s for 15 fits) because it fits eight models.

### 2. Models

| Model | AUC (± spread over shuffles) | Paired gain vs legacy boosting | Calibration error | Superconductors in top 10% | Fit time, 15 fits |
| --- | --- | --- | --- | --- | --- |
| **Tree bag** (8 boosting models, each on a bootstrap of element systems) | **0.826 ± 0.002** | +0.002 (2 of 3 shuffles; −0.005 to +0.006) | 0.046 | 94.3% | 64 s |
| Legacy boosting, refitted here | 0.824 ± 0.004 | reference | 0.057 | 90.5% | 10 s |
| Blend: tree bag + deep ensemble | 0.821 ± 0.004 | −0.004 (1 of 3) | **0.019** | 93.3% | 122 s |
| Legacy scikit-learn MLP (saved scores) | 0.782 ± 0.010 | −0.042 | 0.158 | 91.0% | 23.5 s |
| **Deep ensemble** (new network) | 0.777 ± 0.005 | −0.048 (0 of 3) | 0.026 | 88.5% | 62 s |
| Legacy PyTorch network (saved scores) | 0.756 ± 0.006 | −0.069 | 0.065 | 87.0% | 8.1 s |

What this shows:

- **The refit of legacy boosting reproduces the published scores exactly** (0.8238, 0.8295, 0.8199), and the feature-matrix hash matches `docs/sc-model-benchmark.json`. The comparison is like for like.
- **The tree bag is not a proven accuracy gain.** +0.002 AUC is inside the spread between shuffles. Its real advantages are steadier results across shuffles, a better top decile, and a spread across members that the agent can use.
- **The new network beats the old PyTorch network on all three shuffles (+0.021 AUC)** and is much better calibrated. It is still 0.048 below the trees and slightly below the scikit-learn MLP. Averaging it into the trees lowers AUC, though it gives the best-calibrated probabilities.
- **The network is not cheaper to train.** Eight members cost about the same per member as one legacy network (0.5 s each).

The network is a deep ensemble: eight members trained together as one batched tensor program. Each member has its own starting weights and its own held-out element systems for early stopping. Inputs are quantile-normalised features, missing-value flags, and a learned element encoder with two attention steps over the formula. A second head predicts log Tc for known superconductors. Without the element encoder the same network scores 0.771, so the encoder adds 0.006 AUC (3 of 3 shuffles; run with `--models deep_ensemble_no_elements`).

Network settings were tried on separate shuffles (seeds 1003 and 1004), not the three reported ones. They use the same 2,153 materials, so a small optimistic bias is possible. None of the settings tried moved AUC outside 0.772 to 0.780.

### 3. The selection agent

The existing loop picks candidates by a fixed formula and never tests that formula. `rtsc2` replays the search on materials whose outcome is known: hide the labels, let the agent pick 25, reveal those 25, refit, repeat 20 times (500 picks). Ten campaigns per policy. There are two versions.

**Standard replay** starts from 100 random materials and lets the agent pick anything.

| Policy | Any superconductor (base rate 55%): hits in 500 picks | Tc ≥ 10 K (base rate 4.4%): hits in 500 picks | Tc ≥ 10 K: hits in new element systems |
| --- | --- | --- | --- |
| Random | 283.4 | 23.4 | 20.0 |
| Greedy (highest probability) | 412.9 (1.46×) | 63.7 (2.72×) | 46.3 (2.31×) |
| Diverse (greedy, at most 2 per element system per batch) | 414.5 (1.46×) | 65.3 (2.79×) | 48.0 (2.40×) |
| Thompson (rank by one random ensemble member) | 394.9 (1.39×) | 65.0 (2.78×) | 47.1 (2.35×) |
| Upper confidence bound (mean + member spread) | 389.0 (1.37×) | 62.4 (2.67×) | 44.9 (2.25×) |

In this version, guided picking for "any superconductor" finds no more new element systems than random (226 to 230 against 231). It scores by returning to chemistries already known to work, so the standard replay cannot tell re-finding from discovery.

**Strict replay** removes that route (`replay(..., strict=True)`). The starting set is whole element systems, the agent may query each remaining element system once, and the other materials of a queried system leave the pool unrevealed. Every query is a first look at an element system the model has never seen, so every hit is a hit in a new system.

| Policy | Any superconductor: hits in 500 picks (range over campaigns) | Tc ≥ 10 K: hits in 500 picks (range) |
| --- | --- | --- |
| Random | 269.8 (261–284) | 22.6 (16–27) |
| Greedy | 374.0, 1.39× (362–388) | 50.2, 2.22× (43–53) |
| Thompson | 364.3, 1.35× (358–381) | 52.4, 2.32× (45–58) |
| Upper confidence bound | 363.8, 1.35× (358–373) | 51.0, 2.26× (44–55) |

Diverse is left out because with one query per system it is the same as greedy.

What this shows:

- **The model does generalise to element systems it has never seen.** Guided picking beats random on all ten campaigns for both targets: 1.39× for any superconductor and about 2.2× for Tc ≥ 10 K. This is the result the standard replay could not establish.
- **The gain is smaller than the standard replay suggests.** For Tc ≥ 10 K it drops from about 2.7× to 2.2×. The difference is the share that came from re-finding known chemistries.
- **For the rare target the agent finds most of what exists.** Only 72 of the 1,338 element systems contain a Tc ≥ 10 K material. Greedy finds about 50 of them after querying roughly 40% of the unseen systems.
- **No policy is shown to be better than greedy.** In the strict replay the three guided policies overlap on both targets. In the standard replay the upper-confidence-bound rule is the lowest guided policy for "any superconductor" (389 against 413). An uncertainty bonus has not earned its place in either test.

## Limits

- **Replay uses known outcomes only.** It ranks policies on the labelled set. It says nothing about the hit rate on the 28,205 untested materials, where superconductors are far rarer.
- **Probabilities on untested materials are too high.** The top of `next-batch.csv` scores 0.99. Labelled materials are 55% superconductors because people chose what to test; the untested pool is not. Treat the scores as a ranking only.
- **The label bias is inherited.** "Tested non-superconductor" means someone reported a null result, so the model partly learns what people chose to test.
- **The Tc models, the λ/ωlog models and the QE/EPW steps are not rebuilt here.** This folder covers the classifier, the network and the agent. Those are the parts with enough in-repository data to measure.
- **The next batch is a list of hypotheses for review**, screened to Other-family, metallic, non-magnetic, at most 20 sites, with no element lacking a stable isotope.

One thing found along the way: the existing radioactive-element list in `scripts/chemistry_policy.py` names Tc, Pm, Po, Th and U. It does not name actinium, and the first run here proposed elemental Ac. The screen in `rtsc2` excludes Tc, Pm and every element from Po upward. The existing script was left as it is.

## Layout

| Module | Job |
| --- | --- |
| `rtsc2/data.py` | Load the table, apply the label policy, cache features and element tokens |
| `rtsc2/models.py` | `TreeBag`, `DeepEnsemble`, `Blend`; one `fit`/`predict` interface and a registry |
| `rtsc2/evaluate.py` | Standard folds, metrics, paired comparison with the saved legacy scores |
| `rtsc2/agent.py` | Selection policies, `replay`, `compare_policies`, `propose` |
| `rtsc2/cli.py` | Cached stages and the command line |

The feature definition is imported from `scripts/train_mp_sc_classifier.py` rather than copied. An identical matrix is what makes the numbers comparable.

## What to do next

1. Decide whether to switch the production classifier to the tree bag. The case rests on stability and calibration, not AUC.
2. Stop spending effort on network architecture for this table. Two architectures, a tuned search and an ensemble all land between 0.756 and 0.782. The lever is the input: pretrained crystal-structure embeddings or more labels.
3. Score every future policy change with the strict replay. It is the version that measures discovery.
