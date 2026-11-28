# Overnight GPU run: crystal-structure networks against the ranking of record

**Main idea:** this is the first properly trained test of whether a
material's crystal structure helps predict superconductivity in this
project.
- The earlier graph networks were 5-epoch CPU pilots: 5,773 rows, Tc
  regression, random-forest readouts (`docs/modeling.md`). They did not
  settle it.
- This run trains crystal-graph networks (CGCNN) on the owner's GPU for
  about 8 hours, on exactly cycle 6's folds. It also pretrains them on
  30,000 Materials Project structures.
- This session then judges the results against cycle 6 with paired
  intervals, by a rule written here before any result.

Written 2026-11-22, before any GPU result.

## What is new, and what is not

| Part | File | Role |
| --- | --- | --- |
| Structure bundle | `scripts/structure_gpu.py export` writes `data/processed/local-gpu/structures.npz` (10.7 MB) | crystal graphs, labels, folds, pretraining targets, candidates |
| Trainer | `scripts/structure_gpu_train.py` | numpy and PyTorch only; 5 models plus pretraining |
| Driver | `scripts/overnight_gpu.py` | the job queue, the time budget, resume, STOP, keeping Windows awake, `share/` |
| Judge | `scripts/structure_gpu.py compare <folder>` | paired comparison of every job with cycle 6; the decision rule |
| Formula bundle | `scripts/local_gpu.py export` (`bundle.npz`, from the previous step, `docs/local-gpu.md`) | rows and features for the tabular jobs and the `*_all` models |

### The structure bundle

**Training graphs:** 4,931 crystal graphs for 4,806 compositions.
- Of these compositions, 4,263 are development rows and 543 train-only
  rows.
- They come from the 3DSC dataset: SuperCon compositions matched to
  Materials Project structures, with doping applied to the structure.
- Labels are the table's (cycle 6), not 3DSC's.
- A composition with several candidate structures has several graphs.
  Each graph is weighted 1 / (number of graphs), and their predictions
  are averaged.

**Graphs**
- Each site links to its 12 nearest neighbours within 8 Å.
- A site with mixed occupancy carries up to 4 elements, each with its
  occupancy.
- Element features are the project's own element table plus a learned
  embedding.

**Folds**
- Cycle 6's own fit, early-stopping and test rows
  (`run_scratchnet.fold_rows`), restricted to compositions that have a
  structure.
- Per fold: about 3,600 fit, 360 early-stopping and 850 test
  compositions.

**Pretraining:** 29,913 Materials Project structures with five calculated
targets: formation energy, band gap, energy above the hull, metal or not,
magnetic or not.
- There are no superconductivity labels.
- 438 structures are left out because their composition is a locked
  composition, or the parent of a locked 3DSC structure.

**Candidates:** the 26,911 made metals and shortlist 3's 571 compounds,
scored on their own structures.

**Locked rows:** not in the bundle. 839 locked 3DSC structures are
dropped; the export and the trainer both refuse a bundle that contains
one.

### The five models

All five share one code path, so the comparisons inside a track differ
only in the inputs.

| Model | Rows | Inputs |
| --- | --- | --- |
| `formula_all` | every row, as cycle 6 (14,861, including the NIMS rows) | the 155 formula features; **control** for `hybrid_all` |
| `hybrid_all` | every row | formula features, plus the crystal graph where one exists and a learned "no structure" vector elsewhere |
| `formula` | rows with a structure | formula features; **control** for `cgcnn` and `hybrid` |
| `cgcnn` | rows with a structure | the crystal graph only |
| `hybrid` | rows with a structure | graph plus formula features |

## The run

`scripts/overnight_gpu.py --hours 8` runs these jobs in this order. The
estimated times for the RTX 5070 Laptop GPU are [Speculative]; the driver
replaces them with measured times as jobs finish.

| # | Job | What | Estimated time |
| --- | --- | --- | --- |
| 1 | `check` | 2 epochs of `cgcnn` on fold 0. If it fails, nothing else runs | 1 min |
| 2 | `tab-default-s20` | the tabular network of `docs/local-gpu.md`, 20 networks per fold | 5 min |
| 3 | `tab-wide-s20` | the same, wider (512, 256, 128), dropout 0.2 | 5 min |
| 4 | `A-formula_all` | track A control, 3 networks per fold | 5 min |
| 5 | `A-hybrid_all` | track A, from scratch | 45 min |
| 6 | `pretrain` | the encoder on the Materials Project targets, up to 60 epochs | 15 min |
| 7 | `A-hybrid_all-pre` | track A, fine-tuned from job 6 | 45 min |
| 8 | `S-formula` | track S control | 3 min |
| 9–12 | `S-cgcnn`, `S-hybrid`, `S-cgcnn-pre`, `S-hybrid-pre` | track S | 25 min each |
| 13 | `sweep-*` | around the better of jobs 5 and 7, one setting changed at a time (12 settings), 2 networks per fold, while time allows | up to about 3 h |
| 14 | `final-A` | the track A model (job 5, 7 or a sweep job) with the **lowest mean validation loss**, retrained with new seeds, 3 to 10 networks per fold as time allows | 1 to 2 h |
| 15 | `final-S` | the same for track S (jobs 9 to 12), 2 to 6 networks per fold | 20 to 40 min |

**What the driver does**
- **Time:** it holds back enough time for both final jobs. A job that
  would eat into that time is skipped and recorded as skipped.
- **Restarts:** run the same command again after a crash, a reboot or a
  STOP. Finished jobs are skipped.
- **Stopping:** a file named `STOP` in the run folder stops the run after
  the current job.
- **Out of memory:** a structure job that runs out of GPU memory is
  retried once with half the batch size.
- **Sleep:** on Windows the driver asks the system not to sleep while it
  works. Closing a laptop lid may still put it to sleep: keep the lid
  open, or set "When I close the lid" to "Do nothing" while plugged in.

## How the results will be judged (fixed before any result)

**Selection**
- Jobs are selected only on each fold's early-stopping rows
  (`mean_validation_bce` in each `run.json`).
- No test row is scored on the owner's machine.
- The two final jobs are chosen by that number alone, and the driver
  records which one it chose.

**The rule for the ranking of record**
- `final-A` scores every development row, as cycle 6 does. It joins or
  replaces the ranking of record only if two things hold:
  - its paired non-cuprate AUC interval against cycle 6 lies entirely
    above zero;
  - the owner accepts it.
- The same test is reported for its 50/50 blend with cycle 6.
- Every other job is exploratory and is reported as such.

**Secondary questions**, also fixed now:
- Does structure add anything to the formula features on the same rows
  and in the same code?
  - Track A: `final-A` minus `A-formula_all`.
  - Track S: `final-S` minus `S-formula`.
- Does `final-S` beat cycle 6 on the rows it scores (about 4,263
  development rows with structures)?

The 1,950 locked rows are not used. They are scored once, only if a model
passes the rule.

**Judging command, here:**

```bash
python scripts/structure_gpu.py compare data/benchmarks/local-gpu/overnight-20261122
```

### Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| `final-A` beats cycle 6 under the rule (interval above zero, all development rows) | 10% |
| The blend of `final-A` and cycle 6 beats cycle 6 (interval above zero) | 20% |
| `final-A` beats `A-formula_all` (structure adds something; interval above zero) | 30% |
| `final-S` beats `S-formula` on the same rows (interval above zero) | 35% |
| `final-S` beats cycle 6 on its rows (interval above zero) | 15% |
| Pretraining lowers the validation loss: `S-cgcnn-pre` below `S-cgcnn` | 65% |
| The 20-network tabular bank still trails cycle 6 (interval below zero) | 85% |
| Every core job (1 to 12) finishes within the 8 hours | 75% |

**Why the chances are low**
- Gradient boosting has beaten every network in this project, and in the
  hackathon repository's own tests.
- Only 44% of the development rows have a structure.
- Many 3DSC structures are a doped parent shared by a whole series of
  compositions, so the graph adds less than it seems.

**The case for running it anyway:** a graph network sees bonding and
coordination, which formula features cannot. Pretraining on 30,000
structures is the usual fix for small labelled sets, and a GPU night
costs nothing but time.

## Commands for the owner's machine (Windows PowerShell)

The same commands, with notes, are in the chat message that came with
this file. In short:

1. **Get the code:** clone or pull, on branch
   `claude/practical-tesla-ctf6oq`.
2. **Put the two bundles in place** (sent by chat), as
   `data\processed\local-gpu\bundle.npz` and
   `data\processed\local-gpu\structures.npz`. Check them with
   `Get-FileHash`:
   - `bundle.npz`:
     `1327a6d6756b81f3bb4f9bf41354961a61773722feb1c11b89c8d8208d8e2b2d`
   - `structures.npz`:
     `f216fc5c5dd2ad610d6df142596522c26da552629c526adadabee5f06f53c918`
3. **Check the environment:**
   `python -c "import torch; print(torch.cuda.is_available())"`.
4. **Rehearse** (about 5 minutes):
   `python scripts\overnight_gpu.py --out runs\rehearsal --profile smoke --device cuda`.
5. **The run:**
   `python scripts\overnight_gpu.py --out runs\overnight-20261122 --hours 8 --device cuda`.
6. **In the morning:**
   - Copy `runs\overnight-20261122\share` to
     `data\benchmarks\local-gpu\overnight-20261122`.
   - Commit it on a new branch, `gpu-results-20261122`, and push it.
   - It holds predictions and logs only: no weights, no Hazel results.

## Results, 2026-11-23

**Main idea:** crystal structure does not help.
- [Certain, on these rows and folds] No network beat cycle 6, and no
  blend with cycle 6 did either.
- Adding the crystal graph to the formula features made no measurable
  difference to the AUC, in either track.
- The rule is not met, so cycle 6 stays the ranking of record.

**The run:** the owner's RTX 5070 Laptop GPU (8 GB, compute capability
12.0; PyTorch 2.15.0+cu132 on Windows 11).
- Ran 01:28 to 07:16 local time (5.8 hours), exit code 0.
- All 26 jobs finished; none was skipped.
- Results: `data/benchmarks/local-gpu/overnight-20261122/`.
- Comparison: `docs/local-gpu/overnight-20261122.json`, from
  `python scripts/structure_gpu.py compare data/benchmarks/local-gpu/overnight-20261122`.
- The checksums match the bundles exported here.

### Against cycle 6, non-cuprate AUC (positive favours the job)

The `check` job is left out: it was the 2-epoch test.

| Job | Rows | Early-stopping loss | AUC | Cycle 6 on the same rows | Job minus cycle 6 (95%) | 50/50 blend minus cycle 6 (95%) |
| --- | --- | --- | --- | --- | --- | --- |
| **final-A** (from `sweep-batch64`, 10 networks per fold) | 9,765 | 0.4644 | 0.766 | 0.788 | −0.022 (−0.035, −0.012) | +0.002 (−0.004, +0.006) |
| A-formula_all (control) | 9,765 | 0.4443 | 0.767 | 0.788 | −0.020 (−0.032, −0.011) | +0.001 (−0.004, +0.006) |
| A-hybrid_all | 9,765 | 0.4697 | 0.759 | 0.788 | −0.029 (−0.040, −0.016) | +0.001 (−0.005, +0.007) |
| A-hybrid_all-pre | 9,765 | 0.4839 | 0.751 | 0.788 | −0.036 (−0.053, −0.022) | −0.000 (−0.008, +0.006) |
| tab-default-s20 | 9,765 | 0.4430 | 0.770 | 0.788 | −0.018 (−0.031, −0.009) | +0.001 (−0.004, +0.006) |
| tab-wide-s20 | 9,765 | 0.4442 | 0.770 | 0.788 | −0.018 (−0.030, −0.009) | +0.001 (−0.004, +0.005) |
| **final-S** (from `S-hybrid`, 6 networks per fold) | 4,263 | 0.5329 | 0.757 | 0.802 | −0.045 (−0.061, −0.027) | −0.009 (−0.017, +0.000) |
| S-formula (control) | 4,263 | 0.5026 | 0.755 | 0.802 | −0.047 (−0.067, −0.029) | −0.010 (−0.019, −0.001) |
| S-hybrid | 4,263 | 0.5255 | 0.760 | 0.802 | −0.042 (−0.061, −0.022) | −0.007 (−0.015, +0.003) |
| S-hybrid-pre | 4,263 | 0.5386 | 0.762 | 0.802 | −0.040 (−0.060, −0.020) | −0.002 (−0.012, +0.008) |
| S-cgcnn (structure only) | 4,263 | 0.5689 | 0.725 | 0.802 | −0.077 (−0.102, −0.054) | −0.017 (−0.028, −0.007) |
| S-cgcnn-pre | 4,263 | 0.5602 | 0.732 | 0.802 | −0.069 (−0.101, −0.042) | −0.011 (−0.025, +0.001) |

The 12 sweep jobs all lie between −0.029 and −0.041 against cycle 6, and
none has a blend interval above zero. Every row is in the JSON.

### The questions fixed before the run

| Question | Result |
| --- | --- |
| Does `final-A` beat cycle 6 (the rule)? | **No.** −0.022 (−0.035, −0.012); the interval lies below zero |
| Does its blend with cycle 6 beat cycle 6? | **No.** +0.002 (−0.004, +0.006) |
| Does structure add anything in track A (`final-A` minus `A-formula_all`)? | **No.** −0.002 (−0.011, +0.007) |
| Does structure add anything in track S (`final-S` minus `S-formula`)? | **No, for ranking:** AUC +0.002 (−0.014, +0.018). Tc error on the log scale is lower by 0.035 (0.003 to 0.080), but still 0.10 worse than cycle 6's |
| Does `final-S` beat cycle 6 on its own rows? | **No.** −0.045 (−0.061, −0.027) |

### Predictions against results

| Prediction | Chance given | Result |
| --- | --- | --- |
| `final-A` beats cycle 6 under the rule | 10% | no |
| The blend of `final-A` and cycle 6 beats cycle 6 | 20% | no |
| `final-A` beats `A-formula_all` | 30% | no |
| `final-S` beats `S-formula` | 35% | no |
| `final-S` beats cycle 6 on its rows | 15% | no |
| Pretraining lowers validation loss (`S-cgcnn-pre` below `S-cgcnn`) | 65% | yes: 0.560 against 0.569 |
| The 20-network tabular bank still trails cycle 6 | 85% | yes |
| Every core job finishes within 8 hours | 75% | yes: 5.8 hours in all |

All eight landed on the side given more than 50%. Eight predictions are
too few to judge calibration.

### What I make of it

- [Certain] **The graph adds no ranking information beyond what the
  formula features already carry.**
  - Structure alone (`S-cgcnn`) is the worst model in the table.
  - Pretraining on 29,913 Materials Project structures helped the
    structure-only network a little, and hurt the track A network.
  - [Likely] The reason is the data. Many 3DSC structures are one doped
    parent shared by a whole series of compositions, so the graph mostly
    repeats the formula.
- [Certain] **The network family as a whole trails cycle 6** by about
  0.02 AUC on all rows and about 0.045 on the rows that have structures.
  - The best network is still the plain tabular bank (−0.018), which
    matches the CPU baseline (`docs/local-gpu.md`).
  - Twenty networks per fold instead of five, a wider network and an
    overnight sweep moved it by less than 0.002.
- [Likely] **Compute was not the limit.** Five times more networks,
  pretraining and a 12-setting sweep left the gap where it was.
- **Selection worked as intended.** The configuration chosen by
  early-stopping loss (`sweep-batch64`) was not the one with the best
  test AUC (`sweep-wd1e-3`, −0.029). The difference is inside the noise,
  and the test rows were never used to choose.

**Decision:** cycle 6 stays the ranking of record. The structure networks
are not added as a second opinion: they trail the tabular network, which
already fills that role.
