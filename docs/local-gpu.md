# Training on the owner's own GPU

**Main idea:** the owner's machine trains networks; this session builds
what they train on and judges what comes back. Both halves use exactly
the rows, folds and early-stopping splits that cycle 6's boosting (the
ranking of record) used. So any difference in the comparison comes from
the model, not from the data.

Written 2026-11-22.

## The loop

| Step | Where | Command | Output |
| --- | --- | --- | --- |
| 1. Export | here | `python scripts/local_gpu.py export` | `data/processed/local-gpu/bundle.npz` (11 MB) and `bundle.json` |
| 2. Hand over | here → owner | the bundle is sent to the owner as a file. It is not committed: it holds the licensed training table (see `.gitignore`) | — |
| 3. Train | owner's machine | `python scripts/local_gpu_train.py bundle.npz --out results/<tag> --device cuda` | `oof.npz`, `candidates_*.npz`, `run.json` |
| 4. Return | owner → repository | commit `results/<tag>/` as `data/benchmarks/local-gpu/<tag>/` on a branch and push it. It holds predictions only, about 1 MB | — |
| 5. Judge | here | `python scripts/local_gpu.py compare <tag>` | `docs/local-gpu/<tag>.json` |

`scripts/local_gpu_train.py` needs only numpy and PyTorch. It imports
nothing from this repository, so it can be copied to any machine on its
own.

## What the bundle holds, and what it leaves out

**Rows**
- Every table row except the locked ones: 14,010 rows (9,765 development
  rows plus 4,245 train-only rows).
- Plus the 851 NIMS rows that cycle 4 added.
- The 1,950 locked rows are not in the file at all. The export stops if
  one reaches it, and the trainer refuses a bundle that contains one.

**Features:** the same 155 formula features the boosting uses.

**Folds:** for each of the 5 development folds, the row numbers that
cycle 6 fitted on, stopped early on, and scored on.
- NIMS rows are added to a fold only by cycle 4's rule: never a row of
  the held-out fold, and never a row tied to a group in that fold's
  early-stopping rows.

**Candidate lists:** predicted with every fold's networks, then averaged.
- The 26,911 made metals.
- Shortlist 3's 571 filtered compounds.

**Checksum**
- `bundle.json` records a checksum of the arrays.
- The trainer records the same checksum in `run.json`.
- `compare` refuses results trained on a different bundle.

## How results are judged

`compare` scores three sets of predictions on the same development rows:
- the network;
- cycle 6;
- a 50/50 blend of the two.

The intervals come from resampling whole groups, as in every learning
cycle.

**The rule, fixed before any GPU result:** a network replaces or joins
the ranking of record only if two things hold:
- its paired non-cuprate AUC interval against cycle 6 lies above zero;
- the owner accepts it.

Otherwise it is a second opinion, as the network already is (`docs/scratchnet.md`, "The ranking of record").

The locked rows are not used here. They are scored once, only if a
network passes the rule above.

## Owner's machine: setup

1. Install Python 3.10 to 3.13.
2. Install the CUDA build of PyTorch from pytorch.org. The hackathon
   repository's `gpu/README.md` has the steps for the laptop.
3. Install numpy.

Then, from the folder that holds the bundle:

```bash
python local_gpu_train.py bundle.npz --out results/mlp-default --device cuda
```

**Other things to try:**
- Wider networks: `--hidden 512,256,128`.
- More networks per fold: `--seeds 20`.
- A weaker Tc head: `--aux-weight 0.1`.

Name each run, for example `results/mlp-wide-s20`. Each run writes its
settings to `run.json`, so every result carries its own recipe.

## What a GPU is for, and what it is not for

- [Certain, measured here] **The default network does not need a GPU.**
  - It is a 2-layer network on 12,000 rows.
  - A full run took about 3 minutes on this session's 4 CPU cores: 5
    folds, 5 networks each, about 40 seconds per fold.
  - A GPU can only shorten that run to seconds.
- [Likely] **The GPU pays off on work that is about 100 times bigger:**
  - sweeps of hundreds of settings;
  - banks of 20 or more networks;
  - networks over crystal structures, such as the hackathon repository's
    CGCNN.
- [Certain, from this repository's history] **Compute was not what held
  the network back.**
  - The network trailed boosting by 0.035 AUC (95% interval 0.020 to
    0.050) when it was built (`docs/scratchnet.md`).
  - The hackathon's own overnight run found the same thing on its data:
    single networks were no better than trees, and only an average of
    networks matched them.
  - [Likely] The limit is the labels: about 10,000 development rows, some
    still wrong.

## Baseline to beat: the default network on this session's CPU, 2026-11-22

Run: `cpu-baseline`, the default settings (two layers of 256 and 128, 5
networks per fold, about 3.5 minutes on 4 CPU cores). Comparison:
`docs/local-gpu/cpu-baseline.json`. All 9,765 development rows were
scored. A positive AUC difference favours the first model.

| Rows | Network AUC | Cycle 6 AUC | Blend AUC | Network minus cycle 6 (95% interval) | Blend minus cycle 6 |
| --- | --- | --- | --- | --- | --- |
| All | 0.764 | 0.790 | 0.789 | −0.026 (−0.038 to −0.014) | −0.001 (−0.006 to +0.005) |
| Not cuprates | 0.768 | 0.788 | 0.789 | −0.019 (−0.031 to −0.009) | +0.001 (−0.004 to +0.006) |

**What it shows**
- [Certain] The network loses to cycle 6 on these rows. Averaging the
  two adds nothing that can be measured.
- The network's Tc error, on the log scale, is also worse: +0.06
  (+0.04 to +0.08).
- So under the rule above, it stays a second opinion.

**Compared with the first network**
- [Likely] The gap is about half the first network's, which was −0.035
  in `docs/scratchnet.md`.
- The likely reasons are the bank of 5 and the NIMS rows.
- That comparison is not paired: it used an earlier table.

**What a GPU run must show:** a non-cuprate AUC interval against cycle 6
that lies above zero, about +0.02 better than this run.
- [Speculative] Chance that wider networks or more seeds alone do it:
  15%.
- Better odds lie with inputs boosting cannot use, such as crystal
  structures. That is a different bundle, not yet built.

## Next: crystal-structure networks overnight, 2026-11-22

The owner asked for a structure bundle and an 8-hour run. That run is set
up in `docs/overnight-gpu.md`:
- `scripts/structure_gpu.py`;
- `scripts/structure_gpu_train.py`;
- `scripts/overnight_gpu.py`.

It includes the two 20-network tabular banks that this section's
baseline left as the next step.

## Next: one test of a tabular foundation model, 2026-11-23

A literature search for better training methods found one family with
independent evidence at this data size: tabular foundation models. The
review, the planned test and its commands are in
`docs/training-methods-review.md`.
- The runner is `scripts/foundation_gpu_train.py`.
- Its output is judged by `python scripts/local_gpu.py compare`, like
  every run in this document.

## Sharing the laptop GPU with the NCSU-Hackathon pipeline, 2026-11-25

The owner wants to run this project's training and the NCSU-Hackathon
pipeline on the laptop at the same time. Before that, the other project's
agent answers the read-only questions in `docs/shared-gpu-questions.md`.
The deciding answer is how busy the GPU is when each pipeline runs alone.
