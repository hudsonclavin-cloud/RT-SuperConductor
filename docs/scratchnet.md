# The from-scratch neural network

A neural network, written by hand in numpy, that reads a chemical formula and
answers two questions: is this material a superconductor, and if so at what
temperature (Tc)? This page says what it is, how it was tested, what it scored
and what the scores do and do not mean.

Short version of the result is in [the last section](#what-the-results-mean).

## Words used on this page

| Word | Meaning |
|---|---|
| Weight | One adjustable number inside the network. Training means adjusting the weights. |
| Step | One adjustment of all the weights, from one batch of 128 materials. |
| Epoch | One pass through all the training materials. |
| Early stopping | Watching the error on materials held back from training, and keeping the weights from the epoch where that error was lowest. Past that point the network memorizes instead of learning. |
| AUC | Take one superconductor and one non-superconductor at random. AUC is how often the model scores the superconductor higher. 0.5 is a coin flip, 1.0 is perfect. |
| Tc error | Average gap, in kelvin, between the predicted and the measured Tc, over materials that do superconduct. |
| Fold | One fifth of the development data, held out in turn so every material is predicted by a model that never saw it. |
| Locked test set | Materials set aside before any training and scored exactly once, at the very end. Because no choice was ever made by looking at them, their score is the honest one. |
| Group | A set of materials too alike to separate (see below). Splits are always made by whole groups. |
| Gradient boosting | The existing method in this project: many small decision trees, each correcting the last. It is the method to beat. |
| Ensemble | Several models trained separately whose predictions are averaged. |

## What was built

Everything in `src/rt_superconductor/scratchnet/` uses numpy only. No
PyTorch, no scikit-learn.

| File | What it holds |
|---|---|
| `layers.py` | The building blocks, each with a forward pass (compute the output) and a hand-derived backward pass (work out how each weight should change): linear layer, embedding, layer normalization, SiLU activation, dropout, self-attention, attention pooling, and the two loss functions. |
| `model.py` | The network assembled from those blocks. |
| `optim.py` | The AdamW optimizer, the learning-rate schedule, gradient clipping, and the tools used to check gradients. |
| `data.py` | Formula parsing and turning compositions into arrays. |
| `train.py` | The training loop, early stopping, weight averaging, prediction and scoring. |

How the network reads a formula such as YBa2Cu3O7:

1. Each element becomes a list of 64 numbers built from three things: a
   learned vector for that element, 28 fixed physical properties
   (electronegativity, valence electrons, melting point and so on, from
   `data/training/element-features.csv`), and how much of the element there is.
2. Two self-attention blocks let each element adjust its numbers in light of
   the others. Copper next to oxygen is not the same as copper next to zinc.
3. A pooling step merges the elements into one vector, weighted by amount.
4. Two outputs read that vector: the chance of superconductivity, and Tc.

The basic network has about 108,000 weights. The version chosen in round 2
adds two things and has about 127,000:

- a **formula branch**: the 155 whole-formula features that gradient boosting
  uses are also fed in, alongside the per-element view;
- an **auxiliary task**: besides the superconductor data, the network is
  shown 26,911 Materials Project compositions and asked to predict five
  calculated properties of each (formation energy, band gap, metal or not,
  volume per atom, magnetization). These are not superconductivity labels.
  The point is that learning them teaches the shared layers chemistry for
  thousands of element combinations the superconductor data never covers,
  which is where the candidates are.

### How the hand-written maths was checked

A hand-derived backward pass is easy to get subtly wrong, and a wrong one
still trains, just badly. So every layer and the whole network are checked
two independent ways (`tests/test_scratchnet_layers.py`,
`tests/test_scratchnet_model.py`):

- against finite differences: nudge one weight, see how the error changes,
  compare with what the backward pass says;
- against PyTorch's automatic differentiation, where the two agree to eight
  significant figures.

One real finding came out of this: a bias term in the attention layer has a
gradient of exactly zero, by the mathematics of softmax, so it was removed.

## The data, and the leak that was closed

`scripts/build_scratchnet_data.py` builds one row per distinct composition
from the SuperCon-lineage sources already in this project: 15,960
compositions, of which 11,819 superconduct, 3,708 are recorded with Tc = 0,
and 433 appear only on Hosono's list of unsuccessful attempts (kept, but not
used as negatives).

**Groups.** Superconductor data is full of doping series: the same compound
with 1%, 2%, 3% of something swapped in. If one of those is in training and
its neighbour is in the test set, the model is being asked something it has
effectively already been told, and every score comes out flattering.

- The first grouping used here (elements above 5 atomic percent) still left
  **11% of locked-test rows with a near-twin in training.**
- The grouping now used ties together any two compositions that differ by at
  most 0.02 on every element, and chains those ties. Near-twin pairs across
  groups: **zero** (`near_twin_pairs_split_across_groups` in
  `docs/scratchnet-data.json`).

One consequence: the cuprate doping series all chain into a single group of
4,245 rows. It is too big to hold out, so it is used for training only. The
tests below therefore say little about the famous cuprate families and a lot
about everything else, which suits the goal, since the candidate pool is
about 1% cuprates.

| Part | Rows | Groups | Use |
|---|---|---|---|
| Training only | 4,245 | 1 | The cuprate chain. Never tested on. |
| Development | 9,765 | 2,768 | Five folds. All choices are made here. |
| Locked test set | 1,950 | 539 | Scored once, at the end. |

Only 80 locked-test superconductors have Tc above 40 K. Claims about
high-Tc materials rest on thin evidence, and the intervals show it.

## The comparison

Three models are trained on identical rows and tested on identical held-out
groups (`scripts/run_scratchnet.py cv`):

- **the network**, an ensemble of independently trained copies;
- **gradient boosting** on 155 whole-formula features (the project's Magpie
  and shell-filling features), with a classifier for "does it superconduct"
  and a second model for Tc;
- **nearest neighbour**: look up the most similar known compositions and
  copy their answer. Anything that cannot beat this has learned nothing
  beyond the lookup table.

Scores are also reported by **distance**: the share of atoms you would have
to swap to turn a material into its nearest training material. "Near" is
under 0.1; "far" is 0.3 or more. The candidate pool sits mostly far (median
about 0.26), so the far rows are the ones that resemble the real job.

## Round 1: train, test, see what is wrong

Basic network, two independently trained copies averaged, five folds.

| Model | AUC | Tc error (K) |
|---|---|---|
| Gradient boosting | 0.790 | 4.09 |
| Network (attention) | 0.755 | 4.16 |
| Network without attention | 0.751 | 4.40 |
| Nearest neighbour | 0.694 | 5.01 |
| Network + boosting, averaged | 0.791 | 3.94 |

What round 1 showed:

- The network beat the lookup baseline clearly and lost to boosting clearly
  (AUC gap -0.035, 95% interval -0.050 to -0.020).
- Averaging it with boosting changed nothing measurable.
- One copy alone scored 0.733 to 0.741; two averaged scored 0.755. A single
  network is noisy, so an ensemble is worth having.
- The best epoch was usually between 6 and 24 out of 80: the network starts
  memorizing early. More epochs would not help; this is why "keep training
  the same network" is not a route to a better model here.
- On far rows everything got worse: boosting 0.727, network 0.671, lookup
  0.555 (barely above a coin flip).

Full numbers: `docs/scratchnet-round1.json`.

## Round 2: four changes, one chosen

Each change was tried on the same development folds, three copies each.

| Network variant | AUC | Tc error (K) | What changed |
|---|---|---|---|
| Round 1 network, three copies | 0.762 | 4.34 | Reference |
| Tuned | 0.756 | 4.30 | More regularization, weight averaging. **No gain.** |
| Multitask | 0.775 | 3.98 | Auxiliary Materials Project task. |
| Hybrid | 0.769 | 4.35 | Formula branch. |
| Hybrid + multitask | 0.784 | 4.18 | Both. |
| Gradient boosting (its own settings tuned on a seven-setting grid) | 0.790 | 3.91 | |
| Hybrid-multitask network + boosting, averaged | 0.799 | 3.92 | |

The auxiliary task was the change that mattered: +0.019 AUC on its own. It
is also the one thing here that a network can do and boosting on formula
features cannot.

The choice of variant and of how to mix it with boosting was made by a rule
written into the code before the scores were seen (best ordering of
materials by probability x Tc on non-cuprate development rows). It picked
hybrid + multitask, mixed 50/50 with boosting. On the development folds that
mix beat boosting alone by +0.009 AUC (0.004 to 0.015).

Full numbers: `docs/scratchnet-round2.json`.

### A review before opening the locked set

Before the final test an independent review of the code was run, with each
claimed defect then attacked by a second reviewer. It found no error in the
hand-written gradients and no leak of test rows into training. It did find
that the "use the locked set only once" guard could be bypassed by pointing
the output at a different folder, plus several smaller defects (listed in
the commit message). All were fixed and tested before the locked set was
touched.

## The locked test set, scored once

Final ensemble: 25 networks (5 seeds x 5 validation folds) and 5 boosting
models, trained on all development and training-only rows. 1,950 rows in 539
groups that no model and no decision had seen.

| Model | AUC (95% interval) | Tc error (K) | Tc error, log scale |
|---|---|---|---|
| Gradient boosting | 0.832 (0.793 to 0.875) | 4.54 | 0.351 |
| Network | 0.815 (0.776 to 0.857) | 4.22 | 0.382 |
| Network + boosting, averaged | 0.834 (0.795 to 0.876) | 4.35 | 0.357 |
| Nearest neighbour | 0.708 (0.663 to 0.755) | 5.85 | 0.460 |

Differences from boosting, with 95% intervals from resampling whole groups:

| | AUC | Tc error, log scale (negative is better) |
|---|---|---|
| Network alone | -0.019 (-0.038 to +0.003) | +0.033 (+0.001 to +0.060) |
| Network + boosting | +0.001 (-0.008 to +0.011) | +0.007 (-0.009 to +0.021) |

On the 188 far rows, the ones most like the candidate pool: boosting 0.760,
network 0.744, mix 0.765, lookup 0.625. With 171 labelled rows these are
too few to separate the three.

Full numbers: `docs/scratchnet-lockbox.json`.

## The candidate ranking, and why not to trust its top

`scripts/run_scratchnet.py rank` scores the 26,911 distinct compositions in
the Materials Project pool (25,019 of them untested) and writes three lists
of 200, at most three per set of elements:

- `docs/scratchnet-candidates-top.csv`: everything;
- `docs/scratchnet-candidates-top-not-cuprates.csv`: without copper-oxygen
  compounds;
- `docs/scratchnet-candidates-top-metals.csv`: calculated metals with no
  copper and no iron-pnictogen chemistry, the only kind the electron-phonon
  pipeline in this project can check.

Each row carries the network's rank and boosting's rank separately, how much
the 25 networks disagree, the distance to the nearest training material, and
the nearest known material with its Tc.

**Read the top of these lists as a description of what the models learned,
not as a list of discoveries.**

- 94 of the top 200 contain copper: 49 are copper oxides that look like known
  cuprates, and 45 are copper fluorides with no oxygen (RbCuF3, CsCuF4 and
  similar). The fluorides trace back to a single known oxyfluoride family at
  46 K. No copper fluoride is known to superconduct. The models have learned
  "copper plus a strongly electron-attracting partner plus a large metal ion
  resembles a cuprate", which is a statement about the training table.
- 131 of the top 200 are far (0.3 or more) from anything the models trained
  on, the regime where measured accuracy was lowest.
- 118 of the top 200 are not metals according to their calculated band
  structure.
- The metals-only list tops out at an expected Tc of 26 K and is led by
  silver and palladium fluorides and alkali iridates, again by resemblance
  to cuprate chemistry.
- The "probability" column is a ranking score. Three quarters of the
  training table are superconductors, because people record what works;
  among arbitrary compounds the true share is far lower. Expected Tc values
  are therefore inflated and only their order means anything.

## What the results mean

1. **The network works.** It is built from nothing but numpy, its gradients
   are verified two ways, and it beats the lookup baseline by a wide margin
   on materials it never saw (0.815 against 0.708).
2. **It does not beat gradient boosting.** It is about 0.02 AUC behind on the
   locked set, with an interval that only just includes zero, and is worse
   at Tc on the log scale. This matches the literature: no published result
   was found in which a composition-only network beats boosted trees on a
   grouped superconductor split.
3. **Mixing the two is a tie with boosting alone.** The +0.009 seen on the
   development folds shrank to +0.001 on the locked set. That gap is the
   reason the locked set exists: a gain picked out of 25 variant-and-weight
   combinations was partly luck.
4. **The bottleneck is the data, not the model.** Two very different models
   land in the same place and both fall off at the same rate with distance
   from known materials. A third model family is unlikely to change that.
   What would: information the formula does not contain (crystal structure,
   calculated electron-phonon coupling) and trustworthy negative results.
5. **The one neural-network-specific gain was the auxiliary task.** That is
   the direction worth extending if this network is developed further: more
   calculated properties, and structure as an input.

## Not done

Stated so nobody assumes otherwise: no weighting of rows by group size (the
cuprate chain is 35% of training rows); the 19 zero-Tc rows that Materials
Project marks as wide-gap insulators were kept as negatives; the multitask
and hybrid variants were only tried with the "tuned" training settings, which
on their own did not help; probabilities are not calibrated; only 80 locked
superconductors exceed 40 K, so nothing here tests high-Tc prediction well;
elements He, Ne, Ar, At, Rn and Fr are not in the element table, and seven
pool structures are skipped for that reason.

## Running it

```bash
python scripts/build_scratchnet_data.py
```

```bash
python scripts/run_scratchnet.py cv --name round2 --variants attention,tuned,multitask,hybrid,hybrid-multitask --seeds 0,1,2 --jobs 3 --tune-boosting
```

```bash
python scripts/run_scratchnet.py final --name round2 --selection docs/scratchnet-round2.json --seeds 0,1,2,3,4 --jobs 4
```

```bash
python scripts/run_scratchnet.py rank --name round2
```

The first needs the SuperCon exports
(`python scripts/download_training_data.py --cif-limit 0`). The third
refuses to run while `docs/scratchnet-lockbox.json` exists: the locked set
has been used. Add `--rehearse` to exercise that step on development fold 0
instead. Set `OMP_NUM_THREADS=1` when running several jobs. On four busy
cores the cross-validation took 68 minutes and the final step 18.

Model files and the full ranking are written under
`data/processed/scratchnet/` and are not committed.
