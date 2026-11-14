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

The network has about 108,000 weights.

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
