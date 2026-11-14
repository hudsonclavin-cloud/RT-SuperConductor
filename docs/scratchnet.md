# The from-scratch neural network

A neural network, written by hand in numpy, that reads a chemical formula and
answers two questions: is this material a superconductor, and if so at what
temperature (Tc)? This page says what it is, how it was tested, what it scored
and what the scores do and do not mean.

Short version of the result is in [the last section](#what-the-results-mean).

**Decision, 13 November 2026:** candidates are ranked by gradient boosting
alone; the network is shown beside it as a second opinion. See
[the ranking of record](#the-ranking-of-record-boosting-with-the-network-as-a-second-opinion).

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
| Labelled | A row whose answer is known: recorded either as a superconductor or as not one. |
| Rank correlation | How alike two orderings of the same materials are: 1 is the same order, 0 no relation. |

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

*Correction added 13 November 2026: measured against the materials the final
models trained on, the pool's median distance is 0.29 and 49% of it is far.
Of boosting's top 200 candidates, 164 are far.*

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

*Note added 13 November 2026: the three lists described in this section are
ordered by the 50/50 mix of network and boosting. They are kept for
comparison, and `rank --order mix` reproduces them (with three extra columns
at the end). The ranking of record is now boosting's, described in the next
section. One statement below is also narrowed there: "only their order
means anything" holds for near and middle-distance rows; for far rows (131
of this list's 200) the order has not been verified either.*

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

## The ranking of record: boosting, with the network as a second opinion

On the locked test set the mix was a tie with boosting alone, so the simpler
model sets the order. `scripts/run_scratchnet.py rank` now ranks by boosting
and writes:

- `docs/scratchnet-candidates-boosting-top.csv`: everything;
- `docs/scratchnet-candidates-boosting-top-not-cuprates.csv`;
- `docs/scratchnet-candidates-boosting-top-metals.csv`: calculated metals
  with no copper and no iron-pnictogen chemistry;
- `docs/scratchnet-candidates-boosting.json`: most of the counts quoted
  below; the rest can be recounted from the three CSV files.

How to read a row:

| Column | Meaning |
|---|---|
| `rank`, `expected_tc_k`, `p_superconductor`, `tc_if_superconductor_k` | Boosting's view. `expected_tc_k` is probability x Tc and sets the order. |
| `network_rank`, `network_expected_tc_k`, `network_p`, `network_tc_k` | The network's view of the same material. |
| `network_view` | The second opinion in one word, from the two figures in the row. `rejects`: `network_expected_tc_k` is below 0.3 of `expected_tc_k`. `agrees`: it is at least half. `lukewarm`: in between. |
| `network_disagreement` | How much the 25 networks differ among themselves on this material. |
| `distance_to_nearest_training` | 0 is a known material, 1 shares no atoms with anything the models saw. |

### What the second opinion is worth

Short answer: **on known materials close to the training data, a rejection
by the network was a real warning. For materials far from known ones, which
is most of the list, nothing could be measured.** So the label is
information, not a tested filter for this list.

The check uses the development rows, where the answers are known
(`scripts/run_scratchnet.py second-opinion`, results in
`docs/scratchnet-second-opinion.json`; the locked set is not read). Of the
9,399 development rows with a known answer, take the ones boosting ranks
highest and apply the same rule as the label.

Boosting has its own way of doubting a pick: a low `p_superconductor`. A
second opinion is worth having only if it flags failures that boosting's own
doubt misses. So set aside, as boosting's doubted picks, the same number of
picks that boosting itself rates least probable, and compare the picks **only
the network** rejects with the picks **neither** doubts. Gaps are in
percentage points, with 95% intervals from resampling whole groups
(materials in one group are too alike to count separately).

| Boosting's top | Network's figure is below | Picks rejected (copper oxides among them) | Superconduct: kept | rejected | only the network rejects | neither doubts | Gap between the last two |
|---|---|---|---|---|---|---|---|
| 10% (940 picks) | half of boosting's | 53 (35) | 89% | 47% | 21 of 42 | 762 of 845 | +40.2 (+21.8 to +66.1) |
| 10% | 0.3 of boosting's, the `rejects` rule | 17 (13) | 88% | 12% | 2 of 14 | 804 of 909 | +74.2 (+55.3 to +90.0) |
| 5% (470 picks) | half of boosting's | 19 (16) | 91% | 26% | 5 of 17 | 397 of 434 | +62.1 (+34.4 to +91.1) |
| 5% | 0.3 of boosting's | 6 (6) | 89% | 0% | 0 of 6 | 410 of 458 | too few picks for an interval |

What this shows, and what it does not:

- **On these rows a rejection was a real warning.** Picks that only the
  network rejected superconducted 2 times in 14 at the `rejects` rule,
  against 88% for picks neither model doubted.
- **It flags different picks from boosting's own doubt.** Of the 17 picks
  the network rejects at that rule, 3 are among boosting's 17 least
  probable. As a filter used alone, the network's rejection cannot be shown
  to beat boosting's probability (boosting's 17 least probable picks
  superconduct 41% of the time against the network's 12%; the difference
  between the two kept-against-rejected gaps, +30 points, has an interval
  from -9 to +62). Its value is as an addition.
- **Most of the rejected picks are copper oxides** (13 of the 17). That is
  where the skill shows: spotting copper oxides that resemble cuprates
  without being superconductors. For other kinds of material the evidence is
  a handful of picks.
- **These rows are close to known materials.** Of boosting's 940 top
  development picks, 655 are near, 283 are at middle distance and 2 are far,
  so the rule cannot be run for far materials at all. A looser version can:
  inside the far band alone (boosting's top tenth of the 1,068 far rows,
  with the network's own top tenth of them as the cut), picks only the
  network rejected superconducted 19 times in 29 against 37 in 48 for picks
  neither doubted. The gap, +11.6 points, has an interval from -13.6 to
  +34.3. No effect could be seen.
- **"Agrees" is the ordinary case, not a bonus.** On these rows the network
  agrees with 887 of boosting's 940 picks, so agreement tells you little.
  "Lukewarm" sits in between: of the 36 picks where the network's figure was
  between 0.3 and half of boosting's, 23 superconducted.
- **The check was run on the rows the network was chosen on,** the same rows
  that made the 50/50 mix look better than it was. On the locked set, adding
  the network to boosting changed nothing (+0.001 AUC). Read the gaps above
  as upper estimates.

The JSON also holds the kept-against-rejected comparison with "reaches
20 K" as the outcome, and all of the checks with rules based on rank instead
of size. They point the same way.

### What boosting's top 200 looks like

- 139 contain fluorine; 80 contain rubidium, 72 caesium, 58 potassium. 137
  are fluorides of an alkali metal. Half of those (68) are fluorides of
  copper, silver or palladium, which fill the top of the list (27 of the
  first 50 rows); the rest are fluorides of other metals. The first two rows
  are copper oxides close to known cuprates.
- 46 contain copper: 21 copper oxides and 25 copper fluorides with no oxygen.
- 164 are far (0.3 or more) from anything the models trained on, 26 are at
  middle distance and 10 are near.
- 149 are not metals according to their calculated band structure.
- The network agrees on 47, is lukewarm on 48 and rejects 105.
- **The 105 rejected rows are not the kind the check covers.** 91 are far
  and 14 are at middle distance; 76 are fluorides; none contains copper. The
  14 at middle distance are oxides of silver, gold, cadmium, tin, lead and
  antimony, 13 of them with an alkali metal (KAuO2, CsAgO2, K2Cd2O3 and
  similar) and one with strontium (SrAgO2). They are the rows
  for which the rejection has the most behind it, and even there the
  evidence came mostly from copper oxides.
- **37 of the 47 rows the network agrees on contain copper**: 15 of the 21
  copper oxides and 22 of the 25 copper fluorides. The two models were
  trained on the same answers, and this network is also fed the same 155
  formula features, so they share the habit of treating copper with a
  strongly electron-attracting partner as cuprate-like. Their agreement
  here is one opinion stated twice, not two.
- Row 2, Sr2Cu2O5, is worth a second look for the opposite reason: it is a
  near copper oxide on which the network's figure is 17 K against boosting's
  55 K. In the check, 15 of the 35 copper oxides with the network below half
  of boosting superconducted.
- The metals list reaches down to boosting's place 1,655 and tops out at an
  expected Tc of 47 K. The network agrees on 32 of its 200 rows, is lukewarm
  on 53 and rejects 115, of which 93 are far and 22 at middle distance.

**The two models disagree about size in a way they rarely did on known
materials.** Across the whole untested pool their orders are similar (rank
correlation 0.85), though only 55 of boosting's 200 listed materials are in
the network's own first 200. On the 536 development rows where boosting
expected 25 K or more, the network's figure was typically 90% of boosting's
and was below half of it in 4% of cases. In the top 200 candidates the
network's figure is typically 28% of boosting's and is below half in 153
cases. Boosting expects 30 K or more for 106 of the 200, the network for 16.
For far materials the development data cannot say which is right. Both
models' high predictions for far materials failed there (boosting expected
20 K or more for 2 far rows and the network for 1; none reached 20 K), and
of the 1,068 far development rows with a known answer only 2 reach 20 K and
none reaches 30 K. What the size gap does show is that the top of this list
is outside the territory where either model has been checked.

One more limit. Inside the top tenth of near and of middle-distance
development rows, boosting's order still tracks the true Tc (rank
correlation 0.46 and 0.39). Inside the top tenth of far rows no such order
could be seen (0.02, interval -0.20 to +0.27). That group is not much like
the list either: its median true Tc is 4 K. So for far candidates the order
is unverified, not shown to be empty. The list says "boosting puts this in
its top group"; whether "this one is better than the row below" has not been
tested.

### A sensible way to use the lists

1. Read a list as boosting's top group, and treat `expected_tc_k` as
   membership of that group, not as a temperature. For far rows
   (`distance_to_nearest_training` of 0.3 or more) do not lean on the order
   within the list either.
2. Take `network_view = rejects` seriously for near and middle-distance
   rows, most of all for copper oxides: on known materials of that kind such
   picks mostly failed. For far rows it is untested, so there it is a reason
   to look harder, not a verdict.
3. Do not read `agrees` as confirmation. It mostly marks the copper
   look-alikes.
4. `p_superconductor` is boosting's own measure of doubt and can be used the
   same way: of the 17 top development picks it rated least probable, 7
   superconducted, against 88% of the picks neither model doubted. In the
   top 200 it runs from 0.46 to 0.97, and 19 rows are below 0.7.
5. Choose among far candidates on other grounds: stability
   (`energy_above_hull_ev_per_atom`), whether it is a metal, and what the
   nearest known material is.
6. For anything headed to the electron-phonon pipeline, use the metals list,
   and expect the calculation to reject most of it. A calculated "metal" flag
   is unreliable for copper and silver fluorides and oxides, which standard
   band-structure calculations often wrongly describe as metallic.

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
   Decision taken on this (13 November 2026): rank by boosting alone and
   show the network as a second opinion.
4. **The bottleneck is the data, not the model.** Two very different models
   land in the same place and both fall off at the same rate with distance
   from known materials. A third model family is unlikely to change that.
   What would: information the formula does not contain (crystal structure,
   calculated electron-phonon coupling) and trustworthy negative results.
5. **The one neural-network-specific gain was the auxiliary task.** That is
   the direction worth extending if this network is developed further: more
   calculated properties, and structure as an input.
6. **As a second opinion, the network is worth something near known
   materials and unproven where the candidates are.** Its view is printed
   beside boosting's, as decided. On known materials close to the training
   data, the picks it rejected mostly failed, and they were mostly not the
   picks boosting itself doubted. For materials far from known ones, where
   most of the listed candidates sit (164 of boosting's top 200), neither
   its agreement nor its rejection made a measurable difference.

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

```bash
python scripts/run_scratchnet.py second-opinion --name round2
```

The first needs the SuperCon exports
(`python scripts/download_training_data.py --cif-limit 0`). The third
refuses to run while `docs/scratchnet-lockbox.json` exists: the locked set
has been used. Add `--rehearse` to exercise that step on development fold 0
instead. Set `OMP_NUM_THREADS=1` when running several jobs. On four busy
cores the cross-validation took 68 minutes and the final step 18.

Model files and the full ranking are written under
`data/processed/scratchnet/` and are not committed.

`rank` orders by boosting unless told otherwise: `--order mix` gives the
50/50 mix (the first version of the lists) and `--order network` the network
alone. `second-opinion` needs the saved predictions of the `cv` run.
