# The blend on the locked rows: plan, fixed before any look

**Main idea:** on the development rows, the 50/50 blend of TabICLv2 and
cycle 6 beat cycle 6 by +0.013 non-cuprate AUC (+0.007 to +0.020).
- That was a secondary result, so by the rule it could not change the
  ranking of record alone (`docs/training-methods-review.md`, "Result,
  2026-11-26").
- The owner chose to test it on the locked rows (2026-11-26).
- This plan is written and pushed before the locked rows' answers are read.

Written 2026-11-26.

## What is scored

**Rows:** the 1,950 locked rows.
- 1,730 of them are not cuprates. 1,663 of those have a label (counted
  without reading any label's value).
- The locked rows have been scored once before, in round 2 of the network
  (`docs/scratchnet-lockbox.json`). This is the second and last planned use.
- No model in this project has trained on them.

**Each model predicts them the way it predicts new compounds today:**

| Model | How it predicts a locked row |
| --- | --- |
| Cycle 6 | The mean of its five saved final models (`data/processed/learning-cycles/models/cycle6/`). Checked here: they reproduce cycle 6's published shortlist-3 probabilities to within 0.0005 |
| TabICLv2 | The mean of its five fold models, the same way it predicted the shortlist candidates. Run on the owner's laptop, with the same bundle as the development run |
| Blend | Their 50/50 average: probabilities directly, Tc on the log scale |

- [Certain] One difference favours cycle 6. Each of its final models learnt
  from every development fold but one. Each TabICLv2 fold model saw only the
  fit and validation rows of its fold, about 80% of the development rows.
- The laptop receives the locked rows' features and composition keys only
  (`scripts/lockbox_blend.py export`). Their labels stay here and are read
  once, by `scripts/lockbox_blend.py score`.
- That script refuses to score a second time.

## The rule

The blend becomes the ranking of record only if both hold:
1. **Its non-cuprate AUC on the locked rows is at least cycle 6's.** This is
   the same test as step 2 of the original plan.
2. The owner accepts it.

Also reported, without deciding anything:
- the paired intervals;
- all rows, cuprates included;
- the Tc error;
- TabICLv2 alone, which cannot be adopted because it failed step 1.

[Likely] Why the point estimate, not an interval:
- 1,663 labelled rows give an interval about 2.3 times as wide as the
  development one, roughly ±0.015.
- So a true gain of +0.013 would usually be inside it, and an interval test
  would mostly test the sample size.
- The development rows already supplied the interval evidence. This look
  checks the direction on rows nobody has tuned on.

## What follows

| Result | What happens |
| --- | --- |
| The blend passes and the owner accepts | It becomes the ranking of record. The made metals and shortlist 3 are re-ranked with it, from predictions the laptop has already made. Any change to the Hazel queue is proposed separately, not made automatically |
| The blend fails | Cycle 6 stays. The blend stays a second opinion. The locked rows are not used again for this question |

Either way, every new blend ranking needs the laptop's GPU (about 25
minutes). This cloud session cannot run TabICLv2.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| The blend's non-cuprate AUC on the locked rows is at least cycle 6's (the rule) | 80% |
| The blend's paired interval on the locked rows lies above zero | 35% |
| TabICLv2 alone is at least cycle 6 on the locked rows | 60% |
| Cycle 6's locked non-cuprate AUC is within ±0.03 of its development 0.788 | 75% |
| The blend's Tc error on the locked rows is lower than cycle 6's | 75% |
| The laptop run takes under 10 minutes | 80% |

## Commands

**Here, already done (2026-11-26):**

```bash
python scripts/lockbox_blend.py export
```

That wrote `data/processed/local-gpu/locked-features.npz`:
- 1,950 rows and 155 features, with no label, Tc, fold or group;
- SHA-256 `028d92a2ec1d8c234d39032ebd281449fcca771734b7751268fa06a934ac0e1d`.

The file is sent to the owner in chat. It is not committed, because it
comes from the licensed table.

**On the laptop** (Windows PowerShell, in `C:\Users\hudso\RT-SuperConductor`):

```powershell
git checkout claude/practical-tesla-ctf6oq
```
```powershell
git pull origin claude/practical-tesla-ctf6oq
```

Put the downloaded file in place:

```powershell
Move-Item $HOME\Downloads\locked-features.npz data\processed\local-gpu\
```

This must show `028D92A2EC1D8C234D39032EBD281449FCCA771734B7751268FA06A934AC0E1D`:

```powershell
Get-FileHash data\processed\local-gpu\locked-features.npz -Algorithm SHA256
```

This must list no `python.exe`:

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
```powershell
.venv\Scripts\python.exe scripts\foundation_gpu_train.py data\processed\local-gpu\bundle.npz --model tabicl --extra-candidates data\processed\local-gpu\locked-features.npz --candidates locked --out runs\lockbox-tabicl --device cuda
```

Then send the predictions back:

```powershell
git checkout -b lockbox-results-20261126
```
```powershell
Copy-Item runs\lockbox-tabicl data\benchmarks\local-gpu\lockbox-tabicl-20261126 -Recurse
```
```powershell
git add data/benchmarks/local-gpu/lockbox-tabicl-20261126
```
```powershell
git commit -m "TabICLv2 predictions for the locked rows (features only)"
```
```powershell
git push -u origin lockbox-results-20261126
```

**Here, once:**

```bash
python scripts/lockbox_blend.py score lockbox-tabicl-20261126
```

## Result, 2026-11-27

**Main result: the blend passes.** Its non-cuprate AUC on the locked rows is
0.827, against cycle 6's 0.820. By the rule above it becomes the ranking of
record if the owner accepts it. Until then, cycle 6 stays.

**The run**
- The owner's laptop: 6.7 minutes, tabicl 2.2.0 (now recorded by the runner
  itself), no context cap.
- Same training bundle (`daed58cb…`) and the locked-features file exported
  here (`028d92a2…`).
- [Certain] Its fold models are the development run's. Its development
  predictions match that run's exactly (largest difference in probability
  0.0).
- [Certain] Cycle 6's saved models reproduced its published shortlist to
  within 0.0005 before scoring.
- Scored once: `python scripts/lockbox_blend.py score lockbox-tabicl-20261126`,
  written to `docs/local-gpu/lockbox-blend.json`.

**Scores on the locked rows**

| | Cycle 6 | TabICLv2 | Blend |
| --- | --- | --- | --- |
| AUC, not cuprates (1,730 rows, 1,663 labelled): **the rule** | 0.820 | 0.823 | **0.827** |
| AUC, all 1,950 rows | 0.824 | 0.827 | 0.832 |
| Brier score, not cuprates (lower is better) | 0.127 | 0.129 | 0.125 |
| Tc error, log scale, not cuprates (lower is better) | 0.321 | 0.325 | 0.316 |
| Rank correlation of probability × Tc with true Tc, not cuprates | 0.742 | 0.750 | 0.755 |
| Share of the true highest-Tc tenth found in the model's own top tenth, not cuprates | 0.759 | 0.705 | 0.723 |

**Paired against cycle 6** (median, 95% interval, 300 resamples of whole
groups; reported, not deciding)

| | Not cuprates | All rows |
| --- | --- | --- |
| Blend AUC | +0.007 (−0.002 to +0.016) | +0.008 (+0.0002 to +0.017) |
| TabICLv2 AUC | +0.004 (−0.015 to +0.019) | +0.003 (−0.013 to +0.019) |
| Blend Tc error | −0.004 (−0.011 to +0.003) | −0.008 (−0.015 to −0.0002) |

**What it shows**
- [Certain] The direction seen on the development rows holds on rows that
  no one had tuned on. The blend gained +0.013 there and +0.007 here.
- [Likely] The gain is real but small: about +0.01 AUC. Here, the interval
  on all rows just clears zero, and the non-cuprate interval just includes
  it, as the plan expected for this many rows.
- [Certain] Every model scores higher here than on the development rows
  (cycle 6: 0.820 against 0.788). The locked rows seem easier. The
  comparison is paired, so this does not favour either model.
- **The one figure that went the other way:** the share of the true
  highest-Tc tenth that the blend puts in its own top tenth. It is −0.036
  against cycle 6.
  - This check was not planned. Its interval is −0.075 to +0.048, so it is
    within noise.
  - On the development rows the blend was slightly ahead on the same figure
    (0.702 against 0.698).
  - [Likely] It is noise, but it is the figure closest to what a shortlist
    uses, so it is worth watching.
- Ranked by probability alone, the top tenth is all superconductors for
  both the blend and cycle 6. That figure cannot tell them apart.

**Predictions against results**

| Outcome | Chance given | Result |
| --- | --- | --- |
| The blend's non-cuprate AUC is at least cycle 6's (the rule) | 80% | **Yes** (0.827 against 0.820) |
| The blend's paired interval lies above zero | 35% | No (−0.002 to +0.016) |
| TabICLv2 alone is at least cycle 6 | 60% | Yes (0.823 against 0.820) |
| Cycle 6's locked AUC is within ±0.03 of its development 0.788 | 75% | No (0.820, 0.032 above) |
| The blend's Tc error is lower than cycle 6's | 75% | Yes (0.316 against 0.321) |
| The laptop run takes under 10 minutes | 80% | Yes (6.7 minutes) |

- Four of six went the favoured way. The miss on cycle 6's own score was
  in the direction of easier locked rows, not of a weaker model.

**Correction to "Result, 2026-11-26" in
`docs/training-methods-review.md`** (labelled here, original kept): its
table row "Share of superconductors in the top tenth" is the same figure as
the last row of the scores table above. It is the share of the true
highest-Tc tenth that a model puts in its own top tenth by probability × Tc.

**What follows**
- The owner decides whether to accept the blend as the ranking of record.
- If accepted:
  - the made metals and shortlist 3 are re-ranked with the blend, from the
    predictions the laptop has already made;
  - any change to the Hazel queue is proposed separately;
  - each new blend ranking needs the laptop's GPU, about 25 minutes.
- The locked rows have now been used twice. They are not used again for
  this question.
