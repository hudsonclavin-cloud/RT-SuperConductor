# An outside test of the ranking of record on NEMAD compounds: plan, fixed before any look

**Main idea:** test the ranking of record (the 50/50 blend of cycle 6 and
TabICLv2) on 2,833 NEMAD compositions that neither model has seen. The
question is whether the blend beats cycle 6 alone outside this project's
own data, as it did on the development rows and on the locked rows.

Written 2026-11-27, before TabICLv2 has predicted these compounds. The owner
approved the test the same day.

## The rows

The usable NEMAD subset (`docs/nemad-superconductor.md`): records flagged
experimental, at normal or unstated pressure. Of those, only compositions
new to the project: not in the table (locked rows included), not in the
NIMS file, and not a near twin of either (`build_nemad_additions.select_new`).

| | Compositions | Superconductors | Not superconductors |
| --- | --- | --- | --- |
| All | 2,833 | 2,233 | 600 |
| Not cuprates (the main reading) | 2,004 | 1,528 | 476 |
| Intermetallic | 1,505 | 1,264 | 241 |
| Not cuprates, groups with no training row (the outside reading) | 1,012 | 754 | 258 |
| Not cuprates, groups with training rows | 992 | 774 | 218 |

**What the selection guarantees, and what it does not.**
- Guaranteed:
  - no row matches a composition either model trained on, or any locked row;
  - every row differs from each of those by more than 0.02 atomic fraction
    on at least one element. The closest is 0.021.
- Not guaranteed: that a row's group is new.
  - Group here means the table's rule: main elements, tied together through
    near twins, applied to the training and test compositions together.
  - By that rule, 1,626 of the 2,833 rows share a group with training rows.
    That includes 992 of the 2,004 non-cuprates.
  - Development and locked rows never share a group with training rows.
    Those rows are more like "new compositions in known families", which is
    how the ranking is used on shortlists.
- So the test has two readings, both fixed now: all non-cuprates, and the
  1,012 non-cuprates in groups with no training row. This was found by an
  independent review before the laptop run, and added then.

- The 750 compositions cycle 7 left out are included here: 726 tied to
  locked groups and 24 tied to more than one fold. This test trains
  nothing, so ties do not matter.
- **What is already known.** Cycle 6 alone has been scored on the 2,083
  rows cycle 7 added: AUC 0.784 on their 1,826 non-cuprates
  (`docs/learning-cycles/cycle6-on-nemad-rows.json`). Its scores on the
  other 750 have not been looked at, and nobody has seen TabICLv2's or the
  blend's scores on any of the 2,833.
- **Expected run time.** About 7 to 9 minutes on the laptop. The locked-row
  run, with 1,950 rows, took 6.66.
- **The labels are NEMAD's.** The paper check found about one wrong in ten
  for this subset.
  - Both models are scored against the same labels, so noise shrinks the
    differences between them.
  - [Speculative] It could favour one model if that model shares NEMAD's
    mistakes. There is no way to check that here.

## How each model predicts them

The same way as for the locked rows (`docs/lockbox-blend.md`):

| Model | Prediction |
| --- | --- |
| Cycle 6 | The mean of its five saved final models, applied here |
| TabICLv2 | The mean of its five fold models, run on the owner's laptop against the same bundle as before (`daed58cb…`) |
| Blend | 50/50: probabilities averaged, Tc averaged on the log scale |

- The laptop gets the features and composition keys only
  (`python scripts/nemad_outside_test.py export`).
- The labels are read here once, by
  `python scripts/nemad_outside_test.py score <folder>`, which refuses to
  run a second time.
- Intervals resample whole groups, formed by the table's rule over the
  training and test compositions together. Test rows chained through a
  training row therefore share a group: 1,245 groups for the 2,833 rows.
- `score` accepts only a TabICLv2 run that matches the ranking of record's:
  - the same bundle;
  - all five folds;
  - 8 estimators, no context limit, seed 0.

## Reading, fixed now

This is a check, not an adoption test: the blend is already the ranking of
record. There are two readings:
- **main:** all non-cuprates;
- **outside:** the non-cuprates in groups with no training row.

| Result | What follows |
| --- | --- |
| On the main reading, the blend's AUC is at least cycle 6's | The decision is supported by a third set of rows, from outside the project's data. Nothing changes |
| The blend's is lower, but its paired interval includes zero | Nothing changes. The result is recorded as a mark against the blend |
| On either reading, the blend's paired interval lies entirely below zero | I propose going back to cycle 6 alone. The owner decides |

- The outside reading is reported beside the main one.
- If the two point the opposite way, the write-up says so plainly. It does not
  pick the reading that favours the blend.

Also reported:
- TabICLv2 alone;
- all rows and the intermetallic rows;
- the Tc error.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| The blend's non-cuprate AUC is at least cycle 6's | 65% |
| The blend's paired non-cuprate interval lies above zero | 30% |
| The blend's paired non-cuprate interval lies entirely below zero | 7% |
| Outside reading: the blend's AUC is at least cycle 6's | 60% |
| On either reading, the paired interval lies entirely below zero | 10% |
| TabICLv2 alone is at least cycle 6 (non-cuprates) | 50% |
| The blend is at least cycle 6 on the intermetallic rows | 60% |
| The laptop run takes under 12 minutes | 80% |

## Commands

**Here, already done (2026-11-27):**

```bash
python scripts/nemad_outside_test.py export
```

That wrote `data/processed/local-gpu/nemad-features.npz`:
- 2,833 rows and 155 features, with no label;
- SHA-256 `1b8b4d6bea398e4d0704305b4db39e896eca0b88f4978a5368a3e45e9e2310a6`.

The file is sent to the owner in chat and is not committed.

**On the laptop** (Windows PowerShell, in `C:\Users\hudso\RT-SuperConductor`):

```powershell
git checkout claude/practical-tesla-ctf6oq
```
```powershell
git pull origin claude/practical-tesla-ctf6oq
```
```powershell
Move-Item $HOME\Downloads\nemad-features.npz data\processed\local-gpu\
```

This must show `1B8B4D6BEA398E4D0704305B4DB39E896ECA0B88F4978A5368A3E45E9E2310A6`:

```powershell
Get-FileHash data\processed\local-gpu\nemad-features.npz -Algorithm SHA256
```

This must list no `python.exe`:

```powershell
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```
```powershell
.venv\Scripts\python.exe scripts\foundation_gpu_train.py data\processed\local-gpu\bundle.npz --model tabicl --extra-candidates data\processed\local-gpu\nemad-features.npz --candidates nemad --out runs\nemad-tabicl --device cuda
```

Then send the predictions back:

```powershell
git checkout -b nemad-results-20261127
```
```powershell
Copy-Item runs\nemad-tabicl data\benchmarks\local-gpu\nemad-tabicl-20261127 -Recurse
```
```powershell
git add data/benchmarks/local-gpu/nemad-tabicl-20261127
```
```powershell
git commit -m "TabICLv2 predictions for the NEMAD outside test (features only)"
```
```powershell
git push -u origin nemad-results-20261127
```

**Here, once:**

```bash
python scripts/nemad_outside_test.py score nemad-tabicl-20261127
```
