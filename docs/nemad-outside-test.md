# An outside test of the ranking of record on NEMAD compounds: plan, fixed before any look

**Main idea:** test the ranking of record (the 50/50 blend of cycle 6 and
TabICLv2) on 2,833 NEMAD compositions that neither model has seen. The
question is whether the blend beats cycle 6 alone outside this project's
own data, as it did on the development rows and on the locked rows.

Written 2026-11-27, before TabICLv2 has predicted these compounds. The owner
approved the test the same day.

## Amendments before the laptop run (2026-11-27)

A second review found the problems below. It came before the laptop run and
before any score. **This section overrides anything later in the plan that
disagrees with it.** The earlier text is kept and marked "superseded".

**1. Training rows are the labelled ones.**
- 366 bundle rows have no label, and neither model fits them.
- Counting only labelled rows as training:

| | Compositions | Superconductors | Not superconductors |
| --- | --- | --- | --- |
| Not cuprates, groups with no training row (the outside reading) | 1,037 | 767 | 270 |
| Not cuprates, groups with training rows | 967 | 761 | 206 |

- 1,601 of the 2,833 rows share a group with training rows (was 1,626).
- Still 1,245 groups.

**2. What was already known, in full.** The plan said nobody had seen
TabICLv2's or the blend's scores on these rows. That was wrong.
- **The made-metals list.**
  - 245 of the rows are on it: 232 non-cuprates, 131 of them in the outside
    slice.
  - The ranking of record's TabICLv2 run predicted that list. Those
    predictions are committed (`data/benchmarks/local-gpu/fm-tabicl-20261123/`).
  - Four of the rows appear with their probabilities in the committed blend
    top 50 (`docs/learning-cycles/blend6-made-metals-top50.csv`), and two in
    cycle 6's.
  - Nobody has compared any of these with NEMAD's labels.
- **Cycle 6's side of the outside reading was mostly known when the reading
  was added.**
  - Cycle 6's AUC on nearly the same two slices was already on file
    (`docs/learning-cycles/cycle6-on-nemad-rows.json`):
    - 0.777 on cycle 7's 859 "new group" non-cuprates;
    - 0.788 on its 967 non-cuprates that joined a development group.
  - Those rows cover 884 of the 1,037 outside rows and 942 of the 967 others.
  - The reading was added because a review asked for it, not because of
    those numbers. But it was not chosen blind.
  - The commit message that added it ("no result seen") overstates this.

**3. The 245 rows double as a check.**
- `score` reports how far the laptop's TabICLv2 is from the record run on
  those rows.
- "Reproduced" means the largest probability gap is at most 0.02.
- If the gap is larger, the result is still reported, flagged as not quite
  the record's TabICLv2.

**4. The outcome, in order; the first match wins.** This replaces the
reading table further down. `score` writes it as `reading.outcome`.

| | Result | What follows |
| --- | --- | --- |
| 1 | On either reading, the blend's paired interval lies entirely below zero | I propose going back to cycle 6 alone. The owner decides |
| 2 | On both readings, the blend's AUC is at least cycle 6's | Supported. Nothing changes |
| 3 | Main at least cycle 6's; outside lower, interval includes zero | Nothing changes. Recorded as a mark on the outside reading |
| 4 | Main lower, interval includes zero | Nothing changes. Recorded as a mark against the blend |

**5. Wording.**
- "From outside the project's data" overstated it. These are compositions
  new to the project, from NEMAD. About half are in families the models
  trained on, and 245 are on the project's made-metals list.
- "An independent review" means separate review agents run in this session,
  not reviewers outside it.
- "Those rows are more like 'new compositions in known families'" meant the
  rows that share a group with training rows, not the development and
  locked rows.
- "Intermetallic" is the family "Other": no cuprates, not iron-based, no
  oxygen.
  - 672 of its 1,505 rows (45%) contain B, C, N, P, S, Se, Te, H or a
    halogen.
  - It is now labelled "other families (mostly intermetallic)".

**6. Added predictions [Speculative].**

| Outcome | Chance |
| --- | --- |
| TabICLv2 reproduces the record run on the 245 rows (largest gap ≤ 0.02) | 85% |
| Outcome 2, supported | 45% |
| Outcome 3, mark on the outside reading | 20% |
| Outcome 4, mark against the blend | 25% |
| Outcome 1, propose going back | 10% |

## The rows

The usable NEMAD subset (`docs/nemad-superconductor.md`): records flagged
experimental, at normal or unstated pressure. Of those, only compositions
new to the project: not in the table (locked rows included), not in the
NIMS file, and not a near twin of either (`build_nemad_additions.select_new`).

| | Compositions | Superconductors | Not superconductors |
| --- | --- | --- | --- |
| All | 2,833 | 2,233 | 600 |
| Not cuprates (the main reading) | 2,004 | 1,528 | 476 |
| Intermetallic (superseded: other families, mostly intermetallic) | 1,505 | 1,264 | 241 |
| Not cuprates, groups with no training row (the outside reading; superseded: 1,037) | 1,012 | 754 | 258 |
| Not cuprates, groups with training rows (superseded: 967) | 992 | 774 | 218 |

**What the selection guarantees, and what it does not.**
- Guaranteed:
  - no row matches a composition either model trained on, or any locked row;
  - every row differs from each of those by more than 0.02 atomic fraction
    on at least one element. The closest is 0.021.
- Not guaranteed: that a row's group is new.
  - Group here means the table's rule: main elements, tied together through
    near twins, applied to the training and test compositions together.
  - By that rule, 1,626 of the 2,833 rows share a group with training rows.
    That includes 992 of the 2,004 non-cuprates. (Superseded: 1,601 and 967,
    counting only labelled rows as training.)
  - Development and locked rows never share a group with training rows.
    Those rows are more like "new compositions in known families", which is
    how the ranking is used on shortlists. (Superseded wording: "those rows"
    means the rows that share a group with training rows.)
- So the test has two readings, both fixed now: all non-cuprates, and the
  1,012 non-cuprates in groups with no training row. This was found by an
  independent review before the laptop run, and added then. (Superseded:
  1,037 rows; the review ran in this session; see amendment 2 on what was
  already known.)

- The 750 compositions cycle 7 left out are included here: 726 tied to
  locked groups and 24 tied to more than one fold. This test trains
  nothing, so ties do not matter.
- **What is already known.** Cycle 6 alone has been scored on the 2,083
  rows cycle 7 added: AUC 0.784 on their 1,826 non-cuprates
  (`docs/learning-cycles/cycle6-on-nemad-rows.json`). Its scores on the
  other 750 have not been looked at, and nobody has seen TabICLv2's or the
  blend's scores on any of the 2,833. (Superseded: wrong for 245 rows; see
  amendment 2.)
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

Superseded by amendment 4, which orders these rows and settles overlaps:

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
- all rows and the intermetallic rows (superseded: other families, mostly intermetallic);
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
| The blend is at least cycle 6 on the intermetallic rows (other families) | 60% |
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
