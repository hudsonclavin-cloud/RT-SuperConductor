# Learning cycles for the ranking of record

A learning cycle needs new evidence (`docs/active-learning.md`): retraining on
unchanged data only checks that the result reproduces. Here the new evidence is
**label corrections from the literature**: rows of the training table labelled
"not a superconductor" that published measurements show to be superconductors
(and, later, verified nulls). Each cycle adds a batch of corrections, retrains
the ranking of record (gradient boosting, `docs/scratchnet.md`), scores it, and
re-ranks the made metals.

## The owner's rule for corrections (decided 2026-11-16)

The owner chose this rule on 2026-11-16, in place of reviewing each row
before it enters training:

- A correction is applied only when two independent readers confirm it from
  the paper itself, with the DOI and the page, table or figure recorded.
- A third, sceptical reader then tries to refute it.
- Each correction is recorded in `data/training/negative_evidence.csv`, with
  the reviewer column saying it was applied under this rule.
- Each cycle's pull request lists every change, and the owner can veto any
  of them before it is merged.

What "confirm" means, fixed before the first batch was read:
- The paper's own text states superconductivity for that composition, within
  0.15 at.% on every element, in a bulk sample at ambient pressure, with a Tc.
- Or the composition lies inside a single-phase composition series whose
  measured samples bracket it within 2 at.% on both sides, all of them
  superconducting.
- A correction is recorded against a composition already in the table. The
  cycle script refuses a batch that adds, removes or moves a material between
  the development, locked and train-only rows.

The literature search uses the FastTrack Literature Open connector (OpenAlex
and Semantic Scholar, about 250 million records; abstracts, DOI checks against
Crossref and open-access PDF links) and arXiv full text.

## Procedure

```bash
python scripts/build_mp_candidates.py
```

```bash
python scripts/build_scratchnet_data.py --skip-elements
```

```bash
python scripts/run_learning_cycle.py --name cycle1 --previous cycle0
```

The cycle script (`scripts/run_learning_cycle.py`):
1. compares the table with the previous cycle's and lists every changed label;
2. runs grouped five-fold cross-validation of boosting on the development
   rows, with round 2's chosen settings;
3. scores the previous cycle's saved predictions against both answer keys,
   so a change splits into "the answer key changed" and "the model changed";
4. trains the final boosting models and ranks the made metals;
5. compares the order with the previous cycle's.

**The locked rows are neither trained on nor scored.** The locked test set
was used once (`docs/scratchnet-lockbox.json`) and stays that way. Two of the
first batch's corrections (W3Re2C and NbS) sit in it: they change the locked
labels on file, not any reported score.

Results: `docs/learning-cycles/<name>.json` and
`docs/learning-cycles/<name>-made-metals-top50.csv`. Predictions, models and
the full order are under `data/processed/learning-cycles/` and are not
committed. Each cycle takes about 2 minutes on four cores.

What a cycle does not do:
- It retrains boosting only. The network second opinion (`docs/scratchnet.md`)
  stays round 2's. Its saved models refuse a changed table, so
  `run_scratchnet.py rank --name round2` will not run after the first
  correction.
- It does not retune boosting. The settings stay round 2's, so a change comes
  from the labels and nothing else.

## Cycle 0: the baseline, 2026-11-16 02:30 UTC

Cycle 0 is the same procedure on the unchanged table. It reproduces round 2
exactly. [Certain]
- The out-of-fold boosting predictions are identical to
  `round2-oof.npz` on all 9,765 development rows (largest difference 0.0).
- The order of all 7,693 untested made metals is identical, position by
  position, to the one in `docs/bercx-published-check.md`. For example,
  Zr6Ga2Ni is 9th, Ti3SnH 511th and PtPb3Bi 221st.
- Cross-validated AUC on development rows: 0.790 (group-resampled interval
  0.769 to 0.810); non-cuprates 0.789.

## Cycle 1: plan and predictions, written before any cycle 1 result, 2026-11-16 02:40 UTC

**Batch:** the seven wrong negatives listed in `docs/negative-labels.md`
("Wrong negatives found in November 2026" and the blinded check):
- YRu3B2, LuRu3B2, ScS (development rows);
- W3Re2C and NbS (locked rows);
- WB (development, medium confidence);
- Ti3V22 (development; its known evidence is for V0.90Ti0.10, so it enters
  only if a measurement at or bracketing Ti 12 at.% is found).

Each goes through the rule above: two independent readers and a sceptical
verifier, run on 2026-11-16 before the table was rebuilt.

**Predictions** [Speculative]:

| Outcome | Chance |
| --- | --- |
| Cross-validated AUC on development rows changes by less than 0.005 | 85% |
| The paired difference between this model and cycle 0's, on today's labels, has an interval that includes zero | 85% |
| The made-metals top 50 keeps at least 45 of cycle 0's 50 | 85% |
| Zr6Ga2Ni stays in the top 20 | 90% |
| At least one made metal in a corrected chemistry (Ru–B, Sc–S, W–Re–C, W–B, Nb–S, V–Ti) moves up by 100 places or more | 50% |

Reasoning: five to seven corrected rows among about 9,400 labelled development
rows cannot move a pooled score much. Their effect should show near their own
chemistry.
