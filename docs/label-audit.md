# Label audit: a scanner for duplicates and labels that look wrong

**Main idea:** one command, also run weekly by a GitHub Action, lists the
training rows whose label is most likely wrong. It decides nothing and
changes no label. A person checks each lead under the owner's correction
rule. It exists because label errors cap the AUC any model can reach: with
10% of labels wrong on both sides, even a perfect model scores about 0.90.

Written 2026-11-27, at the owner's request.

## How to run it

**Here or on the laptop:**

```bash
python scripts/label_audit.py
```

It writes `data/processed/label-audit/`, which is not committed:
- `label-audit.md`: the summary;
- `flags.csv`: every flag;
- `label-audit.json`: everything.

It takes about 6 seconds. `--include-reviewed` also lists compositions that
have already been triaged.

**On GitHub:** `.github/workflows/label-audit.yml`.
- **When it runs:** every Monday, on demand from the Actions tab, and on any
  push that changes `data/training/` or the scanner.
- **What it does:**
  1. rebuilds the table from the pinned sources;
  2. refuses to go on unless the table's checksum matches
     `docs/scratchnet-data.json`;
  3. writes the report to the run's summary page, and keeps the full lists
     as an artifact for 30 days.
- **The issue:** on the weekly and on-demand runs it also opens one issue,
  "Label audit: leads to check", or updates it if it is open. Runs on push
  never touch the issue.
- **Main branch only, for some of it.** GitHub runs scheduled workflows, and
  offers the on-demand button, only for workflow files on the default
  branch. Those two start once this file is merged into `main`.

## What it flags

It scans only development and train-only rows. Locked rows are dropped
before any label is read, because their labels are kept for one-time
scoring. A near twin is always in the same group, and groups are locked
whole, so no scanned row has a locked twin.

| Flag | Signals, one point each |
| --- | --- |
| **Probable duplicate** | Same elements, at most 0.005 apart on every element, opposite labels |
| **Possible false negative** (label 0) | A near twin with the same elements is a superconductor<br>Another source reports it superconducting: the NIMS SuperCon file, NEMAD's usable subset, or SuperCon2 (text-mined, the weakest)<br>The ranking of record gives it at least 0.9 out of fold |
| **Possible false positive** (label 1) | A near twin with the same elements is a null<br>NIMS or NEMAD hold only nulls for it<br>The ranking of record gives it at most 0.1 out of fold<br>Its Tc is above its family's ambient-pressure record (cuprates 140 K, iron-based 60 K, other oxides 40 K, everything else 40 K) |

- **Twin conflicts are deliberately narrowed.** They count only between rows
  with the same elements, and never for cuprates.
  - A dopant, often a magnetic one, can really switch superconductivity off.
  - A cuprate's doping series really crosses from insulator to
    superconductor.
  - In the first version these accounted for 823 of 831 false-negative
    flags, nearly all of them physics rather than errors.
- **Each kind of signal counts once.** Many disagreeing twins are still one
  signal.
- **Reviewed compositions are counted, not listed.** That means anything in:
  - the correction records (`data/training/negative_evidence.csv`);
  - the cycle evidence and triage files under
    `data/benchmarks/learning-cycles/`;
  - the review queues;
  - **`data/training/label-audit-reviewed.csv`**: add a checked composition
    here to stop it being listed again.
- **Sources that are missing are skipped and named.** On GitHub that is NIMS
  and NEMAD: one is a manual download and the other needs an API key.
  - So that GitHub sees the same evidence, their labels for the scanned
    compositions only are committed in
    `data/training/label-audit-external.csv.gz` (108 KB, 15,710 rows). The
    scanner uses that file when the raw files are absent.
  - The file holds no locked composition, and a test checks that.
  - Refresh it with `python scripts/label_audit.py --write-extract` after the
    table or either source changes.
  - Both sources are CC BY 4.0: NIMS SuperCon (doi:10.48505/nims.4487) and
    NEMAD (Itani et al., *J. Magn. Magn. Mater.* 652, 174189, 2026).
  - The first GitHub run, before the extract, found 5 possible false
    negatives against 130 here; with the extract the counts are the same as
    here.
- **The ranking of record's out-of-fold scores** come from two committed
  files: `data/benchmarks/learning-cycles/cycle6-oof.npz` and
  `data/benchmarks/local-gpu/fm-tabicl-20261123/oof.npz`. They cover
  development rows only.

## First run, 2026-11-27 (all sources available here)

- **Rows scanned:** 13,644.
- **Probable duplicates with opposite labels:** 102.
- **Possible false negatives:** 130.
  - By number of signals: 5 with three, 25 with two, 100 with one.
  - 502 more were already triaged.
- **Possible false positives:** 644.
  - By number of signals: 2 with three, 8 with two, 634 with one.
  - 17 more were already reviewed.

**What the top leads look like:**
- **Famous borderline cuprates** among the false negatives: Nd2CuO4,
  La2CuO4 and PrBa2Cu3O7. They superconduct only when doped, with extra
  oxygen, or in disputed single-crystal reports.
  - So the right label depends on oxygen content the formula may not
    carry. These need a reader's judgment, not a flip.
- **Text-mining errors on the other side:** polonium and potassium, which
  SuperCon2 or NEMAD call superconducting at ambient pressure.
  - [Likely] Both are wrong. This is why a flag is only a lead.
- **A probable real error:** C0.938Rb0.062 labelled a 29 K superconductor.
  - That ratio is Rb4C60, an insulator. The 29 K superconductor is Rb3C60
    (C0.952Rb0.048).
  - [Likely] The label belongs to a different compound.
- Most one-signal false positives are a superconductor next to a null twin
  that was triaged and kept. They are the lowest priority.

**Using it:**
1. Work down the list from the most signals.
2. Apply the owner's rule (two independent readers and a sceptical
   verifier).
3. Record confirmed corrections in `data/training/negative_evidence.csv`,
   and checked-but-unchanged rows in
   `data/training/label-audit-reviewed.csv`.
4. Corrections then enter training through a learning cycle, judged as
   usual.
