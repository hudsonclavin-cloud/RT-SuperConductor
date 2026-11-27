# The NEMAD superconductor database: what is new, and how much of it is right

**Main idea:** NEMAD holds about 2,900 compositions this project does not
have that look usable. They are the records that the database's own
"experimental" flag marks as measured, at normal or unstated pressure. In a
check against the source papers, that flag separated good records from bad
ones almost perfectly. Records without it were never right.

Written 2026-11-27. No training table was changed.

## The download

- **Source:** Itani, Zhang, Itani and Zang, "Large language models for
  superconductor discovery", *J. Magn. Magn. Mater.* 652, 174189 (2026);
  https://nemad.org.
  - The site describes the data as CC BY 4.0.
  - Every record was extracted from papers by language models.
- **How:** with the owner's API key, one request per element.
  - Command: `python scripts/nemad_download.py fetch`, then `table`.
  - The records stay under `data/raw/nemad/` and `data/processed/nemad/`,
    which git ignores.

| Step | Count |
| --- | --- |
| Records the site reports | 78,203 |
| Records returned, duplicates removed | 76,700 |
| Names that read as a formula | 52,360 |
| Different compositions among them | 13,392 |
| Labelled superconductor / not / no label | 46,893 / 5,091 / 376 records |

**Names that could not be used (24,340).** They are left out, not guessed:
- words, such as "Yttrium Barium Copper Oxide";
- unknown doping, such as "BaFe2-xNixAs2";
- nicknames, such as "Bi2212" or "Cu1234";
- notes that change the compound, such as "(K-doped)".

Subscripts, δ for small oxygen off-stoichiometry, and a doping value given
in the name, such as "(x=0.15)", are read (`scripts/nemad_download.py`,
`clean_name`).

## What is new to this project

Command: `python scripts/nemad_report.py`, written to
`docs/nemad-superconductor-check.json`. "Held" means our table, including
its locked rows, or the NIMS file. "Near twin" means within 0.02 atomic
fraction of a held composition on every element.

| Filter | New compositions | Superconductor | Not | Intermetallic superconductor | Intermetallic not | Near twins |
| --- | --- | --- | --- | --- | --- | --- |
| All records | 4,466 | 3,616 | 808 | 2,203 | 339 | 3,957 |
| Superconductor reports at known high pressure dropped | 3,943 | 3,073 | 828 | 1,770 | 355 | 3,733 |
| **Normal or unstated pressure, and flagged experimental** | **2,861** | **2,240** | **600** | **1,264** | **241** | 3,344 |

**Label agreement where we hold a label we may read** (never a locked
row's):

| Our label | NEMAD agrees | NEMAD differs |
| --- | --- | --- |
| Superconductor | 3,501 | 46 |
| Null | 342 | 310 |

- [Certain] NEMAD agrees with our superconductors almost always (99%), and
  with our nulls only half the time.
- The paper check below says the nulls disagree mostly because NEMAD is
  wrong. Some of our nulls are wrong too.

## The paper check: 40 records against their papers

A fixed random sample was drawn (seed 20261127):

| Group | Records | What it tests |
| --- | --- | --- |
| A | 15 | New superconductors in intermetallic chemistry, the kind our candidate lists come from |
| B | 10 | New non-superconductors |
| C | 15 | Compositions NEMAD calls superconductors where our table says null |

**How it was checked**
- Eight reader agents each took five records. They found each paper by its
  DOI and read the full text where they could reach it, otherwise the
  abstract.
- A skeptic agent then tried to overturn each of the 20 "wrong" verdicts.
  It overturned none:
  - 18 were upheld from the paper;
  - 2 were upheld only on the first reading, because the full text was
    paywalled.
- Every verdict, with its evidence: `docs/nemad-paper-check.json`.

| Group | NEMAD right | Wrong | Cannot tell |
| --- | --- | --- | --- |
| A: new intermetallic superconductors | 4 | 7 | 4 |
| B: new non-superconductors | 3 | 1 | 6 |
| C: NEMAD superconductor, our null | 3 | 9 | 3 |
| **By NEMAD's experimental flag: flagged** | **10** | **2** | 8 |
| **By NEMAD's experimental flag: not flagged** | **0** | **15** | 5 |

**How the wrong records go wrong**
- **A value cited from another paper**, not studied there (6). For
  example, LaSH6 at 35 K is quoted from a 300 GPa prediction.
- **A calculated Tc** (4). Ta3Bi, Li3Be and LaIr2B2C carry DFT estimates.
- **The superconductor is a doped or layered version of the formula** (4).
  For example, CeOFeAs and TbOFeAs are parents whose fluorine-doped forms
  superconduct, and CaCuO2's 60 K was measured in a heterostructure.
- **Superconducting only under pressure** (1): (Ge,Sn,Pb)Te at 30 GPa.

**What it shows**
- [Likely] The experimental flag, together with dropping known high
  pressure, is a usable filter.
  - Records that pass it were right 10 times out of 11 where the paper
    could be checked. The 95% interval is roughly 62% to 98%: few records.
  - Records without the flag were never right where the paper could be
    checked (0 of 15).
- [Likely] Where NEMAD calls one of our nulls a superconductor, our null
  usually holds: 5 confirmed, against 3 found wrong.
- **Three of our nulls are wrong by their papers.** They are candidates for
  correction under the learning-cycle rule. Nothing has been changed:

| Composition | Where in our table | Paper |
| --- | --- | --- |
| Nb3Ge | development row, labelled null | The classic A15 superconductor; this paper (Testardi et al. 1967) calls it one |
| YBa2Cu3O6.32 | train-only row, labelled null | Measured superconducting at the edge of the ortho-II phase (cond-mat/0003077) |
| SmFeAs0.47P0.53O0.85H0.15 | development row, labelled null | Tc 4 K from the paper's own resistivity (Matsuishi et al. 2014) |

- **Limits**
  - 40 records, 15 of them undecided, mostly because the full text was
    paywalled.
  - The flag's error rate is estimated from 11 decided records.

## What it is worth to the model [Speculative]

- The usable subset adds about 2,840 labelled compositions to the 11,580 per
  fold that cycle 6 trains on, a quarter more. That is about 0.32 of a
  doubling.
  - By the learning curve (`docs/learning-curve.md`), that is worth about
    +0.01 AUC.
  - The 3,344 near twins could add a little more.
  - A label error rate near 10% takes some of it back.
- **Estimated gain: +0.005 to +0.015 non-cuprate AUC.** It is the largest
  addition so far, but still below what this project's intervals (about
  ±0.007 on the development rows) can reliably confirm in one test.
- **The 600 new non-superconductor compositions may matter more than the
  count suggests.** Measured nulls are what this project has been short of.
  - Of the 7 sampled with the flag, 3 were confirmed as measured nulls and
    none was wrong.

## What would come next (the owner decides)

A planned learning cycle (cycle 7), under the same guards as cycle 4. Its
plan and predictions would be written before any model is trained:
- add the usable subset's new compositions to cycle 6's training rows, by
  cycle 4's placement rule (never into a held-out fold or a locked group);
- judge it on the development rows against cycle 6, with placebos;
- if it passes, the owner's laptop reruns TabICLv2 on the extended bundle,
  so that the blend can be rebuilt;
- the three null corrections above go in the same cycle, or in their own,
  as the owner prefers.

## Outcome of cycle 7, 2026-11-27

Cycle 7 added the usable subset to training and failed its rule: the
model improved on NEMAD's own compounds (+0.08 AUC) and got slightly worse
on the development rows (−0.007). It was not adopted. Details:
`docs/learning-cycles.md`, "Cycle 7: result".

**Correction to "Three of our nulls are wrong by their papers" above**
(labelled here, original kept). That was one reader's finding. Under the
owner's rule, with two independent readers and a sceptical verifier, only
one of the three stands:
- YBa2Cu3O6.32 was corrected to a superconductor.
- Nb3Ge at exactly 3:1 superconducts only as a thin film; bulk samples are
  germanium-poor. The null stays.
- For the samarium compound the paper contradicts itself. The null stays.

**The "worth to the model" estimate above did not hold.** It assumed rows
from NEMAD behave like more rows from our own table. They do not.
