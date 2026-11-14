# Published calculations as the fallback check (Bercx et al. 2025)

**Owner's decision, 2026-11-14:** if Stage 1 fails, candidates are judged
from Bercx et al.'s published electron-phonon calculations instead of this
project's EPW runs. This file records the source, what was taken from it,
and how it will be used. The usage rules below were written before any
candidate from this project was matched against the data.

## Source

- Paper: M. Bercx, S. Poncé, Y. Zhang et al., "Charting the landscape of
  Bardeen-Cooper-Schrieffer superconductors in experimentally known
  compounds", arXiv:2503.10943. The LaTeX source was fetched on 2026-11-14
  (gzip dated 2025-07-22); its supplementary section 7 has one file per
  material.
- Data: Materials Cloud Archive record 2025.39
  (doi:10.24435/materialscloud:n9-a7), CC BY 4.0. Tables A to D were
  downloaded on 2026-11-14. The 19 GB AiiDA archives were not downloaded.
- Their method: Quantum ESPRESSO and EPW (the same codes as this project),
  PBEsol, PseudoDojo norm-conserving pseudopotentials, Allen-Dynes and
  Migdal-Eliashberg Tc at μ* = 0.13. Compounds come from the Materials Cloud
  3D database of experimentally known structures (at most 12 atoms in the
  primitive cell).
- Neither the raw tables nor the parsed values are committed.
  `scripts/bercx_published.py` downloads and parses them into
  `data/raw/bercx-2025/`, which git ignores.

| File | Rows | sha256 |
| --- | --- | --- |
| Table_A.csv (unique experimental structures) | 10,195 | c497700e6295bd2267f204666fe0978fb6fe248f700a1f1ebea135306bc5eef2 |
| Table_B.csv (non-magnetic metals screened) | 4,533 | 1d643a42ae14fe76ff53248e0b26d7402ef02ed0a0318e784ec4b935b689f323 |
| Table_C.csv (coarse Allen-Dynes Tc above 1 K) | 1,599 | 4604d9753e3cb3eabd1987a3e79a35689fd820266fe8000f3efdac5628299856 |
| Table_D.csv (passed to the EPW stage) | 949 | e71a430645bc9aee72de2baf20b30a451f0a3c1f5120e7e465d054f211ea9e6b |

**The tables list only formula, source database, source id and space group.**
The calculated values for the final 250 materials (λ, ωlog, Allen-Dynes,
isotropic and anisotropic Eliashberg Tc, category, grids) were parsed from
the supplementary LaTeX. That gives 250 materials. The 82 tagged
`known-exp-structure` match the 82 the paper reports.

## What their niobium says about this project's benchmark

Bercx et al.'s own niobium entry, from the same codes, is:

| Setting or result | Value |
| --- | --- |
| Phonons | k 18³, smearing 272 meV (0.02 Ry), q 6³, cutoff 98 Ry |
| EPW fine grids | k and q 34³ |
| λ | 1.497 |
| ωlog | 12.56 meV = 145.7 K |
| Tc | Allen-Dynes 15.1 K, isotropic 19.7 K, anisotropic 21.2 K (μ* = 0.13); measured 9.25 K |

- [Certain] That λ is 20% above the 1.248 this project's pass rule uses as
  its reference (Cerqueira et al.). It is outside the ±15% band, while ωlog
  is inside it.
- [Likely] So an EPW calculation on grids close to the stop rule's can land
  above the band without anything being broken: the reference (ph.x's own
  coupling, 0.05 Ry double-delta smearing) and EPW differ in method.
- **Revised prediction for the niobium rerun** [Speculative], written before
  its result and labelled as a revision: λ inside 1.061 to 1.435 drops from
  45% (written 19:27 UTC) to 30%. The 45% stays on record.

## How the data will be used (written before matching)

**Matching.** A candidate matches a Bercx row when the reduced compositions
are equal and the space-group numbers are equal (the candidate's space group
comes from the Materials Project snapshot in `data/raw/mp-wide/`). Equal
composition with a different or missing space group is reported as
"formula only" and is not used for a verdict.

**Evidence tiers**, for structure matches only:

| Tier | Where the compound sits in Bercx et al. | Reading |
| --- | --- | --- |
| T4 | Among the 250, isotropic Tc of at least 5 K, category not `problematic` | Calculated superconductor. For matched known structures in that list, 74 of 82 really superconduct, but calculated Tc ran well above measured, so Tc is read as an upper bound. |
| T4-low | Among the 250, isotropic Tc below 5 K (including "<1") | Calculated, weak or no superconductivity |
| T3 | In Table D (passed to EPW), not among the 250 | Screened as promising, no final result (failed, unstable, magnetic or poor interpolation): no verdict |
| T2 | In Table C, not in Table D | Coarse Allen-Dynes Tc above 1 K, not selected |
| T1 | In Table B, not in Table C | Coarse Allen-Dynes Tc of 1 K or less: weak evidence against |
| T0 | Not in Table B | Outside their scope (magnetic, insulating in PBEsol, more than 12 atoms, or not in the database): no evidence |

**What is reported.**
- The tier of each of the 7,693 made metals in this project's boosting order,
  and the tier counts for the top 500 against the rest.
- **Test:** among structure-matched made metals, is the share in T4 higher in
  the top 500 than in the rest? Two-sided Fisher exact test.
  - Prediction [Speculative]: the top 500 has the higher share, odds ratio
    1.5 to 3; p below 0.05 at 50%.
- Nothing is relabelled, retrained or added to training. Calculated results
  are not experimental outcomes.

## Limits known before matching

- Bercx et al. relaxed with PBEsol and screened only non-magnetic compounds
  with at most 12 atoms, so many made metals will be T0.
- Their selection into Table C, Table D and the 250 is by their own coarse
  Allen-Dynes Tc, so T1 to T3 are graded by a coarse calculation.
- Tables A to D give one space group per entry. The same formula in another
  structure is a different material.

## Results, 2026-11-14 20:25 UTC

Produced by `python scripts/bercx_published.py join`; full table in
`docs/bercx-made-metals-join.csv` (matched rows only), counts in
`docs/bercx-made-metals-summary.json`. Ranks are positions in the uncapped
made-metals order (7,693 compounds, by boosting probability); the committed
top-200 file caps compounds per element system, so its row numbers differ.

**Coverage.** 1,598 of the 7,693 made metals (21%) match a Bercx et al.
structure (formula and space group). 6,095 are T0, outside their scope; 58
of those share a formula with a Bercx entry but not its space group.

| Tier | Top 500 | Ranks 501 to 7,693 |
| --- | --- | --- |
| T4 (calculated superconductor, isotropic Tc ≥ 5 K) | 3 | 11 |
| T4-low (calculated, isotropic Tc < 5 K) | 5 | 13 |
| T3 (sent to EPW, no final result) | 13 | 76 |
| T2 (coarse Tc > 1 K, not selected) | 36 | 130 |
| T1 (coarse Tc ≤ 1 K) | 132 | 1,179 |
| T0 (not screened) | 311 | 5,784 |

One detail the rules above did not spell out: entries Bercx et al. tag
`problematic` with an isotropic Tc of 5 K or more are counted as T3 (no
verdict), not T4.

**The pre-registered test.** Among matched compounds, 3 of 189 in the top
500 are T4 (1.6%) against 11 of 1,409 in the rest (0.8%): odds ratio 2.05,
two-sided Fisher p = 0.22. [Certain] Not significant. Against the
prediction: the direction and the odds-ratio range were right; "p below
0.05" (given 50%) was wrong. With 14 T4 compounds in all, even a real
doubling could not be shown.

**Exploratory, not pre-registered.** Counting every matched compound whose
coarse Allen-Dynes Tc exceeded 1 K (T2 to T4 and T4-low): 57 of 189 in the
top 500 (30%) against 230 of 1,409 in the rest (16%), odds ratio 2.2,
p = 1×10⁻⁵. [Likely] So the ranking and a coarse physics calculation agree
more than chance, on the 21% of made metals that both cover. It was chosen
after seeing the tier table, so treat it as a lead.

**Calculated superconductors among the made metals that Bercx et al. list
as never measured** (category `unknown`; their literature survey, 2025, not
re-checked here):

| Made-metals rank | Compound | Space group | λ | Isotropic Tc (K) | Anisotropic Tc (K) |
| --- | --- | --- | --- | --- | --- |
| 9 | Zr6Ga2Ni | 189 | 0.70 | 5.5 | 5.8 |
| 511 | Ti3SnH | 221 | 1.18 | 14.6 | 18.7 |
| 1,002 | ZrSbRu (their half-Heusler highlight) | 216 | 0.96 | 11.3 | 13.2 |
| 1,055 | TiAlPt | 194 | 0.75 | 6.2 | 5.3 |
| 1,658 | Y2B3C2 | 65 | 0.53 | 5.6 | 7.7 |
| 2,028 | ZrS | 129 | 1.13 | 13.4 | 18.8 |
| 2,105 | NbCoSb | 216 | 1.73 | 20.4 | 21.4 |
| 5,992 | Ti2Ga | 194 | 0.69 | 7.1 | 8.3 |

- **Calculated Tc runs high.** For the matched known superconductors, the
  isotropic Tc is 1.1 to 2.8 times the measured value: LiGa2Ir 5.6 against
  2.9 K, LiGa2Rh 6.8 against 2.4 K, YSn3 7.7 against 7.0 K, LiGePd2 2.8
  against 2.0 K, TaH2 8.3 against 3.0 K. LiZrSe2 is about 6 times (5.0
  against 0.5 to 1.05 K). Read these Tc values as upper bounds.
- **False positives exist.** TaB2 and RuO2 are T4 but are not
  superconductors in experiment. This project's ranking puts them at 3,242
  and 5,761.
- **Zr6Ga2Ni** is the only compound that both this ranking (top 10) and the
  published calculation call a superconductor without a measurement. Its
  calculated Tc (5.5 K isotropic) is low enough that a real Tc near 2 to
  3 K is plausible [Speculative, from the ratios above]. Its relatives
  Zr6Ga2Co (rank 11) and Hf6Ga2Co (rank 98) are calculated at 4.7 and 3.0 K
  (T4-low).

Nothing was relabelled, retrained or added to training.
