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

## Literature check of Zr6Ga2Ni: decided 2026-11-14 20:40 UTC, prediction before the result

**Owner's decision:** Zr6Ga2Ni goes to a literature check, the first of the
eight never-measured calculated superconductors.

**Method.** Three readers search independently, each by a different route:
- scholarly databases, for the synthesis paper behind ICSD 635001 / MPDS
  S1613081;
- the general web and arXiv;
- the Zr6MX2 family, for context.

Each gives a verdict for Zr6Ga2Ni itself from the fixed list in
`data/benchmarks/literature-check/PROVENANCE.md`. A sceptical second reader
then tries to open every cited source and gives the final verdict. Bercx et
al.'s calculation does not count as evidence.

**Prediction** [Speculative], written before any reader returned:
- `no_low_temperature_data_found`: 70%;
- a measured superconductor (ambient, or pressure/doping only): 15%;
- `measured_no_superconductivity`: 10%;
- magnetic or insulating: 5%.

## Corrections after an independent review (2026-11-14, committed with the fixes)

Three independent reviewers looked at this file, the script and the join:
- one recomputed everything from the raw inputs, including an independent
  re-parse of all 250 pages;
- one reviewed the code;
- one argued against the interpretation.

Every committed number reproduced. The sections above are kept as committed;
these notes take precedence over them.

**Timestamps.** Three section headings give clock times later than the
commits that contain them. The commit order itself is intact:

| Heading said | Committed in | Commit time (UTC) |
| --- | --- | --- |
| runbook "Fallback decided … 20:20 UTC" | 36a67a5 | 20:14:05 |
| "Results, 2026-11-14 20:25 UTC" | 9a50e2d | 20:20:03 |
| "Literature check … decided 2026-11-14 20:40 UTC" | 6657bb2 | 20:23:35 |

The commit times are the ones to trust. The literature-check prediction
was committed before any reader returned.

**Code fixes. None changes a committed number;** the join table and the
summary are byte-identical after the fixes.
- **The same structure can appear more than once among the 250.** In/139
  appears twice, and TaS2/194 three times (one copy `problematic`). Records
  are now kept as lists, and a compound whose records disagree on the tier
  gets no verdict (T3).
- **Space group.** 13 supplementary pages print a lower symmetry than Table
  D gives for the same structure (for example V 139 against 229, AgB2 1
  against 191). Each record is now also filed under Table D's space group.
- **No verdict for any `problematic` entry**, whatever its Tc. This makes
  the code match its comment; the only affected entry, ScAs, matches no made
  metal.
- **Smaller fixes:**
  - source ids containing digits (`mc3d-…`) now parse;
  - the experimental-Tc field is matched directly and its whitespace
    normalised;
  - the arXiv source is pinned to v2 and checked against sha256
    e2355f7a79b984c819b52cdb22e3ec50f7a6f308fccb0e1c8b0e5abc5a7243f0;
  - the parse must give exactly 250 materials.
- Tests were added for each of these, and for the made-metals order against
  the committed top-200 file.

**Wording.**
- **"58 share a formula with a Bercx entry but not its space group":** that
  means a Table B or supplementary entry. A further **446 made metals (58 in
  the top 500) match a Table A structure exactly** but were screened out
  before Table B, as magnetic or non-metallic in PBEsol.
- **"The committed top-200 file caps compounds per element system, so its
  row numbers differ" is wrong.** The cap never applies within the first
  200. Row numbers agree with that file for every compound it holds; a test
  now checks this.
- **LiZrSe2 is 5 to 10 times** its measured range, not "about 6".
- **The exploratory agreement is entirely at Bercx et al.'s coarse first
  screen.** Split step by step among matched made metals, top 500 against
  the rest:

  | Step | Top 500 | Rest | Odds ratio | p |
  | --- | --- | --- | --- | --- |
  | Table B to C (coarse Tc above 1 K) | 57/189 | 230/1,409 | 2.2 | 1×10⁻⁵ |
  | C to D (sent to EPW) | 21/57 | 100/230 | 0.76 | 0.45 |
  | D to the 250 | 9/21 | 24/100 | 2.4 | 0.10 |
  | The 250 to T4 | 3/9 | 11/24 | 0.59 | 0.70 |

  So once a compound passes the coarse screen, a top-500 rank does not
  predict that it passes the careful steps, and the T4 verdict the fallback
  uses is the pre-registered null. The agreement is also weak in size: the
  boosting probability's AUC for the coarse step is 0.60, against 0.64 for
  element presence alone. It does survive controls for element presence,
  element count and cell size (permutation p = 0.005; Mantel-Haenszel odds
  ratio 2.4 by element count), so it is not simply shared chemistry.
- **"Zr6Ga2Ni is the only compound that both … call a superconductor"
  holds only under the tier rules** (5 K isotropic cut, `problematic`
  excluded):
  - KZrS2, at rank 60, has isotropic Tc 10.2 K and is `problematic` only
    because its source paper was not found;
  - Ti3SnH (14.6 K) sits at rank 511, just outside the top 500;
  - Zr6Ga2Ni (5.54 K) and Zr6Ga2Co (4.68 K) are on either side of the 5 K
    cut by less than 1 K.
- **Bercx et al.'s +20% on niobium need not be the method.** Their niobium
  phonons used k 18³ at 0.02 Ry, the regime this project found unconverged
  for niobium's soft modes. So the +20% may come from the method (EPW against
  ph.x's double-delta at 0.05 Ry) or from their phonon grid. The [Likely]
  label above is lowered to [Speculative].
- **The revised niobium prediction left the set incoherent.** It cut λ
  inside the band to 30% but kept "λ and ωlog both inside: 40%" and "full
  pass about 14%". The revised set, labelled as a revision and still written
  before the result:
  - λ inside: 30%;
  - λ above 1.435: 60%;
  - λ below 1.061: 10%;
  - λ and ωlog both inside: 27%;
  - full pass: 9%.

## Literature check of Zr6Ga2Ni: result

**Verdict: `no_low_temperature_data_found`, medium confidence, "not found in
the time allowed" (the search was not exhaustive).**
- All three readers gave this verdict independently.
- The second reader re-opened every cited source and rejected every claim it
  could not confirm, including the idea that Bercx et al.'s `unknown`
  category is evidence either way. Their paper defines it as "no other
  results are found connecting this material to superconductivity".
- The prediction gave this verdict 70%.

**What was found.**
- [Certain, full text] Superconductors measured in Zr6Ga2Ni's family
  (Zr6CoAl2 type, P-62m), all tellurides, antimonides or bismuthides:

  | Compound | Tc | Source |
  | --- | --- | --- |
  | Sc6MTe2 (seven M) | up to 4.7 K | Shinoda et al., J. Phys. Soc. Jpn. 92, 103701 (2023) |
  | Zr6FeTe2 | 0.76 K | H. Matsumoto et al., 93, 023705 (2024) |
  | Zr6CoTe2 | 0.13 K | same paper |
  | Zr6FeSb2 | 1.3 K | R. Matsumoto et al., 93, 065001 (2024); abstract only |
  | Zr6RuBi2 | 4.9 K | Yuchi et al., 94, 013701 (2025) |
  | Zr6FeBi2 | 1.4 K | same paper |
  | Zr6RuTe2 | 1.1 K | Yuchi et al., 94, 085001 (2025) |

  Yuchi et al. (2025) write that over 100 members have been made and only
  about ten are reported superconducting.
- No measurement on any Al, Ga, In or Sn member was found.
- The Ga members are known from hydrogen-storage and structure work:
  - Zavaliy, Pecharsky, Miller, Akselrud, J. Alloys Compd. 283, 106 (1999);
    abstract only;
  - Zavalii, Materials Science 34, 528 (1998), whose title includes
    "magnetic properties" for Me = Fe, Co, Ni and X = Al, Ga, Sn. This could
    not be opened (link.springer.com is blocked here).
- **Do not confuse** Zr6NiGa2 with equiatomic ZrNiGa (Tc 0.42 K) or Heusler
  ZrNi2Ga (Tc 2.9 K), which are different compounds.

**What a person with library access should read to settle it:**
1. Zavalii 1998 (doi:10.1007/BF02360705): is Zr6NiGa2 among the compounds,
   and how low do its magnetic measurements go? A susceptibility curve to
   about 2 K would decide between "no data" and "measured, no
   superconductivity".
2. Zavaliy et al. 1999 (doi:10.1016/S0925-8388(98)00877-9), full text.
3. The synthesis paper behind ICSD 635001 / MPDS S1613081, and Markiv and
   Storozhenko 1974 on the Zr-Ni-Ga system.
4. NIMS SuperCon, for Zr6NiGa2.

**Reading.** [Speculative] Zr6Ga2Ni looks like a reasonable sample for an
experimental partner:
- it has been made;
- it is apparently unmeasured at low temperature;
- its family superconducts at 0.1 to 5 K;
- the published calculation gives 5.5 K, which past ratios suggest means a
  real Tc of about 2 to 3 K if it superconducts.

But no family member with a group-13 element has been reported as a
superconductor, and the calculation sits only 0.5 K above the cut. Nothing
was relabelled.
