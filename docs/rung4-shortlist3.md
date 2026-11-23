# Rung 4, shortlist 3: the same filters, the cycle 6 model, verdicts carried forward

**Main idea:** shortlist 2 was ranked with cycle 3. Cycles 4 to 6 changed the
ranking of record (851 NIMS rows, then 16 label corrections). This shortlist
re-ranks the same filtered pool with cycle 6. It also applies the two fixes
shortlist 2 asked for, so the top 10 holds new names, not repeats.

Plan written 2026-11-22, before the pool was ranked with cycle 6.

## Steps

| Step | What | Why |
| --- | --- | --- |
| 1 to 6 | Shortlist 2's filters, unchanged: on the hull; no radioactive or open-4f element; no Be, Cd, Hg, Tl, As or Os; pseudopotentials available; formation energy −0.10 eV per atom or lower; 12 atoms or fewer. The 571 survivors are reused (`data/benchmarks/rung4/shortlist2-filtered.csv`). | So the only differences from shortlist 2 are the model and the two steps below. |
| 8 (new) | Drop every compound that an earlier reader or calculation has already given a verdict: the 50 literature-checked in shortlists 1 and 2, and the 5 pre-checked for Hazel. | Shortlist 2's own lesson: half its top 10 were repeats. Their cycle 6 places are reported, but they do not take a slot. |
| 9 (new) | Drop element systems that readers reported as solid solutions or alloys rather than compounds: Ir-Re, Re-Rh, Os-Ru and Os-Re. | Shortlist 2's second lesson. Os is already excluded by step 3; the rule is written out so it is explicit. |
| Rank | The five final models of cycle 6 (`data/processed/learning-cycles/models/cycle6/`), averaged, as in shortlist 2. | The ranking of record since PR #72. |
| Cap | Top 10, at most 2 per structure prototype and 2 per element system. | As in shortlist 2. |
| Read | One reader per compound, titles and abstracts plus their own searches: made or not, any low-temperature measurement, any superconductivity. Verdicts as in `literature-check2.csv`. | As in shortlist 2. |

Not run: the force-field stability screen. It failed its validation in
shortlist 2 (28% of made compounds flagged), so it would only add a column
that does not tell stable from unstable.

Not in this step: the Quantum ESPRESSO pre-check. Whichever never-made
compounds survive the reading are proposed for it, to run on the owner's
machine or here.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| Rank correlation between the cycle 3 and cycle 6 orders of the 571 is 0.85 or higher | 70% |
| At least 4 of the new top 10 turn out to have been made already | 55% |
| All four Hazel candidates (NaMgPb2, LiMgPb2, SbTe2, CsIr2P2) stay in cycle 6's top 50 of the 571 | 45% |
| At least 1 of the new top 10 is never made and has 6 atoms or fewer | 60% |

## Risks

- **Biggest risk:** after removing 55 checked compounds and the
  solid-solution systems, what is left ranks lower and is duller.
- **Most likely failure:** most of the new top 10 have already been made.
  Shortlist 2 lost 3 to 4 of 10 that way.
- **What would change the plan:** a real lab's list of what it can make
  replaces step 3.

## Results, 2026-11-22

**Main idea:** the new top 10 holds no repeats, as planned. But 7 of the
10 had already been made. Two never-made compounds with 6 atoms or fewer
survive: LiGaIr and SrAl3Ag. They are proposed for the Quantum ESPRESSO
pre-check. Three made compounds have never been measured at low
temperature: V2GaN, SrIn3Au and SrNiSn2. Those are cheap measurements
for a lab, not calculations.

Files:
- `data/benchmarks/rung4/shortlist3.csv`
- `data/benchmarks/rung4/shortlist3-ranked.csv` (all 571)
- `data/benchmarks/rung4/literature-check3.csv`
- `docs/rung4/shortlist3-summary.json`
- Script: `scripts/rung4_shortlist3.py`

### Steps

| Step | Compounds left |
| --- | --- |
| Shortlist 2's filters (steps 1 to 6) | 571 |
| 8: verdicts carried forward | 552. Only 19 of the 55 already-checked compounds were in the filtered pool; the rest had been dropped by the filters |
| 9: solid-solution systems dropped | 551 (one Ir-Re entry) |
| Cap: 2 per prototype, 2 per element system | top 10 below |

The caps did work. The pool is crowded with Heuslers
(0.5-0.25-0.25 | Fm-3m) and with one hexagonal type
(0.5-0.25-0.25 | P63/mmc). Only 2 of each got through.

### The top 10 and what the readers found

| Rank of 571 | Compound | Atoms | p (cycle 6) | Rank under cycle 3 | Verdict | Source |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | TaAlRu2 (Heusler) | 4 | 0.938 | 13 | made, measured, not a superconductor (a semimetal; how low the measurement went is not stated in the abstract) | 10.1103/physrevb.96.125106 |
| 6 | V2GaN (MAX phase) | 8 | 0.934 | 25 | made, no low-temperature data | 10.1021/acs.inorgchem.2c00200 |
| 9 | VGaRu2 (Heusler) | 4 | 0.929 | 9 | made, measured, not a superconductor (read from the title and keywords) | 10.1016/j.intermet.2023.107958 |
| 33 | VAuS2 | 8 | 0.893 | 38 | made at 3 GPa; measured to 4.2 K, not a superconductor | 10.1103/physrevb.66.085106 |
| 35 | LiGaIr | 6 | 0.892 | 42 | **not found as made**. Its relative LiGa2Ir superconducts at 2.94 K (10.1038/s41598-021-95944-1) | — |
| 41 | SrIn3Au (BaNiSn3 type) | 5 | 0.890 | 54 | made, no low-temperature data. Its relative SrAuSi3 superconducts near 1.5 K | 10.1002/zaac.200700191 |
| 44 | SrAl3Ag (BaNiSn3 type) | 5 | 0.889 | 49 | **not found as made** | — |
| 60 | Ca(GePt)2 | 10 | 0.880 | 72 | made, nothing above 2 K. It was reported as monoclinic P21, not the tetragonal P4/nmm cell here | 10.6843/nthu.2013.00560 (a thesis) |
| 72 | TePtPb | 12 | 0.872 | 144 | **not found as made**; possibly cannot be made (PtBiTe could not be) | — |
| 84 | SrNiSn2 | 8 | 0.861 | 76 | made, no low-temperature data | 10.1021/ic060613y |

**The "theoretical" flag leaks**
- [Certain, for these 10] The Materials Project marks all 571 compounds
  in the pool "theoretical", meaning not matched to a known experimental
  structure.
- Even so, 7 of the top 10 had been made.
- So the flag is a weak filter for "never made". Shortlist 2 found the
  same thing, losing 3 to 4 of its 10.

### Places of the compounds checked earlier

These no longer take a slot. Their cycle 6 places among the 571:

- **Top four:** SbTe2 1, NaMgPb2 2, LiMgPb2 3 and Re3Ir 4. Re3Ir failed
  its stability pre-check at the zone corner (`docs/hazel-candidates.md`).
- **Further down the top 20:** Hf2CS 7, ReIr3 8, CaAl3Au 10,
  Nb2GeC 12 (a known superconductor), ScTaN2 13 and Ti2NbAl 14.
- **CsIr2P2:** fell from the cycle 3 top 50 to 107th.
  - [Likely] This does not change the Hazel queue. CsIr2P2 is there for
    its family and its calculation, not for its rank.
  - It is still a point against it.

### Predictions against results

| Prediction | Chance given | Result |
| --- | --- | --- |
| Rank correlation between cycle 3 and cycle 6 is 0.85 or higher | 70% | **yes**, 0.95 |
| At least 4 of the new top 10 had been made | 55% | **yes**, 7 of 10 |
| All four Hazel candidates stay in cycle 6's top 50 | 45% | **no**: CsIr2P2 is 107th; the other three are 1 to 3 |
| At least 1 of the new top 10 is never made and has 6 atoms or fewer | 60% | **yes**, LiGaIr (6) and SrAl3Ag (5) |

Three of four came true. That is about what the chances given imply (2.3
expected). Four predictions are too few to say whether the chances were
well judged.

### Next

1. **Quantum ESPRESSO pre-check of SrAl3Ag and LiGaIr.** Use the same
   single-point stability test as the Hazel candidates (Γ and the zone
   boundary points), on this machine or the owner's.
   - [Speculative] Chance at least one passes: 40%. Two earlier
     compounds failed away from the zone centre: Re3Ir and the layered
     NaMgPb2.
2. **Ask a lab about V2GaN, SrIn3Au and SrNiSn2.** All three are made and
   have never been measured below 2 K. A resistivity run to 2 K is far
   cheaper than any calculation, and the model puts all three in its top
   15% of the pool.
3. **For shortlist 4:** read "made or not" for the top 30 before
   applying the caps. Otherwise most of each new top 10 will be made
   compounds again.

## Quantum ESPRESSO pre-check of LiGaIr and SrAl3Ag: plan, written before any result, 2026-11-22

The owner asked for it on 2026-11-22. The two compounds are the never-made
compounds of the top 10 with 6 atoms or fewer.

**Method:** as for shortlist 2 (`docs/rung4-shortlist2.md`, step 4). The
packages are `ligair` and `sral3ag`, built by
`scripts/prepare_phonon_candidate.py`.

**Setup**
- Starting structures: the Materials Project's (`mp-aaacqrxq`,
  `mp-aaacrihz`), symmetrised.
- Same pseudopotential set as every package (ONCV PBE, scalar
  relativistic). Sr, Al and Ag are added from the same source; the Ga file
  fetched again matches the one already in `pp/` byte for byte.
- 90 Ry, smearing 0.02 Ry.
- Each compound is relaxed (vc-relax) first. The package's own `scf.in`
  is then used unchanged:
  - LiGaIr: k 12×12×8;
  - SrAl3Ag: k 10×10×12, in the body-centred primitive cell.

**Points**
- The zone centre, plus one point from each symmetry-distinct
  zone-boundary point with components 0 or ½ in reciprocal-lattice units.
  That is all of them.
- This is more than shortlist 2's two points per compound. Shortlist 2
  chose its second point from the force field, and that screen has since
  failed its validation.

| Compound | Space group | Points, in reciprocal-lattice units of the package cell |
| --- | --- | --- |
| LiGaIr | P6₃/mmc | Γ; A (0 0 ½); M (0 ½ 0); L (0 ½ ½) |
| SrAl3Ag | I4mm | Γ; (0 ½ 0), star of 4; (0 0 ½), star of 2; (½ ½ ½), star of 1 |

**Order and stopping**
- Both zone centres run first, then the boundary points in the order
  above.
- A compound stops at its first failure: one failure is enough under the
  rule.

**Rule:** unchanged. A mode below −5 cm⁻¹ fails, except the three
whole-crystal translations at the zone centre.

**Predictions [Speculative]**

| Outcome | Chance |
| --- | --- |
| LiGaIr passes at every point | 45% |
| SrAl3Ag passes at every point | 35% |
| At least one of the two passes everywhere | 60% |
| The relaxed cell volume is within 3% of the Materials Project's, for both | 80% |

**Why SrAl3Ag gets the lower chance**
- Its I4mm structure is polar, with free z coordinates on every site.
- No compound of this family has been made with Al as the main element
  ([Likely], from the shortlist 3 reading).

**What a pass means, and does not mean**
- A pass means the compound is stable at these points, at these settings.
- It does not show the compound can be made.
- It does not give the size of its Tc. The coupling would come from a
  full Hazel run.
- A full run waits on the owner's say-so and on ORC.

### Pre-check results so far, 2026-11-22 10:45 UTC

**Interruption and reordering**
- The container restarted while the session was idle and stopped both
  runs. The finished LiGaIr relaxation and ground state were kept on disk.
- The runs were restarted at 08:46 UTC with LiGaIr's points first.
  SrAl3Ag's relaxation alone takes several hours here: its mesh has 319
  k-points against LiGaIr's 95.
- The points and the rule are unchanged; only the order is.

**LiGaIr, relaxed**
- Settles at 0.0 kbar after 6 steps: a = 4.081 Å, c = 5.464 Å.
- The volume is 78.8 Å³, 2.4% above the Materials Project's 77.0 Å³.
- That is within the 3% predicted.

**LiGaIr at the zone centre: passes**

| Point | Lowest counted mode (cm⁻¹) | Imaginary? |
| --- | --- | --- |
| Γ (0 0 0) | 63.7 (B₁g) | No |

- [Certain] The three lowest raw modes, −5.5 (×2, E₁u) and −4.2
  (A₂u), are the three whole-crystal translations.
  - E₁u plus A₂u is the acoustic set in D₆h.
  - With the acoustic sum rule (`dynmat.x`, `asr='crystal'`) all three
    go to 0.00, and every other mode is unchanged to within 1.2 cm⁻¹.
- Every other mode is positive: 63.7 to 543.1 cm⁻¹.
- [Likely] This rules out the buckling of the honeycomb Ga–Li layers,
  which would lower the symmetry to P6₃mc (the LiGaGe type). That
  distortion is a zone-centre mode.

**Next:** A (0 0 ½), M (0 ½ 0) and L (0 ½ ½), started 10:40 UTC; then
SrAl3Ag.

**LiGaIr at A (0 0 ½): passes, 2026-11-22 14:08 UTC**
- [Certain] All 18 modes are positive: 65.2 cm⁻¹ (×4) to 541.9 cm⁻¹.
- This point was rerun from the start. The first attempt stopped at
  11:27 UTC when this machine's disk filled with phonon scratch files.
  - Older scratch was cleared, and each point's scratch is now deleted
    once it finishes.
  - The runner also had a fault that would have skipped M and L: the MPI
    launcher read the point list from standard input. That is fixed.
- M (0 ½ 0) started 14:08 UTC; L (0 ½ ½) follows.

**LiGaIr at M (0 ½ 0): passes, 2026-11-22 19:21 UTC**
- [Certain] All 18 modes are positive: 42.4 to 550.2 cm⁻¹.
- The lowest, 42.4 cm⁻¹, is the softest mode found in LiGaIr so far.
  It is well above the line, and above the 13.0 cm⁻¹ the layered NaMgPb2
  had at A.
- That is 3 of 4 points passed. L (0 ½ ½), the last, started 19:21 UTC.

**LiGaIr at L (0 ½ ½): passes, 2026-11-23 01:04 UTC**
- [Certain] All 18 modes are positive: 87.7 to 545.5 cm⁻¹.

### Pre-check verdict: LiGaIr passes at every point; SrAl3Ag not run here

| Point | Lowest counted mode (cm⁻¹) | Imaginary? |
| --- | --- | --- |
| Γ (0 0 0) | 63.7 | No (the three translations go to 0 under the sum rule) |
| A (0 0 ½) | 65.2 | No |
| M (0 ½ 0) | 42.4 | No |
| L (0 ½ ½) | 87.7 | No |

**What the pass means**
- [Certain] LiGaIr in P6₃/mmc, relaxed with the package settings, has no
  imaginary mode at the zone centre or at any of the three
  symmetry-distinct zone-boundary points with components 0 or ½.
- Its softest mode anywhere checked is 42.4 cm⁻¹, at M.

**What it does not show**
- [Likely] Stability at points that are not of that kind.
  - K (⅓ ⅓ 0) was not checked. A √3 × √3 distortion would show there
    first.
  - The full Hazel run's 4 × 4 × 2 phonon mesh does not hold K either.
    Its phonon interpolation would hint at a soft mode near K, but not
    prove one.
- That LiGaIr can be made. No synthesis has been reported, and its
  relative LiGa2Ir has been made and superconducts at 2.94 K.
- Its coupling or its Tc. That needs the full run, and the result stays
  private under the 2026-11-19 rule.

**SrAl3Ag**
- Not run here. Its relaxation alone takes several hours on this
  machine: 319 k-points against LiGaIr's 95.
- The container restarts whenever the session is idle, and a relaxation
  cannot resume after a restart.
- A Hazel package (`sral3ag`) is registered in
  `scripts/prepare_phonon_candidate.py`. It needs a relaxed cell first,
  which the owner can run on Hazel or ask for here.

**Predictions against results**

| Prediction | Chance given | Result |
| --- | --- | --- |
| LiGaIr passes at every point | 45% | **yes** |
| SrAl3Ag passes at every point | 35% | not run |
| At least one of the two passes everywhere | 60% | **yes** (LiGaIr) |
| Relaxed volume within 3% of the Materials Project's, both | 80% | LiGaIr yes (+2.4%); SrAl3Ag not relaxed |

**Next, if the owner agrees**
1. Add LiGaIr to the Hazel queue for its full run (phonons and coupling).
   Place it after CsIr2P2 and before SbTe2, because it passed its
   pre-check at all four points.
2. Optional: a one-point check at K (⅓ ⅓ 0) here, about 6 hours of this
   machine, before Hazel time is spent.
