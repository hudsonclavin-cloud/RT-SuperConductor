# Rung 4, shortlist 2: filters first, ranking second

**Main idea:** the first shortlist ranked first and checked afterwards, and
lost five of its six never-made compounds to problems a filter could have
caught. This one applies the filters first: can a lab make it, does the
chemistry form a compound, is the structure stable. Only then does the model
rank what is left.

Plan written 2026-11-16 16:55 UTC, before any filter was applied to the pool.

## What went wrong the first time

| Shortlist 1 compound (not found as made) | What stopped it |
| --- | --- |
| AcYbTl2 | Radioactive, and f-electrons. |
| NaYbPb2, CaYbPb2 | Ytterbium's f-electrons: not calculable with this method. |
| La3Pb | No pseudopotential available here. |
| Re3Ir | Unstable phonon mode at full settings; and the two metals form solid solutions, not this compound. |
| SbTe2 | Still open after one point. |

Also, 9 of the 20 had in fact been made already.

## The filters, in order

Start: the frozen pool of 4,014 (`data/benchmarks/rung4/pool.csv`).

| Step | Keep a compound only if | Why |
| --- | --- | --- |
| 1 | It is on the hull (energy above hull 0). | The most likely to form. |
| 2 | No element is radioactive (Tc, Pm, Po, At, Rn, Fr, Ra, Ac, Th, Pa, U, Np, Pu) or has an open 4f shell (Ce to Yb). | Makeable and calculable. |
| 3 | No element is Be, Cd, Hg, Tl, As or Os. | Toxic enough that a first partner lab would probably refuse. [Speculative: my list, not a lab's.] |
| 4 | Every element has a pseudopotential in the set used by the Hazel packages. | Otherwise step 5 of the rung 4 plan cannot run. |
| 5 | Formation energy is −0.10 eV per atom or lower. | Separates compounds from solid solutions. At 1,500 K the mixing entropy of a 3:1 alloy is worth about 0.07 eV per atom, so a compound bound by less than that tends to disorder. Re3Ir would have been caught here if its formation energy is as small as I expect; I have not looked. |
| 6 | The cell has 12 atoms or fewer. | Affordable on Hazel. |
| 7 | It passes the force-field stability screen below, if the screen passes its own validation. | Catches the Re3Ir kind of failure in seconds. |

## Step 7: the force-field stability screen

- **Tool:** CHGNet 0.4.2, a machine-learned force field trained on Materials
  Project calculations. It runs here in seconds per compound.
- **Method:** relax the structure with the force field, then get the force
  constants by displacing each atom in a 2×2×2 supercell, and the phonon
  frequencies at the eight points that supercell holds exactly.
- **Flag as unstable:** any mode below −20 cm⁻¹, leaving out the three
  whole-crystal translations at the zone centre.
- **Validation, before it is used as a filter.** Run it on 60 compounds that
  have been made (experimental structures in the Materials Project: metals,
  non-magnetic, on the hull, 12 atoms or fewer, in the same prototypes as the
  pool), chosen at random with a fixed seed. They exist, so nearly all should
  come out stable.
  - **It is a filter only if it flags 20% or fewer of them.** Otherwise it is
    reported as a column and filters nothing.
- **What I had already seen:** one trial on Re3Ir. The force field calls it
  unstable at seven of eight points, including the zone centre (−69 cm⁻¹),
  where the full calculation found it stable (+78). So it found the right
  compound partly for the wrong reason. [Likely] Force fields of this kind
  are known to make phonons too soft, which is why the validation matters.

## After the filters

1. Rank the survivors with the ranking of record (cycle 3 boosting).
2. Take the top 10, with at most 2 per structure prototype and at most 2 per
   element system. Shortlist 1 was dominated by Re, Os and Ru alloys.
3. Literature check: one low-tier reader per compound, as before.
4. For the best-ranked compounds not found as made (up to 3, 6 atoms or
   fewer): the single-point Quantum ESPRESSO pre-check at the zone centre and
   the softest zone-boundary point, with the rule in
   `docs/hazel-candidates.md`.
5. A Hazel package for each survivor.

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| The force-field screen passes its validation (flags 20% or fewer of made compounds) | 35% |
| Fewer than 300 of the 4,014 survive steps 1 to 6 | 60% |
| At least 4 of the 10 shortlisted turn out to have been made already | 50% |
| At least 1 never-made compound survives the Quantum ESPRESSO pre-check | 40% |

## Risks

- **Biggest risk:** the filters leave only dull compounds. Requiring a strong
  formation energy and no f-elements removes much of the pool's variety.
- **Most likely failure:** the force-field screen fails its validation and
  step 7 does nothing, so stability is checked only for three compounds.
- **Opportunity cost:** about a day of session time that could go to
  collecting post-2020 papers for a blind test of the models.
- **What would change the plan:** a real lab's list of what it can and cannot
  handle replaces step 3.

## Results, 2026-11-16 17:39 UTC

Written while the Quantum ESPRESSO pre-check (step 4) was still running, so
its result is not here.

**Main result:** the filters cut the pool from 4,014 to 571 before ranking,
but they did not fix the first shortlist's two real problems. Half of the new
top 10 are the same compounds as before, and three of the ten are in metal
pairs that form solid solutions. Nine compounds are now "not found as made";
three of them are being checked for stability.

### The filters

| Step | Left |
| --- | --- |
| Frozen pool | 4,014 |
| 1. On the hull | 2,399 |
| 2. Not radioactive, no open 4f shell | 1,032 |
| 3. No toxic element | 703 |
| 4. Pseudopotentials available (drops La and Lu) | 624 |
| 5. Formation energy −0.10 eV per atom or lower | 589 |
| 6. Twelve atoms or fewer | 571 |

- The 571 cover 31 structure prototypes and 540 element systems.
- **A wrong expectation, corrected.** The plan said step 5 would have caught
  Re3Ir "if its formation energy is as small as I expect". It is −0.105 eV per
  atom, so Re3Ir passes step 5, by 0.005. The limit I reasoned out does not
  separate this solid-solution system from a compound.

### Step 7: the force-field screen failed its validation

- **17 of 60 made compounds were flagged unstable (28%). The limit was 20%.**
  By the rule it filters nothing and is a column only.
- The flagged ones include ordinary, well-known phases (TiCo, HfRu, HfAl2,
  HfCo2Sn). [Likely] That is the force field making phonons too soft, as the
  plan warned.
- It is cheap: a median of 6 seconds per compound.
- **A change to the order, made because the screen filters nothing.** The
  plan ran the screen on every survivor before ranking. With no filtering to
  do, that would cost time for no effect, so the screen covers the 40
  best-ranked survivors only. It flags 16 of those 40.
- **One fault on the way.** The first validation run was killed for memory at
  compound 39 (13.7 GB). Forces are now computed a few structures at a time,
  and the screen resumes from a saved file. No rule or limit changed.

### The top 10, ranked by the ranking of record

"Champion" is the generation 7 model's percentile among the 571
(`docs/children-experiment.md`). It is a second opinion and orders nothing.

| Rank | Compound | Structure | Chance | Champion | Force-field lowest mode (cm⁻¹) | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Re3Ir | Cu3Au type | 0.952 | 98.6 | −69 (flagged) | **Failed** the full pre-check on 2026-11-16. Carried over from shortlist 1. |
| 2 | Ti2NbAl | Cmcm | 0.944 | 98.4 | +52 | Made; no low-temperature data found (shortlist 1). A rung 3 candidate. |
| 3 | NaMgPb2 | Heusler | 0.944 | 99.8 | −28 (flagged) | **Not found as made.** Pre-check running. |
| 4 | SbTe2 | CdI2 type | 0.941 | 100.0 | −20 | Not found as made; passed one point; Hazel package ready. |
| 5 | ReIr3 | hexagonal | 0.940 | 71.3 | +43 | Not found as made. Re and Ir form solid solutions here. |
| 6 | LiMgPb2 | Heusler | 0.938 | 98.9 | −15 | **Not found as made.** Pre-check queued. |
| 8 | ScTaN2 | hexagonal | 0.921 | 99.6 | +36 | Made; reported as a small-gap semiconductor (shortlist 1). |
| 10 | Re3Rh | hexagonal | 0.918 | 99.1 | −35 (flagged) | Not found as made. Re and Rh are reported as solid solutions. |
| 11 | Nb2GeC | MAX phase | 0.917 | 98.8 | −29 (flagged) | Made; a 2017 paper claims superconductivity (shortlist 1). |
| 12 | YRhIn5 | 115 type | 0.917 | 90.4 | −23 (flagged) | Unclear: a 2007 abstract implies it was grown. |

- Ranks 7 and 9 were skipped by the caps (two per prototype, two per element
  system). They are NbGaRu2 and VGaRu2, two more Heusler compounds; the two
  Heusler places had gone to NaMgPb2 and LiMgPb2. Neither has been read by a
  reader. [Speculative] Both would be far easier for a lab than the
  lead compounds: every element in them can be arc-melted.
- **The two models agree on these.** Eight of the ten are in the champion's
  top 5%. Over all 571 the two orderings correlate at 0.71.
- The force field flags Nb2GeC, which exists, and passes ReIr3, which
  probably does not. That is the failed validation showing in practice.

### Literature check

Ten low-tier readers, one compound each: the five in the top 10 that shortlist
1 had not checked, and five replacements.
- **A change to the plan:** it said "the top 10". Five of those already had
  verdicts, so the next five by rank (same caps) were read in their place.
- Everything is in `data/benchmarks/rung4/literature-check2.csv`. Every
  reader saw titles and abstracts only.
- Two readers each made one stray tool call to send a message. Both were
  denied and nothing was sent.

| Compound | Verdict | What the reader found |
| --- | --- | --- |
| NaMgPb2 | Not found as made | Made neighbours exist: Na2MgPb and MgNaPb. |
| ReIr3 | Not found as made | Ir and Re are reported as solid solutions with a gap near 20 to 30% Re. |
| LiMgPb2 | Not found as made | Made neighbour: Li2MgPb. |
| Re3Rh | Not found as made | Re-Rh solid solutions, including a disordered alloy at this composition. |
| YRhIn5 | Unclear | A 2007 paper on RRhIn5 crystals names R = Y; no Y-specific data seen. |
| CaAuAl3 (rank 16) | Unclear | A 1992 structure paper reports CaAu0.9Al3.1. |
| ZrCoGa5 (rank 22) | Not found as made | Sister compounds exist; YCoGa5 does not superconduct down to 1.75 K. |
| BaPdIn2 (rank 31) | Made, no low-temperature data | Reported in 2001 with its Rh, Ir and Pt relatives. |
| LiRuN (rank 32) | Not found as made | No lithium ruthenium nitride found at all. |
| CsIr2P2 (rank 35) | Not found as made | Relatives superconduct: BaIr2P2 at 2.1 K, SrIr2As2 at 2.9 K. |

### What I make of the nine "not found as made"

| Compound | My read |
| --- | --- |
| Re3Ir | Out: failed the full calculation. |
| ReIr3, Re3Rh | [Likely] Not real ordered compounds. Both pairs form solid solutions. |
| NaMgPb2, LiMgPb2 | [Speculative] Plausible, since a neighbouring compound exists in each system. Sodium, lithium and magnesium are volatile and air-sensitive, so no arc melting: sealed-tube work. |
| CsIr2P2 | [Speculative] The most interesting chemically: its barium and strontium relatives superconduct near 2 to 3 K. The champion is lukewarm (68th percentile). Caesium needs a glovebox. |
| SbTe2 | Open, as before. |
| ZrCoGa5 | [Speculative] Weak: the yttrium sister does not superconduct. |
| LiRuN | [Speculative] Weak: nothing like it has been made. |

### Predictions against results

| Prediction | Chance given | Result |
| --- | --- | --- |
| The force-field screen passes its validation | 35% | no (28% flagged) |
| Fewer than 300 survive steps 1 to 6 | 60% | no (571) |
| At least 4 of the 10 turn out to have been made | 50% | 3 for certain; 4 if YRhIn5 counts |
| At least 1 never-made compound survives the pre-check | 40% | pending |

### What the filters got wrong, and what I would change

- **Carry verdicts forward.** A compound that already failed, or is already
  known to be made, should leave the pool. Five of the top 10 were repeats.
- **The formation-energy limit is too loose for metal-metal pairs.** Re3Ir,
  ReIr3 and Re3Rh are at −0.10 to −0.17 eV per atom and all look like solid
  solutions. [Speculative] A limit near −0.20 would remove them; it would also
  remove real compounds. A phase-diagram lookup is the right tool, and I do
  not have one.
- **The stability screen needs a better force field or the real
  calculation.** This one cannot tell stable from unstable well enough.
- **The lab constraint is still a guess.** It should come from a lab.

### Step 4, running: the Quantum ESPRESSO pre-check

- **Compounds:** NaMgPb2, LiMgPb2 and CsIr2P2: the best-ranked compounds
  not found as made with six atoms or fewer, leaving out Re3Ir (failed) and
  SbTe2 (done).
- **Points:** the zone centre, and the point where the force field found
  each compound softest: (½ ½ 0) for NaMgPb2, (½ 0 ½) for LiMgPb2,
  (½ ½ ½) for CsIr2P2, in reciprocal-lattice units.
- **Rule:** as in `docs/hazel-candidates.md`. A mode below −5 cm⁻¹ fails,
  except the three translations at the zone centre.
- **Settings:** each compound is relaxed first, then its package's own
  ground-state input is used unchanged (k mesh 8×8×8, 90 Ry, smearing 0.02
  Ry).

### Step 4, first result: NaMgPb2 passes at both points, 2026-11-16 22:49 UTC

**Result.** With the package's own ground-state settings on the relaxed cell
(k mesh 8×8×8, 90 Ry), no mode is imaginary at either point.

| Point | Frequencies (cm⁻¹) | Lowest | Imaginary? |
| --- | --- | --- | --- |
| Zone centre | 18.1 (×3), 60.4 (×3), 160.3 (×3), 182.5 (×3) | 18.1 | No |
| (½ ½ 0), where the force field was softest | 26.5 (×2), 49.0 (×2), 55.6, 57.4, 150.6 (×2), 159.6, 182.4, 190.2 (×2) | 26.5 | No |

- Verdict under the rule in `docs/hazel-candidates.md`: **NaMgPb2 passes
  the pre-check.** Its Hazel package is `data/benchmarks/candidates/namgpb2/`.
- [Certain] The force field had flagged it (−28 cm⁻¹). The real calculation
  does not agree, as with 17 of the 60 made compounds in the screen's
  validation.
- [Likely] The 18.1 triplet at the zone centre is the three whole-crystal
  translations, shifted up from zero because ph.x does not impose the sum
  rule. All four atoms move in the same direction in those modes.
- **A limit of the report script, found here.** It recognises a translation
  only if every atom moves by nearly the same amount (within 10%). With atoms
  as different in mass as Na and Pb the amounts differ, so the three modes
  were not set aside. That does not change this verdict, because they are
  positive. For a compound of unequal masses whose translations came out
  negative, the script would wrongly call it unstable. Not changed here; it
  should be fixed with a test before the next such case.
- [Certain] What this does not show: two points are not the whole zone, and
  a stable lattice says nothing yet about superconductivity. The full run on
  Hazel gives both.

**Prediction.** "At least 1 never-made compound survives the pre-check"
(40%): yes, NaMgPb2. LiMgPb2 and CsIr2P2 are still to run.

**Note on the ranking of record.** This shortlist was ranked with cycle 3.
The owner accepted cycle 4 on 2026-11-16; the shortlist is not re-ranked.

### Step 4, second result: LiMgPb2 passes at both points, 2026-11-16 23:33 UTC

| Point | Frequencies (cm⁻¹) | Lowest | Imaginary? |
| --- | --- | --- | --- |
| Zone centre | 14.7 (×3), 72.3 (×3), 178.1 (×3), 195.9 (×3) | 14.7 | No |
| (½ 0 ½), where the force field was softest | 33.3 (×2), 47.8 (×2), 56.7, 65.5, 169.6 (×2), 179.4, 204.6, 204.8 (×2) | 33.3 | No |

- Verdict under the rule: **LiMgPb2 passes the pre-check.** Its package is
  `data/benchmarks/candidates/limgpb2/`.
- [Likely] As with NaMgPb2, the lowest triplet at the zone centre is the
  three translations, shifted up by the missing sum rule.
- [Certain] NaMgPb2 and LiMgPb2 are the same structure with one element
  swapped, so they are not two independent successes. If one is run on Hazel,
  the other adds less.
- The run was cut by a container restart during the second point and
  continued from its saved work; the output ends normally.
- CsIr2P2 is running.

### Step 4: CsIr2P2 at the zone centre, and a sum-rule check of all three, 2026-11-17 03:39 UTC

**CsIr2P2, zone centre.** Raw frequencies from ph.x (cm⁻¹): −9.4, 11.9 (×2),
50.8 (×2), 71.6, 126.5, 163.4 (×2), 307.1 (×2), 312.0 (×2), 333.5, 410.6.

- The −9.4 mode is the whole-crystal translation along the long axis: all
  five atoms move the same way by the same amount (0.43 to 0.45). The report
  script classifies it as a translation, so the rule sets it aside.
- Verdict at this point: **no imaginary mode.** The second point, (½ ½ ½),
  is still running; CsIr2P2 has no verdict until it finishes.

**Sum-rule check (added after seeing the numbers; labelled so).** The rule
excludes the three zone-centre translations, and the last section noted
that the script cannot always recognise them. Quantum ESPRESSO's own tool
settles it: `dynmat.x` with `asr = 'crystal'` imposes the acoustic sum rule
on a zone-centre result. The three translations then come out at exactly
zero, and what is left are the real modes.

| Compound | Raw lowest three (cm⁻¹) | After the sum rule | Lowest real mode (cm⁻¹) |
| --- | --- | --- | --- |
| NaMgPb2 | 18.1, 18.1, 18.1 | 0, 0, 0 | 60.4 |
| LiMgPb2 | 14.7, 14.7, 14.7 | 0, 0, 0 | 72.3 |
| CsIr2P2 | −9.4, 11.9, 11.9 | 0, 0, 0 | 50.8 |

- [Certain] In all three the lowest raw triplet is the three translations.
  The earlier sections said "[Likely]"; this confirms it.
- [Certain] No verdict changes. NaMgPb2 and LiMgPb2 passed on positive raw
  numbers already.
- The sum rule also moves the upper modes of the two Heusler compounds down
  by 8 to 10 cm⁻¹ (160.3 to 150.1 and 182.5 to 174.8 for NaMgPb2). The raw
  values in the tables above are ph.x's, before the correction.
- Suggested for the report script, not done here: use this check at the
  zone centre in place of the equal-amounts test.

### Step 4, final: CsIr2P2 passes; all three pass, 2026-11-17 06:04 UTC

**CsIr2P2, second point (½ ½ ½).** Frequencies (cm⁻¹): 28.2 (×2), 45.5,
46.9 (×2), 63.7, 126.5, 162.8 (×2), 306.3 (×2), 310.6 (×2), 335.2, 405.8.
None is imaginary. With the zone centre above, **CsIr2P2 passes the
pre-check.** Its package is `data/benchmarks/candidates/csir2p2/`.

**All three.**

| Compound | Structure | Lowest real mode, zone centre (cm⁻¹) | Lowest mode, second point (cm⁻¹) | Verdict |
| --- | --- | --- | --- | --- |
| NaMgPb2 | Heusler | 60.4 | 26.5 at (½ ½ 0) | Passes |
| LiMgPb2 | Heusler | 72.3 | 33.3 at (½ 0 ½) | Passes |
| CsIr2P2 | I4/mmm (ThCr2Si2 type, like its relatives) | 50.8 | 28.2 at (½ ½ ½) | Passes |

- The zone-centre column is after the sum rule; the raw ph.x numbers are in
  `data/benchmarks/candidates/precheck.csv`.
- [Certain] Three of three passed. With Re3Ir (failed) and SbTe2 (passed
  one point), four of the five never-made compounds checked so far show no
  instability at the points tested.
- [Certain] That is a weak filter by design: two points per compound. A
  compound can still be unstable elsewhere in the zone, and none of this
  says whether any of them superconducts.
- [Likely] The two Heusler compounds are one result, not two.
- The CsIr2P2 run took about 6 hours on four cores and was cut four times
  (three time limits and one container restart). Each time it continued
  from saved work; both outputs end normally.

**Prediction.** "At least 1 never-made compound survives the pre-check"
(40%): yes, three did. I set that chance too low: I was anchored on Re3Ir's
failure and on the force field's flags, which its own validation had already
shown to be unreliable.

**Step 5.** All survivors have a Hazel package: `sbte2`, `namgpb2`,
`limgpb2`, `csir2p2`. Suggested order after the Zr6Ga2Ni benchmark:
NaMgPb2 (highest ranked, 4 atoms), CsIr2P2 (a different structure),
SbTe2, then LiMgPb2 only if NaMgPb2 is interesting.

### An outside search of the three names, and what it changes, 2026-11-19 00:44 UTC

The owner searched the three compound names with a search engine's AI
summary. I checked each claim against the Materials Project snapshot here,
the training data and the one paper it cited.

| Claim in the search summary | Check | Verdict |
| --- | --- | --- |
| NaMgPb2, LiMgPb2 and CsIr2P2 have not been made | Same as both literature reads here; all three are "theoretical" in the Materials Project | Agrees [Likely] |
| Na2MgPb is a known, made compound in the same system | In the Materials Project with an ICSD entry (icsd-238202), on the convex hull; the hull NaMgPb2 sits on already includes it | True; no change |
| Na2Mg3Pb5 is a known compound | The source's identifier (agm…) looks like a computed Alexandria entry, not an experiment. It is not in the Materials Project snapshot used here | Probably wrong as stated [Likely]. If it is a real low-energy phase, it could make NaMgPb2 unstable against decomposition [Speculative] |
| Li2MgPb is a known compound | In the Materials Project with two ICSD entries | True; LiMgPb2 is on the hull with it included |
| A 2025 paper (arXiv 2509.05909) studies LiMgPb2 | Read here: a machine-learning paper on magnetism. LiMgPb2 appears once, in a table, called diamagnetic by valence counting. No calculation of phonons or superconductivity | No bearing on these calculations or on novelty [Certain] |
| NaMgPb2 is a heavy analogue of MgB2 | The structure computed here is a cubic Heusler type, not the MgB2 type | Not applicable [Certain] |
| CsIr2P2 is a ThCr2Si2-type 122 compound; BaIr2P2 superconducts | Matches the package (I4/mmm) and the training data (BaIr2P2, 2.1 K) | Agrees |

**One thing this turned up that matters more than the summary.** None of
the three element systems has any record in either training table. The
nearest data are lead-rich binaries, whose reported Tc values cluster near
lead's own 7.2 K: NaPb3 at 5.62 K, Na0.28Pb0.72 at 7.2 K, LiPb at 7.2 K
(a 1932 report), Mg2Pb at 5.6 K.

- [Likely] A sample of any of these candidates that contains a little free
  lead will show a superconducting signal near 7.2 K that is not the
  compound's. A lab check would have to rule that out, for example by
  showing the signal's size and that it moves with field differently from
  lead's.
- [Speculative] Some of those binary records may themselves be lead
  impurity. If so, part of the model's preference for lead-rich compounds is
  learned from contaminated labels. This is a reason to trust the Hazel
  calculation over the ranking for these three, and a check worth doing on
  the training data later.
