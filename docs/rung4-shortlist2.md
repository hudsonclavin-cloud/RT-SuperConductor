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
