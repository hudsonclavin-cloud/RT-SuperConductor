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
