# A second physics proxy, trained on about 9,000 calculated couplings

**Main idea:** the first proxy learned from one group's 4,366 screening
results and failed to carry over to experiments. Two larger public sets of
full electron-phonon calculations are now on this machine. This trains a new
proxy on them and puts it through the same tests the first one took.

Plan written 2026-11-16 18:09 UTC, before any model was trained on the new data. The
owner asked for it. The Alexandria download was still running (4 of 17
archives), so the data counts are filled in by the data step, not here.

## What a proxy is, and what the first one did

A proxy never sees an experiment. It learns, from the formula alone, what a
physics calculation would say about a compound. Its output then goes to the
main model as extra inputs.

| Test of the first proxy (`docs/rung4-plan.md`) | Result |
| --- | --- |
| T1: does it learn the calculation? (AUC within Bercx et al.'s screening table) | 0.825, pass |
| T2: does it carry to experiments? (882 made metals with measured labels) | 0.531, fail |
| Added to the ranking of record | AUC down 0.069 |
| As the `proxy` child in generation 6 | +0.008 on intermetallics, within noise |

The diagnosis then: its training table has no lanthanide or actinide
compounds, and 587 of the 882 experimental rows do.

## The new data

| Set | What it holds | Source | On disk |
| --- | --- | --- | --- |
| Alexandria | About 8,200 compounds: λ, ωlog and Tc from full phonon and electron-phonon calculations (PBEsol) | Cerqueira, Sanna and Marques, Materials Cloud Archive record 3kbt5-r3n56 | `data/raw/alexandria-epc/` (git ignores it); only the small result files are kept |
| JARVIS | 1,058 compounds: λ, ωlog, Tc, structure | Choudhary and Garrity, figshare file 38307921, sha256 `4213651c…` | `data/raw/jarvis-epc/` (git ignores it) |

## Choices fixed here

- **One row per composition.** Several structures of one formula are averaged
  (in logarithms), because the proxy sees only the formula.
- **Kept rows:** 0 < λ ≤ 4 and ωlog > 0; from JARVIS only rows marked stable.
  A larger λ nearly always means an unstable lattice, where the number is not
  physical.
- **One Tc formula for both sets:** Allen-Dynes at μ* = 0.10 from each row's λ
  and ωlog, so the two sets are on one scale.
- **Four outputs:** predicted log λ, log ωlog, log(1 + Tc), and the chance
  that calculated Tc is above 1 K.
- **Inputs:** the 155 formula features. Learner: the ranking of record's
  boosting settings. Nothing is tuned.
- **Cross-fitting:** five folds by element system, the same fold rule as the
  first proxy, so a compound's own chemistry is never in the model that
  scores it.
- **The frozen champion is not touched.** The blind test's models keep the
  first proxy. This work lives in its own script and files.

## Tests, in order

| Test | Question | Measure | Pass mark |
| --- | --- | --- | --- |
| P1 | Does it learn the calculation? | Cross-fitted AUC for "calculated Tc above 1 K" on the new data; and the rank correlation of predicted with calculated λ | AUC 0.70 (the first proxy's mark) |
| P1b | Do the calculations agree with each other? | Trained on the new data, AUC on Bercx et al.'s table (their Tc-above-1-K flag), leaving out formulas the new data shares | reported, no mark |
| P2 | Does it carry to experiments? | AUC on the same 882 made metals as T2, leaving out formulas in the new data | 0.60 (the first proxy's mark) |
| P3 | Does it help the model? | The generation 6 test: baseline plus the new proxy, on the four NIMS selection folds, intermetallics AUC; and the champion with the new proxy swapped in for the old | gain over the baseline with a 99% interval above zero |

- **P2 and P3 are second uses.** The 882 rows were read once for the first
  proxy, and the four folds chose the champion. Neither is blind. The blind
  table is reserved for the frozen champion and is not used here.

## What happens with each outcome

| Outcome | Action |
| --- | --- |
| P3 passes, and the champion with the new proxy is at least as good as the champion | The new proxy becomes a candidate for the next generation. It still needs blind data of its own. |
| P1 passes, P2 and P3 do not | Same finding as before, on thirty times the detailed data: learning a calculation from the formula does not predict experiments. The next step would be structure-aware inputs, not more data. |
| P1 fails | The formula alone cannot represent these calculations. Stop. |

## Predictions [Speculative]

| Outcome | Chance |
| --- | --- |
| P1 passes (AUC 0.70 or more) | 75% |
| P2 passes (0.60 or more on experiments; the first proxy got 0.531) | 30% |
| The new proxy's gain in P3 is larger than the first proxy's +0.008 | 55% |
| P3 passes (99% interval above zero) | 20% |
| The champion with the new proxy beats the champion | 45% |

## Risks

- **Hidden assumption:** that the first proxy failed for lack of data. The
  stronger explanation is that a formula does not determine a coupling
  strength; the structure does. More data cannot fix that.
- **Biggest risk:** the two sets disagree with each other and with Bercx et
  al. (different functionals and grids), so pooling them blurs the target.
  P1b measures this.
- **Opportunity cost:** small. It is local computation, with no reader runs.

## Amendment to test P3, 2026-11-16 18:35 UTC, before anything was trained

**The change.** P3 as written scores the new proxy on the whole of the four
selection folds. Generations 8 to 10 (`docs/children-experiment.md`) now keep
half of each fold as a lockbox that no choice may see. P3 on whole folds
would show the new proxy the lockbox.

- P3 is therefore run as part of generation 8, on the **selection halves** of
  five folds (A to E): the baseline with the new proxy added, and the old
  champion with the new proxy swapped in for the first.
- Its pass mark is unchanged in kind: a gain over the baseline with a 99%
  interval above zero, now on the selection halves.
- P1, P1b and P2 are unchanged. They use no NIMS fold.
- Whether the new proxy reaches the lockbox depends on whether evolution
  keeps it.

## Results of P1, P1b and P2, 2026-11-16 19:34 UTC

**Main result:** the new proxy learns the calculations well (P1 passes) and
carries to experiments better than the first proxy did, but it misses P2's
pass mark by 0.006. P3 is run inside generation 8 and is reported in
`docs/children-experiment.md`.

Numbers are from `docs/physics-proxy2/data.json`, `p1.json` and `p2.json`
(commands: `python scripts/physics_proxy2.py data`, `p1`, `p2`).

**Data.**

| | Count |
| --- | --- |
| Alexandria calculation folders downloaded | 8,249 |
| Alexandria rows kept (46 folders unreadable) | 8,203 |
| JARVIS rows kept, of 1,058 (432 not marked stable, 3 out of range) | 623 |
| Distinct compositions after merging | 7,027 |
| Element systems | 4,957 |
| Calculated Tc above 1 K | 2,855 |

- Two of the 17 Alexandria archives arrived cut short on the first download
  (306 and 359 folders of 500). The download script had marked them complete.
  They were fetched again before anything was trained, and the script now
  accepts an archive only if the stream ended cleanly.

**Tests.**

| Test | Measure | Result | Pass mark | Verdict |
| --- | --- | --- | --- | --- |
| P1 | Cross-fitted AUC, calculated Tc above 1 K (7,027 compositions) | 0.907 (95% 0.899 to 0.915) | 0.70 | Passes |
| P1 | Rank correlation, predicted with calculated λ | 0.81 | none | |
| P1b | AUC on Bercx et al.'s flag, 3,970 compositions, shared formulas left out of training | 0.698 (0.679 to 0.717) | none | The first proxy, trained on those rows, scored 0.825 within them |
| P2 | AUC on 814 made metals (355 superconductors), shared formulas left out | 0.594 (0.550 to 0.634) | 0.60 | Fails by 0.006 |
| P2 | The same, ranking by predicted λ in place of the Tc flag | 0.622 | none | Reported only |
| P2 | The first proxy on these rows | 0.531 | | |

**What it shows.**

- [Certain] A formula is enough to predict these calculations across element
  systems: 0.907, against the first proxy's pass mark of 0.70.
- [Likely] The calculations disagree between sources. A proxy trained on
  Alexandria and JARVIS ranks Bercx et al.'s compounds at 0.698, well below
  what it reaches on its own kind (0.907). This was the plan's biggest risk.
- [Likely] More and better calculations do carry further towards
  experiments: 0.594 against 0.531. But the interval reaches down to 0.550,
  and the pass mark was not met. By the rule the proxy fails P2.
- [Speculative] Predicted λ ranks experiments a little better than the Tc
  flag (0.622). That was not the planned measure and is not a pass.

**Predictions.**

| Outcome | Chance given | Result |
| --- | --- | --- |
| P1 passes | 75% | Yes (0.907) |
| P2 passes | 30% | No (0.594, mark 0.60) |
