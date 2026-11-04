# CHGNet/phonopy reference-control validation

**Result: diagnostic only.** No tested setting satisfies the predeclared rule
because all three settings incorrectly flag experimentally stable fcc Al below
the fixed −0.5 THz cutoff. The protocol remains excluded from candidate
ranking and cannot reject a material as physically unstable.

## Benchmark and method

The validation used 24 experimentally established ambient-pressure ordered
stable controls: nine fcc metals (Al, Cu, Ni, Ag, Au, Pd, Pt, Rh, Ir), seven
bcc metals (Nb, Mo, Ta, W, V, Cr, Fe), six hcp metals (Mg, Ti, Zr, Co, Ru,
Zn), plus the A15 Nb3Sn and AlB2-type MgB2 compounds. The ordered structures
were made from the stated prototype and lattice parameters in
[`scripts/validate_phonon_screen.py`](../scripts/validate_phonon_screen.py);
no Materials Project API or external DFT result was used.

Six known 0 K dynamical-instability structures were also tested: bcc Zr, bcc
Ti, bcc γ-U, B2 AuZn, B2 NiTi, and B2 FeV. FeV's cited instability is reported
in a high-pressure interval, so it is retained as a condition-mismatch control
but excluded from the matched-pressure false-negative denominator; the screen
relaxes at 0 GPa. Thus the per-setting instability sensitivity denominator is
five. Published references are listed below. These phase labels concern
specific structures and conditions; they do not mean every phase of those
compositions is unstable.

All 30 structures were processed at each of three settings, for 90 completed
CHGNet 0.3.0/phonopy runs. Settings used 150 maximum relaxation steps, a 0 GPa
target, atomic-force tolerance 0.05 eV/Å, residual-stress tolerance 0.1 GPa,
the unchanged −0.5 THz cutoff, Gamma frequencies, and a centered coarse q mesh.
The supercell and mesh were both 3×3×3, 4×4×4, and 5×5×5, respectively. There
were zero blocked controls. The recorded results and software/model hashes are
in [`phonon-screen-validation.json`](phonon-screen-validation.json).

## Measured classification

| setting | stable correct | stable false positives | false-positive rate | matched unstable detected | false negatives | false-negative rate | blocked | Al flagged? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| current, 3×3×3 | 22/24 (91.7%) | 2/24 | 8.3% | 4/5 | 1/5 | 20.0% | 0 | yes |
| finer, 4×4×4 | 22/24 (91.7%) | 2/24 | 8.3% | 4/5 | 1/5 | 20.0% | 0 | yes |
| finer, 5×5×5 | 22/24 (91.7%) | 2/24 | 8.3% | 4/5 | 1/5 | 20.0% | 0 | yes |

The two stable false positives are fcc Al and bcc Fe. The missed matched
instability is bcc Zr, whose relaxed structure remained bcc and whose lowest
frequency stayed near zero instead of crossing the threshold. The four detected
matched unstable controls are bcc Ti, bcc γ-U, B2 AuZn, and B2 NiTi. The FeV
high-pressure control showed no instability after a 0 GPa relaxation and is
not counted as a false negative because that comparison changes pressure
conditions.

Al's minimum frequency was −0.722, −1.241, and −1.052 THz at the current,
4×4×4, and 5×5×5 settings. The false positive persists at every resolution;
changing the mesh did not fix it. Each setting reaches the requested 80% stable
control accuracy, but fails the explicit Al acceptance condition. Therefore no
setting passes, and the screen remains **diagnostic only**. The cutoff was not
adjusted against candidates or these validation outcomes.

`ml_dynamic_status` and related phonon fields are informational metadata only.
`rank_candidates.py` and the physics-target ranking do not use them to change
candidate scores, remove candidates, or promote a candidate. All current
shortlist rows label the screen diagnostic-only. A missing or clean ML phonon
screen is not evidence of DFT dynamic stability, experimental stability,
synthesizability, or superconductivity.

## Literature sources for unstable reference phases

- bcc Zr: [Accurate quantum mechanical treatment of phonon instability: body-centred cubic zirconium](https://doi.org/10.1088/0953-8984/14/43/101), *Journal of Physics: Condensed Matter*.
- bcc Ti: [First-principles calculations on bcc–hcp transition of titanium](https://doi.org/10.1016/S0921-5093(00)01905-5), *Materials Science and Engineering A*.
- bcc γ-U: [High-temperature phonon stabilization of γ-uranium from relativistic first-principles theory](https://doi.org/10.1103/PhysRevB.85.060301), *Physical Review B*.
- B2 AuZn: [Dynamic Stabilization of Cubic AuZn](https://doi.org/10.1016/j.matpr.2015.07.349), *Materials Today: Proceedings*.
- B2 NiTi: [Martensitic transformation in TiNi based shape memory alloys – a first principles study](https://doi.org/10.25560/10729), Imperial College London thesis. Its repository abstract reports imaginary frequencies at experimentally observed soft-mode locations in B2.
- B2 FeV: [Thermally frustrated phase transition at high pressure in B2-ordered FeV](https://doi.org/10.1063/5.0219881), *Journal of Applied Physics*. The published instability is pressure-qualified, hence its separate treatment above.
