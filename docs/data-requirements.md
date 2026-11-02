# Data required for useful RT-superconductor screening

The target is a reproducibly synthesizable material with a transition between
0 and 25 °C (273.15–298.15 K) at 1 atm or a pressure compatible with a metal
cell. More columns do not automatically make a better model. Each field needs
clear units, provenance, and a definition of what an unknown value means.

## Priority 0: make each reported transition trustworthy

These fields should be required before a literature row enters the empirical
training set:

| Group | Fields |
| --- | --- |
| Identity | reported formula, normalized formula, phase/polymorph, stoichiometry variables, defects or vacancies, chemical family |
| Transition | Tc onset, midpoint, and zero-resistance temperature; transition width; measurement method; magnetic-field value; evidence for bulk shielding or heat capacity |
| Pressure | numeric pressure in GPa, pressure medium, calibration method, pressure uncertainty, and whether the value is ambient, unknown, or not reported |
| Sample | single crystal, polycrystal, thin film, powder, grain orientation, density, sample dimensions, and batch identifier |
| Provenance | DOI or source identifier, page/table/figure location, extraction method, reviewer, review date, and confidence |

Unknown pressure must remain unknown. A missing pressure field must never be
converted into an ambient-pressure negative or positive label.

## Priority 1: synthesis and phase reproducibility

The model should learn whether a candidate can actually be made, not only its
predicted Tc. Capture precursor formulas and purities, weighed ratios,
reaction temperature and time, atmosphere, pressure and cell type, heating and
cooling rates, annealing, quenching, compression history, and post-processing.
Record phase-purity measurements, impurity phases, composition maps, oxygen or
hydrogen content, and the exact sample batch connected to each measurement.

These fields also enable an experimental partner to repeat a promising result.

## Priority 2: crystal structure and defects

For each phase, retain the original CIF or an equivalent structure record,
space group, lattice parameters, volume, atomic coordinates, occupancies,
displacement parameters, coordination numbers, bond lengths and angles,
layered or three-dimensional connectivity, and polymorph identity. Store the
structure source and whether it is experimental, relaxed, predicted, or only a
prototype.

Vacancies, substitutions, interstitial hydrogen, site disorder, strain, and
charge ordering can change superconductivity. They should be represented
explicitly rather than hidden inside a single idealized formula.

## Priority 3: electronic and magnetic physics

Once a real structure exists, useful calculated features include metallicity,
band gap or overlap, density of states at the Fermi level, orbital-resolved DOS,
Fermi-surface summaries, carrier density, effective masses, spin polarization,
magnetic moments, spin order, spin-orbit coupling, and any chosen Hubbard U/J
parameters. Every value needs the code, functional, pseudopotential or basis,
k-point mesh, smearing, and convergence information.

For conventional electron-phonon candidates, also retain phonon stability,
imaginary-mode flags, electron-phonon coupling λ, logarithmic phonon
frequency, Coulomb pseudopotential μ*, and the Allen-Dynes or Eliashberg Tc
estimate. These are hypotheses until reproduced with converged settings.

## Priority 4: charge, bonding, and localized-orbital descriptors

These are valuable after the structure and electronic calculation are valid:

- Bader or another charge-partition result, with the partition method and
  reference density.
- ELF grids or summaries, with spin treatment and grid settings.
- COHP/ICOHP for named atom pairs, including the energy window and LOBSTER
  basis quality.
- Wannier centers, spreads, orbital labels, hopping amplitudes, band
  interpolation error, and electron-phonon matrix elements when EPW is used.

Periodic crystals do not have a unique molecular sigma/pi bond count. For this
project, bond lengths, coordination, COHP, ELF, and Wannier hoppings are more
reproducible than forcing a Lewis structure onto a solid. The existing Lewis
and shell fields are neutral-atom proxies and must stay labeled as proxies.

## Priority 5: engineering and safety constraints

For the final pressure-limited target, track pressure required for the claimed
Tc, pressure-cell geometry, maximum safe pressure, temperature cycling,
chemical compatibility, air and moisture sensitivity, toxicity, element
scarcity, precursor cost, and whether the synthesis uses equipment available
to the collaborating lab.

## Labels, negatives, and leakage controls

The training table should retain failed syntheses, nonsuperconducting controls,
and rejected predictions with their reasons. A failed experiment is different
from an untested material. Keep publication date and experiment date when
available so chronological evaluation is possible. Split by chemical family,
prototype structure, and near-duplicate composition before reporting model
accuracy. Never let a structure or calculation derived from the test record
flow into training features.

## What the next repository run will collect

The structure workflow can currently collect Materials Project IDs, reference
CIFs, stability values, and symmetry. The next useful additions are: an
experimental-versus-computed structure flag, polymorph and defect metadata,
structure provenance, and a calculation manifest that records the exact DFT
settings and output file hashes. Bader, ELF, COHP, and Wannier values should be
added only after a real external DFT run produces them.
