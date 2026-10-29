# Candidate variables and open data sources

The Actions workflow is short because it performs metadata/API requests, parses
13 structures, writes compact manifests, and uploads files. It does not run DFT.
Adding arbitrary waiting time would not improve the science. Better workflow
growth means more records, coverage summaries, cross-source reconciliation,
and calculation stages that have real compute resources.

## Additional sources to ingest

`data/training/external-data-sources.csv` records useful source repositories,
what they contribute, and their limitations. The strongest additions for this
project are:

- NIMS SuperCon2, SuperMat, and 3DSC for paper-level evidence and structure
  links; extracted records need DOI-level review.
- COD for openly available experimental CIFs; ICSD may help through institutional
  access if NC State provides it.
- OQMD, AFLOW, and JARVIS-DFT for independent computed structure,
  thermodynamic, electronic, magnetic, and mechanical properties.
- NOMAD for reusable raw DFT workflows and provenance.
- Materials Cloud Archive for study-specific phonon and electron-phonon data.
- Materials Project for reference structures, stability, band gap, metallicity,
  magnetization, density, and volume. These are computed database results, not
  observed superconductivity labels.

## Mechanism-informed variables

Store features in interpretable groups and preserve calculation settings:

1. **Evidence and measurement:** Tc onset/midpoint/zero resistance, transition
   width, susceptibility shielding fraction, heat-capacity anomaly, magnetic
   field, pressure medium, pressure calibration/error, sample batch, publication
   date, and evidence confidence.
2. **Composition and defects:** stoichiometry, valence/doping estimates with
   method, vacancies, substitution disorder, hydrogen/oxygen content, isotope,
   carrier concentration, and charge balance. Keep inferred variables labeled
   as inferred.
3. **Structure:** phase/prototype, space group, lattice and volume, coordination,
   bond lengths/angles, layer spacing, dimensionality, octahedral/tetrahedral
   distortions, occupancy, strain, and disorder.
4. **Electronic:** metallicity, DOS at EF and orbital projections, Fermi-surface
   sheets/nesting, carrier density, effective mass, bandwidth, orbital character,
   spin-orbit coupling, magnetic moments/order, and validated charge transfer.
5. **Lattice dynamics/conventional pairing:** phonon stability and frequencies,
   imaginary modes, electron-phonon lambda, omega-log, Eliashberg spectral
   function alpha2F, Coulomb pseudopotential assumptions, and Tc estimate.
6. **Correlated/unconventional pairing:** magnetic exchange couplings, spin
   fluctuations/susceptibility, competing charge or magnetic order, correlation
   method and U/J, quantum geometry/Berry curvature, and pair-symmetry evidence.
   These are mechanism-specific and are not interchangeable labels.
7. **Synthesis and engineering:** precursors, temperature/time/atmosphere,
   pressure path, anneal/quench, phase purity, repeatability, toxicity, element
   scarcity/cost, and feasible pressure-cell conditions.

Do not add all of these as model features immediately. A feature becomes usable
only when enough records have it, its definition and units are stable, and its
provenance is known. Missingness often reflects which materials were studied,
so report coverage and missingness by family and outcome. Avoid leakage from
target-derived or post-publication calculations into held-out predictions.

## Workflow runtime

The current minute-scale Action is an orchestration and packaging job. A richer
screen can collect broader database properties and process more reviewed
records, but DFT relaxation, dense k-mesh calculations, phonons, Bader/ELF,
LOBSTER, and Wannier/EPW work can range from hours to days and need licensed or
validated inputs plus appropriate CPU/memory allocation. That workload belongs
on an NC State cluster or another configured compute system. GitHub Actions
should stage, validate, and ingest those jobs, then summarize actual runtimes
and convergence rather than simulate longer execution with delays.
