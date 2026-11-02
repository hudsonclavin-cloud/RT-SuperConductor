# Structure-aware discovery phase checklist

This checklist is the detailed execution plan for the next RT-SuperConductor
phase. The long-term target is a reproducibly synthesizable material with
Tc between 273.15 and 298.15 K at 1 atm, or below a separately configured
engineering pressure ceiling. A formula, prediction, or computed value is a
hypothesis until its provenance and validation are recorded.

## 1. Deduplicate the structure registry

- Deduplicate by Materials Project ID.
- Preserve every source record, DOI, Tc, pressure, method, review state, and
  provenance that maps to each structure.
- Distinguish unique structures, duplicate reports, synthetic demo rows, and
  unresolved records.
- Create `unique-structures.csv`, `structure-lineage.csv`, and a JSON report.
- Do not count duplicate records as independent structures.

## 2. Resolve missing structures conservatively

- Investigate unresolved formulas such as YBa2Cu3O6.94.
- Map a structure only when composition and phase identity justify it.
- Label approximate mappings as `proxy_structure_requires_review`.
- Leave ambiguous cases unresolved rather than guessing.

## 3. Validate every CIF

Record reconstructed formula, site count, cell volume, lattice parameters,
space group, symmetry, occupancies, disorder, duplicate sites, unrealistic
distances, missing oxidation states, hydrogen ambiguity, and primitive versus
conventional representation. Assign `valid`, `warning`, or `invalid` and
exclude invalid structures from production calculations.

## 4. Derive structure-aware features

Add density, volume per atom, coordination, bond-length and bond-angle
statistics, dimensionality, layer connectivity, vacancies, disorder, space
group, prototype similarity, formation energy, energy above hull, and
decomposition information. Store units, source, method, and missing-value
reason for every feature. Do not force molecular sigma/pi bond counts onto
periodic crystals.

## 5. Generate real DFT inputs from CIFs

Replace placeholder coordinates with CIF-derived lattice vectors and atomic
positions. Generate relaxation, SCF, NSCF, phonon, and electron-phonon inputs.
Require an explicit pseudopotential manifest and record functional, cutoffs,
k-mesh, smearing, convergence, structure ID, and MP ID. Missing inputs must
block a job rather than produce an incomplete calculation.

## 6. Define external-compute execution

Create a reproducible interface for an NC State cluster, workstation, or Slurm
system. Track job status, commands, software versions, pseudopotential hashes,
input hashes, expected outputs, convergence, and failure reasons. GitHub Actions
may prepare and validate jobs but must not pretend to run expensive DFT.

## 7. Add electronic-density and orbital stages

After converged DFT, support Bader charge, ELF, COHP/ICOHP, Wannier90, and EPW
metadata. Record program versions, grids, spin treatment, atom pairs, energy
windows, projections, spreads, centers, hopping values, and interpolation error.
Missing outputs remain `blocked` or `pending`; never write fabricated values.

## 8. Train without leakage

Integrate composition, shell proxies, structure, and electronic features only
after provenance validation. Exclude synthetic rows from empirical training.
Group duplicate MP IDs and near-identical prototypes. Use family-held-out and
chronological tests, retain unknown pressure as unknown, and report uncertainty.
Compare composition-only, formula-descriptor, structure-aware, and
structure-plus-electronic models.

## 9. Rank candidates under pressure and engineering constraints

Combine predicted Tc, uncertainty, pressure, stability, novelty, synthesis
feasibility, element availability, safety, evidence quality, structure quality,
and calculation status. Produce separate ambient, low-pressure, high-pressure
control, unresolved, and experimental-validation rankings. Do not treat
synthetic placeholders as serious candidates.

## 10. Produce a research decision package

Publish the registry, QC report, calculation manifest, DFT-ready list, blocked
list, unresolved mappings, model report, ranked candidates, assumptions, next
calculation, next experiment, and highest-value missing information. Separate
literature observations, reference structures, ML predictions, calculations,
and experimental evidence. Commit all implementation and documentation,
validate the workflows, and report exactly what was completed versus still
required.

Current artifact baseline: 26 records, 21 CIF files, 13 unique MP structures,
four synthetic IDs skipped, and one unresolved YBa2Cu3O6.94 record. The current
artifact contains structures only; no Bader, ELF, COHP, Wannier, or EPW results.
