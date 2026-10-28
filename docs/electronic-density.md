# Crystal structures and electronic-density analysis

The AI model can rank candidates, but it cannot infer bonding or electron
localization from a formula alone. The next physics layer therefore keeps
structure retrieval, density calculations, and interpretation as separate,
auditable stages.

## What the pipeline produces

The **Structure analysis** GitHub Actions workflow uses the
`Materials_Project_API` secret to enrich harmonized records, retrieve reference
Materials Project structures as CIF files, and create an electronic-analysis
manifest. It can be run from an iPad through **Actions → Structure and density
inputs → Run workflow**. The artifact contains the CIFs and manifests.

Materials Project structures are computed reference structures. They are useful
starting geometries, not experimental proof and not necessarily the phase that
produced a reported transition.

## Meaning of each analysis

- **Bader charge** partitions a converged charge density into atomic basins. It
  needs a converged density and a consistent all-electron or reconstructed
  density, not just a CIF.
- **ELF** measures electron localization from a converged electronic
  calculation. It can help inspect localized pairs, metallic regions, and
  bonding motifs, but it is not a direct superconductivity label.
- **COHP** resolves bonding and antibonding contributions between selected atom
  pairs. A compatible wavefunction calculation and LOBSTER basis are needed;
  the result depends on the chosen bonds and energy window.
- **Wannier analysis** constructs localized orbitals from band calculations.
  Wannier90, and EPW when electron-phonon interpolation is required, need
  converged k/q meshes, projections, and disentanglement choices.

## What still requires a compute target

The repository does not contain Quantum ESPRESSO, VASP, LOBSTER, or Wannier90
executables, pseudopotentials, or an NC State cluster allocation. Therefore the
workflow deliberately stops after creating real structure inputs and a status
manifest. It does not fabricate Bader, ELF, COHP, or Wannier values.

Before running production calculations, replace the old QE placeholder inputs
with coordinates from a retrieved CIF, relax the structure, document the
pseudopotential set and convergence settings, then run SCF/NSCF and the
analysis-specific post-processing. Keep the output files and settings with the
record ID so the result can be reviewed and fed back into model training.

The implementation is in `scripts/fetch_materials_project_structures.py` and
`scripts/prepare_electronic_density_manifest.py`. The latter marks rows as
`pending_external_dft` only when a real CIF was retrieved, and marks demo or
unmatched rows as blocked.
