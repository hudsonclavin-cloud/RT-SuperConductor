# Nb3Sn Quantum ESPRESSO and EPW benchmark

This is a small, reproducible electron–phonon workflow benchmark. Its experimental
reference target is **Tc ≈ 18 K**. The primary report by Matthias, Geballe,
Geller, and Corenzwit states:

> “the superconducting transition temperature of Nb3Sn at 18°K is the highest one known.”

Source: B. T. Matthias, T. H. Geballe, S. Geller, and E. Corenzwit,
“Superconductivity of Nb3Sn,” *Physical Review* **95**, 1435 (1954),
[DOI 10.1103/PhysRev.95.1435](https://doi.org/10.1103/PhysRev.95.1435),
[APS article](https://journals.aps.org/pr/abstract/10.1103/PhysRev.95.1435).
The 18 K value is the experimental comparison target, not a promised output of
the current QE/EPW templates. A computed Tc must record the electron–phonon
method, convergence, and Coulomb pseudopotential assumptions before comparing it
with experiment.

## Prepare the hand-off

Run from the repository root after Materials Project API access is configured:

```bash
python scripts/prepare_epw_benchmark.py \
  --output-dir data/processed/epw-benchmark/nb3sn \
  --pseudopotential-dir data/pseudopotentials/upf
```

The script selects an exact-composition Nb3Sn entry from Materials Project,
preferring the lowest reported energy above hull, fetches its reference
structure, verifies the reduced composition, and writes:

- `nb3sn-mp-reference.cif` — the computed reference structure and its SHA-256.
- `relax.in` and `vc-relax.in` — fixed-cell and variable-cell relaxation
  templates using coordinates from that structure.
- `scf.in`, `nscf.in`, `ph.in`, `q2r.in`, `matdyn.in`, and `epw.in` — the
  staged electronic, phonon, and electron–phonon hand-off inputs.
- `benchmark-manifest.json` and `generation-summary.json` — source ID,
  structure/input hashes, prerequisite status, and explicit `dft_executed: false`.

The generated QE files use `Nb.UPF` and `Sn.UPF` under the supplied
pseudopotential directory. Replace those placeholders with reviewed,
checksum-recorded UPF files. Merely having a PSLibrary generator-family entry
does not satisfy this requirement. By default, the manifest remains
`blocked_prerequisites` because those files and Hazel compute access/allocation
are not confirmed. `--hazel-access-confirmed` records an operator confirmation;
it does not submit or run a job. Even when both prerequisites are present, the
status is only `prepared_for_manual_review`, and **no DFT is executed**.

## Scientific checks before execution

The Materials Project structure is a calculated reference structure, not proof
of the experimentally realized phase or its measured transition temperature.
Review the structure and phase choice before calculating. The current inputs
are templates with configurable PBE functional, cutoffs (Ry), k meshes,
smearing, and degauss (Ry). These defaults are starting values, not converged settings.
Validate QE/EPW syntax against the installed versions, review the metallic
smearing and k/q meshes, choose and converge pseudopotentials and cutoffs,
relax the cell and internal positions, and establish the appropriate
electronic bands and Wannier windows. The EPW template also needs expert review
of Wannier projections/windows, interpolation meshes, and the Coulomb
pseudopotential used to derive Tc. Do not compare an unconverged or
underspecified EPW result with the 18 K experimental reference as a reproduction.

Expected input chain:

```text
Materials Project reference structure
       ↓
QE relax → converged QE SCF / NSCF and Wannier representation
       ↓
QE phonons → EPW electron–phonon coupling and Tc estimate
       ↓
compare against the reported experimental Tc ≈ 18 K
```

GitHub Actions and this preparation script only create auditable input files;
they do not provide Hazel access, run QE/EPW, or establish a successful
calculation.
