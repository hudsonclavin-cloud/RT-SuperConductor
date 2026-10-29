# External DFT execution and result ingestion

GitHub Actions retrieves structures, validates manifests, and packages jobs. It
does not run Quantum ESPRESSO, Bader, ELF, LOBSTER, Wannier90, or EPW. The
calculation manifest is blocked by default until a researcher supplies a
reviewed pseudopotential manifest and calculation settings. This prevents
placeholder coordinates or guessed pseudopotentials from becoming results.

For each structure, use its `structure_id` directory on an NC State cluster,
Slurm system, or workstation. Preserve the CIF and manifest, record software
versions and commands, and return outputs under `results/<structure_id>/`.
Expected files and SHA-256 hashes are recorded by
`scripts/ingest_dft_results.py`. QE output detection checks its convergence
message; other output presence checks are preliminary and require scientific
review before being labeled valid results.

Run the manifest builder with the registry and CIF directory:

```bash
python scripts/build_external_dft_manifest.py \
  --registry data/processed/structure-analysis/unique-structures.csv \
  --cif-dir data/processed/structure-analysis/structures \
  --pseudopotential-manifest pseudopotentials.csv \
  --output calculation-manifest.csv
```

The pseudopotential manifest must include at least an `element` column. A
production system should extend it with file path, checksum, functional, and
license/source metadata. Do not start a calculation from this minimal manifest
until those fields and convergence settings are reviewed.

After copying actual outputs into a results directory, run:

```bash
python scripts/ingest_dft_results.py \
  --results-root results \
  --output data/processed/dft-results/result-manifest.csv
```

The ingestion report validates file presence, hashes files, and performs a
basic QE convergence-message check. It does not replace inspection of residuals,
forces, stress, k-point/cutoff convergence, imaginary phonons, ELF grids, COHP
basis quality, or Wannier interpolation accuracy.
