# JARVIS tantalum electron–phonon benchmark

`scripts/prepare_tantalum_epw_benchmark.py` prepares QE/EPW input templates
from the exact JARVIS-SuperconDB structure `JVASP-1014` (elemental Ta). The
published DFT row reports electron–phonon coupling `lambda = 0.7572` and
`omega_log = 164.9922 K`; the generated manifest records the requested rounded
target `omega_log = 164.99 K`, along with the unrounded value. The source row
also reports `Tc = 7.6395 K` at `0 GPa`; this is a dataset value, not a measured
validation target.

The source is JARVIS-SuperconDB, Figshare v4,
[dataset DOI 10.6084/m9.figshare.21370572.v4](https://doi.org/10.6084/m9.figshare.21370572.v4),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The paper
is Choudhary and Garrity, *npj Computational Materials* 8, 244 (2022),
[DOI 10.1038/s41524-022-00933-1](https://doi.org/10.1038/s41524-022-00933-1).
The generator verifies the downloaded source archive against both the
publisher MD5 and repository SHA-256 before using this entry.

```bash
python scripts/prepare_tantalum_epw_benchmark.py \
  --dataset data/raw/jarvis-supercond/jarvis_epc_data_figshare_1058.json.zip \
  --output-dir data/processed/epw-benchmark/tantalum-jvasp-1014
```

It writes a reference CIF, fixed-cell and variable-cell relaxations, SCF,
NSCF, phonon, q2r, matdyn, and EPW templates plus a checksummed manifest. The
job remains blocked until a reviewed Ta pseudopotential and the site's
scheduler/HPC access are confirmed. The generator performs no DFT.
