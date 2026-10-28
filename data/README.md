# Local research data

Raw datasets are kept under `data/raw/` for local processing and are excluded
from Git because they may be large or subject to source-specific terms.

The checked-in [manifest](manifest.json) records source identifiers, archive
checksums, and the observed contents of the files received for this project.
Do not edit the original archives or commit extracted raw data.

## Current sources

- `raw/uci/`: UCI Superconductivity Data, containing `train.csv` and
  `unique_m.csv` with 21,263 records each.
- `raw/3dsc/`: 3DSC_MP, containing a CSV with a one-line preamble followed by
  multiline CSV records, plus Materials Project CIF structures. The received
  archive parses to 5,773 CSV records and contains 10,904 CIF files.
- `raw/supermat/`: SuperMat 2.0 literature annotations with free-text `Tc`,
  pressure, measurement method, and source-paper filename fields.
- `raw/supercon2/`: NIMS SuperCon2 automatic literature extraction with
  pressure, measurement-method, synthesis-related, and bibliographic fields.

The 3DSC archive also contains macOS metadata entries; those are ignored during
extraction. The 3DSC_ICSD structure archive has not been added.

Run the normalization command in the repository README to create derived files
under `data/processed/`. Use `docs/reviewing-data.md` to curate duplicate
composition groups without altering the source archives.
