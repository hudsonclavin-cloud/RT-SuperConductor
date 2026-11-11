# EPW Pb example (Stage 0 package)

These inputs are copied **unchanged** from the official Quantum ESPRESSO
repository so that Stage 0 tests the tools, not our own settings.

| Item | Value |
| --- | --- |
| Source | https://github.com/QEF/q-e, `EPW/examples/pb/` (`pp/` and `woSOC/`) |
| Commit | `61569eb480231b649c47f631df9c5c83537df461` (2026-07-09) |
| License | GNU GPL v2 (Quantum ESPRESSO `License` file) |
| Tutorial | [EPW tutorial 04: superconductivity in Pb and MgB2](https://docs.epw-code.org/tutorials/tutorial_04/index.html) |

SHA-256 of the copied files:

| File | SHA-256 |
| --- | --- |
| `pp/pb_s.UPF` | `24fe8cc951a89521da24dc6bd936b462078927a6970132848a355857a82d27af` |
| `woSOC/phonons/scf.in` | `e3d03f46b42eaba5e7dd7afa29e9130319d18e5590acbbb99406742f7a55d06a` |
| `woSOC/phonons/ph.in` | `9cf915df289c35e866df00415db07b3fd8e8f127efff2f172339f6674782fea0` |
| `woSOC/phonons/pp.py` | `5aef3d150ed78ac12f3e1c1632cfa57e73c4f5696e7457148e237b6bcb88c3d6` |
| `woSOC/epw/scf.in` | `b858551313da526937b4d9f837931ecb124440be0753c3ee27519fb2f4ca6c68` |
| `woSOC/epw/nscf.in` | `b394d8bc14211b5c5f467f73fc268a9729f9095458638b9441adc0ac8cc366d4` |
| `woSOC/epw/epw.in` | `fc5fcbc01db6d6791f3e3f6d57c74156da98563ec9afd2c729a44316311d8eb6` |

Files added by this project, not part of the original example:

- `pb-fcc.cif`: fcc Pb with a = 4.88035 Å, the same lattice as `celldm(1) = 9.2225583816` bohr in the inputs. `promote_dft_results.py` uses it to confirm the formula.
- `review.template.json`: copy to `results/pb-tutorial/review.json` after the run and fill it in.
- `chain.conf` and `smoke-edits.tsv`: read by `scripts/hpc/submit_epw_chain.sh`. The first names the folders and the QE prefix; the second lists the mesh changes smoke mode makes to the copies in the run directory.

The calculation order, as the example's directory names imply: `phonons/scf` →
`phonons/ph` → `phonons/pp.py` (collects dynamical matrices and dvscf files
into `phonons/save`) → `epw/scf` → `epw/nscf` → `epw/epw`.
`scripts/hpc/submit_stage0_pb.sh` submits that chain on Hazel.
Since 2026-11-11 it does so through the shared `scripts/hpc/submit_epw_chain.sh`.
