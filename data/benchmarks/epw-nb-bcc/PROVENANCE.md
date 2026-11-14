# Niobium EPW benchmark (Stage 1, second element)

The second check of the **method**, after tantalum: run niobium, whose
electron-phonon numbers are published, and see whether this setup reproduces
them. Judge it by the rule in `docs/hazel-runbook.md`, section 7 ("Stage 1
pass rule"), written on 2026-11-14 before any niobium result was read. The
reference is Cerqueira, Sanna and Marques 2023 (arXiv:2307.10728), which uses
the same double-delta definition of λ: **λ = 1.248, ωlog = 147.1 K**. The
result is never added to training.

> **Status, 2026-11-14: not validated.** On a 4×4×4 phonon grid this setup
> gave λ = 3.35 and ωlog = 28 K: two interpolated acoustic modes near
> 1 meV carried more than half of λ, an artifact of the coarse phonon grid
> (the electrons were interpolated well). This package holds the 6×6×6
> phonon grid and 12×12×12 electron grid that were run next; see the runbook
> for the result.
>
> **Status, 2026-11-14 14:05 UTC: fails the rule.** The 6×6×6 run gave
> λ = 1.767 (band 1.061 to 1.435) and ωlog = 120.7 K (band 125.0 to
> 169.2 K), with no imaginary mode. The excess comes from soft transverse
> modes near Γ; see "Niobium 6×6×6 result" in `docs/hazel-runbook.md`.

The inputs follow the tantalum package (`data/benchmarks/epw-ta-jvasp-1014/`),
changed only where niobium needs it.

## Pseudopotential

| Item | Value |
| --- | --- |
| File | `pp/Nb_ONCV_PBE_sr.upf` |
| Source | https://github.com/pipidog/ONCVPSP, file `abinit/Nb_ONCV_PBE_sr.upf`, fetched 2026-11-14 (re-fetched and checksum confirmed the same day) |
| Kind | Norm-conserving (ONCV), scalar-relativistic, PBE |
| Valence | 13 electrons: 4s² 4p⁶ 4d⁴ 5s¹ (the file's reference configuration) |
| Made with | ONCVPSP 3.3.0, as the file's own header records |
| License | The header asks for a citation of D. R. Hamann, Phys. Rev. B 88, 085117 (2013); ONCVPSP is GPL |

| Check (4 cores, QE 7.6) | Result |
| --- | --- |
| Lattice constant at zero pressure, 16×16×16 k-points | 6.2568 bohr = 3.311 Å at 90 Ry (pressure −0.01 kbar); 70 Ry gave 6.2416 bohr, so 70 Ry is not converged for this file. Measured: 3.300 Å |
| Band energies at that lattice | Fermi level 17.96 eV; bottom of the valence bands 12.07 eV; band 11 starts at 27.08 eV |
| Wannier interpolation (4×4×4 run, 8×8×8 coarse grid) | Mean error 12.9 meV within 0.4 eV of E_F, 95th percentile 36.8 meV, worst 49.8 meV; DOS at E_F +1.0% at 0.1 eV smearing, +6.2% at 0.05 eV |

## Settings

| Setting | Value | Why |
| --- | --- | --- |
| Cell | bcc, `celldm(1) = 6.2568` bohr | The zero-pressure lattice for this pseudopotential |
| Cutoff | `ecutwfc = 90` Ry | 70 Ry moved the lattice constant by 0.25% |
| Smearing | Marzari-Vanderbilt, 0.02 Ry | As for tantalum |
| Phonon k-mesh, q-mesh | 16×16×16, **6×6×6** (16 irreducible q-points) | 4×4×4 gave spurious near-zero acoustic modes after interpolation |
| EPW coarse k-mesh | **12×12×12**, listed in full in `epw/nscf.in`, 18 bands | Must contain the q-mesh |
| Wannier functions | 9: Nb d, s and p (`proj` lines in `epw/epw.in`) | As for tantalum, where six were too few |
| Skipped bands | 1 to 4 | The 4s and 4p semicore states |
| Energy windows | outer 11 to 45 eV, frozen 11 to 25.5 eV | Frozen window from below the valence bottom (12.07 eV) to below band 11 (27.08 eV) |
| Fine meshes | k 24×24×24, q 12×12×12 with `mp_mesh_q` (one point per symmetry set), `fsthick = 0.4` eV, `degaussw = 0.1` eV | As for tantalum |
| λ formula | `delta_approx = .true.` (double delta) | The standard definition; without it EPW broke symmetry badly on tantalum |

The windows are absolute energies for this pseudopotential and lattice
constant; change either and find them again.

`smoke-edits.tsv` turns the package into a quick setup check (phonon q-mesh
2×2×2, k-mesh 8×8×8, fine q-mesh 6×6×6). Only `CHAIN_MODE=full` counts.

## Running it

```bash
CHAIN_MODE=full scripts/hpc/submit_stage1_nb.sh <run_dir> <partition> <qos> 16
```

On this project's 4-core cloud machine, the 4×4×4 version took 74 minutes for
phonons and 8 for EPW.

## Checksums

| File | sha256 |
| --- | --- |
| `pp/Nb_ONCV_PBE_sr.upf` | `298de1637579813d3a5de5e2b8f24cfa719a27d4520afef2bbb2033b0e2191cd` |
| `phonons/scf.in` | `80e86e2d6b9a9dc03e67b2d22ec65902a3bf012e14330ca228b756869431bba9` |
| `phonons/ph.in` | `0acc27dfa2fa6ae09cdf68571a741b0f8f6103908e2f824024645bbdb1d77af3` |
| `phonons/pp.py` | `5aef3d150ed78ac12f3e1c1632cfa57e73c4f5696e7457148e237b6bcb88c3d6` |
| `epw/scf.in` | `451fa09497ea391e28e5544f8f709bcabfc35828f3480d4aa865f00bda088ee0` |
| `epw/nscf.in` | `27ea1029b414207d7e8ab7fd97070c9615040f0c9d3784234d3fbf86e700a9c1` |
| `epw/epw.in` | `14a0a402ec8f106aae0cb9dad9cb61f76892ef92ae48a7ceb835e0d0ea3c06c2` |
