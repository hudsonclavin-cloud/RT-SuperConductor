#!/usr/bin/env python3
"""Build a ready-to-submit phonon and electron-phonon package for a candidate (``docs/hazel-candidates.md``).

Method: Quantum ESPRESSO's own interpolated electron-phonon coupling (``ph.x`` with
``electron_phonon = 'interpolated'``, then ``q2r.x`` and ``matdyn.x`` with ``la2F``). It needs no Wannier functions,
so nothing has to be hand-tuned per material. It answers the two questions rung 4 asks of a finalist: are there
imaginary phonon modes, and what are lambda and omega_log.

    python scripts/prepare_phonon_candidate.py relax-input zr6ga2ni <dir>    # a vc-relax input from the MP structure
    python scripts/prepare_phonon_candidate.py package zr6ga2ni <relax.out>  # the package, from the relaxed cell

The structure comes from the Materials Project snapshot (``data/raw/mp-structures/structures.jsonl.gz``). It is
relaxed once with the same pseudopotentials and cutoff as the package, off the cluster, and the relaxed cell is
written into every input, so Hazel only runs the ground states and the phonons.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = ROOT / "data/raw/mp-structures/structures.jsonl.gz"
PP_SOURCE = ROOT / "data/benchmarks/candidates/pp"
OUT = ROOT / "data/benchmarks/candidates"
ECUTWFC, SMEARING, DEGAUSS = 90, "mv", 0.02
BOHR = 0.529177210903
# k: ground-state mesh for the phonons; the fit mesh is twice as dense; q must divide k.
CANDIDATES = {
    "zr6ga2ni": {"mp_id": "mp-aaacjlsl", "formula": "Zr6Ga2Ni", "k": (6, 6, 12), "q": (2, 2, 4),
                 "smoke_k": (4, 4, 8), "smoke_q": (1, 1, 2), "matdyn_k": (12, 12, 24), "ph_time": "24:00:00"},
    "re3ir": {"mp_id": "mp-aaacpnuc", "formula": "Re3Ir", "k": (12, 12, 12), "q": (4, 4, 4),
              "smoke_k": (6, 6, 6), "smoke_q": (2, 2, 2), "matdyn_k": (24, 24, 24), "ph_time": "12:00:00"},
    "sbte2": {"mp_id": "mp-aaacrjyk", "formula": "SbTe2", "k": (12, 12, 8), "q": (4, 4, 2),
              "smoke_k": (6, 6, 4), "smoke_q": (2, 2, 2), "matdyn_k": (24, 24, 16), "ph_time": "12:00:00"},
}


def load_structure(mp_id: str) -> dict:
    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["mp_id"] == mp_id:
                return row["structure"]
    raise SystemExit(f"{mp_id} is not in {STRUCTURES}")


def from_mp(structure: dict) -> tuple[list[list[float]], list[tuple[str, list[float]]]]:
    cell = structure["lattice"]["matrix"]
    sites = [(site["species"][0]["element"], [x % 1.0 for x in site["abc"]]) for site in structure["sites"]]
    return cell, sites


def from_relax(text: str) -> tuple[list[list[float]], list[tuple[str, list[float]]]]:
    """The final cell (angstrom) and crystal positions from a vc-relax output."""
    if "End final coordinates" not in text:
        raise SystemExit("the relaxation did not finish (no 'End final coordinates')")
    final = text.split("Begin final coordinates")[-1].split("End final coordinates")[0]
    header = re.search(r"CELL_PARAMETERS\s*\((.*?)\)\s*\n((?:\s*[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s*\n){3})", final)
    unit = header.group(1).strip()
    scale = 1.0
    if unit.startswith("alat"):
        scale = float(unit.split("=")[1]) * BOHR
    elif unit == "bohr":
        scale = BOHR
    elif unit != "angstrom":
        raise SystemExit(f"unexpected cell unit: {unit}")
    cell = [[float(x) * scale for x in line.split()] for line in header.group(2).strip().splitlines()]
    block = re.search(r"ATOMIC_POSITIONS\s*\(crystal\)\s*\n((?:\s*[A-Z][a-z]?\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+.*\n)+)", final)
    sites = [(line.split()[0], [float(x) for x in line.split()[1:4]]) for line in block.group(1).strip().splitlines()]
    return cell, sites


def masses() -> dict[str, float]:
    from pymatgen.core import Element
    return {el.symbol: float(el.atomic_mass) for el in Element}


def pw_input(name: str, calculation: str, cell, sites, kmesh, la2f: bool, pseudo_dir: str = "../pp/") -> str:
    species = sorted({s for s, _ in sites})
    mass = masses()
    lines = [" &control", f"    calculation = '{calculation}'", f"    prefix      = '{name}'",
             f"    pseudo_dir  = '{pseudo_dir}'", "    outdir      = './tmp'", "    tprnfor     = .true.",
             "    tstress     = .true.", " /", " &system", "    ibrav       = 0", f"    nat         = {len(sites)}",
             f"    ntyp        = {len(species)}", f"    ecutwfc     = {ECUTWFC}", "    occupations = 'smearing'",
             f"    smearing    = '{SMEARING}'", f"    degauss     = {DEGAUSS}"]
    if la2f:
        lines.append("    la2F        = .true.")
    lines += [" /", " &electrons", "    mixing_beta = 0.4", "    conv_thr    = 1.0d-12", " /"]
    if calculation == "vc-relax":
        lines += [" &ions", " /", " &cell", "    press_conv_thr = 0.1", " /"]
    lines.append("ATOMIC_SPECIES")
    lines += [f" {s} {mass[s]:.5f} {s}_ONCV_PBE_sr.upf" for s in species]
    lines.append("CELL_PARAMETERS angstrom")
    lines += [" " + " ".join(f"{x:.9f}" for x in row) for row in cell]
    lines.append("ATOMIC_POSITIONS crystal")
    lines += [f" {s} " + " ".join(f"{x:.9f}" for x in abc) for s, abc in sites]
    lines += ["K_POINTS automatic", " " + " ".join(str(k) for k in kmesh) + " 0 0 0", ""]
    return "\n".join(lines)


def ph_input(name: str, q) -> str:
    return "\n".join(["phonons and electron-phonon coupling", " &inputph", f"    prefix   = '{name}'",
                      "    outdir   = './tmp'", f"    fildyn   = '{name}.dyn'", "    fildvscf = 'dvscf'",
                      "    ldisp    = .true.", f"    nq1 = {q[0]}", f"    nq2 = {q[1]}", f"    nq3 = {q[2]}",
                      "    tr2_ph   = 1.0d-14", "    trans    = .true.",
                      "    electron_phonon = 'interpolated'", "    el_ph_sigma  = 0.005", "    el_ph_nsigma = 10",
                      "    recover  = .true.", " /", ""])


def q2r_input(name: str) -> str:
    return "\n".join([" &input", "    zasr   = 'simple'", f"    fildyn = '{name}.dyn'", f"    flfrc  = '{name}.fc'",
                      "    la2F   = .true.", " /", ""])


def matdyn_input(name: str, kmesh) -> str:
    return "\n".join([" &input", "    asr    = 'simple'", f"    flfrc  = '{name}.fc'", f"    flfrq  = '{name}.freq'",
                      "    dos    = .true.", f"    fldos  = '{name}.phdos'", f"    nk1 = {kmesh[0]}",
                      f"    nk2 = {kmesh[1]}", f"    nk3 = {kmesh[2]}", "    ndos   = 500", "    la2F   = .true.",
                      " /", ""])


def cif_text(formula: str, cell, sites) -> str:
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    return str(CifWriter(Structure(Lattice(cell), [s for s, _ in sites], [abc for _, abc in sites]), symprec=0.01))


def cmd_relax_input(name: str, directory: Path) -> None:
    spec = CANDIDATES[name]
    cell, sites = from_mp(load_structure(spec["mp_id"]))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "relax.in").write_text(pw_input(name, "vc-relax", cell, sites, spec["k"], la2f=False,
                                                 pseudo_dir=str(PP_SOURCE) + "/"), encoding="utf-8")
    print(directory / "relax.in")


def cmd_package(name: str, relax_out: Path) -> None:
    spec = CANDIDATES[name]
    cell, sites = from_relax(relax_out.read_text(encoding="utf-8", errors="replace"))
    package = OUT / name
    (package / "run").mkdir(parents=True, exist_ok=True)
    fit = tuple(2 * k for k in spec["k"])
    smoke_fit = tuple(2 * k for k in spec["smoke_k"])
    files = {"run/scf_fit.in": pw_input(name, "scf", cell, sites, fit, la2f=True),
             "run/scf.in": pw_input(name, "scf", cell, sites, spec["k"], la2f=True),
             "run/ph.in": ph_input(name, spec["q"]), "run/q2r.in": q2r_input(name),
             "run/matdyn.in": matdyn_input(name, spec["matdyn_k"])}
    for relative, text in files.items():
        (package / relative).write_text(text, encoding="utf-8")
    checksums = {}
    for species in sorted({s for s, _ in sites}):
        source = PP_SOURCE / f"{species}_ONCV_PBE_sr.upf"           # shared by every package: ../pp
        checksums[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    (package / f"{name}.cif").write_text(cif_text(spec["formula"], cell, sites), encoding="utf-8")
    mesh = lambda m: " " + " ".join(str(x) for x in m) + " 0 0 0"  # noqa: E731
    edits = ["# Smoke-mode edits: file, text to find, replacement (tab-separated). Coarser meshes; tools check only.",
             f"run/scf_fit.in\t{mesh(fit)}\t{mesh(smoke_fit)}", f"run/scf.in\t{mesh(spec['k'])}\t{mesh(spec['smoke_k'])}"]
    edits += [f"run/ph.in\tnq{i + 1} = {spec['q'][i]}\tnq{i + 1} = {spec['smoke_q'][i]}" for i in range(3)
              if spec["q"][i] != spec["smoke_q"][i]]
    edits += [f"run/matdyn.in\tnk{i + 1} = {spec['matdyn_k'][i]}\tnk{i + 1} = {spec['smoke_k'][i] * 2}" for i in range(3)]
    (package / "smoke-edits.tsv").write_text("\n".join(edits) + "\n", encoding="utf-8")
    (package / "chain.conf").write_text("\n".join([
        "# Read by scripts/hpc/submit_phonon_chain.sh. Paths are relative to this directory.",
        f"CHAIN_NAME={name}", f"PREFIX={name}", "RUN_DIR=run", 'COPY_ITEMS="../pp run"',
        f"FULL_PH_TIME={spec['ph_time']}", ""]), encoding="utf-8")
    meta = {"name": name, "formula": spec["formula"], "mp_id": spec["mp_id"], "atoms": len(sites),
            "ecutwfc_ry": ECUTWFC, "smearing": SMEARING, "degauss_ry": DEGAUSS, "k_mesh": spec["k"], "fit_k_mesh": fit,
            "q_mesh": spec["q"], "relaxed_cell_angstrom": cell, "pseudopotential_sha256": checksums,
            "relaxed_from": "Materials Project structure, vc-relax with these pseudopotentials and cutoff"}
    (package / "package.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(package)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    relax = sub.add_parser("relax-input")
    relax.add_argument("name", choices=sorted(CANDIDATES))
    relax.add_argument("directory", type=Path)
    package = sub.add_parser("package")
    package.add_argument("name", choices=sorted(CANDIDATES))
    package.add_argument("relax_out", type=Path)
    args = parser.parse_args()
    if args.command == "relax-input":
        cmd_relax_input(args.name, args.directory)
    else:
        cmd_package(args.name, args.relax_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
