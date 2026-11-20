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
    # Shortlist 2 (docs/rung4-shortlist2.md): the best-ranked compounds not found as made, six atoms or fewer.
    "namgpb2": {"mp_id": "mp-aaabxftj", "formula": "NaMgPb2", "k": (8, 8, 8), "q": (4, 4, 4),
                "smoke_k": (4, 4, 4), "smoke_q": (2, 2, 2), "matdyn_k": (16, 16, 16), "ph_time": "12:00:00"},
    "limgpb2": {"mp_id": "mp-aaacplni", "formula": "LiMgPb2", "k": (8, 8, 8), "q": (4, 4, 4),
                "smoke_k": (4, 4, 4), "smoke_q": (2, 2, 2), "matdyn_k": (16, 16, 16), "ph_time": "12:00:00"},
    "csir2p2": {"mp_id": "mp-aaacqsbd", "formula": "CsIr2P2", "k": (8, 8, 8), "q": (4, 4, 4),
                "smoke_k": (4, 4, 4), "smoke_q": (2, 2, 2), "matdyn_k": (16, 16, 16), "ph_time": "12:00:00"},
    # The layered tetragonal NaMgPb2 (P4/mmm), lower in energy than the cubic form in "namgpb2"
    # (docs/rung4-shortlist2.md, "Alexandria check"). Not in the Materials Project snapshot, so the starting
    # structure is Alexandria's. k spacing about 0.14 per angstrom, a little denser than the cubic package's 0.18.
    "namgpb2tet": {"alexandria_id": "agm001144382", "formula": "NaMgPb2", "k": (12, 12, 6), "q": (4, 4, 2),
                   "smoke_k": (4, 4, 2), "smoke_q": (2, 2, 1), "matdyn_k": (24, 24, 12), "ph_time": "24:00:00"},
}

# Starting structures that are not in the Materials Project snapshot: the Alexandria PBE database
# (alexandria.icams.rub.de, OPTIMADE endpoint /pbe/v1/structures), copied as published. Cell rows in angstrom,
# positions in crystal coordinates.
ALEXANDRIA = {
    "agm001144382": {
        "cell": [[3.76869961, 0.0, 0.0], [0.0, 3.76869961, 0.0], [0.0, 0.0, 7.76526153]],
        "sites": [("Pb", [0.0, 0.0, 0.77283445]), ("Pb", [0.0, 0.0, 0.22716555]),
                  ("Na", [0.5, 0.5, 0.5]), ("Mg", [0.5, 0.5, 0.0])],
    },
}


def load_structure(mp_id: str) -> dict:
    with gzip.open(STRUCTURES, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["mp_id"] == mp_id:
                return row["structure"]
    raise SystemExit(f"{mp_id} is not in {STRUCTURES}")


def initial_structure(spec: dict):
    """The unrelaxed cell and sites: from the Materials Project snapshot, or from ``ALEXANDRIA``."""
    if "mp_id" in spec:
        return from_mp(load_structure(spec["mp_id"]))
    entry = ALEXANDRIA[spec["alexandria_id"]]
    return [list(row) for row in entry["cell"]], [(element, list(xyz)) for element, xyz in entry["sites"]]


def source_fields(spec: dict) -> dict:
    """Where a package's structure came from, for package.json."""
    if "mp_id" in spec:
        return {"mp_id": spec["mp_id"]}
    return {"alexandria_id": spec["alexandria_id"]}


def snap(x: float) -> float:
    """Put a coordinate that is within 1e-5 of a multiple of 1/24 exactly on it, so no symmetry is lost to rounding."""
    nearest = round(x * 24) / 24
    return nearest % 1.0 if abs(x - nearest) < 1e-5 else x


def from_mp(structure: dict) -> tuple[list[list[float]], list[tuple[str, list[float]]]]:
    cell = structure["lattice"]["matrix"]
    sites = [(site["species"][0]["element"], [snap(x % 1.0) for x in site["abc"]]) for site in structure["sites"]]
    return cell, sites


def qe_orientation(cell):
    """Turn a hexagonal cell (a = b, 120 degrees, c perpendicular) to Quantum ESPRESSO's own setting.

    The Materials Project draws a1 at -60 degrees from x. pw.x accepts that, but ph.x then stops with
    "divide_class: Wrong classes for D_3d" when it labels the modes (found on SbTe2). A rigid rotation of the
    cell changes no crystal coordinate and no physics.
    """
    import math

    a1, a2, a3 = cell
    length = [math.sqrt(sum(x * x for x in v)) for v in cell]
    dot = lambda u, v: sum(x * y for x, y in zip(u, v))  # noqa: E731
    hexagonal = (abs(length[0] - length[1]) < 1e-4 * length[0]
                 and abs(dot(a1, a2) / (length[0] * length[1]) + 0.5) < 1e-4
                 and abs(dot(a1, a3)) < 1e-4 * length[0] * length[2]
                 and abs(dot(a2, a3)) < 1e-4 * length[1] * length[2])
    if not hexagonal:
        return cell
    a, c = (length[0] + length[1]) / 2, length[2]
    return [[a, 0.0, 0.0], [-a / 2, a * math.sqrt(3) / 2, 0.0], [0.0, 0.0, c]]


def symmetrise(cell, sites, symprec: float = 1e-2):
    """Put the atoms exactly on their symmetry positions (primitive standard cell).

    Materials Project coordinates are rounded, and some are off by more than ``snap`` can repair: Zr6Ga2Ni came out
    with 2 symmetry operations instead of the 12 of P-62m, which multiplies the cost of every later step.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = Structure(Lattice(cell), [element for element, _ in sites], [xyz for _, xyz in sites])
    refined = SpacegroupAnalyzer(structure, symprec=symprec).get_primitive_standard_structure()
    if len(refined) != len(sites):
        raise SystemExit(f"symmetrising changed the atom count: {len(sites)} -> {len(refined)}")
    return (qe_orientation([[float(x) for x in row] for row in refined.lattice.matrix]),
            [(site.species_string, [snap(float(x) % 1.0) for x in site.frac_coords]) for site in refined])


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
    sites = [(line.split()[0], [snap(float(x)) for x in line.split()[1:4]])
             for line in block.group(1).strip().splitlines()]
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
    cell, sites = symmetrise(*initial_structure(spec))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "relax.in").write_text(pw_input(name, "vc-relax", cell, sites, spec["k"], la2f=False,
                                                 pseudo_dir=str(PP_SOURCE) + "/"), encoding="utf-8")
    print(directory / "relax.in")


def cmd_package(name: str, relax_out: Path) -> None:
    spec = CANDIDATES[name]
    cell, sites = from_relax(relax_out.read_text(encoding="utf-8", errors="replace"))
    cell = qe_orientation(cell)
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
    source = "Materials Project" if "mp_id" in spec else "Alexandria"
    meta = {"name": name, "formula": spec["formula"], **source_fields(spec), "atoms": len(sites),
            "ecutwfc_ry": ECUTWFC, "smearing": SMEARING, "degauss_ry": DEGAUSS, "k_mesh": spec["k"], "fit_k_mesh": fit,
            "q_mesh": spec["q"], "relaxed_cell_angstrom": cell, "pseudopotential_sha256": checksums,
            "relaxed_from": f"{source} structure, vc-relax with these pseudopotentials and cutoff"}
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
