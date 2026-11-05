#!/usr/bin/env python3
"""Reference-control validation for the CHGNet/phonopy diagnostic screen.

The controls deliberately include room-temperature stable phases and phases
with published zero-kelvin dynamical instabilities. They are benchmarks for
this ML-potential protocol, not DFT calculations.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from deep_physics_checks import ScreenBudget, atomic_screen_json, screen_config, screen_structure


STABLE_CONTROLS = {
    # Established ambient-pressure elemental structures. Lattice constants are
    # explicit benchmark inputs (angstrom), not a Materials Project query.
    **{f"{el}_fcc": {"label": "stable", "kind": "elemental_fcc", "element": el, "a": a}
       for el, a in [("Al", 4.05), ("Cu", 3.615), ("Ni", 3.524), ("Ag", 4.086),
                     ("Au", 4.078), ("Pd", 3.889), ("Pt", 3.924), ("Rh", 3.803), ("Ir", 3.839)]},
    **{f"{el}_bcc": {"label": "stable", "kind": "elemental_bcc", "element": el, "a": a}
       for el, a in [("Nb", 3.30), ("Mo", 3.147), ("Ta", 3.30), ("W", 3.165),
                     ("V", 3.03), ("Cr", 2.884), ("Fe", 2.866)]},
    **{f"{el}_hcp": {"label": "stable", "kind": "elemental_hcp", "element": el, "a": a, "c": c}
       for el, a, c in [("Mg", 3.209, 5.211), ("Ti", 2.951, 4.684), ("Zr", 3.232, 5.147),
                        ("Co", 2.507, 4.069), ("Ru", 2.706, 4.282), ("Zn", 2.665, 4.947)]},
    "Nb3Sn_A15": {"label": "stable", "kind": "A15", "a": 5.29,
                   "reference": "A15 Nb3Sn, experimentally established superconducting phase"},
    "MgB2_AlB2": {"label": "stable", "kind": "AlB2", "a": 3.086, "c": 3.524,
                   "reference": "AlB2-type MgB2, experimentally established superconducting phase"},
}

UNSTABLE_CONTROLS = {
    "Zr_bcc_0K": {"label": "unstable", "kind": "elemental_bcc", "element": "Zr", "a": 3.57,
                  "doi": "10.1088/0953-8984/14/43/101",
                  "reference": "Accurate quantum mechanical treatment of phonon instability: body-centred cubic zirconium"},
    "Ti_bcc_0K": {"label": "unstable", "kind": "elemental_bcc", "element": "Ti", "a": 3.30,
                  "doi": "10.1016/S0921-5093(00)01905-5",
                  "reference": "First-principles calculations on bcc-hcp transition of titanium"},
    "U_gamma_bcc_0K": {"label": "unstable", "kind": "elemental_bcc", "element": "U", "a": 3.47,
                       "doi": "10.1103/PhysRevB.85.060301",
                       "reference": "High-temperature phonon stabilization of gamma-uranium from relativistic first-principles theory"},
    "AuZn_B2_0K": {"label": "unstable", "kind": "B2", "elements": ["Au", "Zn"], "a": 3.12,
                    "doi": "10.1016/j.matpr.2015.07.349", "reference": "Dynamic Stabilization of Cubic AuZn"},
    "NiTi_B2_0K": {"label": "unstable", "kind": "B2", "elements": ["Ni", "Ti"], "a": 3.01,
                    "doi": "10.25560/10729",
                    "reference": "Martensitic transformation in TiNi based shape memory alloys – a first principles study",
                    "condition": "B2 austenite at 0 K; published phonons show imaginary frequencies at experimentally observed soft-mode locations"},
    "FeV_B2_high_pressure_0K": {"label": "unstable", "kind": "B2", "elements": ["Fe", "V"], "a": 2.84,
                                 "doi": "10.1063/5.0219881",
                                 "reference": "Thermally frustrated phase transition at high pressure in B2-ordered FeV",
                                 "condition": "0 K DFT instability reported in the experimentally investigated high-pressure range; condition differs from this screen's 0 GPa relaxation"},
}

SETTINGS = (("current_3x3", 3, 3), ("finer_4x4", 4, 4), ("finer_5x5", 5, 5))


def control_structure(name: str):
    """Build a compact primitive/prototype structure from explicit parameters."""
    from ase import Atoms
    from ase.build import bulk
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    row = (STABLE_CONTROLS | UNSTABLE_CONTROLS)[name]
    kind = row["kind"]
    if kind == "elemental_fcc":
        atoms = bulk(row["element"], "fcc", a=row["a"])
        return AseAtomsAdaptor.get_structure(atoms)
    if kind == "elemental_bcc":
        atoms = bulk(row["element"], "bcc", a=row["a"])
        return AseAtomsAdaptor.get_structure(atoms)
    if kind == "elemental_hcp":
        atoms = bulk(row["element"], "hcp", a=row["a"], c=row["c"])
        return AseAtomsAdaptor.get_structure(atoms)
    if kind == "A15":
        a = row["a"]
        frac = [[0, 0, 0], [.5, .5, .5], [0, .25, .5], [.5, 0, .25],
                [.25, .5, 0], [0, .75, .5], [.5, 0, .75], [.75, .5, 0]]
        species = ["Sn", "Sn", "Nb", "Nb", "Nb", "Nb", "Nb", "Nb"]
        return Structure(Lattice.cubic(a), species, frac)
    if kind == "AlB2":
        return Structure(Lattice.hexagonal(row["a"], row["c"]),
                         ["Mg", "B", "B"], [[0, 0, 0], [1/3, 2/3, .5], [2/3, 1/3, .5]])
    if kind == "B2":
        return Structure(Lattice.cubic(row["a"]), row["elements"], [[0, 0, 0], [.5, .5, .5]])
    raise ValueError(f"Unsupported control structure kind {kind!r}")


def score_setting(results: list[dict], setting: str) -> dict:
    """Compute threshold confusion counts; blocked controls are not successes."""
    refs = STABLE_CONTROLS | UNSTABLE_CONTROLS
    subset = [r for r in results if r.get("setting") == setting]
    scored = {r["control_id"]: r for r in subset if r.get("minimum_frequency_thz") is not None}
    unstable_pred = {key for key, row in scored.items() if float(row["minimum_frequency_thz"]) < -.5}
    stable_ids = set(STABLE_CONTROLS)
    # FeV is a pressure-qualified reference: the published instability is at
    # high pressure, whereas this protocol relaxes at 0 GPa. Keep and report
    # its result, but do not count it as a false negative at mismatched P.
    condition_mismatch_ids = {key for key, ref in UNSTABLE_CONTROLS.items() if ref.get("condition") and "high-pressure" in ref["condition"]}
    unstable_ids = set(UNSTABLE_CONTROLS) - condition_mismatch_ids
    fp = len(unstable_pred & stable_ids)
    tn = len((set(scored) & stable_ids) - unstable_pred)
    tp = len(unstable_pred & unstable_ids)
    fn = len((set(scored) & unstable_ids) - unstable_pred)
    blocked_stable = len(stable_ids - set(scored))
    blocked_unstable = len(unstable_ids - set(scored))
    al_result = scored.get("Al_fcc")
    stable_correct_fraction_all = tn / len(stable_ids)
    criterion = bool(al_result is not None and "Al_fcc" not in unstable_pred and stable_correct_fraction_all >= .8)
    return {
        "setting": setting, "stable_total": len(stable_ids), "stable_scored": len(set(scored) & stable_ids),
        "unstable_total": len(unstable_ids), "unstable_scored": len(set(scored) & unstable_ids),
        "true_negative_stable": tn, "false_positive_stable": fp,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "true_positive_unstable": tp, "false_negative_unstable": fn,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "blocked_stable": blocked_stable, "blocked_unstable": blocked_unstable,
        "condition_mismatch_unstable_controls": len(condition_mismatch_ids),
        "stable_correct_fraction_of_full_set": stable_correct_fraction_all,
        "al_false_positive": ("Al_fcc" in unstable_pred) if al_result is not None else None,
        "passes_user_criterion": criterion,
        "screen_classification": "reference-validated-for-this-control-set" if criterion else "diagnostic only",
    }


def validate(output: Path, cache_root: Path, cpu_budget_hours: float = 6.0, relax_steps: int = 150) -> dict:
    import torch
    from chgnet.model.model import CHGNet

    torch.set_num_threads(2)
    warnings.filterwarnings("ignore")
    model = CHGNet.load(model_name="0.3.0", use_device="cpu")
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = json.loads(output.read_text()) if output.exists() else {"results": []}
    results = list(prior.get("results", []))
    control_rows = STABLE_CONTROLS | UNSTABLE_CONTROLS
    for setting, repeat, mesh in SETTINGS:
        config = screen_config(relax_steps=relax_steps, repeat=repeat, mesh=mesh)
        signature = f"{setting}-{config['model_sha256']}-{config['code_sha256']}"
        budget = ScreenBudget(cache_root / f"{setting}.cpu.json", signature, cpu_budget_hours * 3600)
        done = {r["control_id"] for r in results if r.get("setting") == setting}
        for control_id, ref in control_rows.items():
            if control_id in done:
                continue
            record = {"control_id": control_id, "setting": setting, "expected_label": ref["label"],
                      "structure_kind": ref["kind"], "formula": str(control_structure(control_id).composition.reduced_formula),
                      "reference_doi": ref.get("doi"), "reference": ref.get("reference"),
                      "reference_condition": ref.get("condition"), "screen_pressure_gpa": 0.0}
            structure = control_structure(control_id)
            key = f"{setting}-{control_id}"
            try:
                result = screen_structure(structure, model, budget, config,
                                          cache_root / f"{setting}-cache", key)
                record.update({k: result.get(k) for k in (
                    "status", "minimum_frequency_thz", "gamma_min_thz", "max_final_force_ev_angstrom",
                    "residual_stress_max_gpa", "relaxation_steps", "displacements", "cpu_seconds_cumulative",
                    "reason", "method")})
            except TimeoutError as exc:
                record.update(status="blocked_calculation_budget", reason=str(exc))
            except Exception as exc:
                record.update(status="blocked_calculation", reason=f"{type(exc).__name__}: {exc}")
            results.append(record)
            budget.save()
            atomic_screen_json(output, {"schema_version": 1, "results": results,
                                        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(f"{setting} {control_id} {record.get('status')} {record.get('minimum_frequency_thz')}", flush=True)
    summaries = [score_setting(results, setting) for setting, _, _ in SETTINGS]
    protocols = {name: screen_config(relax_steps, repeat, mesh)
                 for name, repeat, mesh in SETTINGS}
    report = {"schema_version": 1, "method": "CHGNet 0.3.0 + phonopy finite-displacement ML-potential screen; not DFT",
              "threshold_thz": -.5, "relax_steps": relax_steps, "target_pressure_gpa": 0.0,
              "controls": {"stable": len(STABLE_CONTROLS), "known_unstable": len(UNSTABLE_CONTROLS)},
              "stable_control_definitions": STABLE_CONTROLS, "unstable_control_definitions": UNSTABLE_CONTROLS,
              "protocols": protocols, "settings": summaries, "results": results,
              "ranking_use": "diagnostic only; not used to exclude or order candidates",
              "decision": "diagnostic only" if not any(x["passes_user_criterion"] for x in summaries) else "reference-validated only for stated control criterion"}
    atomic_screen_json(output, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("docs/phonon-screen-validation.json"))
    p.add_argument("--cache", type=Path, default=Path("/tmp/rt-supercon-phonon-validation"))
    p.add_argument("--cpu-budget-hours", type=float, default=6.0, help="cumulative CPU hours per setting")
    p.add_argument("--relax-steps", type=int, default=150)
    a = p.parse_args()
    report = validate(a.output, a.cache, a.cpu_budget_hours, a.relax_steps)
    print(json.dumps(report["settings"], indent=2))


if __name__ == "__main__":
    main()
