#!/usr/bin/env python3
"""Build the two tables the from-scratch network trains on.

1. ``data/training/element-features.csv`` (committed): one row per element
   from Z = 1 to 96 that has all 22 Magpie properties tabulated (six rare
   elements with gaps are left out rather than filled in with averages), plus
   a one-hot orbital block and the outer-shell occupancy from this project's
   shell descriptors.

2. ``data/processed/scratchnet/compositions.csv.gz`` (not committed; it is
   derived from the SuperCon exports): one row per distinct composition with

   * ``tc_k``: median of the distinct positive Tc values reported for that
     composition (the sources copy each other, so repeats are counted once);
   * ``label``: 1 if anyone reported superconductivity, 0 if every report is
     Tc = 0, empty if the zeros come only from Hosono's unsuccessful-materials
     list (kept, but not used as a negative by default);
   * ``major_group``: the elements that make up at least 5 atomic percent;
   * ``group``: major groups tied together whenever two of their compositions
     differ by at most 0.02 on every element. Splits are made by group, so a
     material never sits on the other side of a split from a near-twin (the
     same compound with slightly different doping);
   * ``split``: ``train_only`` for any group holding more than 5% of the table
     (in practice one chain of cuprate doping series, too large to hold out),
     ``lockbox`` for 15% of the remaining groups, never used for any choice,
     otherwise ``dev`` with a ``fold`` from 0 to 4.

   Split and fold come from a hash of the group name, so they are the same on
   every machine. Adding data can merge groups, so the table's checksum is
   written to ``docs/scratchnet-data.json`` and the training scripts check it.

Labels come from ``build_mp_candidates.known_index`` and so already exclude
the quarantined suspect records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "scripts", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rt_superconductor.scratchnet.data import composition_key, parse_key, stable_hash, write_table  # noqa: E402

MAJOR_FRACTION = 0.05
NEAR_TOLERANCE = 0.02          # compositions this close on every element are tied into one group
TRAIN_ONLY_SHARE = 0.05        # a group with more than this share of all rows is never held out
LOCKBOX_PERCENT = 15
FOLDS = 5
MAGPIE = ["Number", "MendeleevNumber", "AtomicWeight", "MeltingT", "Column", "Row", "CovalentRadius",
          "Electronegativity", "NsValence", "NpValence", "NdValence", "NfValence", "NValence", "NsUnfilled",
          "NpUnfilled", "NdUnfilled", "NfUnfilled", "NUnfilled", "GSvolume_pa", "GSbandgap", "GSmagmom",
          "SpaceGroupNumber"]
ELEMENT_FIELDS = ["symbol"] + ["magpie_" + name for name in MAGPIE] + [
    "block_s", "block_p", "block_d", "block_f", "outer_shell_electrons", "outer_shell_number"]
COMPOSITION_FIELDS = ["composition", "system", "family", "major_group", "group", "group_rows", "n_elements", "label",
                      "tc_k", "tc_max_k", "n_reports", "n_positive", "n_zero", "lineages", "mixed_reports",
                      "hosono_only_zero", "split", "fold"]


def build_element_table(path: Path, max_z: int = 96) -> tuple[int, list[str]]:
    """Write the element table; returns (rows written, elements left out for missing properties)."""
    from matminer.utils.data import MagpieData
    from pymatgen.core import Element

    from rt_superconductor.descriptors import neutral_atom_occupancy

    magpie = MagpieData(impute_nan=False)       # the default silently fills gaps with the column mean
    rows, skipped = [], []
    for z in range(1, max_z + 1):
        element = Element.from_Z(z)
        row = {"symbol": element.symbol}
        values = {name: magpie.get_elemental_property(element, name) for name in MAGPIE}
        if any(value is None or value != value for value in values.values()):
            skipped.append(element.symbol)
            continue
        for name, value in values.items():
            row["magpie_" + name] = float(value)
        for block in "spdf":
            row["block_" + block] = float(element.block == block)
        _, shells = neutral_atom_occupancy(z)
        occupied = [count for count in shells if count > 0]
        row["outer_shell_electrons"] = float(occupied[-1])
        row["outer_shell_number"] = float(len(occupied))
        rows.append(row)
    write_table(path, rows, ELEMENT_FIELDS)
    return len(rows), skipped


def lineage(source: str) -> str:
    if source.startswith("hosono"):
        return "hosono"
    if source.startswith("supermat"):
        return "supermat"
    if source.startswith("audit"):
        return "audit"
    return "supercon"                      # the Stanev export, 3DSC and SuperCon-MTG all descend from SuperCon


def major_group(fractions: dict[str, float]) -> str:
    major = sorted(element for element, fraction in fractions.items() if fraction >= MAJOR_FRACTION)
    return "-".join(major or sorted(fractions))


def near_pairs(fractions: list[dict[str, float]], tolerance: float) -> list[tuple[int, int]]:
    """Index pairs of compositions that differ by at most ``tolerance`` on every element."""
    import numpy as np
    from scipy.spatial import cKDTree

    elements = sorted({element for composition in fractions for element in composition})
    column = {element: i for i, element in enumerate(elements)}
    dense = np.zeros((len(fractions), len(elements)))
    for row, composition in enumerate(fractions):
        for element, value in composition.items():
            dense[row, column[element]] = value
    pairs = cKDTree(dense).query_pairs(r=tolerance + 1e-9, p=np.inf, output_type="ndarray")
    return [(int(a), int(b)) for a, b in pairs]


def merge_groups(major: list[str], pairs: list[tuple[int, int]]) -> list[str]:
    """Tie major groups together through near-twin pairs; name each merged group after its largest member."""
    parent = {name: name for name in major}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for a, b in pairs:
        root_a, root_b = find(major[a]), find(major[b])
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    sizes = Counter(major)
    members: dict[str, list[str]] = defaultdict(list)
    for name in sizes:
        members[find(name)].append(name)
    label = {root: min(names, key=lambda name: (-sizes[name], name)) for root, names in members.items()}
    return [label[find(name)] for name in major]


def assign_split(group: str, group_rows: int, total_rows: int) -> tuple[str, int]:
    """(split, fold): fold is -1 outside the dev folds."""
    if group_rows > TRAIN_ONLY_SHARE * total_rows:
        return "train_only", -1
    h = stable_hash("scratchnet-split:" + group)
    if h % 100 < LOCKBOX_PERCENT:
        return "lockbox", -1
    return "dev", (h // 100) % FOLDS


def build_composition_rows(index: dict, family=None) -> list[dict]:
    merged: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for reports in index.values():
        for fractions, tc, source in reports:
            merged[composition_key(fractions)].append((float(tc), lineage(source)))
    keys = sorted(merged)
    compositions = []
    for key in keys:
        fractions = parse_key(key)
        total = sum(fractions.values())
        compositions.append({element: value / total for element, value in fractions.items()})
    major = [major_group(fractions) for fractions in compositions]
    groups = merge_groups(major, near_pairs(compositions, NEAR_TOLERANCE))
    group_rows = Counter(groups)
    rows = []
    for key, fractions, major_name, group in zip(keys, compositions, major, groups):
        reports = merged[key]
        positive = [tc for tc, _ in reports if tc > 0]
        distinct = sorted({round(tc, 4) for tc in positive})
        zero_lineages = {source for tc, source in reports if tc == 0}
        hosono_only = not positive and zero_lineages == {"hosono"}
        split, fold = assign_split(group, group_rows[group], len(keys))
        rows.append({
            "composition": key, "system": "-".join(sorted(fractions)),
            "family": family(set(fractions)) if family else "", "major_group": major_name, "group": group,
            "group_rows": group_rows[group], "n_elements": len(fractions),
            "label": "" if hosono_only else int(bool(positive)),
            "tc_k": round(statistics.median(distinct), 4) if distinct else "",
            "tc_max_k": round(max(distinct), 4) if distinct else "",
            "n_reports": len(reports), "n_positive": len(positive), "n_zero": len(reports) - len(positive),
            "lineages": ";".join(sorted({source for _, source in reports})),
            "mixed_reports": int(bool(positive) and len(positive) < len(reports)),
            "hosono_only_zero": int(hosono_only), "split": split, "fold": fold,
        })
    return rows


def cross_group_near_pairs(rows: list[dict]) -> int:
    """Near-twin pairs that ended up in different groups. Zero by construction; reported as a check."""
    compositions = [parse_key(row["composition"]) for row in rows]
    return sum(rows[a]["group"] != rows[b]["group"] for a, b in near_pairs(compositions, NEAR_TOLERANCE))


def summarize(rows: list[dict]) -> dict:
    def count(predicate) -> int:
        return sum(1 for row in rows if predicate(row))

    def describe(part: list[dict]) -> dict:
        families = Counter(row["family"] for row in part)
        return {"rows": len(part), "groups": len({row["group"] for row in part}),
                "superconductors": sum(row["label"] == 1 for row in part),
                "negatives": sum(row["label"] == 0 for row in part),
                "tc_above_40k": sum(row["label"] == 1 and float(row["tc_k"]) > 40 for row in part),
                "families": dict(sorted(families.items()))}

    by_split = {"train_only": describe([row for row in rows if row["split"] == "train_only"]),
                "lockbox": describe([row for row in rows if row["split"] == "lockbox"])}
    for fold in range(FOLDS):
        by_split[f"dev fold {fold}"] = describe([row for row in rows if row["split"] == "dev" and row["fold"] == fold])
    sizes = Counter(row["group"] for row in rows)
    return {
        "compositions": len(rows), "superconductors": count(lambda r: r["label"] == 1),
        "negatives": count(lambda r: r["label"] == 0), "unlabelled_hosono_only": count(lambda r: r["label"] == ""),
        "mixed_reports": count(lambda r: r["mixed_reports"] == 1),
        "element_systems": len({row["system"] for row in rows}),
        "major_groups": len({row["major_group"] for row in rows}), "groups": len(sizes),
        "largest_groups": sizes.most_common(5),
        "max_elements": max(row["n_elements"] for row in rows),
        "major_fraction": MAJOR_FRACTION, "near_tolerance": NEAR_TOLERANCE, "train_only_share": TRAIN_ONLY_SHARE,
        "lockbox_percent_of_groups": LOCKBOX_PERCENT, "folds": FOLDS,
        "near_twin_pairs_split_across_groups": cross_group_near_pairs(rows),
        "by_split": by_split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--elements", type=Path, default=ROOT / "data/training/element-features.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/scratchnet/compositions.csv.gz")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/scratchnet-data.json")
    parser.add_argument("--skip-elements", action="store_true", help="keep the committed element table as it is")
    args = parser.parse_args()
    import build_mp_candidates as labels

    skipped: list[str] = []
    if not args.skip_elements:
        _, skipped = build_element_table(args.elements)
    missing = [str(p) for p in (labels.THREEDSC, labels.STANEV) if not p.is_file()]
    if missing:
        raise SystemExit("reference tables missing: " + ", ".join(missing)
                         + "\nrun: python scripts/download_training_data.py --cif-limit 0")
    index = labels.known_index(labels.THREEDSC, labels.STANEV, labels.MTG, labels.AUDIT, labels.QUARANTINE_QUEUE)
    rows = build_composition_rows(index, family=labels.family)
    write_table(args.output, rows, COMPOSITION_FIELDS)
    summary = summarize(rows)
    if not args.skip_elements:
        summary["elements_left_out_for_missing_properties"] = skipped
    summary["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
