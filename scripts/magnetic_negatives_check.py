#!/usr/bin/env python3
"""NEMAD's magnetic datasets (Itani, Zhang and Zang, Nat. Commun. 16, 9415, 2025; github.com/sumanitani/NEMAD-MagneticML)
against this project: are ferromagnets usable as assumed non-superconductors? (docs/data-sources.md, 2026-11-27)

    python scripts/magnetic_negatives_check.py crosstab <their Dataset folder> docs/magnetic-crosstab.json
    python scripts/magnetic_negatives_check.py score    <their Dataset folder> docs/magnetic-ferromagnet-scores.json

crosstab: how many of their compositions are new to us, and how often a magnetic compound we hold is labelled a
superconductor (locked rows' labels are never read). score: how cycle 6 scores 3,000 ferromagnets it has never seen.
Their data are not copied into this repository; only counts are written.
"""
import sys

if len(sys.argv) != 4 or sys.argv[1] not in ("crosstab", "score"):
    raise SystemExit(__doc__)
COMMAND = sys.argv.pop(1)

if COMMAND == "crosstab":
    import csv, json, sys
    from collections import Counter
    from pathlib import Path
    import numpy as np
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
    import build_mp_candidates as M, build_scratchnet_data as B, scratchnet_common as C, novelty_check as N

    data = Path(sys.argv[1])
    ds = C.load_dataset()
    held, valid = N.held_compositions()
    label = {}                                   # labels we may read: never the locked rows'
    for row, split, lab in zip(ds.rows, ds.split, ds.arrays.label):
        if split != "lockbox" and np.isfinite(lab):
            label[row["composition"]] = int(lab)
    nims = json.loads((ROOT / "data/processed/children-knowledge/table.json").read_text())
    locked = {row["composition"] for row, split in zip(ds.rows, ds.split) if split == "lockbox"}
    for key in nims["keys"]:
        if key not in label and key not in locked:
            label[key] = int(any(l for _, l, _ in nims["records"][key]))
    csv.field_size_limit(10**9)

    def read(name, value_col=None):
        out = {}
        with open(data / name, newline="", encoding="utf-8", errors="replace") as h:
            for row in csv.DictReader(h):
                fr = M.fractions(row["Normalized_Composition"])
                if not fr or not set(fr) <= valid: continue
                out[B.composition_key(fr)] = (row.get("Type"), float(row[value_col]) if value_col and row.get(value_col) else None)
        return out

    cls = read("Classification_FM_AFM_NM.csv")
    curie = {k: v[1] for k, v in read("FM_with_curie.csv", "Mean_TC_K").items()}
    neel = {k: v[1] for k, v in read("AFM_with_Neel.csv", "Mean_TN_K").items()}
    names = {"0.0": "FM", "1.0": "AFM", "2.0": "NM (computed, Materials Project)"}
    report = {}
    for code, name in names.items():
        keys = [k for k, v in cls.items() if v[0] == code]
        new = {k: B.parse_key(k) for k in keys}
        where = N.classify(held, new)
        by = Counter(where.values())
        lab = Counter(label[k] for k in keys if k in label)
        report[name] = {"compositions": len(keys), "by_class": {c: by.get(c, 0) for c in N.CLASSES},
                        "held_with_readable_label": sum(lab.values()), "of_those_superconductor": lab.get(1, 0),
                        "of_those_not": lab.get(0, 0)}
    # ferromagnets by Curie temperature: how often is a held one labelled a superconductor?
    bands = [(0, 10), (10, 50), (50, 300), (300, 5000)]
    fm = {}
    for lo, hi in bands:
        ks = [k for k, t in curie.items() if t is not None and lo <= t < hi]
        lab = Counter(label[k] for k in ks if k in label)
        fm[f"Curie {lo}-{hi} K"] = {"compositions": len(ks), "held_with_label": sum(lab.values()), "superconductor": lab.get(1, 0), "not": lab.get(0, 0)}
    report["ferromagnets_by_curie"] = fm
    af = {}
    for lo, hi in bands:
        ks = [k for k, t in neel.items() if t is not None and lo <= t < hi]
        lab = Counter(label[k] for k in ks if k in label)
        af[f"Neel {lo}-{hi} K"] = {"compositions": len(ks), "held_with_label": sum(lab.values()), "superconductor": lab.get(1, 0), "not": lab.get(0, 0)}
    report["antiferromagnets_by_neel"] = af
    # how many would be NEW negatives: ferromagnets, Curie >= 50 K, not held and not a near twin of a held composition
    newfm = {k: B.parse_key(k) for k, t in curie.items() if t is not None and t >= 50}
    w = N.classify(held, newfm)
    report["ferromagnets_curie_50_plus"] = {"compositions": len(newfm), "by_class": dict(Counter(w.values())),
        "by_family_not_held_or_twin": dict(Counter(M.family(set(newfm[k])) for k in newfm if w[k] not in ("held", "near twin")))}
    report["examples_fm_labelled_superconductor"] = [C.formula_text(B.parse_key(k)) + f" (Curie {curie[k]:.0f} K)" for k in sorted(curie) if label.get(k) == 1 and curie[k] is not None][:25]
    print(json.dumps(report, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(report, indent=1) + "\n")
else:
    import csv, json, sys
    from pathlib import Path
    import numpy as np
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
    import build_mp_candidates as M, build_scratchnet_data as B, scratchnet_common as C, novelty_check as N, lockbox_blend as LB
    held, valid = N.held_compositions()
    csv.field_size_limit(10**9)
    curie = {}
    with open(Path(sys.argv[1]) / "FM_with_curie.csv", newline="", encoding="utf-8", errors="replace") as h:
        for row in csv.DictReader(h):
            fr = M.fractions(row["Normalized_Composition"])
            if fr and set(fr) <= valid and len(fr) <= C.MAX_ELEMENTS and row.get("Mean_TC_K"):
                curie[B.composition_key(fr)] = float(row["Mean_TC_K"])
    new = {k: B.parse_key(k) for k, t in curie.items() if t >= 50}
    where = N.classify(held, new)
    keys = sorted(k for k in new if where[k] not in ("held", "near twin"))
    rng = np.random.default_rng(0)
    keys = [keys[i] for i in sorted(rng.choice(len(keys), size=min(3000, len(keys)), replace=False))]
    X, _ = C.formula_features([C.formula_text(new[k]) for k in keys])
    p, tc = LB.record_predict(X)
    fam = np.array([M.family(set(new[k])) for k in keys])
    out = {"ferromagnets_scored": len(keys), "mean_p": float(p.mean()), "median_p": float(np.median(p)),
           "share_p_at_least_0.5": float((p >= 0.5).mean()), "share_p_at_least_0.8": float((p >= 0.8).mean()),
           "by_family": {f: {"n": int((fam == f).sum()), "mean_p": float(p[fam == f].mean()), "share_p_at_least_0.5": float((p[fam == f] >= 0.5).mean())} for f in sorted(set(fam))},
           "highest": [(C.formula_text(new[keys[i]]), round(float(p[i]), 3), round(curie[keys[i]])) for i in np.argsort(-p)[:15]]}
    # for scale: cycle 6's own out-of-fold probabilities on the development rows it may read
    ds = C.load_dataset()
    with np.load(ROOT / "data/processed/learning-cycles/cycle6-oof.npz") as o:
        oof = {str(k): float(v) for k, v in zip(o["compositions"], o["p"])}
    dev = [(oof[r["composition"]], l) for r, s, l in zip(ds.rows, ds.split, ds.arrays.label) if s == "dev" and np.isfinite(l) and r["composition"] in oof]
    out["for_scale_development_rows"] = {"mean_p_superconductors": float(np.mean([a for a, l in dev if l == 1])), "mean_p_nulls": float(np.mean([a for a, l in dev if l == 0])),
                                         "share_of_nulls_p_at_least_0.5": float(np.mean([a >= 0.5 for a, l in dev if l == 0]))}
    print(json.dumps(out, indent=1)); Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
