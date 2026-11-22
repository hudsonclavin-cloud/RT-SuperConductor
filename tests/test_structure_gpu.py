"""The structure bundle, the crystal-graph trainer and the overnight driver, on a small synthetic table."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("scipy")
pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402

import local_gpu as G  # noqa: E402
import structure_gpu as SG  # noqa: E402
import structure_gpu_train as ST  # noqa: E402
from rt_superconductor.scratchnet import data as D  # noqa: E402

ELEMENTS = ["Al", "Ga", "Nb", "Sn", "Ti", "V"]


def fake_world(seed=0):
    """A table of 60 binary compositions (dev in 3 folds, train-only, locked), their structures, and 12 MP entries."""
    rng = np.random.default_rng(seed)
    pairs = [(a, b) for i, a in enumerate(ELEMENTS) for b in ELEMENTS[i + 1:]]
    ratios = [(1, 1), (1, 2), (1, 3), (2, 3)]
    formulas = [f"{a}{x}{b}{y}" for a, b in pairs for x, y in ratios]
    keys = [D.composition_key(D.parse_formula(f)) for f in formulas]
    n = len(keys)
    split = np.array(["dev"] * 36 + ["train_only"] * 12 + ["lockbox"] * (n - 48))
    fold = np.array([i % 3 if i < 36 else -1 for i in range(n)])
    group = np.array([f"g{i // 2}" for i in range(n)])
    label = rng.integers(0, 2, n).astype(float)
    tc = np.where(label == 1, rng.uniform(1, 20, n), np.nan)
    symbols = ELEMENTS
    table = SimpleNamespace(symbols=symbols, index={s: i + 1 for i, s in enumerate(symbols)},
                            features=np.vstack([np.zeros((1, 3)), rng.normal(size=(len(symbols), 3))]).astype(np.float32))
    ds = SimpleNamespace(rows=[{"composition": k, "split": s} for k, s in zip(keys, split)], split=split, fold=fold,
                         arrays=SimpleNamespace(label=label, tc_k=tc, group=group), table=table,
                         family=np.array(["Other"] * n))
    X = rng.normal(size=(n, 5))
    X[::7, 1] = np.nan

    def structure_for(formula, a=3.1):
        comp = D.parse_formula(formula)
        (e1, f1), (e2, f2) = sorted(comp.items())
        return Structure(Lattice.cubic(a), [e1, {e1: 0.5, e2: 0.5}], [[0, 0, 0], [0.5, 0.5, 0.5]])

    structures = {}
    three_d = []
    for i, f in enumerate(formulas):
        if i % 5 == 4:
            continue                                         # some compositions have no structure
        copies = 2 if i % 6 == 0 else 1
        for c in range(copies):
            name = f"{f}-{c}"
            structures[name] = structure_for(f, 3.0 + 0.1 * c)
            three_d.append({"formula_sc": f, "orig_formula_cif": f, "cif": name})
    three_d.append({"formula_sc": "Nb3Ge", "orig_formula_cif": "Nb3Ge", "cif": "missing"})   # not in the table
    mp_ids = [f"mp-{i}" for i in range(12)]
    mp_structures = [Structure(Lattice.cubic(3.0 + 0.05 * i), [ELEMENTS[i % 6]], [[0, 0, 0]]) for i in range(12)]
    locked_formula = formulas[50]
    mp_table = {m: {"formula": f"{ELEMENTS[i % 6]}" if i != 11 else locked_formula, "formation_energy_ev_per_atom": str(-0.1 * i),
                    "band_gap_ev": "0.0", "energy_above_hull_ev_per_atom": "0.0", "is_metal": "True",
                    "is_magnetic": "False"} for i, m in enumerate(mp_ids)}
    mp_X = rng.normal(size=(12, 5))
    candidates = {"pool": mp_ids[:5]}
    return ds, X, three_d, structures, mp_ids, mp_structures, mp_table, mp_X, candidates


def build(world=None):
    ds, X, three_d, structures, mp_ids, mp_structures, mp_table, mp_X, candidates = world or fake_world()
    bundle, report = SG.build(ds, X, [f"f{i}" for i in range(5)], three_d, lambda row: structures[row["cif"]],
                              mp_ids, mp_structures, mp_table, mp_X, candidates)
    return ds, X, bundle, report


def test_structure_bundle_holds_no_locked_rows_or_their_structures():
    ds, _, bundle, report = build()
    assert "lockbox" not in set(bundle["comp_split"].tolist())
    locked = {r["composition"] for r in ds.rows if r["split"] == "lockbox"}
    assert not locked & set(bundle["comp_keys"].tolist())
    assert report["3dsc_locked"] > 0 and report["3dsc_not_in_table"] == 1
    assert "mp-11" not in set(bundle["pre_mp_ids"].tolist())          # its formula is a locked composition
    assert report["mp_left_out_of_pretraining_as_locked"] >= 1
    assert np.allclose(np.bincount(bundle["train_comp"], weights=bundle["train_weight"]), 1.0)


def test_structure_folds_are_cycle6_rows_restricted_to_compositions_with_structures():
    ds, _, bundle, _ = build()
    keys = bundle["comp_keys"]
    for fold in bundle["folds"]:
        fit, valid, test = (set(bundle[f"fold{fold}_{p}"].tolist()) for p in ("fit", "valid", "test"))
        assert not fit & test and not valid & test and not fit & valid
        expected = {r["composition"] for r, s, f in zip(ds.rows, ds.split, ds.fold) if s == "dev" and f == fold}
        assert {str(keys[i]) for i in test} == expected & set(keys.tolist())


def test_graph_edges_are_the_nearest_neighbours_and_local():
    s = Structure(Lattice.cubic(3.0), ["Nb"], [[0, 0, 0]])
    g = SG.graph_of(s, {"Nb": 1})
    assert len(g["src"]) == 12 and set(g["src"].tolist()) == {0} and set(g["dst"].tolist()) == {0}
    dist = np.sort(g["dist"].astype(float))
    assert np.allclose(dist[:6], 3.0, atol=0.01) and np.allclose(dist[6:], 3.0 * np.sqrt(2), atol=0.01)
    mixed = Structure(Lattice.cubic(3.0), [{"Nb": 0.7, "Ti": 0.3}], [[0, 0, 0]])
    g = SG.graph_of(mixed, {"Nb": 1, "Ti": 2})
    assert g["species"][0].tolist()[:2] == [1, 2] and np.allclose(g["occ"][0][:2].astype(float), [0.7, 0.3], atol=1e-3)


def test_batches_put_each_graphs_edges_inside_its_own_nodes():
    torch = pytest.importorskip("torch")
    _, _, bundle, _ = build()
    graphs = ST.Graphs(torch, bundle, torch.device("cpu"))
    b = graphs.batch(torch.tensor([3, 0, 7]))
    sizes = [int(bundle["node_offsets"][g + 1] - bundle["node_offsets"][g]) for g in (3, 0, 7)]
    assert torch.bincount(b["node_graph"]).tolist() == sizes
    assert bool((b["node_graph"][b["src"]] == b["node_graph"][b["dst"]]).all())


def write_bundles(tmp_path):
    ds, X, bundle, _ = build()
    structures = tmp_path / "structures.npz"
    np.savez_compressed(structures, **bundle)
    extra = {"keys": ["Al:0.500;Nb:0.500x"], "X": np.zeros((1, 5)), "label": np.ones(1), "tc_k": np.ones(1) * 3,
             "fold": [0], "tied": [set()]}
    formula = tmp_path / "bundle.npz"
    np.savez_compressed(formula, **G.build_bundle(ds, X, extra))
    return ds, structures, formula


@pytest.mark.parametrize("model", ["cgcnn", "hybrid", "formula", "hybrid_all", "formula_all"])
def test_trainer_runs_every_model_on_cpu(tmp_path, model):
    pytest.importorskip("torch")
    ds, structures, formula = write_bundles(tmp_path)
    out = tmp_path / model
    argv = ["train", str(structures), "--model", model, "--out", str(out), "--device", "cpu", "--seeds", "1",
            "--epochs", "2", "--patience", "1", "--hidden", "16", "--embed", "8", "--head", "16", "--x-hidden", "16",
            "--conv", "2", "--batch", "16"]
    if model.endswith("_all"):
        argv += ["--formula-bundle", str(formula)]
    assert ST.main(argv) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["complete"] and run["bundle_kind"] == "structure" and run["mean_validation_bce"] > 0
    with np.load(out / "oof.npz") as oof:
        keys = set(oof["compositions"].tolist())
        assert np.all((oof["p"] >= 0) & (oof["p"] <= 1))
    dev = {r["composition"] for r, s in zip(ds.rows, ds.split) if s == "dev"}
    assert keys <= dev
    if model.endswith("_all"):
        assert keys == dev                                   # every development row, as cycle 6
    with np.load(out / "candidates_pool.npz") as pool:
        assert len(pool["keys"]) == 5


def test_pretraining_then_fine_tuning(tmp_path):
    pytest.importorskip("torch")
    _, structures, formula = write_bundles(tmp_path)
    small = ["--hidden", "16", "--embed", "8", "--head", "16", "--conv", "2", "--batch", "4"]
    assert ST.main(["pretrain", str(structures), "--out", str(tmp_path / "pre"), "--device", "cpu", "--epochs", "2",
                    *small]) == 0
    assert (tmp_path / "pre" / "encoder.pt").is_file()
    assert ST.main(["train", str(structures), "--model", "hybrid_all", "--formula-bundle", str(formula),
                    "--pretrained", str(tmp_path / "pre" / "encoder.pt"), "--out", str(tmp_path / "ft"), "--device", "cpu",
                    "--seeds", "1", "--epochs", "1", "--patience", "1", "--x-hidden", "16", "--head", "16"]) == 0
    run = json.loads((tmp_path / "ft" / "run.json").read_text())
    assert run["pretrain_config"]["hidden"] == 16 and run["settings"]["hidden"] == 16


def test_trainer_refuses_locked_rows(tmp_path):
    pytest.importorskip("torch")
    _, _, bundle, _ = build()
    bundle["comp_split"] = bundle["comp_split"].copy()
    bundle["comp_split"][0] = "lockbox"
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **bundle)
    with pytest.raises(SystemExit):
        ST.main(["train", str(path), "--model", "cgcnn", "--out", str(tmp_path / "x"), "--device", "cpu"])


def test_overnight_driver_runs_its_smoke_profile_end_to_end(tmp_path):
    pytest.importorskip("torch")
    import overnight_gpu as O

    _, structures, formula = write_bundles(tmp_path)
    out = tmp_path / "night"
    code = O.main(["--out", str(out), "--profile", "smoke", "--device", "cpu", "--hours", "1",
                   "--structures", str(structures), "--formula", str(formula)])
    summary = json.loads((out / "share" / "summary.json").read_text())
    status = {j["name"]: j["status"] for j in summary["jobs"]}
    assert code == 0, status
    for name in ("check", "tab-default-s20", "A-formula_all", "A-hybrid_all", "pretrain", "A-hybrid_all-pre",
                 "S-cgcnn", "final-A", "final-S"):
        assert status.get(name) == "done", (name, status)
    assert (out / "share" / "final-A" / "oof.npz").is_file()
    assert not list((out / "share").rglob("*.pt"))           # no model weights in what is sent back
    # A second run skips everything already finished.
    assert O.main(["--out", str(out), "--profile", "smoke", "--device", "cpu", "--hours", "1",
                   "--structures", str(structures), "--formula", str(formula)]) == 0
    assert "finished in an earlier session" in (out / "driver.log").read_text()


def test_blend_and_paired_comparison_of_identical_predictions():
    ds, _, bundle, _ = build()
    dev = [r["composition"] for r, s in zip(ds.rows, ds.split) if s == "dev"]
    rng = np.random.default_rng(4)
    preds = {k: (float(rng.uniform()), float(rng.uniform(1, 9))) for k in dev}
    out = SG.paired(ds, preds, preds, "a", "b")
    gap = out["not cuprates"]["difference"]["auc"]
    assert gap["low"] == pytest.approx(0.0) and gap["high"] == pytest.approx(0.0)
    mix = SG.blend_of(preds, preds)
    assert all(abs(mix[k][0] - preds[k][0]) < 1e-12 for k in preds)
