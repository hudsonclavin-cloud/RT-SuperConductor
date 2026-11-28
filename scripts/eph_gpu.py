#!/usr/bin/env python3
"""Overnight GPU run: learn the calculated electron-phonon coupling from crystal structures (plan: docs/eph-surrogate.md).

The targets are calculations, not measurements: Alexandria's 2025.08.11 phonon release (QE DFPT, PBEsol, CC BY 4.0),
reduced by scripts/alexandria_eph_2025.py. A network that reproduces them cheaply can rank the 27,000 Materials Project
metals by what a Hazel calculation would probably say, before any Hazel time is spent.

    python scripts/eph_gpu.py export                                   # here: writes data/processed/local-gpu/eph.npz
    python scripts/eph_gpu.py train data/processed/local-gpu/eph.npz --out runs/eph-YYYYMMDD --hours 6 --device cuda
    python scripts/eph_gpu.py train ... --profile smoke --device cpu   # two tiny members, about a minute

Rows: dynamically stable entries with a Tc, without the flags of docs/alexandria-eph-2025.json (lambda > 5 or
omega_log < 100 K). Targets: lambda, ln(omega_log / K) and ln(1 + Tc), Tc by Allen-Dynes at mu* = 0.10.
Split, fixed at export: whole chemical systems (element sets) go to fit (75%), validation (10%) or test (15%), by a
hash of the element set, so no test compound has a sibling of the same elements in training.

The train command trains networks one after another until the time is used: first different settings, then more seeds
of the three best. Members are chosen on validation rows only. The five best are averaged; the test rows are scored once,
at the end, and the candidates are predicted (in a smoke run, validation rows stand in for the test rows). Each finished member is saved, so a stopped run resumes where it was.
Outputs in --out/share (predictions and logs, no weights) are what to commit: copy them to data/benchmarks/local-gpu/.

Time is counted as training time, not wall time: the script asks Windows not to sleep, and if the machine sleeps anyway
(a closed lid), the gap is charged as one ordinary epoch, so the budget is not used up while nothing trains. Keep the
lid open, or set "When I close the lid" to "Do nothing" while plugged in. A finished run (share/result.json) refuses to
train or score again: the test rows are scored once.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

EPH = ROOT / "data/processed/alexandria-eph-2025"
STRUCTURE_BUNDLE = ROOT / "data/processed/local-gpu/structures.npz"   # its graph store and candidate lists are reused
OUT = ROOT / "data/processed/local-gpu/eph.npz"
TARGETS = ["lambda", "ln_wlog_k", "ln1p_tc_ad_mu010"]
TC = 2                                                   # column of ln(1 + Tc): what members are chosen on
ENSEMBLE = 5
GRID = {"conv": [3, 4, 5, 6], "hidden": [128, 192, 256], "lr": [1e-3, 5e-4, 2e-3], "batch": [64, 128, 256],
        "dropout": [0.0, 0.1], "weight_decay": [1e-5, 1e-4]}


def split_of(composition: str) -> str:
    """fit, valid or test for a whole element set, from a hash, so the split never depends on row order."""
    elements = ",".join(sorted(part.split(":")[0] for part in composition.split(";")))
    u = int(hashlib.sha256(elements.encode()).hexdigest()[:12], 16) / 16 ** 12
    return "test" if u < 0.15 else "valid" if u < 0.25 else "fit"


# ---------------------------------------------------------------------------
# Export (needs pymatgen; runs where the Alexandria reduction is)

def _graph(args):
    import structure_gpu as G
    from pymatgen.core import Structure
    structure, index = args
    try:
        return G.graph_of(Structure.from_dict(structure), index)
    except (KeyError, ValueError):
        return None


def cmd_export(workers: int) -> dict:
    import csv
    import gzip
    from multiprocessing import Pool

    import structure_gpu as G

    old = dict(np.load(STRUCTURE_BUNDLE, allow_pickle=False))
    symbols = [str(s) for s in old["elem_symbols"]]
    index = {s: i for i, s in enumerate(symbols)}
    rows = {}
    for r in csv.DictReader(gzip.open(EPH / "table.csv.gz", "rt", newline="", encoding="utf-8")):
        if r["imag"] != "False" or not (r["lambda"] and r["wlog_k"] and r["tc_allen_dynes_mu010"]):
            continue
        lam, wlog, tc = float(r["lambda"]), float(r["wlog_k"]), float(r["tc_allen_dynes_mu010"])
        if lam > 5 or wlog < 100:
            continue
        rows[r["mat_id"]] = (r["composition"], [lam, math.log(wlog), math.log1p(max(tc, 0.0))])
    structures = {}
    for path in sorted(EPH.glob("structures-*.jsonl.gz")):
        for line in gzip.open(path, "rt", encoding="utf-8"):
            d = json.loads(line)
            if d["mat_id"] in rows:
                structures[d["mat_id"]] = d["structure"]
    ids = sorted(set(rows) & set(structures))
    with Pool(workers) as pool:
        made = pool.map(_graph, [(structures[i], index) for i in ids], chunksize=50)
    kept = [(i, g) for i, g in zip(ids, made) if g is not None]
    n_old = len(old["graph_sites"])
    old_graphs = [{"species": old["node_species"][old["node_offsets"][k]:old["node_offsets"][k + 1]],
                   "occ": old["node_occ"][old["node_offsets"][k]:old["node_offsets"][k + 1]],
                   "src": old["edge_src"][old["edge_offsets"][k]:old["edge_offsets"][k + 1]],
                   "dst": old["edge_dst"][old["edge_offsets"][k]:old["edge_offsets"][k + 1]],
                   "dist": old["edge_dist"][old["edge_offsets"][k]:old["edge_offsets"][k + 1]],
                   "volume_per_atom": old["graph_volume_per_atom"][k], "sites": old["graph_sites"][k]}
                  for k in range(n_old)]
    bundle = G.stack_graphs(old_graphs + [g for _, g in kept])
    bundle.update({k: v for k, v in old.items() if k.startswith("cand_") and not k.endswith("_X")})
    bundle.update({"elem_features": old["elem_features"], "elem_symbols": old["elem_symbols"],
                   "cutoff_angstrom": old["cutoff_angstrom"], "max_neighbors": old["max_neighbors"],
                   "eph_graph": np.arange(n_old, n_old + len(kept), dtype=np.int32),
                   "eph_ids": np.array([i for i, _ in kept]),
                   "eph_composition": np.array([rows[i][0] for i, _ in kept]),
                   "eph_targets": np.array([rows[i][1] for i, _ in kept], np.float32),
                   "eph_split": np.array([split_of(rows[i][0]) for i, _ in kept]),
                   "eph_target_names": np.array(TARGETS)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **bundle)
    split = bundle["eph_split"]
    tc = np.expm1(bundle["eph_targets"][:, TC])
    report = {"rows": len(kept), "without_graph": len(ids) - len(kept), "missing_structure": len(rows) - len(ids),
              "split": {s: int((split == s).sum()) for s in ("fit", "valid", "test")},
              "test_tc_at_least_10k": int(((split == "test") & (tc >= 10)).sum()),
              "test_mae_of_fit_median_k": round(float(np.mean(np.abs(tc[split == "test"] - np.median(tc[split == "fit"])))), 3),
              "candidates": {k[5:-6]: int(len(v)) for k, v in bundle.items() if k.startswith("cand_") and k.endswith("_graph")},
              "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    print(json.dumps(report, indent=1))
    return report


# ---------------------------------------------------------------------------
# Training (numpy and PyTorch only; runs on the owner's GPU)

def ranks(x: np.ndarray) -> np.ndarray:
    """0-based ranks, ties given their average rank."""
    order = np.argsort(x, kind="stable")
    xs, r, i = x[order], np.empty(len(x)), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def auc(y: np.ndarray, score: np.ndarray) -> float:
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks(score)[y == 1].sum() - pos * (pos - 1) / 2) / (pos * neg))


def configs(seed: int) -> list[dict]:
    """Every combination of GRID once, in a seeded random order."""
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    order = np.random.default_rng(seed).permutation(len(combos))
    return [combos[k] for k in order]


def keep_awake(on: bool) -> None:
    """Windows only: ask the system not to sleep while this runs (the display may still turn off)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:  # noqa: BLE001 - never let this stop a run
        pass


class Clock:
    """Training time spent, across sessions. A step far longer than the usual epoch (the machine slept) counts as one
    usual epoch, so a sleep cannot use up the budget."""

    def __init__(self, spent: float, budget: float, say=print):
        self.spent, self.budget, self.say = spent, budget, say
        self.last, self.epochs = time.time(), []

    def charge(self, dt: float, epoch: bool) -> float:
        typical = float(np.median(self.epochs[-50:])) if len(self.epochs) >= 5 else None
        if typical is not None and dt > max(10 * typical, 120.0):
            self.say(f"a {dt / 60:.1f} min gap (the machine slept?) is counted as {typical:.0f} s")
            dt = typical
        if epoch:
            self.epochs.append(dt)
        self.spent += dt
        return dt

    def tick(self, epoch: bool = False) -> float:
        now = time.time()
        dt, self.last = now - self.last, now
        return self.charge(dt, epoch)

    def left(self) -> float:
        return self.budget - self.spent


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def train_member(torch, S, graphs, elem, cfg, seed, data, max_epochs, patience, clock, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ns = SimpleNamespace(conv=cfg["conv"], hidden=cfg["hidden"], embed=64, rbf=41, cutoff=8.0)
    encoder = S.build_encoder(torch, elem, ns).to(device)
    head = torch.nn.Sequential(torch.nn.Linear(encoder.out_dim, 128), torch.nn.SiLU(), torch.nn.Dropout(cfg["dropout"]),
                               torch.nn.Linear(128, len(TARGETS))).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    G, Y = data["G"], data["Y"]

    def predict(idx):
        encoder.eval(); head.eval()
        with torch.no_grad():
            return torch.cat([head(encoder(graphs.batch(G[idx[s:s + 1024]]))) for s in range(0, len(idx), 1024)])

    best, best_tc, best_state, best_epoch, waited, epoch = float("inf"), float("inf"), None, 0, 0, 0
    for epoch in range(1, max_epochs + 1):
        encoder.train(); head.train()
        perm = torch.as_tensor(rng.permutation(data["fit"]), device=device)
        for s in range(0, len(perm), cfg["batch"]):
            idx = perm[s:s + cfg["batch"]]
            if len(idx) < 2:
                continue
            loss = torch.nn.functional.mse_loss(head(encoder(graphs.batch(G[idx]))), Y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
        out = predict(data["valid_t"])
        err = (out - Y[data["valid_t"]]) ** 2
        val, val_tc = float(err.mean()), float(err[:, TC].mean())
        if val < best - 1e-5:
            best, best_tc, best_epoch, waited = val, val_tc, epoch, 0
            best_state = {"encoder": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                          "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
        else:
            waited += 1
        clock.tick(epoch=True)
        if waited >= patience or clock.left() < 300:
            break
    return {"best_state": best_state, "valid_mse": round(best, 5), "valid_mse_tc": round(best_tc, 5),
            "best_epoch": best_epoch, "epochs_run": epoch}, (encoder, head)


def cmd_train(args) -> int:
    import torch

    import structure_gpu_train as S

    device = S.pick_device(torch, args.device)
    b = np.load(args.bundle, allow_pickle=False)
    sha = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    if args.expect_sha256 and sha != args.expect_sha256.strip().lower():
        raise SystemExit(f"bundle sha256 {sha} is not the expected {args.expect_sha256}")
    graphs = S.Graphs(torch, b, device)
    elem = b["elem_features"]
    split, raw = b["eph_split"], b["eph_targets"].astype(np.float64)
    fit, valid, test = (np.flatnonzero(split == s) for s in ("fit", "valid", "test"))
    smoke = args.profile == "smoke"
    if smoke:                                    # a rehearsal never touches the test rows: it scores validation rows
        fit, valid = fit[:400], valid[:100]
        test = valid
    if (args.out / "share" / "result.json").is_file():
        raise SystemExit(f"{args.out} is finished (share/result.json exists): the test rows are scored once")
    mean, std = raw[fit].mean(axis=0), raw[fit].std(axis=0)
    data = {"G": torch.as_tensor(b["eph_graph"].astype(np.int64), device=device),
            "Y": torch.as_tensor(((raw - mean) / std).astype(np.float32), device=device),
            "fit": fit, "valid_t": torch.as_tensor(valid, device=device)}
    members_dir = args.out / "members"
    members_dir.mkdir(parents=True, exist_ok=True)
    share = args.out / "share"
    share.mkdir(exist_ok=True)
    log_path = share / "log.txt"

    def say(message: str) -> None:
        S.log(message)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(time.strftime("%Y-%m-%d %H:%M:%S ") + message + "\n")

    state_path = args.out / "state.json"
    settings = {"profile": args.profile, "bundle_sha256": sha, "seed": args.seed, "epochs": args.epochs,
                "patience": args.patience}
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"spent_s": 0.0, "settings": settings}
    if state.get("settings") != settings:                           # a resume must continue the same run
        raise SystemExit(f"{args.out} was started with {state.get('settings')}, not {settings}: use a new --out")
    budget = args.hours * 3600.0
    max_epochs, patience, max_members = (2, 1, 2) if smoke else (args.epochs, args.patience, 10 ** 6)
    grid = configs(args.seed)
    say(f"device {device}; {len(fit)} fit, {len(valid)} validation, {len(test)} test rows; budget {args.hours} h, "
        f"{state['spent_s'] / 3600:.2f} h already spent")

    def done_members():
        found = []
        for path in members_dir.glob("*.json"):
            try:
                m = json.loads(path.read_text())
                if (members_dir / f"{m['index']:03d}.pt").is_file():
                    found.append(m)
                    continue
            except (ValueError, KeyError):
                pass
            say(f"{path.name} is unreadable or has no weights: that member is trained again")
        return sorted(found, key=lambda m: m["index"])

    clock = Clock(state["spent_s"], budget, say)
    while True:
        clock.tick()
        members = done_members()
        typical = float(np.median([m["minutes"] for m in members])) * 60 if members else 0.0
        if len(members) >= max_members or clock.spent + 1.2 * typical + 600 > budget:
            break
        done = {m["index"] for m in members}
        i = min(set(range(len(members) + 1)) - done)
        if clock.spent < 0.5 * budget or len(members) < 3:          # explore settings, one seed each
            cfg, phase = grid[i % len(grid)], "explore"
        else:                                                       # more seeds of the three best settings
            explored = sorted((m for m in members if m["phase"] == "explore"), key=lambda m: m["valid_mse_tc"])[:3]
            cfg, phase = explored[i % len(explored)]["config"], "seeds"
        before = clock.spent
        result, _ = train_member(torch, S, graphs, elem, cfg, args.seed * 1000 + i, data, max_epochs, patience,
                                 clock, device)
        tmp = members_dir / f"{i:03d}.pt.tmp"
        torch.save(result.pop("best_state"), tmp)
        os.replace(tmp, members_dir / f"{i:03d}.pt")
        clock.tick()
        record = {"index": i, "phase": phase, "config": cfg, **result, "minutes": round((clock.spent - before) / 60, 2)}
        write_atomic(members_dir / f"{i:03d}.json", json.dumps(record) + "\n")
        state["spent_s"] = clock.spent
        write_atomic(state_path, json.dumps(state) + "\n")
        say(f"member {i} ({phase}) {cfg}: validation MSE {result['valid_mse']:.4f}, Tc column "
            f"{result['valid_mse_tc']:.4f}, {record['minutes']} min")

    members = done_members()
    if not members:
        raise SystemExit("no member finished")
    chosen = sorted(members, key=lambda m: m["valid_mse_tc"])[:ENSEMBLE]

    def ensemble(gids: np.ndarray) -> np.ndarray:
        outs = []
        for m in chosen:
            ns = SimpleNamespace(conv=m["config"]["conv"], hidden=m["config"]["hidden"], embed=64, rbf=41, cutoff=8.0)
            enc = S.build_encoder(torch, elem, ns).to(device)
            head = torch.nn.Sequential(torch.nn.Linear(enc.out_dim, 128), torch.nn.SiLU(),
                                       torch.nn.Dropout(m["config"]["dropout"]), torch.nn.Linear(128, len(TARGETS))).to(device)
            saved = torch.load(members_dir / f"{m['index']:03d}.pt", map_location=device)
            enc.load_state_dict(saved["encoder"]); head.load_state_dict(saved["head"])
            enc.eval(); head.eval()
            G = torch.as_tensor(gids.astype(np.int64), device=device)
            with torch.no_grad():
                out = torch.cat([head(enc(graphs.batch(G[s:s + 1024]))) for s in range(0, len(G), 1024)])
            outs.append(out.cpu().numpy() * std + mean)
        return np.stack(outs)                                       # members x rows x targets, in target units

    test_out = ensemble(b["eph_graph"][test])                       # the one scoring of the test rows
    pred, spread = test_out.mean(axis=0), test_out[:, :, TC].std(axis=0)
    tc_true, tc_pred = np.expm1(raw[test, TC]), np.clip(np.expm1(pred[:, TC]), 0.0, None)   # Tc is never negative
    tc_fit_median = float(np.median(np.expm1(raw[fit, TC])))
    result = {"plan": "docs/eph-surrogate.md", "bundle_sha256": sha, "profile": args.profile, **S.device_info(torch, device),
              "rows": {"fit": int(len(fit)), "valid": int(len(valid)), "test": int(len(test))},
              "members_trained": len(members), "ensemble": [m["index"] for m in chosen],
              "hours_spent": round(json.loads(state_path.read_text())["spent_s"] / 3600, 2),
              "test": {"tc_mae_k": round(float(np.mean(np.abs(tc_pred - tc_true))), 3),
                       "tc_mae_k_fit_median_baseline": round(float(np.mean(np.abs(tc_fit_median - tc_true))), 3),
                       "tc_spearman": round(float(np.corrcoef(ranks(tc_pred), ranks(tc_true))[0, 1]), 4),
                       "auc_tc_at_least_10k": round(auc((tc_true >= 10).astype(float), tc_pred), 4),
                       "auc_tc_at_least_20k": round(auc((tc_true >= 20).astype(float), tc_pred), 4),
                       "lambda_mae": round(float(np.mean(np.abs(pred[:, 0] - raw[test, 0]))), 4),
                       "ln_wlog_mae": round(float(np.mean(np.abs(pred[:, 1] - raw[test, 1]))), 4)}}
    names = np.array(TARGETS)
    np.savez_compressed(share / "test_predictions.npz", ids=b["eph_ids"][test], true=raw[test], pred=pred,
                        ln1p_tc_spread=spread, target_names=names)
    # For marking candidates unlike the training rows: their size, and how often their rarest element is in fit rows
    node_graph = np.repeat(np.arange(len(b["graph_sites"])), np.diff(b["node_offsets"]))
    present = b["node_occ"].astype(np.float32) > 0
    pairs = np.unique(np.stack([np.repeat(node_graph, 4)[present.ravel()], b["node_species"].ravel()[present.ravel()]], 1), axis=0)
    fit_graphs = np.zeros(len(b["graph_sites"]), bool)
    fit_graphs[b["eph_graph"][fit]] = True
    element_fit_count = np.bincount(pairs[fit_graphs[pairs[:, 0]], 1], minlength=len(elem))
    result["fit_max_sites"] = int(b["graph_sites"][b["eph_graph"][fit]].max())
    for key in (k for k in b.files if k.startswith("cand_") and k.endswith("_graph")):
        name = key[5:-6]
        gids = b[key][:200] if smoke else b[key]
        out = ensemble(gids)
        rarest = np.full(len(gids), np.iinfo(np.int64).max)
        mine = np.isin(pairs[:, 0], gids)
        position = {g: k for k, g in enumerate(gids)}
        for g, species in pairs[mine]:
            k = position[g]
            rarest[k] = min(rarest[k], element_fit_count[species])
        np.savez_compressed(share / f"candidates_{name}.npz", keys=b[f"cand_{name}_keys"][:len(gids)], pred=out.mean(axis=0),
                            ln1p_tc_spread=out[:, :, TC].std(axis=0), target_names=names,
                            sites=b["graph_sites"][gids], rarest_element_fit_graphs=rarest)
        result[f"candidates_{name}"] = int(len(gids))
    write_atomic(share / "members.json", json.dumps(members, indent=1) + "\n")
    write_atomic(share / "result.json", json.dumps(result, indent=1) + "\n")
    say(json.dumps(result["test"]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--workers", type=int, default=3)
    tr = sub.add_parser("train")
    tr.add_argument("bundle", type=Path)
    tr.add_argument("--out", type=Path, required=True)
    tr.add_argument("--hours", type=float, default=6.0)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--profile", choices=["full", "smoke"], default="full")
    tr.add_argument("--epochs", type=int, default=300)
    tr.add_argument("--patience", type=int, default=30)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--expect-sha256", default="")
    args = parser.parse_args(argv)
    if args.command == "export":
        cmd_export(args.workers)
        return 0
    keep_awake(True)
    try:
        return cmd_train(args)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    raise SystemExit(main())
