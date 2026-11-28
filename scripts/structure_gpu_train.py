#!/usr/bin/env python3
"""Crystal-graph networks for superconductivity, trained on a local GPU (``docs/overnight-gpu.md``).

Self-contained: numpy and PyTorch only, so it runs on any machine. Input is the bundle written by
``python scripts/structure_gpu.py export`` (and, for the ``*_all`` models, ``python scripts/local_gpu.py export``).

    python structure_gpu_train.py pretrain structures.npz --out runs/x/pretrain --device cuda
    python structure_gpu_train.py train structures.npz --model hybrid --out runs/x/hybrid --device cuda
    python structure_gpu_train.py train structures.npz --model hybrid_all --formula-bundle bundle.npz \
        --pretrained runs/x/pretrain/encoder.pt --out runs/x/hybrid_all_pre --device cuda

Models (the encoder is a CGCNN: Xie and Grossman, Phys. Rev. Lett. 120, 145301, 2018):

    cgcnn        structure only, on the compositions that have a 3DSC structure
    hybrid       structure plus the 155 formula features, same rows
    formula      formula features only, same rows and same code: the control for "does structure add anything"
    hybrid_all   every row of the formula bundle (as cycle 6): the structure where one exists, a learned
                 "no structure" vector elsewhere
    formula_all  every row of the formula bundle, no structure: the control for hybrid_all

Each development fold uses exactly the fit, early-stopping and test rows of cycle 6. Locked rows are not in either
bundle. Outputs in ``--out``: ``oof.npz`` (compositions, p, tc), ``candidates_<name>.npz`` and ``run.json``, whose
``mean_validation_bce`` (the early-stopping rows only) is what the overnight driver selects on. Test rows are
predicted but never scored here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

GRAPH_MODELS = ("cgcnn", "hybrid", "hybrid_all")
ALL_ROW_MODELS = ("hybrid_all", "formula_all")
MODELS = ("cgcnn", "hybrid", "formula", "hybrid_all", "formula_all")


# ---------------------------------------------------------------------------
# Small helpers shared with local_gpu_train.py (copied so this file stands alone)

def content_sha256(bundle) -> str:
    """A checksum of the bundle's arrays, not of the file, so the same export made twice gives the same value."""
    digest = hashlib.sha256()
    for key in sorted(bundle.files):
        value = np.ascontiguousarray(bundle[key])
        digest.update(f"{key}|{value.dtype.str}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def preprocess_fit(X: np.ndarray) -> dict:
    """Median imputation, missingness indicators and standardisation, fitted on the training rows only."""
    finite = np.isfinite(X)
    median = np.nanmedian(np.where(finite, X, np.nan), axis=0) if len(X) else np.zeros(X.shape[1])
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(finite, X, median)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    indicators = np.flatnonzero((~finite).any(axis=0))
    return {"median": median, "mean": mean, "std": std, "indicators": indicators}


def preprocess_apply(X: np.ndarray, prep: dict) -> np.ndarray:
    finite = np.isfinite(X)
    filled = np.where(finite, X, prep["median"])
    scaled = (filled - prep["mean"]) / prep["std"]
    missing = (~finite[:, prep["indicators"]]).astype(np.float32)
    return np.clip(np.hstack([scaled, missing]), -10, 10).astype(np.float32)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ---------------------------------------------------------------------------
# Items: what one training example is. A composition with several structures gives several items.

class Items:
    """Per-item arrays (numpy): composition key, raw formula features, label, Tc, weight, graph id (-1 = none)."""

    def __init__(self, keys, X, label, tc, weight, graph):
        self.keys, self.X, self.label, self.tc = np.asarray(keys), np.asarray(X, np.float32), label, tc
        self.weight, self.graph = np.asarray(weight, np.float32), np.asarray(graph, np.int64)

    def take(self, rows: np.ndarray) -> "Items":
        return Items(self.keys[rows], self.X[rows], self.label[rows], self.tc[rows], self.weight[rows], self.graph[rows])

    def __len__(self) -> int:
        return len(self.keys)


def subset_items(sb) -> tuple[Items, np.ndarray]:
    """Structure-subset models: one item per training graph. Returns the items and each item's composition index."""
    comp = sb["train_comp"].astype(np.int64)
    items = Items(sb["comp_keys"][comp], sb["comp_X"][comp], sb["comp_label"][comp].astype(np.float32),
                  sb["comp_tc_k"][comp].astype(np.float32), sb["train_weight"], sb["train_graph"])
    return items, comp


def all_row_items(sb, fb, use_graphs: bool) -> tuple[Items, np.ndarray]:
    """Every formula-bundle row; rows with structures expand to one item per structure. Returns items and row index."""
    graphs_of: dict[str, list[int]] = {}
    if use_graphs:
        comp_key = sb["comp_keys"]
        for g, c in zip(sb["train_graph"], sb["train_comp"]):
            graphs_of.setdefault(str(comp_key[c]), []).append(int(g))
    keys, X, label, tc = fb["keys"], fb["X"], fb["label"].astype(np.float32), fb["tc_k"].astype(np.float32)
    rows, graph, weight = [], [], []
    for i, key in enumerate(keys):
        mine = graphs_of.get(str(key), [-1])
        rows += [i] * len(mine)
        graph += mine
        weight += [1.0 / len(mine)] * len(mine)
    rows = np.array(rows)
    return Items(keys[rows], X[rows], label[rows], tc[rows], weight, graph), rows


def fold_item_sets(owner: np.ndarray, bundle, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Item numbers of one fold's fit, early-stopping and test sets, from the bundle's row (or composition) numbers."""
    out = []
    for part in ("fit", "valid", "test"):
        member = np.zeros(int(owner.max()) + 1 if len(owner) else 0, bool)
        rows = bundle[f"fold{fold}_{part}"]
        member[rows[rows < len(member)]] = True
        out.append(np.flatnonzero(member[owner]))
    return tuple(out)


# ---------------------------------------------------------------------------
# The graph store on the device, and batches by graph id

class Graphs:
    def __init__(self, torch, bundle, device):
        self.torch = torch
        t = lambda a, dtype: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)  # noqa: E731
        self.node_off, self.edge_off = t(bundle["node_offsets"], torch.long), t(bundle["edge_offsets"], torch.long)
        self.species = t(bundle["node_species"], torch.long)
        self.occ = t(bundle["node_occ"].astype(np.float32), torch.float32)
        self.src, self.dst = t(bundle["edge_src"], torch.long), t(bundle["edge_dst"], torch.long)
        self.dist = t(bundle["edge_dist"].astype(np.float32), torch.float32)
        g = np.stack([np.log(bundle["graph_volume_per_atom"].astype(np.float64)),
                      np.log(bundle["graph_sites"].astype(np.float64))], axis=1)
        g = (g - g.mean(axis=0)) / np.where(g.std(axis=0) > 0, g.std(axis=0), 1.0)
        self.glob = t(g, torch.float32)
        self.device = device

    def batch(self, gids):
        torch = self.torch
        n0, nn_ = self.node_off[gids], self.node_off[gids + 1] - self.node_off[gids]
        e0, ne = self.edge_off[gids], self.edge_off[gids + 1] - self.edge_off[gids]
        ar = torch.arange(len(gids), device=self.device)
        node_graph = torch.repeat_interleave(ar, nn_)
        node_start = torch.cumsum(nn_, 0) - nn_
        n_nodes = int(nn_.sum())
        node_idx = n0[node_graph] + torch.arange(n_nodes, device=self.device) - node_start[node_graph]
        edge_graph = torch.repeat_interleave(ar, ne)
        edge_start = torch.cumsum(ne, 0) - ne
        edge_idx = e0[edge_graph] + torch.arange(int(ne.sum()), device=self.device) - edge_start[edge_graph]
        return {"species": self.species[node_idx], "occ": self.occ[node_idx],
                "src": self.src[edge_idx] + node_start[edge_graph], "dst": self.dst[edge_idx] + node_start[edge_graph],
                "dist": self.dist[edge_idx], "node_graph": node_graph, "n_graphs": len(gids), "glob": self.glob[gids]}


# ---------------------------------------------------------------------------
# Networks

ENCODER_KEYS = ("conv", "hidden", "embed", "rbf", "cutoff")


def build_encoder(torch, elem_features: np.ndarray, args):
    nn, F = torch.nn, torch.nn.functional

    class Conv(nn.Module):
        def __init__(self, h, r):
            super().__init__()
            self.lin = nn.Linear(2 * h + r, 2 * h)
            self.bn1, self.bn2 = nn.BatchNorm1d(2 * h), nn.BatchNorm1d(h)

        def forward(self, h, src, dst, e):
            z = self.bn1(self.lin(torch.cat([h[src], h[dst], e], dim=1)))
            gate, core = z.chunk(2, dim=1)
            message = torch.sigmoid(gate) * F.softplus(core)
            total = torch.zeros_like(h).index_add_(0, src, message)
            return F.softplus(h + self.bn2(total))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("props", torch.as_tensor(elem_features, dtype=torch.float32))
            self.register_buffer("centres", torch.linspace(0.0, float(args.cutoff), int(args.rbf)))
            self.gamma = 1.0 / (float(args.cutoff) / (int(args.rbf) - 1)) ** 2
            self.embed = nn.Embedding(len(elem_features), args.embed)
            self.inp = nn.Linear(args.embed + elem_features.shape[1] + 1, args.hidden)
            self.convs = nn.ModuleList([Conv(args.hidden, int(args.rbf)) for _ in range(args.conv)])
            self.out_dim = args.hidden + 2

        def forward(self, b):
            occ = b["occ"]
            feats = torch.cat([self.embed(b["species"]), self.props[b["species"]]], dim=-1)
            x = (feats * occ.unsqueeze(-1)).sum(dim=1)
            h = self.inp(torch.cat([x, occ.sum(dim=1, keepdim=True)], dim=1))
            e = torch.exp(-self.gamma * (b["dist"].unsqueeze(-1) - self.centres) ** 2)
            for conv in self.convs:
                h = conv(h, b["src"], b["dst"], e)
            pooled = torch.zeros(b["n_graphs"], h.shape[1], device=h.device).index_add_(0, b["node_graph"], h)
            count = torch.zeros(b["n_graphs"], device=h.device).index_add_(
                0, b["node_graph"], torch.ones(len(b["node_graph"]), device=h.device))
            return torch.cat([pooled / count.clamp(min=1).unsqueeze(1), b["glob"]], dim=1)

    return Encoder()


def build_net(torch, model: str, elem_features: np.ndarray, n_x: int, args):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.encoder = build_encoder(torch, elem_features, args) if model in GRAPH_MODELS else None
            width = 0
            if self.encoder is not None:
                width += self.encoder.out_dim
            if model == "hybrid_all":
                self.no_structure = nn.Parameter(torch.zeros(self.encoder.out_dim))
                width += 1                                   # a has-structure flag
            if model != "cgcnn":
                self.xnet = nn.Sequential(nn.Linear(n_x, args.x_hidden), nn.LayerNorm(args.x_hidden), nn.SiLU(),
                                          nn.Dropout(args.dropout), nn.Linear(args.x_hidden, args.x_hidden), nn.SiLU())
                width += args.x_hidden
            else:
                self.xnet = None
            self.head = nn.Sequential(nn.Linear(width, args.head), nn.SiLU(), nn.Dropout(args.dropout))
            self.logit, self.log_tc = nn.Linear(args.head, 1), nn.Linear(args.head, 1)

        def forward(self, x, gid, graphs):
            parts = []
            if self.encoder is not None:
                has = gid >= 0
                if self.model == "hybrid_all":
                    vec = self.no_structure.expand(len(gid), -1).clone()
                    if bool(has.any()):
                        vec[has] = self.encoder(graphs.batch(gid[has]))
                    parts += [vec, has.float().unsqueeze(1)]
                else:
                    parts.append(self.encoder(graphs.batch(gid)))
            if self.xnet is not None:
                parts.append(self.xnet(x))
            h = self.head(torch.cat(parts, dim=1))
            return self.logit(h).squeeze(-1), self.log_tc(h).squeeze(-1)

    return Net()


# ---------------------------------------------------------------------------
# Training

def weighted_bce(torch, logit, y, w):
    keep = torch.isfinite(y)
    if not bool(keep.any()):
        return logit.sum() * 0.0
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit[keep], y[keep], reduction="none")
    return (loss * w[keep]).sum() / w[keep].sum()


def to_device(torch, items: Items, prep: dict, device) -> dict:
    log_tc = np.where((items.label == 1) & np.isfinite(items.tc), np.log1p(np.nan_to_num(items.tc)), np.nan)
    return {"x": torch.as_tensor(preprocess_apply(items.X, prep), device=device),
            "y": torch.as_tensor(items.label.astype(np.float32), device=device),
            "t": torch.as_tensor(log_tc.astype(np.float32), device=device),
            "w": torch.as_tensor(items.weight, device=device),
            "g": torch.as_tensor(items.graph, device=device)}


def predict_items(torch, nets, data: dict, graphs, batch: int) -> tuple[np.ndarray, np.ndarray]:
    ps, ts = [], []
    with torch.no_grad():
        for net in nets:
            net.eval()
            p_parts, t_parts = [], []
            for start in range(0, len(data["x"]), batch):
                sl = slice(start, start + batch)
                logit, log_tc = net(data["x"][sl], data["g"][sl], graphs)
                p_parts.append(torch.sigmoid(logit).float().cpu().numpy())
                t_parts.append(log_tc.float().cpu().numpy())
            ps.append(np.concatenate(p_parts) if p_parts else np.zeros(0))
            ts.append(np.concatenate(t_parts) if t_parts else np.zeros(0))
    return np.mean(ps, axis=0), np.mean(ts, axis=0)


def by_composition(keys: np.ndarray, weight: np.ndarray, p: np.ndarray, log_tc: np.ndarray):
    """Average the items of each composition (weights 1 / number of its structures)."""
    out_keys, out_p, out_t = [], [], []
    order = np.argsort(keys, kind="stable")
    keys, weight, p, log_tc = keys[order], weight[order], p[order], log_tc[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(keys)]):
        w = weight[a:b] / weight[a:b].sum()
        out_keys.append(str(keys[a]))
        out_p.append(float((w * p[a:b]).sum()))
        out_t.append(float((w * log_tc[a:b]).sum()))
    return np.array(out_keys), np.array(out_p), np.clip(np.expm1(np.array(out_t)), 0.0, None)


def train_one(torch, make_net, fit: dict, valid: dict, graphs, args, seed: int, device, encoder_state=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = make_net().to(device)
    if encoder_state is not None and net.encoder is not None:
        net.encoder.load_state_dict(encoder_state)
    encoder_params = list(net.encoder.parameters()) if net.encoder is not None else []
    ids = {id(p) for p in encoder_params}
    groups = [{"params": [p for p in net.parameters() if id(p) not in ids], "lr": args.lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.lr * (args.encoder_lr_scale if encoder_state else 1.0)})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(3, args.patience // 3))
    huber = torch.nn.functional.huber_loss
    best, best_state, best_epoch, waited, epoch = float("inf"), None, 0, 0, 0
    n = len(fit["x"])
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = torch.as_tensor(rng.permutation(n), device=device)
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            if len(idx) < 2:                             # batch norm needs two rows
                continue
            logit, log_tc = net(fit["x"][idx], fit["g"][idx], graphs)
            loss = weighted_bce(torch, logit, fit["y"][idx], fit["w"][idx])
            pos = torch.isfinite(fit["t"][idx])
            if args.aux_weight > 0 and bool(pos.any()):
                w = fit["w"][idx][pos]
                loss = loss + args.aux_weight * (huber(log_tc[pos], fit["t"][idx][pos], reduction="none") * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            p, _ = predict_items(torch, [net], valid, graphs, 2048)
            val = float(weighted_bce(torch, torch.logit(torch.as_tensor(p, device=device).clamp(1e-6, 1 - 1e-6)),
                                     valid["y"], valid["w"]))
        sched.step(val)
        if val < best - 1e-5:
            best, best_epoch, waited = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_bce": round(best, 5),
                 "seconds": round(time.time() - started, 1)}


def device_info(torch, device) -> dict:
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(device_name=props.name, memory_gb=round(props.total_memory / 2**30, 2),
                    capability=f"{props.major}.{props.minor}", cuda=torch.version.cuda,
                    peak_memory_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 3))
    else:
        info["device_name"] = platform.processor() or platform.machine()
    return info


def pick_device(torch, name: str):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was asked for, but this PyTorch cannot see a CUDA GPU")
    return device


def load_bundle(path: Path):
    bundle = np.load(path, allow_pickle=False)
    for split_key in ("comp_split", "split"):
        if split_key in bundle.files and (bundle[split_key] == "lockbox").any():
            raise SystemExit(f"{path} contains locked rows; refusing to train on it")
    return bundle


def cmd_train(args) -> int:
    import torch

    device = pick_device(torch, args.device)
    started = time.time()
    sb = load_bundle(args.bundle)
    fb = load_bundle(args.formula_bundle) if args.model in ALL_ROW_MODELS else None
    if args.model in ALL_ROW_MODELS and fb is None:
        raise SystemExit(f"--model {args.model} needs --formula-bundle")
    encoder_state, pretrain_config = None, None
    if args.pretrained:
        if args.model not in GRAPH_MODELS:
            raise SystemExit("--pretrained only applies to models with a structure encoder")
        saved = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        encoder_state, pretrain_config = saved["state"], saved["config"]
        for key in ENCODER_KEYS:                        # the encoder's shape comes from the pretrained file
            setattr(args, key, pretrain_config[key])
    if args.model in ALL_ROW_MODELS:
        items, owner = all_row_items(sb, fb, use_graphs=args.model == "hybrid_all")
        fold_source = fb
    else:
        items, owner = subset_items(sb)
        fold_source = sb
    graphs = Graphs(torch, sb, device) if args.model in GRAPH_MODELS else None
    folds = [int(f) for f in fold_source["folds"]]
    if args.folds:
        folds = [f for f in folds if str(f) in args.folds.split(",")]
    candidate_names = sorted(k[5:-6] for k in sb.files if k.startswith("cand_") and k.endswith("_graph"))
    if args.candidates:
        candidate_names = [c for c in candidate_names if c in args.candidates.split(",")]
    cand_items = {}
    for name in candidate_names:
        graph = sb[f"cand_{name}_graph"] if args.model in GRAPH_MODELS else np.full(len(sb[f"cand_{name}_keys"]), -1)
        if args.model in ALL_ROW_MODELS and fb is not None and sb[f"cand_{name}_X"].shape[1] != fb["X"].shape[1]:
            raise SystemExit("the structure bundle's formula features do not match the formula bundle's")
        n = len(graph)
        cand_items[name] = Items(sb[f"cand_{name}_keys"], sb[f"cand_{name}_X"], np.full(n, np.nan, np.float32),
                                 np.full(n, np.nan, np.float32), np.ones(n, np.float32), graph)
    n_x_raw = items.X.shape[1]
    oof_keys, oof_p, oof_t, fold_info, cand_parts = [], [], [], [], {name: [] for name in candidate_names}
    log(f"{args.model}: {len(items)} items, folds {folds}, device {device}, pretrained={bool(args.pretrained)}")
    for fold in folds:
        fit_i, valid_i, test_i = fold_item_sets(owner, fold_source, fold)
        prep = preprocess_fit(items.X[fit_i])
        n_x = n_x_raw + len(prep["indicators"])
        fit, valid, test = (to_device(torch, items.take(rows), prep, device) for rows in (fit_i, valid_i, test_i))
        make = lambda: build_net(torch, args.model, sb["elem_features"], n_x, args)  # noqa: E731
        nets, infos = [], []
        for s in range(args.seeds):
            net, info = train_one(torch, make, fit, valid, graphs, args, seed=args.seed_base + 1000 * fold + s,
                                  device=device, encoder_state=encoder_state)
            nets.append(net)
            infos.append(info)
            log(f"  fold {fold} seed {s}: best epoch {info['best_epoch']} of {info['epochs_run']}, "
                f"validation BCE {info['best_validation_bce']}, {info['seconds']} s")
        p, t = predict_items(torch, nets, test, graphs, args.batch * 4)
        test_items = items.take(test_i)
        k, pp, tt = by_composition(test_items.keys, test_items.weight, p, t)
        oof_keys.append(k)
        oof_p.append(pp)
        oof_t.append(tt)
        for name, citems in cand_items.items():
            data = to_device(torch, citems, prep, device)
            cand_parts[name].append(predict_items(torch, nets, data, graphs, args.batch * 4))
        fold_info.append({"fold": fold, "fit_items": int(len(fit_i)), "valid_items": int(len(valid_i)),
                          "test_items": int(len(test_i)), "test_compositions": int(len(k)), "networks": infos})
        del nets
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "oof.npz", compositions=np.concatenate(oof_keys), p=np.concatenate(oof_p),
                        tc=np.concatenate(oof_t))
    for name, parts in cand_parts.items():
        p = np.mean([a for a, _ in parts], axis=0)
        tc = np.clip(np.expm1(np.mean([b for _, b in parts], axis=0)), 0.0, None)
        np.savez_compressed(args.out / f"candidates_{name}.npz", keys=sb[f"cand_{name}_keys"], p=p, tc=tc)
    networks = [n for f in fold_info for n in f["networks"]]
    run = {"bundle_kind": "structure", "bundle_content_sha256": content_sha256(sb),
           "formula_bundle_content_sha256": content_sha256(fb) if fb is not None else None,
           **device_info(torch, device),
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
           "pretrain_config": pretrain_config, "folds": fold_info,
           "mean_validation_bce": round(float(np.mean([n["best_validation_bce"] for n in networks])), 5),
           "seconds_per_network": round(float(np.mean([n["seconds"] for n in networks])), 1),
           "minutes": round((time.time() - started) / 60.0, 2), "complete": True}
    (args.out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {args.out}: mean validation BCE {run['mean_validation_bce']}, {run['minutes']} min")
    return 0


def cmd_pretrain(args) -> int:
    """Teach the encoder calculated properties of the Materials Project structures. No superconductivity labels."""
    import torch

    device = pick_device(torch, args.device)
    started = time.time()
    sb = load_bundle(args.bundle)
    graphs = Graphs(torch, sb, device)
    gid = sb["pre_graph"].astype(np.int64)
    targets = sb["pre_targets"].astype(np.float32)
    names = [str(n) for n in sb["pre_target_names"]]
    if args.limit:
        gid, targets = gid[:args.limit], targets[:args.limit]
    if len(gid) < 4:
        raise SystemExit("the bundle has too few pretraining graphs")
    binary = np.array([set(np.unique(targets[np.isfinite(targets[:, j]), j])) <= {0.0, 1.0} for j in range(len(names))])
    scale_mean = np.where(binary, 0.0, np.nanmean(targets, axis=0))
    scale_std = np.where(binary, 1.0, np.nanstd(targets, axis=0))
    scaled = (targets - scale_mean) / np.where(scale_std > 0, scale_std, 1.0)
    rng = np.random.default_rng(args.seed_base)
    order = rng.permutation(len(gid))
    n_val = max(2, int(round(0.05 * len(gid))))
    val_i, fit_i = order[:n_val], order[n_val:]
    torch.manual_seed(args.seed_base)
    encoder = build_encoder(torch, sb["elem_features"], args).to(device)
    head = torch.nn.Sequential(torch.nn.Linear(encoder.out_dim, args.head), torch.nn.SiLU(),
                               torch.nn.Linear(args.head, len(names))).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    G = torch.as_tensor(gid, device=device)
    Y = torch.as_tensor(scaled, device=device)
    is_binary = torch.as_tensor(binary, device=device)

    def loss_of(idx):
        out = head(encoder(graphs.batch(G[idx])))
        y = Y[idx]
        known = torch.isfinite(y)
        reg = torch.nn.functional.mse_loss(out, torch.nan_to_num(y), reduction="none")
        cls = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.nan_to_num(y).clamp(0, 1), reduction="none")
        per = torch.where(is_binary, cls, reg) * known
        return per.sum() / known.sum().clamp(min=1)

    best, best_state, best_epoch, waited, epoch, history = float("inf"), None, 0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        encoder.train(); head.train()
        perm = torch.as_tensor(rng.permutation(fit_i), device=device)
        for start in range(0, len(perm), args.batch):
            idx = perm[start:start + args.batch]
            if len(idx) < 2:
                continue
            loss = loss_of(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sched.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            vi = torch.as_tensor(val_i, device=device)
            val = float(np.mean([float(loss_of(vi[s:s + 1024])) for s in range(0, len(vi), 1024)]))
        history.append(round(val, 5))
        log(f"  pretrain epoch {epoch}: validation loss {val:.4f}")
        if val < best - 1e-5:
            best, best_epoch, waited = val, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                break
    args.out.mkdir(parents=True, exist_ok=True)
    config = {key: getattr(args, key) for key in ENCODER_KEYS}
    torch.save({"state": best_state, "config": config, "targets": names}, args.out / "encoder.pt")
    run = {"bundle_kind": "structure", "bundle_content_sha256": content_sha256(sb), **device_info(torch, device),
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
           "config": config, "graphs": int(len(gid)), "validation_graphs": int(n_val), "best_epoch": best_epoch,
           "epochs_run": epoch, "best_validation_loss": round(best, 5), "history": history,
           "minutes": round((time.time() - started) / 60.0, 2), "complete": True}
    (args.out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {args.out}: best validation loss {best:.4f} at epoch {best_epoch}, {run['minutes']} min")
    return 0


def add_common(parser) -> None:
    parser.add_argument("bundle", type=Path, help="structures.npz from scripts/structure_gpu.py export")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--conv", type=int, default=4, help="graph convolutions")
    parser.add_argument("--hidden", type=int, default=128, help="atom feature width")
    parser.add_argument("--embed", type=int, default=64, help="learned element embedding width")
    parser.add_argument("--rbf", type=int, default=41, help="Gaussians for the distance expansion")
    parser.add_argument("--cutoff", type=float, default=8.0, help="angstrom; must match the bundle's graphs")
    parser.add_argument("--head", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    add_common(train)
    train.add_argument("--model", choices=MODELS, required=True)
    train.add_argument("--formula-bundle", type=Path, default=None, help="bundle.npz, for hybrid_all and formula_all")
    train.add_argument("--pretrained", type=Path, default=None, help="encoder.pt from the pretrain command")
    train.add_argument("--encoder-lr-scale", type=float, default=0.3, help="learning-rate factor for a pretrained encoder")
    train.add_argument("--x-hidden", type=int, default=128, help="width of the formula-feature branch")
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--aux-weight", type=float, default=0.3, help="weight of the log(1 + Tc) head's loss")
    train.add_argument("--seeds", type=int, default=3, help="networks per fold; their average is reported")
    train.add_argument("--folds", default="", help="only these folds, comma separated (default: all)")
    train.add_argument("--candidates", default="", help="only these candidate lists, comma separated (default: all)")
    train.set_defaults(func=cmd_train)
    pre = sub.add_parser("pretrain")
    add_common(pre)
    pre.add_argument("--limit", type=int, default=0, help="use only the first N pretraining graphs (smoke runs)")
    pre.set_defaults(func=cmd_pretrain, epochs=60, patience=8, batch=256)
    args = parser.parse_args(argv)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
