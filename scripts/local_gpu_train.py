#!/usr/bin/env python3
"""Train the superconductor network on a local GPU from a bundle made by ``scripts/local_gpu.py export``.

Self-contained: needs only numpy and PyTorch, so it runs on any machine (``docs/local-gpu.md``):

    python local_gpu_train.py bundle.npz --out results/ --device cuda

For each development fold in the bundle it trains a bank of networks (one per seed) on that fold's fit rows, stops
early on its validation rows, and predicts its test rows. It never sees locked rows: they are not in the bundle.

Outputs, all keyed by composition, in ``--out``:
    oof.npz                      out-of-fold predictions on the development rows (compositions, p, tc)
    candidates_<name>.npz        the average of every fold's networks on each candidate list in the bundle
    run.json                     settings, bundle checksum, device, timings and best epochs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


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
    median = np.nanmedian(np.where(finite, X, np.nan), axis=0)
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
    return np.hstack([scaled, missing]).astype(np.float32)


def build_model(torch, n_in: int, hidden: tuple[int, ...], dropout: float):
    nn = torch.nn
    layers, width = [], n_in
    for i, size in enumerate(hidden):
        layers += [nn.Linear(width, size), nn.LayerNorm(size) if i == 0 else nn.Identity(), nn.SiLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        width = size

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*layers)
            self.logit = nn.Linear(width, 1)
            self.log_tc = nn.Linear(width, 1)

        def forward(self, x):
            h = self.body(x)
            return self.logit(h).squeeze(-1), self.log_tc(h).squeeze(-1)

    return Net()


def train_one(torch, X_fit, y_fit, t_fit, X_val, y_val, args, seed: int, device):
    """One network: BCE on labelled rows plus a masked Huber loss on log(1 + Tc) for superconductors."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_model(torch, X_fit.shape[1], args.hidden, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    Xf, yf, tf = (torch.as_tensor(a, device=device) for a in (X_fit, y_fit, t_fit))
    Xv, yv = torch.as_tensor(X_val, device=device), torch.as_tensor(y_val, device=device)
    labelled_f, positive_f = torch.isfinite(yf), torch.isfinite(tf)
    labelled_v = torch.isfinite(yv)
    bce, huber = torch.nn.functional.binary_cross_entropy_with_logits, torch.nn.functional.huber_loss
    best, best_state, best_epoch, waited = float("inf"), None, 0, 0
    n = len(X_fit)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.as_tensor(rng.permutation(n), device=device)
        for start in range(0, n, args.batch):
            idx = order[start:start + args.batch]
            logit, log_tc = model(Xf[idx])
            lab, pos = labelled_f[idx], positive_f[idx]
            loss = bce(logit[lab], yf[idx][lab]) if lab.any() else logit.sum() * 0.0
            if args.aux_weight > 0 and pos.any():
                loss = loss + args.aux_weight * huber(log_tc[pos], tf[idx][pos])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logit_v, _ = model(Xv)
            val = float(bce(logit_v[labelled_v], yv[labelled_v])) if labelled_v.any() else 0.0
        if val < best - 1e-5:
            best, best_epoch, waited = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": seed, "best_epoch": best_epoch, "epochs_run": epoch, "best_validation_bce": round(best, 5)}


def predict(torch, models, X: np.ndarray, device, batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Average probability and the log-mean Tc of a list of networks."""
    ps, ts = [], []
    with torch.no_grad():
        for model in models:
            p_parts, t_parts = [], []
            for start in range(0, len(X), batch):
                logit, log_tc = model(torch.as_tensor(X[start:start + batch], device=device))
                p_parts.append(torch.sigmoid(logit).cpu().numpy())
                t_parts.append(log_tc.cpu().numpy())
            ps.append(np.concatenate(p_parts))
            ts.append(np.concatenate(t_parts))
    p = np.mean(ps, axis=0)
    tc = np.clip(np.expm1(np.mean(ts, axis=0)), 0.0, None)
    return p, tc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps or cpu")
    parser.add_argument("--seeds", type=int, default=5, help="networks per fold (a bank; their average is reported)")
    parser.add_argument("--hidden", default="256,128", help="hidden layer widths, comma separated")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--aux-weight", type=float, default=0.3, help="weight of the log(1 + Tc) head's loss")
    parser.add_argument("--folds", default="", help="only these folds, comma separated (default: all)")
    args = parser.parse_args(argv)
    args.hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    return args


def main(argv=None) -> int:
    import torch

    args = parse_args(argv)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    started = time.time()
    raw = args.bundle.read_bytes()
    bundle = np.load(args.bundle, allow_pickle=False)
    X, label, tc_k, keys = bundle["X"], bundle["label"], bundle["tc_k"], bundle["keys"]
    if "split" in bundle.files and (bundle["split"] == "lockbox").any():
        raise SystemExit("this bundle contains locked rows; refusing to train on it")
    folds = [int(f) for f in bundle["folds"]]
    if args.folds:
        folds = [f for f in folds if str(f) in args.folds.split(",")]
    log_tc = np.where((label == 1) & np.isfinite(tc_k), np.log1p(np.nan_to_num(tc_k, nan=0.0)), np.nan)
    candidate_names = sorted(k[len("candidates_"):-len("_X")] for k in bundle.files
                             if k.startswith("candidates_") and k.endswith("_X"))
    candidate_sums = {name: [] for name in candidate_names}
    oof_keys, oof_p, oof_tc, fold_info = [], [], [], []
    for fold in folds:
        fit, valid, test = bundle[f"fold{fold}_fit"], bundle[f"fold{fold}_valid"], bundle[f"fold{fold}_test"]
        prep = preprocess_fit(X[fit])
        Xf, Xv, Xt = (preprocess_apply(X[rows], prep) for rows in (fit, valid, test))
        y = label.astype(np.float32)
        t = log_tc.astype(np.float32)
        fold_started = time.time()
        models, infos = [], []
        for s in range(args.seeds):
            model, info = train_one(torch, Xf, y[fit], t[fit], Xv, y[valid], args, seed=1000 * fold + s, device=device)
            models.append(model)
            infos.append(info)
        p, tc = predict(torch, models, Xt, device)
        oof_keys += list(keys[test])
        oof_p.append(p)
        oof_tc.append(tc)
        for name in candidate_names:
            candidate_sums[name].append(predict(torch, models, preprocess_apply(bundle[f"candidates_{name}_X"], prep), device))
        fold_info.append({"fold": fold, "fit": int(len(fit)), "valid": int(len(valid)), "test": int(len(test)),
                          "seconds": round(time.time() - fold_started, 1), "networks": infos})
        print(f"fold {fold}: {len(fit)} fit, {len(valid)} valid, {len(test)} test; "
              f"best epochs {[i['best_epoch'] for i in infos]}; {fold_info[-1]['seconds']} s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "oof.npz", compositions=np.array(oof_keys), p=np.concatenate(oof_p),
                        tc=np.concatenate(oof_tc))
    for name, parts in candidate_sums.items():
        p = np.mean([a for a, _ in parts], axis=0)
        tc = np.expm1(np.mean([np.log1p(b) for _, b in parts], axis=0))
        np.savez_compressed(args.out / f"candidates_{name}.npz", keys=bundle[f"candidates_{name}_keys"], p=p, tc=tc)
    run = {"bundle_sha256": hashlib.sha256(raw).hexdigest(), "bundle_content_sha256": content_sha256(bundle),
           "device": str(device),
           "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
           "torch": torch.__version__, "python": platform.python_version(),
           "settings": {k: (list(v) if isinstance(v, tuple) else str(v) if isinstance(v, Path) else v)
                        for k, v in vars(args).items()},
           "folds": fold_info, "minutes": round((time.time() - started) / 60.0, 2)}
    (args.out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} in {run['minutes']} min on {run['device']} ({run['device_name']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
