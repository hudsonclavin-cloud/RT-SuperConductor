#!/usr/bin/env python3
"""Off-the-shelf tabular learners on cycle 6's exact folds (``docs/training-methods-review.md``).

    python scripts/foundation_gpu_train.py data/processed/local-gpu/bundle.npz --model tabicl --out runs/fm/tabicl --device cuda

Reads the bundle written by ``python scripts/local_gpu.py export`` and writes the same outputs as
``local_gpu_train.py`` (``oof.npz``, ``candidates_<name>.npz``, ``run.json``), so
``python scripts/local_gpu.py compare <folder>`` judges it against the ranking of record.

Models. Nothing is installed by this script; each library is installed separately by whoever runs it.

    tabicl    TabICLv2, a tabular foundation model (``pip install tabicl``; code and weights BSD-3). No training:
              the fold's training rows are the context and the test rows are predicted in one pass.
    tabpfn    TabPFN (``pip install tabpfn``). From version 2.5 on its weights carry a non-commercial licence and
              the first use asks for a login; read ``docs/training-methods-review.md`` before using it.
    catboost  CatBoost with its default settings (``pip install catboost``), early stopping on the fold's
              validation rows.
    hgb       scikit-learn's HistGradientBoosting with cycle 6's own settings: a plumbing check. It should land
              on cycle 6's scores; if it does not, the bundle or this script is wrong.
    logistic  Logistic and ridge regression on imputed, scaled features: a floor.

In-context models (tabicl, tabpfn) and ``logistic`` have no early stopping, so they are given the fold's fit rows
and its validation rows together. Test rows are never part of what a model is given. Locked rows are not in the
bundle, and a bundle that holds one is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

MODELS = ("tabicl", "tabpfn", "catboost", "hgb", "logistic")
USES_VALIDATION = {"catboost", "hgb"}                      # the others take fit and validation rows together
# Cycle 6's boosting settings (docs/learning-cycles/cycle6.json), for the plumbing check.
HGB_COMMON = dict(max_iter=1500, early_stopping=True, n_iter_no_change=40, learning_rate=0.05, min_samples_leaf=20,
                  l2_regularization=1.0)
HGB_CLASSIFIER_LEAVES, HGB_REGRESSOR_LEAVES = 31, 63


def content_sha256(bundle) -> str:
    """A checksum of the bundle's arrays, not of the file (the same function as in ``local_gpu_train.py``)."""
    digest = hashlib.sha256()
    for key in sorted(bundle.files):
        value = np.ascontiguousarray(bundle[key])
        digest.update(f"{key}|{value.dtype.str}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


# ---------------------------------------------------------------------------
# One fitted pair per fold: a classifier (superconductor or not) and a regressor (log(1 + Tc) of superconductors)

class Pair:
    """Fit on arrays, predict (probability, Tc in kelvin). Subclasses fill in ``_fit`` and the two predictors."""

    def __init__(self, args, seed: int):
        self.args, self.seed = args, seed

    def fit(self, Xc, yc, Xr, tr, Xc_val=None, yc_val=None, Xr_val=None, tr_val=None) -> None:
        raise NotImplementedError

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def versions() -> dict:
        return {}


class SklearnStyle(Pair):
    """Any pair of estimators with ``fit``, ``predict_proba`` and ``predict``."""

    def make(self) -> tuple:
        raise NotImplementedError

    def fit(self, Xc, yc, Xr, tr, Xc_val=None, yc_val=None, Xr_val=None, tr_val=None) -> None:
        self.classifier, self.regressor = self.make()
        self.classifier.fit(Xc, yc)
        self.regressor.fit(Xr, tr)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        size = self.args.predict_batch
        p = np.concatenate([np.asarray(self.classifier.predict_proba(X[s:s + size]))[:, 1] for s in range(0, len(X), size)])
        t = np.concatenate([np.asarray(self.regressor.predict(X[s:s + size])) for s in range(0, len(X), size)])
        return p.astype(float), np.clip(np.expm1(t.astype(float)), 0.0, None)


def package_version(name: str, module) -> str:
    """The module's own ``__version__``, else the installed package's version (tabicl 2.2.0 has no ``__version__``)."""
    found = getattr(module, "__version__", None)
    if found:
        return str(found)
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


class TabICL(SklearnStyle):
    def make(self) -> tuple:
        from tabicl import TabICLClassifier, TabICLRegressor

        device = None if self.args.device == "auto" else self.args.device
        common = dict(n_estimators=self.args.estimators, device=device, random_state=self.seed)
        return TabICLClassifier(**common), TabICLRegressor(**common)

    @staticmethod
    def versions() -> dict:
        import tabicl

        return {"tabicl": package_version("tabicl", tabicl)}


class TabPFN(SklearnStyle):
    def make(self) -> tuple:
        from tabpfn import TabPFNClassifier, TabPFNRegressor

        device = "auto" if self.args.device == "auto" else self.args.device
        return TabPFNClassifier(device=device), TabPFNRegressor(device=device)

    @staticmethod
    def versions() -> dict:
        import os
        import tabpfn

        return {"tabpfn": package_version("tabpfn", tabpfn),
                "TABPFN_MODEL_VERSION": os.environ.get("TABPFN_MODEL_VERSION", "library default")}


class CatBoost(Pair):
    def fit(self, Xc, yc, Xr, tr, Xc_val=None, yc_val=None, Xr_val=None, tr_val=None) -> None:
        from catboost import CatBoostClassifier, CatBoostRegressor

        extra = {"task_type": "GPU"} if str(self.args.device).startswith("cuda") and self.args.catboost_gpu else {}
        self.classifier = CatBoostClassifier(random_seed=self.seed, verbose=False, **extra)
        self.classifier.fit(Xc, yc, eval_set=(Xc_val, yc_val), early_stopping_rounds=self.args.patience)
        self.regressor = CatBoostRegressor(random_seed=self.seed, verbose=False, loss_function="MAE", **extra)
        self.regressor.fit(Xr, tr, eval_set=(Xr_val, tr_val), early_stopping_rounds=self.args.patience)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(self.classifier.predict_proba(X))[:, 1]
        return p.astype(float), np.clip(np.expm1(np.asarray(self.regressor.predict(X), float)), 0.0, None)

    @staticmethod
    def versions() -> dict:
        import catboost

        return {"catboost": catboost.__version__}


class HGB(Pair):
    def fit(self, Xc, yc, Xr, tr, Xc_val=None, yc_val=None, Xr_val=None, tr_val=None) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        self.classifier = HistGradientBoostingClassifier(**HGB_COMMON, max_leaf_nodes=HGB_CLASSIFIER_LEAVES,
                                                         random_state=self.seed)
        self.classifier.fit(Xc, yc, X_val=Xc_val, y_val=yc_val)
        self.regressor = HistGradientBoostingRegressor(loss="absolute_error", **HGB_COMMON,
                                                       max_leaf_nodes=HGB_REGRESSOR_LEAVES, random_state=self.seed)
        self.regressor.fit(Xr, tr, X_val=Xr_val, y_val=tr_val)

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        return (self.classifier.predict_proba(X)[:, 1].astype(float),
                np.clip(np.expm1(self.regressor.predict(X).astype(float)), 0.0, None))

    @staticmethod
    def versions() -> dict:
        import sklearn

        return {"scikit-learn": sklearn.__version__}


class Logistic(SklearnStyle):
    def make(self) -> tuple:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        front = lambda: (SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())  # noqa: E731
        return (make_pipeline(*front(), LogisticRegression(max_iter=2000)), make_pipeline(*front(), Ridge(alpha=1.0)))

    versions = HGB.versions


PAIRS = {"tabicl": TabICL, "tabpfn": TabPFN, "catboost": CatBoost, "hgb": HGB, "logistic": Logistic}


# ---------------------------------------------------------------------------

def capped(rows: np.ndarray, label: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """At most ``cap`` rows, keeping the share of each label (0 = no cap). For GPUs that cannot hold every row."""
    if not cap or len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    keep = []
    for value in np.unique(label[rows]):
        mine = rows[label[rows] == value]
        keep.append(rng.choice(mine, size=max(1, int(round(cap * len(mine) / len(rows)))), replace=False))
    return np.sort(np.concatenate(keep))


def fold_arrays(bundle, fold: int, model: str, cap: int, seed: int) -> dict:
    """The training, validation and test arrays of one fold, for the classifier and for the Tc regressor."""
    X, label, tc = bundle["X"], bundle["label"], bundle["tc_k"]
    fit, valid, test = (bundle[f"fold{fold}_{part}"] for part in ("fit", "valid", "test"))
    if set(fit.tolist()) & set(test.tolist()) or set(valid.tolist()) & set(test.tolist()):
        raise SystemExit(f"fold {fold}: a test row is among the training rows; the bundle is wrong")
    labelled = np.isfinite(label)
    positive = (label == 1) & np.isfinite(tc)
    log_tc = np.log1p(np.nan_to_num(tc, nan=0.0))
    if model in USES_VALIDATION:
        train_c, train_r = fit[labelled[fit]], fit[positive[fit]]
        val_c, val_r = valid[labelled[valid]], valid[positive[valid]]
    else:
        both = np.concatenate([fit, valid])
        train_c, train_r = both[labelled[both]], both[positive[both]]
        val_c = val_r = np.zeros(0, int)
    train_c = capped(train_c, label, cap, seed)
    train_r = capped(train_r, np.zeros(len(label)), cap, seed)
    return {"Xc": X[train_c], "yc": label[train_c].astype(int), "Xr": X[train_r], "tr": log_tc[train_r],
            "Xc_val": X[val_c], "yc_val": label[val_c].astype(int), "Xr_val": X[val_r], "tr_val": log_tc[val_r],
            "test": test, "counts": {"classifier_rows": int(len(train_c)), "regressor_rows": int(len(train_r)),
                                     "validation_rows": int(len(val_c)), "test_rows": int(len(test))}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bundle", type=Path, help="bundle.npz from scripts/local_gpu.py export")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu (used by tabicl, tabpfn and catboost)")
    parser.add_argument("--estimators", type=int, default=8, help="tabicl: ensemble members (its own default is 8)")
    parser.add_argument("--max-context", type=int, default=0,
                        help="tabicl, tabpfn: use at most this many training rows per fold (0 = all)")
    parser.add_argument("--predict-batch", type=int, default=4096, help="rows predicted per call")
    parser.add_argument("--patience", type=int, default=50, help="catboost: early-stopping rounds")
    parser.add_argument("--catboost-gpu", action="store_true", help="catboost: train on the GPU (default: CPU)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", default="", help="only these folds, comma separated (default: all)")
    parser.add_argument("--candidates", default="", help="only these candidate lists, comma separated (default: all)")
    parser.add_argument("--extra-candidates", type=Path, default=None,
                        help="an .npz holding more lists to predict (candidates_<name>_X and _keys, nothing else), "
                             "for example the locked rows' features from scripts/lockbox_blend.py export")
    return parser.parse_args(argv)


def load_extra_candidates(path: Path, bundle) -> dict:
    """More compounds to predict, from a separate file. Only features and keys are accepted, so no label can arrive this
    way; the columns must be the bundle's own features, and a name the bundle already uses is refused."""
    extra = np.load(path, allow_pickle=False)
    allowed = {"feature_names"} | {k for k in extra.files if k.startswith("candidates_") and k.endswith(("_X", "_keys"))}
    if set(extra.files) - allowed:
        raise SystemExit(f"{path} holds more than features and keys: {sorted(set(extra.files) - allowed)}")
    if "feature_names" not in extra.files or "feature_names" not in bundle.files or \
            list(extra["feature_names"]) != list(bundle["feature_names"]):
        raise SystemExit(f"{path} does not carry the bundle's feature names; re-export it from the same table")
    lists = {}
    for key in extra.files:
        if key.endswith("_X"):
            name = key[len("candidates_"):-len("_X")]
            if f"candidates_{name}_X" in bundle.files:
                raise SystemExit(f"the bundle already has a candidate list called {name!r}")
            X, keys = extra[key], extra[f"candidates_{name}_keys"]
            if X.shape[1] != bundle["X"].shape[1] or len(keys) != len(X):
                raise SystemExit(f"{path}: list {name!r} has the wrong shape")
            lists[name] = (X, keys)
    return lists


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    raw = args.bundle.read_bytes()
    bundle = np.load(args.bundle, allow_pickle=False)
    if "split" in bundle.files and (bundle["split"] == "lockbox").any():
        raise SystemExit("this bundle contains locked rows; refusing to use it")
    keys = bundle["keys"]
    folds = [int(f) for f in bundle["folds"]]
    if args.folds:
        folds = [f for f in folds if str(f) in args.folds.split(",")]
    lists = {k[len("candidates_"):-len("_X")]: (bundle[k], bundle[k[:-len("_X")] + "_keys"])
             for k in bundle.files if k.startswith("candidates_") and k.endswith("_X")}
    extra_raw = args.extra_candidates.read_bytes() if args.extra_candidates else b""
    if args.extra_candidates:
        extra_lists = load_extra_candidates(args.extra_candidates, bundle)
        held = set(keys.tolist())
        for name, (_, extra_keys) in extra_lists.items():
            if held & set(extra_keys.tolist()):
                raise SystemExit(f"list {name!r} shares compounds with the training bundle; refusing")
        lists.update(extra_lists)
    names = sorted(lists)
    if args.candidates:
        names = [n for n in names if n in args.candidates.split(",")]
    pair_class = PAIRS[args.model]
    versions = pair_class.versions()                       # fails here, before any work, if the library is missing
    log(f"{args.model}: folds {folds}, device {args.device}, libraries {versions}")
    oof_keys, oof_p, oof_tc, fold_info = [], [], [], []
    candidate_parts = {name: [] for name in names}
    for fold in folds:
        data = fold_arrays(bundle, fold, args.model, args.max_context if args.model not in USES_VALIDATION else 0,
                           args.seed + fold)
        fold_started = time.time()
        pair = pair_class(args, seed=args.seed + fold)
        pair.fit(data["Xc"], data["yc"], data["Xr"], data["tr"], data["Xc_val"], data["yc_val"], data["Xr_val"],
                 data["tr_val"])
        fitted = time.time() - fold_started
        p, tc = pair.predict(bundle["X"][data["test"]])
        if not (np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))):
            raise SystemExit(f"fold {fold}: the model returned probabilities outside 0 to 1")
        oof_keys += list(keys[data["test"]])
        oof_p.append(p)
        oof_tc.append(tc)
        for name in names:
            candidate_parts[name].append(pair.predict(lists[name][0]))
        fold_info.append({"fold": fold, **data["counts"], "fit_seconds": round(fitted, 1),
                          "seconds": round(time.time() - fold_started, 1)})
        log(f"  fold {fold}: {data['counts']}, {fold_info[-1]['seconds']} s")
        del pair

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "oof.npz", compositions=np.array(oof_keys), p=np.concatenate(oof_p),
                        tc=np.concatenate(oof_tc))
    for name, parts in candidate_parts.items():
        p = np.mean([a for a, _ in parts], axis=0)
        tc = np.expm1(np.mean([np.log1p(b) for _, b in parts], axis=0))
        np.savez_compressed(args.out / f"candidates_{name}.npz", keys=lists[name][1], p=p, tc=tc)
    device_name = platform.processor() or platform.machine()
    try:
        import torch

        if torch.cuda.is_available() and args.device != "cpu":
            device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    run = {"bundle_kind": "formula", "bundle_sha256": hashlib.sha256(raw).hexdigest(),
           "bundle_content_sha256": content_sha256(bundle), "model": args.model, "libraries": versions,
           "context": "fit rows; early stopping on validation rows" if args.model in USES_VALIDATION
                      else "fit and validation rows together; no early stopping",
           "device": args.device, "device_name": device_name, "python": platform.python_version(),
           "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "folds": fold_info, "minutes": round((time.time() - started) / 60.0, 2), "complete": True}
    if args.extra_candidates:
        run["extra_candidates"] = {"file": args.extra_candidates.name, "sha256": hashlib.sha256(extra_raw).hexdigest(),
                                   "lists": {n: int(len(lists[n][1])) for n in names}}
    (args.out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {args.out} in {run['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
