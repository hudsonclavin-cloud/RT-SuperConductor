#!/usr/bin/env python3
"""A tabular neural network for the superconductor classifier (PyTorch, CPU).

Architecture (the starting configuration suggested by the 2026-11-06 external
review, not a tuned optimum):

    inputs (imputed + standardised + missingness indicators)
    Linear(128) -> LayerNorm -> SiLU -> Dropout(0.1)
    Linear(64)  -> SiLU
    heads: P(superconductor)  [+ optional log(1 + Tc) regression]

Training: AdamW, mini-batches, early stopping on an inner validation split
taken from the training data by element-system group (never the outer test
fold); the best validation checkpoint is restored. The optional Tc head is a
masked Huber loss on rows with a reported Tc > 0, so it adds no new labels:
it asks the shared layers to also encode how high Tc is.

Exposes a scikit-learn style estimator: fit(X, y, groups=..., aux_target=...,
sample_weight=...) and predict_proba(X).
"""
from __future__ import annotations

from typing import Any

import numpy as np


class TorchTabularClassifier:
    """Feed-forward classifier with LayerNorm/SiLU and grouped early stopping."""

    def __init__(self, hidden: tuple[int, int] = (128, 64), dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 128, max_epochs: int = 300, patience: int = 25,
                 validation_fraction: float = 0.15, aux_weight: float = 0.0, seed: int = 0):
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.batch_size, self.max_epochs, self.patience = batch_size, max_epochs, patience
        self.validation_fraction, self.aux_weight, self.seed = validation_fraction, aux_weight, seed

    # preprocessing fitted on training rows only
    def _fit_preprocess(self, X: np.ndarray) -> None:
        self.median_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        self.missing_cols_ = np.where(np.isnan(X).any(axis=0))[0]
        filled = np.where(np.isnan(X), self.median_, X)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0

    def _transform(self, X: np.ndarray) -> np.ndarray:
        missing = np.isnan(X[:, self.missing_cols_]).astype(np.float32)
        filled = (np.where(np.isnan(X), self.median_, X) - self.mean_) / self.scale_
        return np.hstack([np.clip(filled, -10, 10), missing]).astype(np.float32)

    def _network(self, n_in: int):
        import torch.nn as nn

        h1, h2 = self.hidden
        trunk = nn.Sequential(nn.Linear(n_in, h1), nn.LayerNorm(h1), nn.SiLU(), nn.Dropout(self.dropout),
                              nn.Linear(h1, h2), nn.SiLU())
        return nn.ModuleDict({"trunk": trunk, "cls": nn.Linear(h2, 1), "tc": nn.Linear(h2, 1)})

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
            aux_target: np.ndarray | None = None, sample_weight: np.ndarray | None = None):
        import torch
        from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y, dtype=np.float32)
        weights = np.ones(len(y), np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
        aux = np.full(len(y), np.nan, np.float32) if aux_target is None else np.asarray(aux_target, np.float32)
        if groups is not None and len(set(groups)) > 5:
            split = GroupShuffleSplit(n_splits=1, test_size=self.validation_fraction, random_state=self.seed)
            train_idx, val_idx = next(split.split(X, y, groups))
        else:
            split = StratifiedShuffleSplit(n_splits=1, test_size=self.validation_fraction, random_state=self.seed)
            train_idx, val_idx = next(split.split(X, y))
        self._fit_preprocess(X[train_idx])
        Xt = torch.from_numpy(self._transform(X))
        yt, wt = torch.from_numpy(y), torch.from_numpy(weights)
        at = torch.from_numpy(np.nan_to_num(np.log1p(np.clip(aux, 0, None)), nan=0.0))
        mask = torch.from_numpy((np.isfinite(aux) & (y > 0)).astype(np.float32))
        net = self._network(Xt.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        huber = torch.nn.HuberLoss(reduction="none", delta=1.0)

        def loss_on(idx: np.ndarray, train: bool):
            net.train(train)
            h = net["trunk"](Xt[idx])
            cls = bce(net["cls"](h).squeeze(1), yt[idx]) * wt[idx]
            total = cls.sum() / wt[idx].sum()
            if self.aux_weight > 0:
                m = mask[idx] * wt[idx]
                if m.sum() > 0:
                    reg = huber(net["tc"](h).squeeze(1), at[idx]) * m
                    total = total + self.aux_weight * reg.sum() / m.sum()
            return total

        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.max_epochs):
            order = rng.permutation(train_idx)
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                opt.zero_grad()
                loss_on(batch, True).backward()
                opt.step()
            with torch.no_grad():
                val = float(loss_on(val_idx, False))
            if val < best - 1e-5:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_, self.best_validation_loss_, self.epochs_ = net, best, epoch + 1
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            h = self.net_["trunk"](torch.from_numpy(self._transform(X)))
            p = torch.sigmoid(self.net_["cls"](h).squeeze(1)).numpy().astype(float)
        return np.column_stack([1 - p, p])


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def torch_factories(seed: int) -> dict[str, Any]:
    torch_threads()
    return {
        "torch_mlp": lambda: TorchTabularClassifier(seed=seed),
        "torch_mlp_multitask_tc": lambda: TorchTabularClassifier(seed=seed, aux_weight=0.5),
        "torch_mlp_tuned": lambda: TunedTorchClassifier(seed=seed),
    }


def torch_threads(n: int = 4) -> None:
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass


class TunedTorchClassifier:
    """Pick hyperparameters by grouped inner-validation loss inside the training data.

    Every candidate configuration is fitted on the same inner split of the
    outer training fold; the one with the lowest validation loss is kept. The
    outer test fold never influences the choice.
    """

    GRID = [
        {"hidden": (128, 64), "dropout": 0.1, "weight_decay": 1e-4, "lr": 1e-3},
        {"hidden": (128, 64), "dropout": 0.3, "weight_decay": 1e-4, "lr": 1e-3},
        {"hidden": (128, 64), "dropout": 0.3, "weight_decay": 1e-2, "lr": 1e-3},
        {"hidden": (256, 128), "dropout": 0.3, "weight_decay": 1e-2, "lr": 1e-3},
        {"hidden": (256, 128), "dropout": 0.5, "weight_decay": 1e-2, "lr": 1e-3},
        {"hidden": (64, 32), "dropout": 0.1, "weight_decay": 1e-3, "lr": 1e-3},
        {"hidden": (128, 64), "dropout": 0.3, "weight_decay": 1e-2, "lr": 3e-4},
        {"hidden": (256, 128), "dropout": 0.3, "weight_decay": 1e-3, "lr": 3e-4},
    ]

    def __init__(self, seed: int = 0, aux_weight: float = 0.0):
        self.seed, self.aux_weight = seed, aux_weight

    def fit(self, X, y, groups=None, aux_target=None, sample_weight=None):
        fitted = [TorchTabularClassifier(seed=self.seed, aux_weight=self.aux_weight, **config)
                  .fit(X, y, groups=groups, aux_target=aux_target, sample_weight=sample_weight)
                  for config in self.GRID]
        scores = [m.best_validation_loss_ for m in fitted]
        self.best_index_ = int(np.argmin(scores))
        self.best_config_ = self.GRID[self.best_index_]
        self.model_ = fitted[self.best_index_]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
