"""Two model families behind one interface.

Every model exposes::

    fit(X, y, *, groups=None, tc=None, tokens=None) -> self
    predict(X, *, tokens=None) -> (mean probability, per-member probabilities)

The per-member probabilities are what the selection agent uses as uncertainty.

``TreeBag``       several gradient-boosting models, each fitted on a bootstrap
                  of element systems. The legacy classifier is a single one.
``DeepEnsemble``  K networks trained together as one batched tensor program.
                  Each member has its own initialisation and its own grouped
                  validation split for early stopping. Inputs are the tabular
                  features (quantile-normalised) plus a learned element-set
                  encoder with attention pooling over the formula.
``Blend``         the plain average of the two; no weights are tuned.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

Tokens = tuple[np.ndarray, np.ndarray]  # element indices, element fractions


def _group_bootstrap(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of a bootstrap drawn over whole groups."""
    unique, inverse = np.unique(groups, return_inverse=True)
    drawn = rng.integers(0, len(unique), len(unique))
    counts = np.bincount(drawn, minlength=len(unique))
    return np.repeat(np.arange(len(groups)), counts[inverse])


class TreeBag:
    """Bagged histogram gradient boosting. ``members=1, bootstrap=False`` is the legacy model."""

    def __init__(self, members: int = 8, seed: int = 0, bootstrap: bool = True, max_iter: int = 400,
                 early_stopping: bool = True, n_jobs: int = 1):
        self.members, self.seed, self.bootstrap = members, seed, bootstrap
        self.max_iter, self.early_stopping, self.n_jobs = max_iter, early_stopping, n_jobs

    def _one(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None, member: int):
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=self.max_iter, max_leaf_nodes=31, min_samples_leaf=20,
            l2_regularization=1.0, early_stopping=self.early_stopping, validation_fraction=0.15,
            random_state=self.seed + member)
        if self.bootstrap:
            rng = np.random.default_rng(self.seed * 1000 + member)
            rows = (_group_bootstrap(groups, rng) if groups is not None else rng.integers(0, len(y), len(y)))
            if len(np.unique(y[rows])) < 2:  # degenerate draw on a tiny set
                rows = np.arange(len(y))
            return model.fit(X[rows], y[rows])
        return model.fit(X, y)

    def fit(self, X, y, *, groups=None, tc=None, tokens=None):
        from joblib import Parallel, delayed

        y = np.asarray(y).astype(int)
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._one)(X, y, groups, member) for member in range(self.members))
        return self

    def predict(self, X, *, tokens=None):
        each = np.stack([m.predict_proba(X)[:, 1] for m in self.models_])
        return each.mean(axis=0), each


class DeepEnsemble:
    """K networks trained in one pass; see the module docstring."""

    def __init__(self, members: int = 8, seed: int = 0, width: int = 128, element_dim: int = 32,
                 dropout: float = 0.2, lr: float = 2e-3, weight_decay: float = 1e-2, batch_size: int = 256,
                 max_epochs: int = 120, patience: int = 20, validation_fraction: float = 0.15,
                 tc_weight: float = 0.3, use_elements: bool = True, threads: int = 2):
        self.members, self.seed, self.width, self.element_dim = members, seed, width, element_dim
        self.dropout, self.lr, self.weight_decay, self.batch_size = dropout, lr, weight_decay, batch_size
        self.max_epochs, self.patience, self.validation_fraction = max_epochs, patience, validation_fraction
        self.tc_weight, self.use_elements, self.threads = tc_weight, use_elements, threads

    # ---- preprocessing: fitted on the training rows passed to fit(), never on test rows ----
    def _fit_preprocess(self, X: np.ndarray) -> None:
        from sklearn.preprocessing import QuantileTransformer

        with np.errstate(all="ignore"):
            median = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(median), median, 0.0)
        self.missing_cols_ = np.where(np.isnan(X).any(axis=0))[0]
        self.quantile_ = QuantileTransformer(n_quantiles=min(200, len(X)), output_distribution="normal",
                                             random_state=self.seed).fit(np.where(np.isnan(X), self.median_, X))

    def _transform(self, X: np.ndarray) -> np.ndarray:
        dense = self.quantile_.transform(np.where(np.isnan(X), self.median_, X))
        missing = np.isnan(X[:, self.missing_cols_])
        return np.hstack([np.clip(dense, -5, 5), missing]).astype(np.float32)

    # ---- parameters carry a leading ensemble dimension K ----
    def _init(self, n_in: int, torch) -> dict[str, Any]:
        K, W, D = self.members, self.width, self.element_dim
        gen = torch.Generator().manual_seed(self.seed)

        def linear(fan_in: int, fan_out: int) -> tuple[Any, Any]:
            bound = 1 / math.sqrt(fan_in)
            weight = (torch.rand(K, fan_in, fan_out, generator=gen) * 2 - 1) * bound
            return weight.requires_grad_(), torch.zeros(K, 1, fan_out, requires_grad=True)

        p: dict[str, Any] = {}
        trunk_in = n_in + (D if self.use_elements else 0)
        p["w1"], p["b1"] = linear(trunk_in, W)
        p["w2"], p["b2"] = linear(W, W // 2)
        p["w_cls"], p["b_cls"] = linear(W // 2, 1)
        p["w_tc"], p["b_tc"] = linear(W // 2, 1)
        p["g1"] = torch.ones(K, 1, W, requires_grad=True)   # LayerNorm gain and bias
        p["h1"] = torch.zeros(K, 1, W, requires_grad=True)
        if self.use_elements:
            from .data import N_ELEMENT_SLOTS

            p["emb"] = (torch.randn(K, N_ELEMENT_SLOTS, D, generator=gen) * 0.1).requires_grad_()
            p["w_mix"], p["b_mix"] = linear(D, D)
            p["query"] = (torch.randn(K, D, 1, generator=gen) * 0.1).requires_grad_()
            p["w_ctx"], _ = linear(D, D)
        return p

    def _forward(self, p: dict[str, Any], x, share, torch, train: bool):
        """x: (n, f). Returns class logits and Tc predictions, each (K, n)."""
        h = x.unsqueeze(0).expand(self.members, -1, -1)
        if self.use_elements:
            # All element work happens on the 119-row vocabulary, not per formula token, so the
            # cost does not grow with formula length. ``share`` is the (n, 119) fraction matrix.
            element = p["emb"] + torch.tanh(torch.bmm(p["emb"], p["w_mix"]) + p["b_mix"])     # (K, 119, D)
            log_share = torch.log(share.clamp_min(1e-9))[None]                                 # absent -> -20.7
            first = torch.softmax(torch.bmm(element, p["query"]).transpose(1, 2) + log_share, dim=2)
            summary = torch.bmm(first, element)                                                # (K, n, D)
            # second look: attention conditioned on the formula's own summary
            relevance = torch.bmm(torch.bmm(summary, p["w_ctx"]), element.transpose(1, 2))     # (K, n, 119)
            pooled = torch.bmm(torch.softmax(relevance + log_share, dim=2), element)
            h = torch.cat([h, summary + pooled], dim=2)
        h = torch.bmm(h, p["w1"]) + p["b1"]
        h = h - h.mean(2, keepdim=True)                                   # LayerNorm, written out
        h = h * torch.rsqrt(h.square().mean(2, keepdim=True) + 1e-5) * p["g1"] + p["h1"]
        h = torch.nn.functional.silu(h)
        if train and self.dropout:
            h = h * (torch.rand_like(h) > self.dropout) / (1 - self.dropout)
        h = torch.nn.functional.silu(torch.bmm(h, p["w2"]) + p["b2"])
        return (torch.bmm(h, p["w_cls"]) + p["b_cls"]).squeeze(2), (torch.bmm(h, p["w_tc"]) + p["b_tc"]).squeeze(2)

    @staticmethod
    def _share(tokens: Tokens, torch):
        """(n, 119) atomic-fraction matrix from padded element tokens; column 0 is padding."""
        from .data import N_ELEMENT_SLOTS

        index, fraction = torch.from_numpy(tokens[0]), torch.from_numpy(tokens[1])
        share = torch.zeros(len(index), N_ELEMENT_SLOTS).scatter_add_(1, index, fraction)
        share[:, 0] = 0.0
        return share

    def _validation_masks(self, y: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
        """(K, n) booleans: True where a row is held out for that member's early stopping."""
        rng = np.random.default_rng(self.seed + 17)
        masks = np.zeros((self.members, len(y)), dtype=bool)
        keys = groups if groups is not None and len(set(groups)) > 5 else np.arange(len(y)).astype(str)
        unique, inverse = np.unique(keys, return_inverse=True)
        for member in range(self.members):
            for _ in range(20):  # redraw until both classes appear on each side
                held = rng.random(len(unique)) < self.validation_fraction
                mask = held[inverse]
                if len(np.unique(y[mask])) == 2 and len(np.unique(y[~mask])) == 2:
                    break
            masks[member] = mask
        return masks

    def fit(self, X, y, *, groups=None, tc=None, tokens: Tokens | None = None):
        import torch

        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        y = np.asarray(y, dtype=np.float32)
        if self.use_elements and tokens is None:
            raise ValueError("DeepEnsemble(use_elements=True) needs tokens=(el_index, el_fraction)")
        self._fit_preprocess(X)
        x = torch.from_numpy(self._transform(X))
        share = self._share(tokens, torch) if self.use_elements else None
        target = torch.from_numpy(y)
        tc = np.full(len(y), np.nan) if tc is None else np.asarray(tc, dtype=float)
        has_tc = torch.from_numpy((np.isfinite(tc) & (y > 0)).astype(np.float32))
        log_tc = torch.from_numpy(np.nan_to_num(np.log1p(np.clip(tc, 0, None)), nan=0.0).astype(np.float32))
        held = torch.from_numpy(self._validation_masks(y, groups))
        fit_weight = (~held).float()

        params = self._init(x.shape[1], torch)
        decay = [v for k, v in params.items() if k.startswith("w")]
        rest = [v for k, v in params.items() if not k.startswith("w")]
        optimiser = torch.optim.AdamW([{"params": decay, "weight_decay": self.weight_decay},
                                       {"params": rest, "weight_decay": 0.0}], lr=self.lr)
        steps = self.max_epochs * math.ceil(len(y) / self.batch_size)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=self.lr, total_steps=steps)
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        huber = torch.nn.functional.huber_loss

        def losses(rows, weight, train: bool):
            """Per-member loss (K,), each row weighted by ``weight`` (K, n_rows)."""
            logit, tc_hat = self._forward(params, x[rows], share[rows] if self.use_elements else None, torch, train)
            cls = bce(logit, target[rows].expand_as(logit), reduction="none")
            total = (cls * weight).sum(1) / weight.sum(1).clamp_min(1)
            if self.tc_weight > 0:
                w = weight * has_tc[rows]
                reg = huber(tc_hat, log_tc[rows].expand_as(tc_hat), reduction="none")
                total = total + self.tc_weight * (reg * w).sum(1) / w.sum(1).clamp_min(1)
            return total

        rng = np.random.default_rng(self.seed)
        best = torch.full((self.members,), float("inf"))
        best_params = {k: v.detach().clone() for k, v in params.items()}
        stale = torch.zeros(self.members, dtype=torch.long)
        everything = torch.arange(len(y))
        for epoch in range(self.max_epochs):
            order = torch.from_numpy(rng.permutation(len(y)))
            for start in range(0, len(order), self.batch_size):
                rows = order[start:start + self.batch_size]
                optimiser.zero_grad()
                losses(rows, fit_weight[:, rows], True).sum().backward()
                optimiser.step()
                schedule.step()
            with torch.no_grad():
                validation = losses(everything, held.float(), False)
            improved = validation < best - 1e-5
            best = torch.where(improved, validation, best)
            stale = torch.where(improved, torch.zeros_like(stale), stale + 1)
            for key, value in params.items():
                best_params[key][improved] = value.detach()[improved]
            if bool((stale >= self.patience).all()):
                break
        self.params_, self.epochs_, self.validation_loss_ = best_params, epoch + 1, best.numpy()
        return self

    def predict(self, X, *, tokens: Tokens | None = None):
        import torch

        x = torch.from_numpy(self._transform(X))
        share = self._share(tokens, torch) if self.use_elements else None
        with torch.no_grad():
            each = torch.sigmoid(self._forward(self.params_, x, share, torch, False)[0]).numpy().astype(float)
        return each.mean(axis=0), each


class Blend:
    """Unweighted average of member models. Nothing is tuned on any test fold."""

    def __init__(self, *models: Any):
        self.models = models

    def fit(self, X, y, *, groups=None, tc=None, tokens=None):
        for model in self.models:
            model.fit(X, y, groups=groups, tc=tc, tokens=tokens)
        return self

    def predict(self, X, *, tokens=None):
        parts = [model.predict(X, tokens=tokens) for model in self.models]
        mean = np.mean([p[0] for p in parts], axis=0)
        return mean, np.concatenate([p[1] for p in parts])


def registry(seed: int = 0, n_jobs: int = 4) -> dict[str, Any]:
    """Name -> factory. Add a model here and every stage can use it."""
    return {
        "legacy_boosting": lambda: TreeBag(members=1, seed=seed, bootstrap=False),
        "tree_bag": lambda: TreeBag(members=8, seed=seed, n_jobs=n_jobs),
        "deep_ensemble": lambda: DeepEnsemble(members=8, seed=seed),
        "deep_ensemble_no_elements": lambda: DeepEnsemble(members=8, seed=seed, use_elements=False),
        "blend": lambda: Blend(TreeBag(members=8, seed=seed, n_jobs=n_jobs), DeepEnsemble(members=8, seed=seed)),
    }
