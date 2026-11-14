"""Training, prediction and scoring for the from-scratch composition network.

Vocabulary used here:

* a **step** is one update of the weights from one batch of compositions;
* an **epoch** is one pass over all the training compositions;
* **early stopping** watches the loss on a held-back slice of the training
  groups each epoch and keeps the weights from the best epoch, so the network
  stops before it starts memorizing;
* **weight averaging** keeps a running average of the weights over recent
  steps and uses that average for validation and prediction. Single steps
  are noisy; their average is steadier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from . import layers as L
from .data import stable_hash
from .model import CompositionNet, Config
from .optim import AdamW, clip_by_global_norm, warmup_cosine


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    patience: int = 15              # epochs without improvement before stopping
    min_epochs: int = 10
    tc_weight: float = 1.0          # weight of the Tc loss relative to the class loss
    huber_delta: float = 1.0
    clip_norm: float = 5.0
    inner_validation_percent: int = 12
    aux_weight: float = 0.5         # weight of the auxiliary-target loss, when the data carries such targets
    ema_decay: float = 0.0          # weight averaging: share of the old average kept at each step; 0 turns it off
    init_class_bias: bool = False   # start the class output at the training share of superconductors
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Arrays:
    """One set of compositions in network form, with targets where known."""

    index: np.ndarray
    fraction: np.ndarray
    mask: np.ndarray
    label: np.ndarray               # 1, 0 or nan (no class label)
    tc_k: np.ndarray                # kelvin, nan where there is no Tc target
    group: np.ndarray               # group name per row
    extra: np.ndarray | None = None  # whole-formula features, already scaled, for models with n_global
    aux: np.ndarray | None = None    # auxiliary targets, already scaled, nan where unknown, for models with n_aux

    def take(self, rows: np.ndarray) -> "Arrays":
        return Arrays(*[None if value is None else value[rows]
                        for value in (getattr(self, f.name) for f in fields(self))])

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def join(cls, parts: list["Arrays"]) -> "Arrays":
        """Stack several sets. A part without auxiliary targets contributes nan (unknown) for them."""
        out = []
        for f in fields(cls):
            values = [getattr(part, f.name) for part in parts]
            if all(value is None for value in values):
                out.append(None)
                continue
            if f.name == "aux":
                width = next(value.shape[1] for value in values if value is not None)
                values = [np.full((len(part), width), np.nan) if value is None else value
                          for part, value in zip(parts, values)]
            elif any(value is None for value in values):
                raise ValueError(f"some parts have {f.name} and some do not")
            out.append(np.concatenate(values))
        return cls(*out)


@dataclass
class TcScale:
    """Tc is learned as standardized log(1 + Tc); this holds the mean and spread used."""

    mean: float
    std: float

    @classmethod
    def fit(cls, tc_k: np.ndarray) -> "TcScale":
        values = np.log1p(tc_k[np.isfinite(tc_k)])
        return cls(float(values.mean()), float(values.std()) or 1.0)

    def encode(self, tc_k: np.ndarray) -> np.ndarray:
        return (np.log1p(np.nan_to_num(tc_k, nan=0.0)) - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return np.clip(np.expm1(np.clip(z * self.std + self.mean, -10.0, 8.0)), 0.0, None)


def inner_split(groups: np.ndarray, percent: int, seed: int, max_share: float = 0.05
                ) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (fit, validate): whole groups are held back for early stopping.

    A group holding more than ``max_share`` of the rows is never held back: one
    huge group would otherwise decide, by the luck of its hash, whether the
    validation set is 1% or 40% of the data. Groups are never cut in two; the
    only exception is an input with a single group, where every n-th row is held.
    """
    names, counts = np.unique(groups, return_counts=True)
    if len(names) < 2:
        held = np.arange(len(groups)) % max(2, round(100 / max(percent, 1))) == 0
        return np.flatnonzero(~held), np.flatnonzero(held)
    allowed = [name for name, count in zip(names, counts) if count <= max_share * len(groups)]
    if not allowed:                                   # a small table where every group is "large": spare only the largest
        allowed = [name for name in names if name != names[np.argmax(counts)]]
    hashes = {name: stable_hash(f"inner:{seed}:{name}") for name in allowed}
    chosen = {name for name, value in hashes.items() if value % 100 < percent} or {min(hashes, key=hashes.get)}
    if len(chosen) == len(names):                     # leave at least one group to fit on
        chosen.discard(max(hashes, key=hashes.get))
    held = np.isin(groups, sorted(chosen))
    return np.flatnonzero(~held), np.flatnonzero(held)


def _losses(model: CompositionNet, data: Arrays, scale: TcScale, config: TrainConfig,
            backward: bool) -> tuple[float, float, float]:
    """(total, class, Tc) losses for one batch; with ``backward`` the gradients are left in the model.

    The auxiliary loss, when the model has auxiliary outputs and the batch has
    targets for them, is part of the total only.
    """
    logit, tc_z = model.forward(data.index, data.fraction, data.mask, data.extra)
    has_label = np.isfinite(data.label)
    has_tc = np.isfinite(data.tc_k) & (data.label == 1)
    dtype = logit.dtype
    loss_c, d_logit = L.bce_with_logits(logit, np.nan_to_num(data.label, nan=0.0).astype(dtype),
                                        has_label.astype(dtype))
    loss_r, d_tc = L.huber(tc_z, scale.encode(data.tc_k).astype(dtype), has_tc.astype(dtype), config.huber_delta)
    loss_a, d_aux = 0.0, None
    if model.config.n_aux and data.aux is not None:
        known = np.isfinite(data.aux)
        loss_a, d_flat = L.huber(model.aux.ravel(), np.nan_to_num(data.aux, nan=0.0).astype(dtype).ravel(),
                                 known.astype(dtype).ravel(), config.huber_delta)
        d_aux = config.aux_weight * d_flat.reshape(data.aux.shape)
    if backward:
        model.backward(d_logit, config.tc_weight * d_tc, d_aux)
    return loss_c + config.tc_weight * loss_r + config.aux_weight * loss_a, loss_c, loss_r


def evaluate_loss(model: CompositionNet, data: Arrays, scale: TcScale, config: TrainConfig,
                  batch_size: int = 2048) -> dict[str, float]:
    """Mean losses in evaluation mode (dropout off), weighted by how many rows carry each target."""
    was_training = model.layers["input_drop"].training
    model.train(False)
    n_label = n_tc = 0
    sum_c = sum_r = 0.0
    for start in range(0, len(data), batch_size):
        part = data.take(np.arange(start, min(start + batch_size, len(data))))
        _, loss_c, loss_r = _losses(model, part, scale, config, backward=False)
        labelled = int(np.isfinite(part.label).sum())
        with_tc = int((np.isfinite(part.tc_k) & (part.label == 1)).sum())
        sum_c += loss_c * labelled
        sum_r += loss_r * with_tc
        n_label += labelled
        n_tc += with_tc
    model.train(was_training)
    loss_c = sum_c / max(n_label, 1)
    loss_r = sum_r / max(n_tc, 1)
    return {"total": loss_c + config.tc_weight * loss_r, "class": loss_c, "tc": loss_r}


def fit(model: CompositionNet, train: Arrays, config: TrainConfig, valid: Arrays | None = None,
        scale: TcScale | None = None, log=None) -> dict:
    """Train with AdamW and early stopping. Leaves the best weights in ``model``.

    Early stopping watches ``valid`` when it is given; otherwise whole groups
    are held back from ``train`` for the purpose.
    """
    rng = np.random.default_rng(config.seed)
    if valid is None:
        fit_rows, valid_rows = inner_split(train.group, config.inner_validation_percent, config.seed)
        fit_data, valid_data = train.take(fit_rows), train.take(valid_rows)
    else:
        fit_data, valid_data = train, valid
    scale = scale or TcScale.fit(fit_data.tc_k[fit_data.label == 1])
    if config.init_class_bias:
        rate = float(np.clip(np.nanmean(fit_data.label), 1e-3, 1 - 1e-3))
        model.layers["head_class"].params["b"][...] = np.log(rate / (1.0 - rate))
    average = model.state() if config.ema_decay > 0 else None
    names = [name for name, _, _ in model.parameters()]
    optimizer = AdamW(names, lr=config.lr, weight_decay=config.weight_decay, decay_names=model.decay_names())
    steps_per_epoch = int(np.ceil(len(fit_data) / config.batch_size))
    total_steps = config.epochs * steps_per_epoch
    warmup = config.warmup_epochs * steps_per_epoch
    history, best, best_state, best_epoch, since_best, step = [], float("inf"), model.state(), 0, 0, 0
    for epoch in range(config.epochs):
        model.train(True)
        order = rng.permutation(len(fit_data))
        running = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start:start + config.batch_size]
            loss, _, _ = _losses(model, fit_data.take(rows), scale, config, backward=True)
            parameters = model.parameters()
            clip_by_global_norm([grad for _, _, grad in parameters], config.clip_norm)
            optimizer.step(parameters, lr=warmup_cosine(step, total_steps, config.lr, warmup, floor=config.lr * 0.02))
            running += loss * len(rows)
            step += 1
            if average is not None:
                keep = min(config.ema_decay, step / (step + 9.0))      # a short memory at first, so the start is forgotten
                for name, value, _ in parameters:
                    average[name] *= keep
                    average[name] += (1.0 - keep) * value
        if average is not None:                                       # validate the averaged weights, then carry on
            current = model.state()
            model.load_state(average)
        valid = evaluate_loss(model, valid_data, scale, config)
        record = {"epoch": epoch + 1, "train_loss": running / len(fit_data), "valid_loss": valid["total"],
                  "valid_class": valid["class"], "valid_tc": valid["tc"]}
        history.append(record)
        if log:
            log(record)
        if valid["total"] < best - 1e-5:                 # gains smaller than this do not count as progress
            best, best_state, best_epoch, since_best = valid["total"], model.state(), epoch + 1, 0
        else:
            since_best += 1
        if average is not None:
            model.load_state(current)
        if epoch + 1 >= config.min_epochs and since_best >= config.patience:
            break
    model.load_state(best_state)
    model.train(False)
    return {"history": history, "best_epoch": best_epoch, "best_valid_loss": best, "epochs_run": len(history),
            "fit_rows": int(len(fit_data)), "inner_validation_rows": int(len(valid_data)),
            "tc_scale": {"mean": scale.mean, "std": scale.std}}


def predict(model: CompositionNet, data: Arrays, scale: TcScale) -> tuple[np.ndarray, np.ndarray]:
    """(probability of superconductivity, predicted Tc in kelvin if it superconducts)."""
    logit, tc_z = model.predict(data.index, data.fraction, data.mask, data.extra)
    return L.sigmoid(logit.astype(np.float64)), scale.decode(tc_z.astype(np.float64))


# ---------------------------------------------------------------------------
# Scores, in plain numpy so they can be tested against scikit-learn and scipy.

def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with ties given the mean of the ranks they span."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def roc_auc(label: np.ndarray, score: np.ndarray) -> float:
    """Chance that a random superconductor is scored above a random non-superconductor."""
    label = np.asarray(label).astype(bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = average_ranks(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation: 1 when two lists put the items in the same order."""
    if len(a) < 3:
        return float("nan")
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def score(label: np.ndarray, tc_k: np.ndarray, probability: np.ndarray, tc_pred: np.ndarray) -> dict[str, float]:
    """All the numbers reported for one set of predictions."""
    labelled = np.isfinite(label)
    sc = (label == 1) & np.isfinite(tc_k)
    expected = probability * tc_pred
    truth_all = np.where(label == 1, np.nan_to_num(tc_k, nan=0.0), 0.0)
    out = {
        "rows": int(len(label)), "labelled_rows": int(labelled.sum()), "superconductor_rows": int(sc.sum()),
        "auc": roc_auc(label[labelled], probability[labelled]),
        "brier": float(np.mean((probability[labelled] - label[labelled]) ** 2)) if labelled.any() else float("nan"),
        "tc_mae_k": float(np.mean(np.abs(tc_pred[sc] - tc_k[sc]))) if sc.any() else float("nan"),
        "tc_log_mae": float(np.mean(np.abs(np.log1p(tc_pred[sc]) - np.log1p(tc_k[sc])))) if sc.any() else float("nan"),
        "tc_spearman": spearman(tc_pred[sc], tc_k[sc]) if sc.any() else float("nan"),
        # Ranking everything by probability x Tc against the true Tc (0 for non-superconductors):
        "expected_tc_spearman": spearman(expected[labelled], truth_all[labelled]) if labelled.any() else float("nan"),
    }
    if labelled.sum() >= 20:
        # Of the true top tenth by Tc, the share found in the model's top tenth by expected Tc.
        k = max(1, int(round(0.1 * labelled.sum())))
        truth, guess = truth_all[labelled], expected[labelled]
        top_true = set(np.argsort(-truth, kind="mergesort")[:k])
        top_guess = set(np.argsort(-guess, kind="mergesort")[:k])
        out["top_decile_recall"] = len(top_true & top_guess) / k
    return out


def build_model(config: Config, element_features: np.ndarray, seed: int) -> CompositionNet:
    return CompositionNet(config, element_features, seed=seed)
